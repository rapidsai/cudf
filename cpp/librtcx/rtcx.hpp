/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sha256.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#define RTCX_DEFER__CONCATENATE_DETAIL(x, y) x##y
#define RTCX_DEFER__CONCATENATE(x, y)        RTCX_DEFER__CONCATENATE_DETAIL(x, y)
#define RTCX_DEFER(...)                      ::rtcx::defer RTCX_DEFER__CONCATENATE(defer_, __COUNTER__)(__VA_ARGS__)

extern "C" {
typedef struct CUlib_st* CUlibrary;    // NOLINT(modernize-use-using)
typedef struct CUkern_st* CUkernel;    // NOLINT(modernize-use-using)
typedef struct CUstream_st* CUstream;  // NOLINT(modernize-use-using)
}

namespace rtcx {

inline constexpr std::size_t CACHELINE_ALIGNMENT =
  64;  // = std::hardware_destructive_interference_size */

/**
 * @brief RAII utility to execute a callable at the end of a scope.
 */
template <typename T>
struct defer {
 private:
  T func_;

 public:
  template <typename... Args>
  defer(Args&&... args) : func_{static_cast<Args&&>(args)...}
  {
  }
  defer(defer const&)            = delete;
  defer& operator=(defer const&) = delete;
  defer(defer&&)                 = delete;
  defer& operator=(defer&&)      = delete;
  ~defer() { func_(); }
};

template <typename T>
defer(T) -> defer<T>;

template <typename Signature>
struct func;

/**
 * @brief Zero-copy, type-erased reference to a callable entity (e.g. lambda, function pointer) that
 * can be invoked with the given signature.
 */
template <typename R, typename... Args>
struct [[nodiscard]] func<R(Args...)> {
 private:
  void* _user_data;
  R (*_thunk)(void*, Args...);

 public:
  func(void* user_data, R (*thunk)(void*, Args...)) : _user_data{user_data}, _thunk{thunk} {}

  func(R (*func_ptr)(Args...))
    : _user_data{reinterpret_cast<void*>(func_ptr)},
      _thunk{+[](void* user_data, Args... args) -> R {
        auto func = reinterpret_cast<R (*)(Args...)>(user_data);
        return func(std::forward<Args>(args)...);
      }}
  {
  }

  R operator()(Args... args) const { return _thunk(_user_data, std::forward<Args>(args)...); }

  template <typename Lambda>
  static func from_functor(Lambda& lambda)
  {
    return func{static_cast<void*>(std::addressof(lambda)),
                +[](void* user_data, Args... args) -> R {
                  auto& lambda = *static_cast<std::remove_reference_t<Lambda>*>(user_data);
                  return lambda(std::forward<Args>(args)...);
                }};
  }
};

template <typename R, typename... Args>
func(void*, R (*)(void*, Args...)) -> func<R(Args...)>;

template <typename R, typename... Args>
func(R (*)(Args...)) -> func<R(Args...)>;

struct [[nodiscard]] sha256_hasher {
  constexpr std::uint64_t operator()(sha256 const& obj) const
  {
    struct u64x4 {
      alignas(16) std::uint64_t v[4];  // NOLINT(modernize-avoid-c-arrays)
    };

    auto value = std::bit_cast<u64x4>(obj);
    auto h0    = value.v[0];
    auto h1    = value.v[1];
    auto h2    = value.v[2];
    auto h3    = value.v[3];

    auto mix = [](std::uint64_t seed, std::uint64_t v) {
      seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
      return seed;
    };

    return mix(mix(mix(h0, h1), h2), h3);
  }
};

enum class binary_type : std::int8_t { LTO_IR = 0, CUBIN = 2, FATBIN = 3, PTX = 4 };

/**
 * @brief An heap-allocated statically-sized buffer. Its contents are not guaranteed to be
 * initialized.
 */
template <typename T>
  requires(!std::is_const_v<T> && std::is_trivially_copyable_v<T> &&
           std::is_trivially_destructible_v<T>)
struct buffer {
 private:
  T* _data;
  std::size_t _size;

  buffer(T* data, std::size_t size) : _data(data), _size(size) {}

 public:
  /**
   * @brief Creates a new buffer of the given size, with uninitialized contents.
   */
  static buffer make(std::size_t size)
  {
    T* data = static_cast<T*>(malloc(size * sizeof(T)));
    if (data == nullptr) { throw std::bad_alloc(); }
    return buffer{data, size};
  }

  buffer() : buffer{nullptr, 0} {}  //< Default constructor. Creates an empty buffer

  buffer(buffer const&) = delete;

  buffer& operator=(buffer const&) = delete;

  /**
   * @brief Move constructor. Transfers ownership of the buffer from the source to the new object.
   * After the move, the source buffer is left in an empty state (data pointer is null and size is
   * zero).
   */
  buffer(buffer&& other) noexcept : _data(other._data), _size(other._size)
  {
    other._data = nullptr;
    other._size = 0;
  }

  /**
   * @brief Move assignment operator. Transfers ownership of the buffer from the source to the
   * current object.
   */
  buffer& operator=(buffer&& other) noexcept
  {
    if (this == &other) [[unlikely]] { return *this; }
    this->~buffer();
    new (this) buffer(std::move(other));
    return *this;
  }

  ~buffer() noexcept { free(_data); }

  /**
   * @brief Returns a pointer to the buffer's data
   * @return A pointer to the buffer's data
   */
  [[nodiscard]] T* data() const { return _data; }

  /**
   * @brief Returns the size of the buffer
   * @return The size of the buffer in number of elements
   */
  [[nodiscard]] std::size_t size() const { return _size; }

  /**
   * @brief Returns an iterator to the beginning of the buffer
   * @return An iterator to the beginning of the buffer
   */
  [[nodiscard]] T* begin() { return _data; }

  /**
   * @brief Returns an iterator to the end of the buffer
   * @return An iterator to the end of the buffer
   */
  [[nodiscard]] T* end() { return _data + _size; }

  /**
   * @brief Returns a const iterator to the beginning of the buffer
   * @return A const iterator to the beginning of the buffer
   */
  [[nodiscard]] T const* begin() const { return _data; }

  /**
   * @brief Returns a const iterator to the end of the buffer
   * @return A const iterator to the end of the buffer
   */
  [[nodiscard]] T const* end() const { return _data + _size; }

  /**
   * @brief Returns a const iterator to the beginning of the buffer
   * @return A const iterator to the beginning of the buffer
   */
  [[nodiscard]] T const* cbegin() const { return _data; }

  /**
   * @brief Returns a const iterator to the end of the buffer
   * @return A const iterator to the end of the buffer
   */
  [[nodiscard]] T const* cend() const { return _data + _size; }

  /**
   * @brief Releases ownership of the buffer's data and returns a pointer to it. After calling this
   * function, the buffer is left in an empty state (data pointer is null and size is zero).
   * @return A pointer to the buffer's data
   */
  [[nodiscard]] T* release()
  {
    T* data = _data;
    _data   = nullptr;
    _size   = 0;
    return data;
  }
};

using byte_buffer = buffer<std::uint8_t>;

/**
 * @brief Represents an immutable blob view
 * @details Manages the lifetime of the binary data via a user-provided deallocator function. This
 * enables zero-copy view of binary data stored in various forms (e.g., std::vector, mmap'd file,
 * etc.).
 */
struct [[nodiscard]] blob_t {
 private:
  using deallocator = func<void(std::uint8_t const*, std::size_t)>;

  static void noop_deallocator(std::uint8_t const*, std::size_t) {}

  std::uint8_t const* data_;
  std::size_t size_;
  deallocator deallocator_;

  blob_t(std::uint8_t const* data, std::size_t size, deallocator deallocator)
    : data_(data), size_(size), deallocator_(deallocator)
  {
  }

 public:
  blob_t() : data_(nullptr), size_(0), deallocator_(noop_deallocator) {}

  blob_t(blob_t const&)            = delete;
  blob_t& operator=(blob_t const&) = delete;

  blob_t(blob_t&& other) noexcept
    : data_(other.data_), size_(other.size_), deallocator_(other.deallocator_)
  {
    other.data_        = nullptr;
    other.size_        = 0;
    other.deallocator_ = noop_deallocator;
  }

  blob_t& operator=(blob_t&& other) noexcept
  {
    if (this == &other) [[unlikely]] { return *this; }
    this->~blob_t();
    new (this) blob_t(std::move(other));
    return *this;
  }

  ~blob_t() { deallocator_(data_, size_); }

  [[nodiscard]] std::span<std::uint8_t const> view() const { return {data_, size_}; }

  static blob_t from_parts(std::uint8_t const* data, std::size_t size, deallocator deallocator)
  {
    return blob_t{data, size, deallocator};
  }

  static blob_t from_buffer(byte_buffer&& buffer);

  static blob_t from_static_data(std::span<std::uint8_t const> data);

  static std::optional<blob_t> from_file(char const* path);
};

using blob = std::shared_ptr<blob_t>;

/**
 * @brief Represents the occupancy configuration for a kernel. This information can be used to
 * optimize kernel launches for maximum performance on the GPU.
 */
struct [[nodiscard]] kernel_occupancy_config {
  std::uint32_t min_grid_size = 0;  //< Minimum grid size to achieve the maximum occupancy
  std::uint32_t block_size =
    0;  //< Number of threads per block to achieve the min_grid_size occupancy
};

/**
 * @brief Represents the dimensions of a CUDA grid or block, with x, y, and z components. This
 * struct is used to specify the configuration of kernel launches on the GPU.
 */
struct cuda_dim3 {
  std::uint32_t x = 1;  //< Value for the x dimension
  std::uint32_t y = 1;  //< Value for the y dimension
  std::uint32_t z = 1;  //< Value for the z dimension

  [[nodiscard]] constexpr bool is_valid() const { return x > 0 && y > 0 && z > 0; }
};

/**
 * @brief Represents a compiled kernel that can be launched on the GPU.
 */
struct [[nodiscard]] kernel_ref {
 private:
  CUkernel handle_;

 public:
  explicit kernel_ref(CUkernel handle) : handle_(handle) {}

  /**
   * @brief Computes the maximum occupancy configuration for the kernel, given the specified dynamic
   * shared memory usage and block size limit. This function queries the CUDA driver for the optimal
   * block size and minimum grid size to achieve maximum occupancy of the kernel on the GPU.
   */
  kernel_occupancy_config max_occupancy_config(std::size_t dynamic_shared_memory_bytes,
                                               std::int32_t block_size_limit) const;

  /**
   * @brief Launches the kernel on the GPU with the specified grid and block dimensions,
   * dynamic shared memory size, stream, and kernel parameters. This function wraps the CUDA driver
   * kernel launch API, providing a convenient interface for executing the kernel with the desired
   * configuration.
   * @param grid_dim The dimensions of the grid
   * @param block_dim The dimensions of the block
   * @param shared_mem_bytes The amount of dynamic shared memory (in bytes) to allocate for the
   * kernel
   * @param stream The CUDA stream on which to launch the kernel
   * @param kernel_params A pointer to an array of pointers representing the kernel parameters to be
   * passed to the kernel at launch time
   */
  void launch(cuda_dim3 grid_dim,
              cuda_dim3 block_dim,
              std::uint32_t shared_mem_bytes,
              CUstream stream,
              void** kernel_params) const;

  /**
   * @brief Launches the kernel on the GPU in cooperative mode with the specified grid and block
   * dimensions, dynamic shared memory size, stream, and kernel parameters. This function wraps the
   * CUDA driver cooperative kernel launch API, providing a convenient interface for executing the
   * kernel with the desired configuration.
   * @param grid_dim The dimensions of the grid
   * @param block_dim The dimensions of the block
   * @param shared_mem_bytes The amount of dynamic shared memory (in bytes) to allocate for the
   * kernel
   * @param stream The CUDA stream on which to launch the kernel
   * @param kernel_params A pointer to an array of pointers representing the kernel parameters to be
   * passed to the kernel at launch time
   */
  void launch_cooperative(cuda_dim3 grid_dim,
                          cuda_dim3 block_dim,
                          std::uint32_t shared_mem_bytes,
                          CUstream stream,
                          void** kernel_params) const;

  /**
   * @brief Retrieves the underlying CUDA kernel handle
   * @return The CUDA kernel handle associated with this kernel reference
   */
  [[nodiscard]] CUkernel get() const { return handle_; }
};

/**
 * @brief Represents a loaded RTC library containing compiled kernels
 */
struct [[nodiscard]] library_t {
 private:
  CUlibrary handle_;

 public:
  explicit library_t(CUlibrary handle) : handle_(handle) {}
  library_t(library_t const&)            = delete;
  library_t(library_t&&)                 = delete;
  library_t& operator=(library_t const&) = delete;
  library_t& operator=(library_t&&)      = delete;
  ~library_t();

  /**
   * @brief Retrieves the underlying CUDA library handle
   */
  [[nodiscard]] CUlibrary get() const { return handle_; }

  /**
   * @brief Retrieve a kernel from the library by name
   */
  [[nodiscard]] kernel_ref get_kernel(char const* name) const;
};

using library = std::shared_ptr<library_t>;

/**
 * @brief Parameters for compiling source code into a binary blob using NVRTC
 */
struct [[nodiscard]] compile_params {
  char const* name   = nullptr;                            //< Debug name for the compilation unit
  char const* source = nullptr;                            //< Source code to be compiled
  std::span<char const* const> header_include_names = {};  //< Header file names
  std::span<char const* const> headers              = {};  //< Header file contents
  std::span<char const* const> options              = {};  //< NVRTC compilation options
  std::span<char const* const> name_expressions     = {};  //< Name expressions to be instantiated
  binary_type target_type                           = binary_type::LTO_IR;  //<  Output binary type
};

/**
 * @brief Represents a binary fragment in memory to be linked into a library
 */
struct memory_fragment {
  std::span<std::uint8_t const> data = {};                  //< Binary data for the fragment
  binary_type type                   = binary_type::CUBIN;  //< Binary type of the fragment data
  char const* name                   = nullptr;             //< Debug name for the fragment
};

/**
 * @brief Represents a binary fragment to be linked into a library
 */
struct file_fragment {
  char const* path = nullptr;             //< Path to the binary fragment file
  binary_type type = binary_type::CUBIN;  //< Binary type of the fragment data
};

/**
 * @brief Parameters for linking multiple compiled fragments into a single library
 */
struct [[nodiscard]] link_params {
  char const* name                              = nullptr;  //< Debug name for the linked library
  binary_type output_type                       = binary_type::CUBIN;  //< Output binary type
  std::span<file_fragment const> file_fragments = {};  //< Binary data for each fragment
  std::span<memory_fragment const> memory_fragments =
    {};                                            //< Memory-resident binary fragments to link
  std::span<char const* const> link_options = {};  //< NVJITLink options
};

namespace detail {

template <typename T>
struct alignas(CACHELINE_ALIGNMENT) lru_memory_cache {
  struct entry {
    std::uint64_t last_touched_tick = 0;
    T value;

    void hit(std::uint64_t tick) { last_touched_tick = tick; }
  };

  std::unordered_map<sha256, entry, sha256_hasher> entries_ = {};
  std::size_t limit_;

  explicit lru_memory_cache(std::size_t limit) : limit_{limit}
  {
    // reserve space to avoid rehashing
    entries_.reserve(limit * 2);
  }

  void purge()
  {
    if (entries_.empty()) { return; }

    auto num_to_purge = (entries_.size() + 1) / 2;

    std::vector<std::pair<sha256, std::uint64_t>> rankings;
    rankings.reserve(entries_.size());

    for (auto& [key, entry] : entries_) {
      rankings.emplace_back(key, entry.last_touched_tick);
    }

    std::sort(
      rankings.begin(), rankings.end(), [](auto& a, auto& b) { return a.second < b.second; });

    // purge least recently used half
    rankings.resize(num_to_purge);

    for (auto [key, _] : rankings) {
      entries_.erase(key);
    }
  }

  void insert(sha256 const& sha, T&& value, std::uint64_t tick)
  {
    if (limit_ == 0) { return; }

    if ((entries_.size() + 1) > limit_) { purge(); }

    entries_.emplace(sha, entry{tick, std::move(value)});
  }
};

struct cache_stats_counter {
  struct alignas(CACHELINE_ALIGNMENT) entry {
    std::uint64_t value_ = 0;

    void incr()
    {
      std::atomic_ref c{value_};
      c.fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard]] std::uint64_t get() const
    {
      std::atomic_ref c{value_};
      return c.load(std::memory_order_relaxed);
    }

    void reset()
    {
      std::atomic_ref c{value_};
      c.store(0, std::memory_order_relaxed);
    }
  };

  entry blob_mem_hits;
  entry blob_mem_misses;
  entry blob_disk_hits;
  entry blob_disk_misses;
  entry library_mem_hits;
  entry library_mem_misses;
  entry library_disk_hits;
  entry library_disk_misses;
};

};  // namespace detail

struct [[nodiscard]] cache_stats {
  std::uint64_t blob_mem_hits       = 0;
  std::uint64_t blob_mem_misses     = 0;
  std::uint64_t blob_disk_hits      = 0;
  std::uint64_t blob_disk_misses    = 0;
  std::uint64_t library_mem_hits    = 0;
  std::uint64_t library_mem_misses  = 0;
  std::uint64_t library_disk_hits   = 0;
  std::uint64_t library_disk_misses = 0;
};

struct [[nodiscard]] cache_limits {
  std::uint32_t num_mem_blobs     = 16'384;
  std::uint32_t num_mem_libraries = 16'384;
  std::uint32_t num_disk_entries  = 131'072;
};

using blob_compile_func    = func<blob()>;
using library_compile_func = func<std::tuple<library, blob>()>;

/**
 * @brief Thread-safe user-managed compile cache for compiled blobs and libraries
 *
 * @details Provides in-memory and on-disk caching of compiled RTC artifacts.
 * The cache uses an LRU eviction policy when the number of cached items exceeds user-defined
 * limits. In-memory cache is implemented using a thread-safe LRU cache that supports concurrent
 * reads. The on-disk cache also allows concurrent access and stores cached items in files within a
 * specified directory. Writing to disk is atomic to prevent corruption from concurrent writes or
 * process interruptions. In addition, the cache maintains statistics on cache hits and misses for
 * both in-memory and on-disk caches to help monitor cache performance in benchmarking and
 * debugging. The interface is zero-copy, using shared pointers, mmap, and spans to avoid
 * unnecessary data copying across threads and disk.
 */
struct cache_t {  // NOLINT
 private:
  bool enabled_;

  std::string cache_dir_;

  std::string tmp_dir_;

  cache_limits limits_;

  std::mutex lock_;

  detail::lru_memory_cache<std::shared_future<blob>> blobs_cache_;

  detail::lru_memory_cache<std::shared_future<library>> libraries_cache_;

  detail::cache_stats_counter counter_;

  alignas(CACHELINE_ALIGNMENT) std::uint64_t tick_;  // NOLINT(modernize-use-default-member-init)

 public:
  /**
   * @brief Construct a new cache_t object with the specified cache directory, limits, and options
   * for preloading and enabling the cache.
   * @param cache_dir The directory path to be used for on-disk caching of compiled blobs and
   * libraries
   * @param tmp_dir The directory path to be used for temporary files during atomic writes to the
   * on-disk cache
   * @param limits A cache_limits struct specifying the maximum number of blobs and libraries to
   * store in the cache before eviction occurs
   * @param preload A boolean flag indicating whether to preload the cache from disk during
   * initialization, allowing for faster retrieval of previously compiled kernels at runtime
   * @param disable A boolean flag indicating whether to disable the cache entirely, preventing any
   * caching of compiled blobs and libraries in memory
   * @param materialize_all A boolean flag indicating whether to make the compiled kernels fully
   * materialized in memory during preloading to improve runtime stability at the cost of increased
   * memory usage
   */
  cache_t(std::string cache_dir,
          std::string tmp_dir,
          cache_limits const& limits,
          bool preload,
          bool disable);
  cache_t(cache_t const&)            = delete;
  cache_t& operator=(cache_t const&) = delete;
  cache_t(cache_t&&)                 = delete;
  cache_t& operator=(cache_t&&)      = delete;
  ~cache_t()                         = default;

  /**
   * @brief Get the directory path used for on-disk caching
   * @return String reference to the cache directory path
   */
  [[nodiscard]] std::string const& get_cache_dir();

  /**
   * @brief Get the directory path used for temporary files during atomic writes to the on-disk
   * cache
   * @return String reference to the temporary directory path
   */
  [[nodiscard]] std::string const& get_tmp_dir();

  /**
   * @brief Query the cache for a compiled blob by its SHA-256 hash, or insert it if not present
   * @param sha SHA-256 hash of the blob to query or insert
   * @param compile Function to compile the blob if it's not found in the cache
   * @return A shared future that will hold the compiled blob once it's available
   */
  [[nodiscard]] std::shared_future<blob> get_or_add_blob(sha256 const& sha,
                                                         blob_compile_func compile);

  /**
   * @brief Query the cache for a compiled library by its SHA-256 hash and binary type, or insert
   * it if not present
   * @param sha SHA-256 hash of the library to query or insert
   * @param type Binary type of the library (e.g., CUBIN, PTX)
   * @param compile Function to compile the library if it's not found in the cache
   * @return A shared future that will hold the compiled library once it's available
   */
  [[nodiscard]] std::shared_future<library> get_or_add_library(sha256 const& sha,
                                                               library_compile_func compile);

  /**
   * @brief Retrieve current cache performance statistics, including hits and misses for both
   * in-memory and on-disk caches
   *
   * @return A cache_statistics struct containing the current cache performance metrics
   */
  cache_stats get_stats();

  /**
   * @brief Clear the current cache performance statistics, resetting all hit and miss counters to
   * zero
   */
  void clear_stats();

  /**
   * @brief Retrieve the current cache limits for blobs and libraries
   *
   * @return A cache_limits struct containing the maximum number of blobs and libraries that can be
   * stored in the cache before eviction occurs
   */
  cache_limits get_limits();

  /**
   * @brief Get the current number of blobs stored in the in-memory cache
   *
   * @return The number of blobs currently stored in the in-memory cache
   */
  [[nodiscard]] std::size_t get_blob_count();

  /**
   * @brief Get the current number of libraries stored in the in-memory cache
   *
   * @return The number of libraries currently stored in the in-memory cache
   */
  [[nodiscard]] std::size_t get_library_count();

  /**
   * @brief Clear all entries from the in-memory cache, removing all cached blobs and libraries
   * without affecting the on-disk cache
   *
   * @details This function is useful for freeing up memory without losing the benefits of the
   * on-disk cache, which can still be used to retrieve cached items in the future.
   */
  void clear_memory_store();

  /**
   * @brief Clear all entries from the on-disk cache, removing all cached blobs and libraries
   * stored on disk without affecting the in-memory cache
   * @details This function is useful for freeing up disk space or resetting the on-disk cache
   * without losing the benefits of the in-memory cache, which can still be used to retrieve cached
   * items in the future.
   */
  void clear_disk_store();

  /***
   * @brief Pre-load the JIT program cache from disk into memory during initialization, allowing for
   * faster retrieval and execution of previously compiled kernels at runtime.
   */
  void preload_from_disk();

  /***
   * @brief Enable the cache, allowing it to store and retrieve compiled blobs and libraries in
   * memory.
   * @param enabled A boolean flag indicating whether to enable (true) or disable (false) the cache.
   */
  void enable(bool enabled);

  /**
   * @brief Get whether the cache is currently enabled or disabled.
   * @return A boolean value indicating whether the cache is currently enabled (true) or disabled
   */
  [[nodiscard]] bool is_enabled();
};

/**
 * @brief Compile source code into a binary blob
 *
 * @param params Compilation parameters including source code, headers, options, and target binary
 * type
 * @return A buffer of bytes containing the compiled binary blob
 */
[[nodiscard]] byte_buffer compile(compile_params const& params);

/**
 * @brief Load a compiled library from binary data
 *
 * @param binary Span of bytes containing the compiled library binary data
 * @return A library object representing the loaded library with launchable kernels
 */
[[nodiscard]] library load_library(std::span<std::uint8_t const> binary);

/**
 * @brief Load a compiled library from binary data
 *
 * @param path Path to the file containing the library binary data
 * @return A library object representing the loaded library with launchable kernels
 */
[[nodiscard]] library load_library_from_file(char const* path);

/**
 * @brief Link multiple compiled binary fragments into a single binary blob containing the linked
 * library
 *
 * @param params Linking parameters including the binary fragments to be linked and the target
 * binary type
 * @return A buffer of bytes containing the linked library binary
 */
[[nodiscard]] byte_buffer link_library(link_params const& params);

/**
 * @brief Demangle a CUDA symbol name into a human-readable form
 * @param mangled_name The mangled CUDA symbol name to be demangled
 * @return A string containing the demangled, human-readable symbol name corresponding to the input
 * mangled name
 */
[[nodiscard]] std::string demangle_cuda_symbol(char const* mangled_name);

/**
 * @brief Initialize the RTCX library, setting up necessary resources and state for subsequent
 * operations
 * @details This function must be called before using any other functions in the RTCX library. It
 * performs necessary initialization tasks such as setting up CUDA contexts, initializing caches,
 * and preparing any global state required for compilation, linking, and kernel management
 * operations. Failure to call this function before using other RTCX functions may result in
 * undefined behavior or runtime errors.
 * This function is thread-safe.
 *
 */
void initialize();

/**
 * @brief Teardown the RTCX library, releasing any resources and cleaning up state used by the
 * library
 * @details This function should be called when RTCX functionality is no longer needed, such as at
 * the end of the program or when cleaning up resources. It performs necessary cleanup tasks such as
 * releasing CUDA contexts, clearing caches, and resetting any global state used by the library.
 * After calling this function, other RTCX functions should not be used unless initialize() is
 * called again to reinitialize the library.
 * This function is not thread-safe.
 */
void teardown();

/**
 * @brief Reflect a boolean value into its CUDA string representation ("true" or "false")
 * @param value The boolean value to be reflected
 * @return A string containing the CUDA representation of the boolean value ("true" or "false")
 */
std::string reflect_bool(bool value);

/**
 * @brief Reflect an integer value into its CUDA string representation
 * @tparam T An integral type (e.g., std::uint8_t, std::int32_t, etc.)
 * @param value The integer value to be reflected
 * @return A string containing the CUDA representation of the integer value
 */
std::string reflect_int(std::uint8_t value);

/**
 * @brief Reflect an integer value into its CUDA string representation
 * @tparam T An integral type (e.g., std::uint8_t, std::int32_t, etc.)
 * @param value The integer value to be reflected
 * @return A string containing the CUDA representation of the integer value
 */
std::string reflect_int(std::uint16_t value);

/**
 * @brief Reflect an integer value into its CUDA string representation
 * @tparam T An integral type (e.g., std::uint8_t, std::int32_t, etc.)
 * @param value The integer value to be reflected
 * @return A string containing the CUDA representation of the integer value
 */
std::string reflect_int(std::uint32_t value);

/**
 * @brief Reflect an integer value into its CUDA string representation
 * @tparam T An integral type (e.g., std::uint8_t, std::int32_t, etc.)
 * @param value The integer value to be reflected
 * @return A string containing the CUDA representation of the integer value
 */
std::string reflect_int(std::uint64_t value);

/**
 * @brief Reflect an integer value into its CUDA string representation
 * @tparam T An integral type (e.g., std::uint8_t, std::int32_t, etc.)
 * @param value The integer value to be reflected
 * @return A string containing the CUDA representation of the integer value
 */
std::string reflect_int(std::int8_t value);

/**
 * @brief Reflect an integer value into its CUDA string representation
 * @tparam T An integral type (e.g., std::uint8_t, std::int32_t, etc.)
 * @param value The integer value to be reflected
 * @return A string containing the CUDA representation of the integer value
 */
std::string reflect_int(std::int16_t value);

/**
 * @brief Reflect an integer value into its CUDA string representation
 * @tparam T An integral type (e.g., std::uint8_t, std::int32_t, etc.)
 * @param value The integer value to be reflected
 * @return A string containing the CUDA representation of the integer value
 */
std::string reflect_int(std::int32_t value);

/**
 * @brief Reflect an integer value into its CUDA string representation
 * @tparam T An integral type (e.g., std::uint8_t, std::int32_t, etc.)
 * @param value The integer value to be reflected
 * @return A string containing the CUDA representation of the integer value
 */
std::string reflect_int(std::int64_t value);

/**
 * @brief Reflect a floating-point value into its CUDA string representation
 * @tparam T A floating-point type (e.g., float, double)
 * @param value The floating-point value to be reflected
 * @return A string containing the CUDA representation of the floating-point value
 */
std::string reflect_float(float value);

/**
 * @brief Reflect a floating-point value into its CUDA string representation
 * @tparam T A floating-point type (e.g., float, double)
 * @param value The floating-point value to be reflected
 * @return A string containing the CUDA representation of the floating-point value
 */
std::string reflect_float(double value);

/**
 * @brief Reflect a value of any type into its CUDA string representation, given the type name as a
 * string
 * @param type The name of the type to be reflected (e.g., "int", "float", "MyStruct", etc.)
 * @param value The string representation of the value to be reflected, which will be used in the
 * resulting CUDA code
 * @return A string containing the CUDA representation of the value with the specified type
 */
std::string reflect_cast(std::string_view type, std::string_view value);

/**
 * @brief Reflect an enumeration value into its CUDA string representation, given the type name as a
 * string
 * @tparam T An enumeration type
 * @param type The name of the enumeration type to be reflected (e.g., "MyEnum")
 * @param value The enumeration value to be reflected, which will be cast to its underlying integer
 * type and represented as a string in the resulting CUDA code
 * @return A string containing the CUDA representation of the enumeration value with the specified
 * type
 */
template <typename T>
  requires(std::is_enum_v<T>)
std::string reflect_enum(std::string_view type, T value)
{
  return reflect_cast(type, reflect_int(static_cast<std::underlying_type_t<T>>(value)));
}

/**
 * @brief Reflect a template instantiation into its CUDA string representation, given the template
 * name and its template arguments as strings
 * @param template_name The name of the template to be reflected (e.g., "MyTemplate")
 * @param template_args A span of strings representing the template arguments to be reflected, which
 * will be used in the resulting CUDA code
 * @return A string containing the CUDA representation of the template instantiation with the
 * specified template name and arguments
 */
std::string reflect_template(std::string_view template_name,
                             std::span<std::string_view const> template_args);

/**
 * @brief Reflect a template instantiation into its CUDA string representation, given the template
 * name and its template arguments as strings
 * @param template_name The name of the template to be reflected (e.g., "MyTemplate")
 * @param template_args A span of strings representing the template arguments to be reflected, which
 * will be used in the resulting CUDA code
 * @return A string containing the CUDA representation of the template instantiation with the
 * specified template name and arguments
 */
std::string reflect_template(std::string_view template_name,
                             std::span<std::string const> template_args);

/**
 * @brief Reflect a template instantiation into its CUDA string representation, given the template
 * name and its template arguments as strings
 * @param template_name The name of the template to be reflected (e.g., "MyTemplate")
 * @param template_args A span of strings representing the template arguments to be reflected, which
 * will be used in the resulting CUDA code
 * @return A string containing the CUDA representation of the template instantiation with the
 * specified template name and arguments
 */
template <typename... TemplateArgs>
  requires((true && ... && std::is_constructible_v<std::string_view, TemplateArgs&>))
std::string reflect_template(std::string_view template_name, TemplateArgs&&... template_args)
{
  std::string_view const tparams[sizeof...(TemplateArgs)] =  // NOLINT(modernize-avoid-c-arrays)
    {std::string_view{template_args}...};
  return reflect_template(template_name, tparams);
}

}  // namespace rtcx
