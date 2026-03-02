
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <format>
#include <future>
#include <memory>
#include <optional>
#include <source_location>
#include <span>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#define RTCX_EXPORT __attribute__((visibility("default")))

#define RTCX_DEFER__CONCATENATE_DETAIL(x, y) x##y
#define RTCX_DEFER__CONCATENATE(x, y)        RTCX_DEFER__CONCATENATE_DETAIL(x, y)
#define RTCX_DEFER(...)                      ::rtcx::defer RTCX_DEFER__CONCATENATE(defer_, __COUNTER__)(__VA_ARGS__)

#define RTCX_EXPECTS(_condition, _reason, _exception_type)                               \
  do {                                                                                   \
    if (!(_condition)) {                                                                 \
      throw _exception_type{::std::format("RTCX failure at: {}:{}: {}",                  \
                                          ::std::source_location::current().file_name(), \
                                          ::std::source_location::current().line(),      \
                                          (_reason))};                                   \
    }                                                                                    \
  } while (0)

#define RTCX_FAIL(_reason, _exception_type)                                            \
  do {                                                                                 \
    throw _exception_type{::std::format("RTCX failure at: {}:{}: {}",                  \
                                        ::std::source_location::current().file_name(), \
                                        ::std::source_location::current().line(),      \
                                        (_reason))};                                   \
  } while (0)

extern "C" {
typedef struct evp_md_ctx_st EVP_MD_CTX;

typedef struct CUlib_st* CUlibrary;
typedef struct CUkern_st* CUkernel;
typedef struct CUstream_st* CUstream;
}

namespace RTCX_EXPORT rtcx {

using u8    = std::uint8_t;
using u16   = std::uint16_t;
using u32   = std::uint32_t;
using u64   = std::uint64_t;
using usize = std::size_t;
using i8    = std::int8_t;
using i16   = std::int16_t;
using i32   = std::int32_t;
using i64   = std::int64_t;
using iszie = std::ptrdiff_t;

inline constexpr usize CACHELINE_ALIGNMENT =
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

struct [[nodiscard]] sha256_hex_string {
  char data_[65];

  constexpr std::string_view view() const { return std::string_view{data_, 64}; }

  constexpr operator std::string_view() const { return view(); }

  [[nodiscard]] char const* data() const { return data_; }

  [[nodiscard]] char const* c_str() const { return data_; }

  static constexpr usize size() { return 64; }

  static sha256_hex_string make(std::span<u8 const, 32> input)
  {
    constexpr char const HEX_CHARS[] = "0123456789abcdef";
    sha256_hex_string hex;
    for (usize i = 0; i < 32; ++i) {
      hex.data_[i * 2]     = HEX_CHARS[(input[i] >> 4) & 0x0F];
      hex.data_[i * 2 + 1] = HEX_CHARS[input[i] & 0x0F];
    }
    hex.data_[64] = '\0';
    return hex;
  }
};

struct [[nodiscard]] sha256 {
  alignas(16) u8 data_[32];

  constexpr bool operator==(sha256 const& hash) const
  {
    return std::equal(std::begin(data_), std::end(data_), std::begin(hash.data_));
  }

  constexpr bool operator!=(sha256 const& hash) const { return !(*this == hash); }

  sha256_hex_string to_hex_string() const { return sha256_hex_string::make(data_); }
};

struct [[nodiscard]] sha256_hasher {
  constexpr u64 operator()(sha256 const& obj) const
  {
    struct u64x4 {
      alignas(16) u64 v[4];
    };

    auto value    = std::bit_cast<u64x4>(obj);
    auto const h0 = value.v[0];
    auto const h1 = value.v[1];
    auto const h2 = value.v[2];
    auto const h3 = value.v[3];

    auto mix = [](u64 seed, u64 v) {
      seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
      return seed;
    };

    return mix(mix(mix(h0, h1), h2), h3);
  }
};

struct [[nodiscard]] sha256_context {
 private:
  EVP_MD_CTX* ectx_;

 public:
  sha256_context();
  sha256_context(sha256_context const& other)            = delete;
  sha256_context& operator=(sha256_context const& other) = delete;
  sha256_context(sha256_context&& other) : ectx_(other.ectx_) { other.ectx_ = nullptr; }

  sha256_context& operator=(sha256_context&& other)
  {
    if (this == &other) [[unlikely]] { return *this; }
    this->~sha256_context();
    new (this) sha256_context(std::move(other));
    return *this;
  }

  ~sha256_context();

  void update(std::span<u8 const> data);

  sha256 finalize();
};

enum class binary_type : i8 { LTO_IR = 0, CUBIN = 2, FATBIN = 3, PTX = 4 };

/**
 * @brief Represents a binary blob
 * @details Manages the lifetime of the binary data via a user-provided deallocator function. This
 * enables zero-copy usage of binary data stored in various forms (e.g., std::vector, mmap'd file,
 * etc.).
 */
struct [[nodiscard]] blob_t {
 private:
  using deallocator = func<void(u8 const*, usize)>;

  static void noop_deallocator(u8 const*, usize) {}

  u8 const* data_;
  usize size_;
  deallocator deallocator_;

  blob_t(u8 const* data, usize size, deallocator deallocator)
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

  [[nodiscard]] std::span<u8 const> view() const { return {data_, size_}; }

  static blob_t from_parts(u8 const* data, usize size, deallocator deallocator)
  {
    return blob_t{data, size, deallocator};
  }

  static blob_t from_vector(std::vector<u8>&& data);

  static blob_t from_static_data(std::span<u8 const> data);

  static std::optional<blob_t> from_file(char const* path);
};

using blob = std::shared_ptr<blob_t>;

struct [[nodiscard]] kernel_occupancy_config {
  i32 min_grid_size = 0;
  i32 block_size    = 0;
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
  kernel_occupancy_config max_occupancy_config(usize dynamic_shared_memory_bytes,
                                               i32 block_size_limit) const;

  /**
   * @brief Launches the kernel on the GPU with the specified grid and block dimensions,
   * dynamic shared memory size, stream, and kernel parameters. This function wraps the CUDA driver
   * kernel launch API, providing a convenient interface for executing the kernel with the desired
   * configuration.
   * @param grid_dim_x The number of blocks in the x-dimension of the grid
   * @param grid_dim_y The number of blocks in the y-dimension of the grid
   * @param grid_dim_z The number of blocks in the z-dimension of the grid
   * @param block_dim_x The number of threads in the x-dimension of each block
   * @param block_dim_y The number of threads in the y-dimension of each block
   * @param block_dim_z The number of threads in the z-dimension of each block
   * @param shared_mem_bytes The amount of dynamic shared memory (in bytes) to allocate for the
   * kernel
   * @param stream The CUDA stream on which to launch the kernel
   * @param kernel_params A pointer to an array of pointers representing the kernel parameters to be
   * passed to the kernel at launch time
   */
  void launch(u32 grid_dim_x,
              u32 grid_dim_y,
              u32 grid_dim_z,
              u32 block_dim_x,
              u32 block_dim_y,
              u32 block_dim_z,
              u32 shared_mem_bytes,
              CUstream stream,
              void** kernel_params) const;

  /**
   * @brief Retrieves the name of the kernel
   */
  [[nodiscard]] std::string_view get_name() const;
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
   * @brief Retrieve a kernel from the library by name
   */
  [[nodiscard]] kernel_ref get_kernel(char const* name) const;

  /**
   * @brief Enumerate all kernels contained in the library, returning a vector of kernel references
   * @return A vector of kernel_ref objects representing all kernels contained in the library
   */
  [[nodiscard]] std::vector<kernel_ref> enumerate_kernels() const;
};

using library = std::shared_ptr<library_t>;

struct [[nodiscard]] compile_params {
  /**
   * @brief Name of the kernel or library being compiled
   */
  char const* name = nullptr;

  /**
   * @brief Source code to be compiled (e.g. PTX or LTO IR)
   */
  char const* source = nullptr;

  /**
   * @brief Include names of each header file provided in the `headers` field, used for resolving
   * #include directives during compilation
   */
  std::span<char const* const> header_include_names = {};

  /**
   * @brief Contents of header files required for compilation
   */
  std::span<char const* const> headers = {};

  /**
   * @brief Sizes of each header file provided in the `headers` field
   */
  std::span<usize const> header_sizes = {};

  /**
   * @brief NVRTC compile options
   */
  std::span<char const* const> options = {};

  /**
   * @brief Desired output binary type (e.g. PTX, CUBIN, etc.)
   */
  binary_type target_type = binary_type::LTO_IR;
};

struct [[nodiscard]] link_params {
  char const* name                                   = nullptr;
  binary_type output_type                            = binary_type::CUBIN;
  std::span<std::span<u8 const> const> fragments     = {};
  std::span<binary_type const> fragment_binary_types = {};
  std::span<char const* const> fragment_names        = {};
  std::span<char const* const> link_options          = {};
};

template <typename T>
struct alignas(CACHELINE_ALIGNMENT) lru_memory_cache {
  struct entry {
    u64 last_touched_tick = 0;
    T value;

    void hit(u64 tick) { last_touched_tick = tick; }
  };

  std::unordered_map<sha256, entry, sha256_hasher> entries_;
  usize limit_;

  explicit lru_memory_cache(usize limit) : entries_{}, limit_{limit}
  {
    // reserve space to avoid rehashing
    RTCX_EXPECTS(limit > 0, "Cache limit must be greater than 0", std::logic_error);
    entries_.reserve(limit * 2);
  }

  void purge()
  {
    if (entries_.empty()) { return; }

    auto num_to_purge = std::max(entries_.size() / 2, static_cast<usize>(1));

    std::vector<std::pair<sha256, u64>> rankings;
    rankings.reserve(entries_.size());

    for (auto const& [key, entry] : entries_) {
      rankings.emplace_back(key, entry.last_touched_tick);
    }

    std::sort(rankings.begin(), rankings.end(), [](auto const& a, auto const& b) {
      return a.second < b.second;
    });

    // purge least recently used half
    rankings.resize(num_to_purge);

    for (auto [key, _] : rankings) {
      entries_.erase(key);
    }
  }

  void insert(sha256 const& sha, T&& value, u64 tick)
  {
    if ((entries_.size() + 1) > limit_) { purge(); }

    entries_.emplace(sha, entry{tick, std::move(value)});
  }
};

struct [[nodiscard]] cache_stats {
  u64 blob_mem_hits       = 0;
  u64 blob_mem_misses     = 0;
  u64 blob_disk_hits      = 0;
  u64 blob_disk_misses    = 0;
  u64 library_mem_hits    = 0;
  u64 library_mem_misses  = 0;
  u64 library_disk_hits   = 0;
  u64 library_disk_misses = 0;
};

struct [[nodiscard]] cache_limits {
  u32 num_blobs     = 1024;
  u32 num_libraries = 1024;
};

struct cache_stats_counter {
  struct alignas(CACHELINE_ALIGNMENT) counter {
    u64 value_ = 0;

    void incr()
    {
      std::atomic_ref c{value_};
      c.fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard]] u64 get() const
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

  counter blob_mem_hits;
  counter blob_mem_misses;
  counter blob_disk_hits;
  counter blob_disk_misses;
  counter library_mem_hits;
  counter library_mem_misses;
  counter library_disk_hits;
  counter library_disk_misses;
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
struct cache_t {
 private:
  std::string cache_dir_;

  cache_limits limits_;

  std::mutex lock_;

  lru_memory_cache<std::shared_future<blob>> blobs_cache_;

  lru_memory_cache<std::shared_future<library>> libraries_cache_;

  cache_stats_counter counter_;

  alignas(CACHELINE_ALIGNMENT) u64 tick_;

 public:
  cache_t(std::string cache_dir, cache_limits const& limits);
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
                                                               binary_type type,
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
  [[nodiscard]] usize get_blob_count();

  /**
   * @brief Get the current number of libraries stored in the in-memory cache
   *
   * @return The number of libraries currently stored in the in-memory cache
   */
  [[nodiscard]] usize get_library_count();

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
};

/**
 * @brief Compile source code into a binary blob
 *
 * @param params Compilation parameters including source code, headers, options, and target binary
 * type
 * @return A vector of bytes containing the compiled binary blob
 */
[[nodiscard]] std::vector<u8> compile(compile_params const& params);

/**
 * @brief Load a compiled library from binary data
 *
 * @param binary Span of bytes containing the compiled library binary data
 * @param type Binary type of the library (e.g., CUBIN, PTX)
 * @return A library object representing the loaded library with launchable kernels
 */
[[nodiscard]] library load_library(std::span<u8 const> binary, binary_type type);

/**
 * @brief Link multiple compiled binary fragments into a single binary blob containing the linked
 * library
 *
 * @param params Linking parameters including the binary fragments to be linked and the target
 * binary type
 * @return A vector of bytes containing the linked library binary
 */
[[nodiscard]] std::vector<u8> link_library(link_params const& params);

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

}  // namespace RTCX_EXPORT rtcx
