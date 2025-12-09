
/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

extern "C" {
typedef struct CUlib_st* CUlibrary;
typedef struct CUkern_st* CUkernel;
typedef struct CUstream_st* CUstream;
}

namespace cudf {
namespace rtc {

enum class binary_type : int8_t { LTO_IR = 0, CUBIN = 2, FATBIN = 3, PTX = 4 };

using blob_view = std::span<uint8_t const>;

/// @brief Represents a binary blob
/// @details Manages the lifetime of the binary data via a user-provided
/// deallocator function. This enables zero-copy usage of binary data stored
/// in various forms (e.g., std::vector, mmap'd file, etc.).
struct [[nodiscard]] blob_t {
  using dealloctor_fn = void (*)(void*, uint8_t const*, size_t);

 private:
  static void noop_deallocator(void*, uint8_t const*, size_t) {}

  uint8_t const* data_;
  size_t size_;
  void* user_data_;
  dealloctor_fn deallocator_;

  blob_t(uint8_t const* data, size_t size, void* user_data, dealloctor_fn deallocator)
    : data_(data), size_(size), user_data_(user_data), deallocator_(deallocator)
  {
  }

 public:
  blob_t() : data_(nullptr), size_(0), user_data_(nullptr), deallocator_(noop_deallocator) {}
  blob_t(blob_t const&)            = delete;
  blob_t& operator=(blob_t const&) = delete;
  blob_t(blob_t&& other) noexcept;
  blob_t& operator=(blob_t&& other) noexcept;
  ~blob_t() { deallocator_(user_data_, data_, size_); }

  [[nodiscard]] blob_view view() const { return blob_view{data_, size_}; }

  static blob_t from_parts(uint8_t const* data,
                           size_t size,
                           void* user_data,
                           dealloctor_fn deallocator);

  static blob_t from_vector(std::vector<uint8_t>&& data);

  static blob_t from_static_data(std::span<uint8_t const> data);
};

using blob = std::shared_ptr<blob_t>;

struct [[nodiscard]] header_map {
  std::span<char const* const> include_names = {};  // null-terminated header include names
  std::span<char const* const> headers       = {};  // null-terminated header contents
  std::span<size_t const> header_sizes = {};  // sizes of each header (excluding null-terminator)
};

struct fragment_t;

using fragment = std::shared_ptr<fragment_t>;

/// @brief Represents a partially compiled RTC kernel (i.e. fragment) in LTO-IR or PTX
struct fragment_t {
 private:
  blob blob_;
  binary_type type_;

 public:
  struct load_params {
    blob blob        = {};
    binary_type type = binary_type::LTO_IR;
  };

  struct compile_params {
    char const* name                     = nullptr;
    char const* source                   = nullptr;
    header_map headers                   = {};
    std::span<char const* const> options = {};
    binary_type target_type              = binary_type::LTO_IR;
  };

  [[nodiscard]] static fragment load(load_params const& params);

  [[nodiscard]] static fragment compile(compile_params const& params);

  fragment_t(blob blob, binary_type type) : blob_(std::move(blob)), type_(type) {}
  fragment_t(fragment_t const&)            = delete;
  fragment_t(fragment_t&&)                 = delete;
  fragment_t& operator=(fragment_t const&) = delete;
  fragment_t& operator=(fragment_t&&)      = delete;
  ~fragment_t()                            = default;

  [[nodiscard]] binary_type get_type() const { return type_; }

  [[nodiscard]] blob const& get_lto_ir() const;

  [[nodiscard]] blob const& get_cubin() const;
};

struct library_t;

using library = std::shared_ptr<library_t>;

struct kernel_ref {
 private:
  CUkernel handle_;

 public:
  explicit kernel_ref(CUkernel handle) : handle_(handle) {}

  void launch(uint32_t grid_dim_x,
              uint32_t grid_dim_y,
              uint32_t grid_dim_z,
              uint32_t block_dim_x,
              uint32_t block_dim_y,
              uint32_t block_dim_z,
              uint32_t shared_mem_bytes,
              CUstream stream,
              void** kernel_params);

  [[nodiscard]] std::string_view get_name() const;
};

/// @brief Represents a loaded RTC library containing compiled kernels
/// Input: CUBIN or PTX binary
/// Output: loaded library with launchable kernels
struct library_t {
 private:
  CUlibrary handle_;

 public:
  explicit library_t(CUlibrary handle) : handle_(handle) {}
  library_t(library_t const&)            = delete;
  library_t(library_t&&)                 = delete;
  library_t& operator=(library_t const&) = delete;
  library_t& operator=(library_t&&)      = delete;
  ~library_t();

  struct load_params {
    blob_view binary = {};
    binary_type type = binary_type::CUBIN;
  };

  struct link_params {
    char const* name                          = nullptr;
    binary_type output_type                   = binary_type::CUBIN;
    std::span<blob_view const> fragments      = {};
    std::span<binary_type const> binary_types = {};
    std::span<char const* const> names        = {};
    std::span<char const* const> link_options = {};
  };

  [[nodiscard]] static library load(load_params const& params);

  [[nodiscard]] static blob link_as_blob(link_params const& params);

  [[nodiscard]] static library link(link_params const& params);

  [[nodiscard]] kernel_ref get_kernel(char const* name) const;

  [[nodiscard]] std::vector<kernel_ref> enumerate_kernels() const;
};

std::string demangle_cuda_symbol(char const* mangled_name);

}  // namespace rtc
}  // namespace cudf
