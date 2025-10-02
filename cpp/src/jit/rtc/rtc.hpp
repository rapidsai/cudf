
/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

extern "C" {
typedef struct nvJitLink* nvJitLinkHandle;
typedef struct CUmod_st* CUmodule;
}

namespace cudf {
namespace rtc {

enum class binary_type : int8_t { LTO_IR = 0, CUBIN = 2, FATBIN = 3, PTX = 4 };

using blob_t = std::vector<uint8_t>;
using blob   = std::shared_ptr<blob_t>;

using blob_view = std::span<uint8_t const>;

struct [[nodiscard]] header_map {
  std::span<char const* const> include_names = {};  // null-terminated header include names
  std::span<char const* const> headers       = {};  // null-terminated header contents
  std::span<size_t const> header_sizes = {};  // sizes of each header (excluding null-terminator)
};

struct fragment_t;

using fragment = std::shared_ptr<fragment_t>;

struct fragment_t {
 private:
  nvJitLinkHandle handle_;
  binary_type type_;

 public:
  struct load_params {
    blob_view source = {};
    char const* name = nullptr;
    binary_type type = binary_type::LTO_IR;
  };

  struct compile_params {
    header_map headers                   = {};
    std::span<char const* const> options = {};
    binary_type target_type              = binary_type::LTO_IR;
  };

  [[nodiscard]] static fragment load(load_params const& params);

  [[nodiscard]] static fragment compile(compile_params const& params);

  fragment_t(fragment_t const&)            = delete;
  fragment_t(fragment_t&&)                 = delete;
  fragment_t& operator=(fragment_t const&) = delete;
  fragment_t& operator=(fragment_t&&)      = delete;
  ~fragment_t();

  [[nodiscard]] blob get_lto_ir() const;

  [[nodiscard]] blob get_cubin() const;

 private:
  fragment_t(nvJitLinkHandle handle, binary_type type) : handle_(handle), type_(type) {}
};

struct module_t;

using module = std::shared_ptr<module_t>;

void mangle_cxx_symbol(char const*);

struct function_ref {
  void launch();
};

struct module_t {
 private:
  CUmodule handle_;

 public:
  module_t(module_t const&)            = delete;
  module_t(module_t&&)                 = delete;
  module_t& operator=(module_t const&) = delete;
  module_t& operator=(module_t&&)      = delete;
  ~module_t();

  struct load_params {
    blob_view cubin = {};
  };

  struct link_params {
    std::span<blob_view const> fragments      = {};
    std::span<binary_type const> binary_types = {};
    std::span<char const* const> names        = {};
    std::span<char const* const> link_options = {};
  };

  [[nodiscard]] static blob link_as_cubin(link_params const& params);

  [[nodiscard]] static module load(blob_view cubin);

  [[nodiscard]] static module link(link_params const& params);

  [[nodiscard]] function_ref get_function(char const* name) const;

 private:
  module_t(CUmodule handle) : handle_(handle) {}
};

}  // namespace rtc
}  // namespace cudf
