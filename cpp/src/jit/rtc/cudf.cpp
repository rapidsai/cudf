
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/utilities/defer.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime.h>

#include <cudf_jit_embed.h>
#include <fcntl.h>
#include <jit/rtc/cache.hpp>
#include <jit/rtc/cudf.hpp>
#include <jit/rtc/rtc.hpp>
#include <runtime/context.hpp>
#include <sys/stat.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <format>
#include <future>

namespace cudf {
namespace rtc {

int32_t get_driver_version()
{
  int32_t driver_version;
  CUDF_EXPECTS(cudaDriverGetVersion(&driver_version) == cudaSuccess,
               "Failed to get CUDA driver version");
  return driver_version;
}

int32_t get_runtime_version()
{
  int32_t runtime_version;
  CUDF_EXPECTS(cudaRuntimeGetVersion(&runtime_version) == cudaSuccess,
               "Failed to get CUDA runtime version");
  return runtime_version;
}

int32_t get_current_device_physical_model()
{
  int32_t device;
  CUDF_EXPECTS(cudaGetDevice(&device) == cudaSuccess, "Failed to get current CUDA device");

  cudaDeviceProp props;
  CUDF_EXPECTS(cudaGetDeviceProperties(&props, device) == cudaSuccess,
               "Failed to get device properties");

  return props.major * 10 + props.minor;
}

void max_occupancy_config()
{
  // TODO: Same as configure_1d_max_occupancy
}

sha256_hash hash_string(std::span<char const> input)
{
  sha256_context ctx;
  ctx.update(std::span{reinterpret_cast<uint8_t const*>(input.data()), input.size()});
  return ctx.finalize();
}

cache_t& get_rtc_cache() { return cudf::get_context().rtc_cache(); }

[[noreturn]] void throw_posix(std::string_view message, std::string_view syscall_name)
{
  auto error_code = errno;
  auto error_str  = std::format(
    "{}. `{}` failed with {} ({})", message, syscall_name, error_code, std::strerror(error_code));
  CUDF_FAIL(+error_str, std::runtime_error);
}

void add_file(char const* dst_path, unsigned char const* data, size_t data_size)
{
  int dst_file = open(dst_path, O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  if (dst_file == -1) {
    if (errno == EEXIST) {
      // file already exists (repeated include)
      return;
    }
    throw_posix(std::format("Failed to create file ({})", dst_path), "open");
  }

  CUDF_DEFER([&] {
    if (close(dst_file) != 0) {
      throw_posix(std::format("Failed to close file ({})", dst_path), "close");
    }
  });

  if (write(dst_file, data, data_size) == -1) {
    throw_posix(std::format("Failed to write file ({})", dst_path), "write");
  }
}

std::vector<unsigned char> read_file(char const* path)
{
  int fd = open(path, O_RDONLY);

  if (fd == -1) { throw_posix(std::format("Failed to open file ({})", path), "open"); }

  CUDF_DEFER([&] {
    if (close(fd) == -1) { throw_posix(std::format("Failed to close file ({})", path), "close"); }
  });

  // get file size
  struct stat file_stat;
  if (fstat(fd, &file_stat) == -1) {
    throw_posix(std::format("Failed to get file status for file ({})", path), "fstat");
  }

  std::vector<unsigned char> contents;
  contents.resize(file_stat.st_size);

  if (read(fd, contents.data(), contents.size()) == -1) {
    throw_posix(std::format("Failed to read file ({})", path), "read");
  }

  return contents;
}

void copy_includes_to_dir(char const* dir)
{
  for (size_t i = 0; i < cudf_jit_embed_sources_file_data.size; ++i) {
    auto const data      = cudf_jit_embed_sources_file_data.elements[i];
    auto const data_size = cudf_jit_embed_sources_file_data.element_sizes[i];
    auto const dst =
      reinterpret_cast<char const*>(cudf_jit_embed_sources_file_destinations.elements[i]);
    auto const dst_path = std::format("{}/{}", dir, dst);

    std::filesystem::create_directories(std::filesystem::path{dst_path}.parent_path());
    add_file(dst_path.c_str(), data, data_size);
  }

  auto hash_path = std::format("{}/state.hash", dir);
  add_file(hash_path.c_str(),
           cudf_jit_embed_sources_file_data_hash.data,
           cudf_jit_embed_sources_file_data_hash.size);
}

void create_new_include_dir(std::string const& include_path)
{
  // directory does not exist, so create it
  char tmp_dir_data[] = "/tmp/jit-includes_XXXXXX";
  char* tmp_dir       = mkdtemp(tmp_dir_data);
  if (tmp_dir == nullptr) {
    throw_posix(
      std::format("Failed to create temporary RTC include directory for ({})", include_path),
      "mkdtemp");
  }

  copy_includes_to_dir(tmp_dir);

  // rename the temporary directory to the target include_path
  if (rename(tmp_dir, include_path.c_str()) == -1) {
    throw_posix(
      std::format("Failed to rename temporary RTC include directory to ({})", include_path),
      "rename");
  }
}

void create_include_dir(std::string const& include_path)
{
  struct stat path_info;

  if (lstat(include_path.c_str(), &path_info) == -1) {
    if (errno != ENOENT) {
      throw_posix(std::format("Failed to get stat for jit_include directory ({})", include_path),
                  "lstat");
    } else {
      create_new_include_dir(include_path);
    }
  } else {
    if (!S_ISDIR(path_info.st_mode)) {
      CUDF_FAIL(+std::format("RTC include path ({}) exists but is not a directory", include_path),
                std::runtime_error);
    } else {
      // verify contents match expected headers
      auto hash_path = std::format("{}/state.hash", include_path);
      auto hash_data = read_file(hash_path.c_str());

      if (hash_data.size() != cudf_jit_embed_sources_file_data_hash.size ||
          std::memcmp(hash_data.data(),
                      cudf_jit_embed_sources_file_data_hash.data,
                      cudf_jit_embed_sources_file_data_hash.size) != 0) {
        CUDF_FAIL(+std::format("RTC include directory ({}) contents do not match expected headers",
                               include_path),
                  std::runtime_error);
      }
    }
  }
}

fragment_t const& compile_fragment(char const* name, char const* source_code_cstr, char const* key)
{
  auto sm              = get_current_device_physical_model();
  auto const cache_key = std::format(R"***(
      fragment_type=LTO_IR,
      key={},
      cuda_runtime={},
      cuda_driver={},
      arch={})***",
                                     key,
                                     get_runtime_version(),
                                     get_driver_version(),
                                     sm);

  auto const cache_key_sha256 = hash_string(cache_key);

  auto& cache = get_rtc_cache();

  if (auto frag = cache.query_fragment(cache_key_sha256); frag.has_value()) {
    return *frag->get();
  } else if (auto disk_frag = cache.query_blob_from_disk(cache_key_sha256); disk_frag.has_value()) {
    std::promise<fragment> prom;
    auto fut = std::shared_future{prom.get_future()};
    {
      cache.store_fragment(cache_key_sha256, fut);
      fragment_t::load_params load_params{.binary = *disk_frag, .type = binary_type::LTO_IR};
      auto frag = fragment_t::load(load_params);
      prom.set_value(std::move(frag));
    }
    return *fut.get();
  }

  std::promise<fragment> prom;
  auto fut = std::shared_future{prom.get_future()};
  cache.store_fragment(cache_key_sha256, fut);

  auto cache_dir   = cache.get_cache_dir();
  auto include_dir = std::format("{}/jit-includes", cache_dir);

  create_include_dir(include_dir);

  std::vector<std::string> include_options;
  include_options.push_back(std::format("-I{}", include_dir));
  for (size_t i = 0; i < cudf_jit_embed_sources_include_directories.size; i++) {
    auto include_path =
      reinterpret_cast<char const*>(cudf_jit_embed_sources_include_directories.elements[i]);
    include_options.push_back(std::format("-I{}/{}", include_dir, include_path));
  }

  std::vector<char const*> options;
  auto embed_options = reinterpret_cast<const char* const*>(cudf_jit_embed_options.elements);
  std::copy(
    embed_options, embed_options + cudf_jit_embed_options.size, std::back_inserter(options));
  auto arch_flag = std::format("-arch=sm_{}", sm);
  options.push_back(arch_flag.c_str());
  options.push_back("-dlto");
  options.push_back("-rdc=true");
  options.push_back("--split-compile=0");
  options.push_back("-default-device");

  for (auto const& include_option : include_options) {
    options.push_back(include_option.c_str());
  }

  auto const params = fragment_t::compile_params{.name        = name,
                                                 .source      = source_code_cstr,
                                                 .headers     = {},
                                                 .options     = options,
                                                 .target_type = binary_type::LTO_IR};

  auto frag = fragment_t::compile(params);

  prom.set_value(std::move(frag));
  return *fut.get();
}

fragment_t const& compile_library_fragment()
{
  return compile_fragment("cudf_lto_library",
                          R"***(
 #include "jit/lto/library.inl.cuh"
 )***",
                          "cudf_lto_library");
}

fragment_t const& compile_udf_fragment(char const* source_code_cstr, char const* key)
{
  return compile_fragment("cudf_udf_fragment", source_code_cstr, key);
}

kernel_ref compile_and_link_udf(char const* name,
                                char const* kernel_name,
                                char const* kernel_key,
                                char const* udf_code,
                                char const* udf_key)
{
  auto sm                       = get_current_device_physical_model();
  auto library_key              = std::format(R"***(
      fragment_types=LTO_IR,
      target=CUBIN,
      kernel={},
      udf={},
      cuda_runtime={},
      cuda_driver={},
      arch={})***",
                                 kernel_key,
                                 udf_key,
                                 get_runtime_version(),
                                 get_driver_version(),
                                 sm);
  auto const library_key_sha256 = hash_string(library_key);

  auto& cache = get_rtc_cache();

  // TODO: (atomicity) should probably use query_or_insert
  if (auto lib = cache.query_library(library_key_sha256); lib.has_value()) {
    return lib->get()->get_kernel(kernel_name);
  } else if (auto disk_lib = cache.query_blob_from_disk(library_key_sha256); disk_lib.has_value()) {
    std::promise<library> prom;
    auto fut = std::shared_future{prom.get_future()};

    {
      cache.store_library(library_key_sha256, fut);
      library_t::load_params load_params{.binary = (*disk_lib)->view(), .type = binary_type::CUBIN};
      auto lib = library_t::load(load_params);
      prom.set_value(std::move(lib));
    }

    return fut.get()->get_kernel(kernel_name);
  }

  auto const& library_frag = compile_library_fragment();
  auto const& udf_frag     = compile_udf_fragment(udf_code, udf_key);

  std::promise<library> prom;
  auto fut = std::shared_future{prom.get_future()};
  cache.store_library(library_key_sha256, fut);

  blob_view const link_fragments[]          = {library_frag.get_cubin()->view(),
                                               udf_frag.get_cubin()->view()};
  binary_type const fragment_binary_types[] = {binary_type::LTO_IR, binary_type::LTO_IR};

  char const* const fragment_names[] = {"cudf_lto_library", "cudf_udf_fragment"};

  auto arch_flag                   = std::format("-arch=sm_{}", sm);
  char const* const link_options[] = {"-lto", arch_flag.c_str()};

  auto const params = library_t::link_params{.name                  = name,
                                             .output_type           = binary_type::CUBIN,
                                             .fragments             = link_fragments,
                                             .fragment_binary_types = fragment_binary_types,
                                             .fragment_names        = fragment_names,
                                             .link_options          = link_options};

  auto lib = library_t::link(params);
  prom.set_value(std::move(lib));

  return fut.get()->get_kernel(kernel_name);
}

}  // namespace rtc
}  // namespace cudf
