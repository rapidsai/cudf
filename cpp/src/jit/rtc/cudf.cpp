
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/defer.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime.h>

#include <cudf_jit_embed.h>
#include <fcntl.h>
#include <jit/rtc/cache.hpp>
#include <jit/rtc/cudf.hpp>
#include <jit/rtc/rtc.hpp>
#include <jit/rtc/sha256.hpp>
#include <runtime/context.hpp>
#include <sys/file.h>
#include <sys/stat.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <format>
#include <future>
#include <iostream>

#define CUDFRTC_CHECK_CUDART(msg, ...)                                              \
  do {                                                                              \
    ::cudaError_t __result = (__VA_ARGS__);                                         \
    if (__result != ::cudaSuccess) {                                                \
      auto __errstr = ::std::format("(cudart) Call {} failed, with error ({}): {}", \
                                    #__VA_ARGS__,                                   \
                                    static_cast<::int64_t>(__result),               \
                                    ::cudaGetErrorString(__result));                \
      CUDF_FAIL(+std::format("{}. {}", msg, __errstr), ::std::runtime_error);       \
    }                                                                               \
  } while (0)

namespace cudf {
namespace rtc {

namespace {

int32_t get_driver_version()
{
  int32_t driver_version;
  CUDFRTC_CHECK_CUDART("Failed to get CUDA driver version", cudaDriverGetVersion(&driver_version));

  return driver_version;
}

int32_t get_runtime_version()
{
  int32_t runtime_version;
  CUDFRTC_CHECK_CUDART("Failed to get CUDA runtime version",
                       cudaRuntimeGetVersion(&runtime_version));

  return runtime_version;
}

int32_t get_current_device_physical_model()
{
  int32_t device;
  CUDFRTC_CHECK_CUDART("Failed to get current CUDA device", cudaGetDevice(&device));

  cudaDeviceProp props;
  CUDFRTC_CHECK_CUDART("Failed to get device properties", cudaGetDeviceProperties(&props, device));

  return props.major * 10 + props.minor;
}

/*
void max_occupancy_config()
{
  CUDF_FAIL("Not implemented", std::logic_error);
  // TODO: Same as configure_1d_max_occupancy
}*/

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

void add_file(char const* dst_path, std::span<unsigned char const> contents)
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

  if (write(dst_file, contents.data(), contents.size()) == -1) {
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

static constexpr char const* HASH_FILENAME = ".sha256.hash";

void copy_includes_to_dir(char const* dst_dir)
{
  CUDF_FUNC_RANGE();

  auto const files_data = cudf_jit_embed_sources_file_data.bytes.data;
  auto const destinations_data =
    reinterpret_cast<char const*>(cudf_jit_embed_sources_file_destinations.bytes.data);
  for (size_t i = 0; i < cudf_jit_embed_sources_file_data.num_ranges; ++i) {
    auto const file_data_range   = cudf_jit_embed_sources_file_data.ranges[i];
    auto const destination_range = cudf_jit_embed_sources_file_destinations.ranges[i];

    auto const file_data = std::span{files_data + file_data_range.offset, file_data_range.size};
    auto const destination =
      std::string_view{destinations_data + destination_range.offset, destination_range.size};

    auto const destination_path = std::format("{}/{}", dst_dir, destination);

    std::filesystem::create_directories(std::filesystem::path{destination_path}.parent_path());
    add_file(destination_path.c_str(), file_data);
  }

  {
    // write out the state hash file
    auto hash_path = std::format("{}/{}", dst_dir, HASH_FILENAME);
    add_file(hash_path.c_str(),
             std::span{cudf_jit_embed_sources_hash.data, cudf_jit_embed_sources_hash.size});
  }
}

std::string get_include_dir(char const* base_dir)
{
  auto sha256_str = sha256_hex_string::make(
    std::span{cudf_jit_embed_sources_hash.data, cudf_jit_embed_sources_hash.size});

  return std::format("{}/{}", base_dir, sha256_str.view());
}

void create_new_include_dir(char const* base_dir)
{
  CUDF_FUNC_RANGE();

  // directory does not exist, so create it
  char tmp_dir_data[] = "/tmp/jit-includes_XXXXXX";
  char* tmp_dir       = mkdtemp(tmp_dir_data);
  if (tmp_dir == nullptr) {
    throw_posix(std::format("Failed to create temporary RTC include directory for ({})", base_dir),
                "mkdtemp");
  }

  copy_includes_to_dir(tmp_dir);

  auto include_dir = get_include_dir(base_dir);

  // rename the temporary directory to the target include_dir
  if (rename(tmp_dir, include_dir.c_str()) == -1) {
    throw_posix(
      std::format("Failed to rename temporary RTC include directory to ({})", include_dir),
      "rename");
  }
}

std::string install_includes_to(char const* base_dir)
{
  CUDF_FUNC_RANGE();

  auto include_dir = get_include_dir(base_dir);

  struct stat path_info;
  if (lstat(include_dir.c_str(), &path_info) == -1) {
    if (errno != ENOENT) {
      throw_posix(std::format("Failed to get stat for directory ({})", include_dir), "lstat");
    } else {
      std::filesystem::create_directories(base_dir);
      create_new_include_dir(base_dir);
    }
  } else {
    // directory exists, perform important sanity checks
    if (!S_ISDIR(path_info.st_mode)) {
      CUDF_FAIL(+std::format("Include dir ({}) exists but is not a directory", include_dir),
                std::runtime_error);
    } else {
      // verify contents match expected headers
      auto hash_path = std::format("{}/{}", include_dir, HASH_FILENAME);
      auto hash_data = read_file(hash_path.c_str());

      CUDF_EXPECTS(std::equal(hash_data.begin(),
                              hash_data.end(),
                              cudf_jit_embed_sources_hash.data,
                              cudf_jit_embed_sources_hash.data + cudf_jit_embed_sources_hash.size),
                   +std::format("RTC include dir ({}) is corrupted", include_dir),
                   std::runtime_error);
    }
  }

  return include_dir;
}

}  // namespace

void install_includes(char const* cache_dir)
{
  CUDF_FUNC_RANGE();

  auto install_dir = std::format("{}/jit-install", cache_dir);
  install_includes_to(install_dir.c_str());
}

fragment_t const& compile_fragment(char const* name, char const* source_code_cstr, char const* key)
{
  CUDF_FUNC_RANGE();

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
    std::cout << "Loading RTC base library from memory\n";
    return *frag->get();
  } else if (auto disk_frag = cache.query_blob_from_disk(cache_key_sha256); disk_frag.has_value()) {
    std::cout << "Loading RTC base library from disk cache\n";
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

  std::cout << "Compiling and linking RTC base library\n";
  std::promise<fragment> prom;
  auto fut = std::shared_future{prom.get_future()};
  cache.store_fragment(cache_key_sha256, fut);

  auto begin       = std::chrono::high_resolution_clock::now();
  auto cache_dir   = cache.get_cache_dir();
  auto install_dir = std::format("{}/jit-install", cache_dir);
  auto include_dir = get_include_dir(install_dir.c_str());

  std::vector<std::string> include_options;
  include_options.push_back(std::format("-I{}", include_dir));

  auto include_directories_data =
    reinterpret_cast<char const*>(cudf_jit_embed_sources_include_directories.bytes.data);
  for (size_t i = 0; i < cudf_jit_embed_sources_include_directories.num_ranges; i++) {
    auto range                  = cudf_jit_embed_sources_include_directories.ranges[i];
    auto dest_include_directory = include_directories_data + range.offset;
    include_options.push_back(std::format("-I{}/{}", include_dir, dest_include_directory));
  }

  std::vector<char const*> options;
  auto embed_options_data = reinterpret_cast<const char*>(cudf_jit_embed_options.bytes.data);

  for (size_t i = 0; i < cudf_jit_embed_options.num_ranges; i++) {
    auto range  = cudf_jit_embed_options.ranges[i];
    auto option = embed_options_data + range.offset;
    options.push_back(option);
  }

  auto arch_flag = std::format("--gpu-architecture=sm_{}", sm);
  options.push_back(arch_flag.c_str());
  options.push_back("--dlink-time-opt");
  options.push_back("--relocatable-device-code=true");
  // options.push_back("--split-compile=0");
  // options.push_back("--fdevice-time-trace=jit_comp_trace.json");
  // options.push_back("--minimal");
  // options.push_back("--time=compile_trace.json");
  // options.push_back("-time");
  // --fast-compile
  options.push_back("--pch");
  options.push_back(
    "--pch-dir=/tmp/cudf-rtc-pch");  // [ ] fix; make it consistent (hashing of header contents?)

  options.push_back("--device-as-default-execution-space");

  for (auto const& include_option : include_options) {
    options.push_back(include_option.c_str());
  }

  auto const params = fragment_t::compile_params{.name        = name,
                                                 .source      = source_code_cstr,
                                                 .headers     = {},
                                                 .options     = options,
                                                 .target_type = binary_type::LTO_IR};

  auto frag = fragment_t::compile(params);

  auto view = frag->get(binary_type::LTO_IR)->view();

  auto end = std::chrono::high_resolution_clock::now();
  auto dur = end - begin;
  std::cout << "RTC fragment compilation for `" << name << "` took "
            << std::chrono::duration_cast<std::chrono::microseconds>(dur).count() << " us\n";

  cache.store_blob_to_disk(cache_key_sha256, view);

  prom.set_value(std::move(frag));
  return *fut.get();
}

fragment_t const& compile_library_fragment()
{
  CUDF_FUNC_RANGE();

  return compile_fragment("cudf_lto_library",
                          R"***(
 #include "jit/lto/library.inl.cuh"
 )***",
                          "cudf_lto_library");
}

fragment_t const& compile_udf_fragment(char const* source_code_cstr, char const* key)
{
  CUDF_FUNC_RANGE();

  return compile_fragment("cudf_udf_fragment", source_code_cstr, key);
}

kernel_ref compile_and_link_udf(char const* name,
                                char const* kernel_name,
                                char const* kernel_key,
                                char const* udf_code,
                                char const* udf_key)
{
  CUDF_FUNC_RANGE();

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

  // [ ] we also need to use the include dirs as part of the key
  auto& cache = get_rtc_cache();

  // TODO: (atomicity) should probably use query_or_insert
  if (auto lib = cache.query_library(library_key_sha256); lib.has_value()) {
    std::cout << "Loading kernel from memory\n";
    return lib->get()->get_kernel(kernel_name);
  } else if (auto disk_lib = cache.query_blob_from_disk(library_key_sha256); disk_lib.has_value()) {
    std::cout << "Loading kernel from disk cache\n";
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

  std::cout << "Compiling and linking library\n";
  auto const& library_frag = compile_library_fragment();

  auto const& udf_frag = compile_udf_fragment(udf_code, udf_key);

  std::promise<library> prom;
  auto fut = std::shared_future{prom.get_future()};
  // cache.store_library(library_key_sha256, fut);

  auto begin = std::chrono::high_resolution_clock::now();

  blob_view const link_fragments[]          = {library_frag.get(binary_type::LTO_IR)->view(),
                                               udf_frag.get(binary_type::LTO_IR)->view()};
  binary_type const fragment_binary_types[] = {binary_type::LTO_IR, binary_type::LTO_IR};

  char const* const fragment_names[] = {"cudf_lto_library", "cudf_udf_fragment"};

  // TODO: run compilation tests at program startup

  // TODO: optimization flags
  // TODO: split compile
  // TODO: split-compile-extended
  // TODO: lineinfo and debug info options
  // TODO: -kernels-used=
  // TODO: sass dump
  // TODO: time dump
  // TODO: env variable to control options
  // TODO: fma
  // TODO: variables-used
  // TODO: -optimize-unused-variables
  // TODO: -nocache
  // TODO: -device-stack-protector
  auto arch_flag                   = std::format("-arch=sm_{}", sm);
  char const* const link_options[] = {         // "-split-compile=0",
                                      "-lto",  // TODO: full flag names

                                      // "-optimize-unused-variables",
                                      "-kernels-used=transform_kernel",
                                      arch_flag.c_str()};

  auto const params = library_t::link_params{.name                  = name,
                                             .output_type           = binary_type::CUBIN,
                                             .fragments             = link_fragments,
                                             .fragment_binary_types = fragment_binary_types,
                                             .fragment_names        = fragment_names,
                                             .link_options          = link_options};
  // TODO: compilation flow logging with time taken, should be disabled when not in use
  auto lib = library_t::link(params);

  auto end = std::chrono::high_resolution_clock::now();
  auto dur = end - begin;
  std::cout << "RTC library linking for `" << name << "` took "
            << std::chrono::duration_cast<std::chrono::microseconds>(dur).count() << " us\n";
  // TODO: store to disk cache
  prom.set_value(std::move(lib));

  return fut.get()->get_kernel(kernel_name);
}

}  // namespace rtc
}  // namespace cudf
