
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/defer.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime.h>

#include <fcntl.h>
#include <jit/rtc/cache.hpp>
#include <jit/rtc/cudf.hpp>
#include <jit/rtc/rtc.hpp>
#include <jit/rtc/sha256.hpp>
#include <jit_embed/cudf_jit_embed/embed.hpp>
#include <lz4.h>
#include <runtime/context.hpp>
#include <sys/file.h>
#include <sys/stat.h>

#include <cerrno>
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

void install_file(char const* dst_path, std::span<unsigned char const> contents)
{
  int dst_file = open(dst_path, O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  if (dst_file == -1) {
    if (errno == EEXIST) {
      // file already exists, so just return
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

void install_file_set(char const* target_dir,
                      jit_bytes_t const& compressed_binary,
                      size_t uncompressed_size,
                      std::span<jit_byte_range_t const> file_ranges,
                      jit_bytes_array_t const& dst)
{
  std::vector<unsigned char> decompressed;
  decompressed.resize(uncompressed_size);

  int errc = LZ4_decompress_safe(reinterpret_cast<char const*>(compressed_binary.data),
                                 reinterpret_cast<char*>(decompressed.data()),
                                 compressed_binary.size,
                                 uncompressed_size);

  CUDF_EXPECTS(errc == static_cast<int64_t>(uncompressed_size),
               "Failed to decompress embedded RTC source files",
               std::runtime_error);

  auto const files_data = decompressed.data();

  for (size_t i = 0; i < file_ranges.size(); ++i) {
    auto const file_data_range = file_ranges[i];
    auto const dst_range       = dst.ranges[i];
    auto const file_data = std::span{files_data + file_data_range.offset, file_data_range.size};
    auto const dst_path  = std::string_view{
      reinterpret_cast<char const*>(dst.bytes.data) + dst_range.offset, dst_range.size};

    auto const target_path = std::format("{}/{}", target_dir, dst_path);

    std::filesystem::create_directories(std::filesystem::path{target_path}.parent_path());
    install_file(target_path.c_str(), file_data);
  }
}

void install_cudf_jit(char const* target_dir)
{
  install_file_set(target_dir,
                   cudf_jit_embed_blobs_binary,
                   cudf_jit_embed_blobs_uncompressed_size,
                   cudf_jit_embed_blobs_ranges,
                   cudf_jit_embed_blobs_file_destinations);

  install_file_set(target_dir,
                   cudf_jit_embed_sources_binary,
                   cudf_jit_embed_sources_uncompressed_size,
                   cudf_jit_embed_sources_ranges,
                   cudf_jit_embed_sources_file_destinations);
}

void create_and_install_cudf_jit(char const* target_dir)
{
  // directory does not exist, so create it
  char tmp_dir_[] = "/tmp/cudf-jit-tmpdir_XXXXXX";
  char* tmp_dir   = mkdtemp(tmp_dir_);
  if (tmp_dir == nullptr) {
    throw_posix(
      std::format("Failed to create temporary JIT install directory for ({})", target_dir),
      "mkdtemp");
  }

  install_cudf_jit(tmp_dir);

  // rename the temporary directory to the target install directory
  if (rename(tmp_dir, target_dir) == -1) {
    throw_posix(std::format("Failed to rename temporary JIT install directory to ({})", target_dir),
                "rename");
  }
}

}  // namespace

jit_bundle::jit_bundle(std::string install_dir) : install_dir_{std::move(install_dir)}
{
  ensure_installed();
  preload_lto_library();
}

void jit_bundle::ensure_installed() const
{
  CUDF_FUNC_RANGE();

  auto expected_hash = get_hash();
  auto expected_path = std::format("{}/{}", install_dir_, expected_hash);

  struct stat path_info;

  if (lstat(expected_path.c_str(), &path_info) == -1) {
    if (errno != ENOENT) {
      throw_posix(std::format("Failed to get stat for directory ({})", expected_path), "lstat");
    } else {
      // ensure base install directory exists
      std::filesystem::create_directories(install_dir_);
      create_and_install_cudf_jit(expected_path.c_str());
    }
  } else {
    // directory exists, perform minor sanity check
    CUDF_EXPECTS(S_ISDIR(path_info.st_mode),
                 +std::format("JIT install path ({}) exists but is not a directory", expected_path),
                 std::runtime_error);
  }
}

void jit_bundle::preload_lto_library()
{
  auto& cache = get_rtc_cache();

  auto bundle_hash = get_hash();

  auto const cache_key = std::format(R"***(
      fragment_type=FATBIN,
      key={},
      bundle={}
      )***",
                                     "cudf_lto_library",
                                     bundle_hash);

  auto const cache_key_sha256 = hash_string(cache_key);

  auto compile = [&] {
    auto directory = get_directory();
    auto path      = std::format("{}/{}", directory, "cudf_lto_library.fatbin");
    auto cubin     = blob_t::from_file(path.c_str());
    CUDF_EXPECTS(cubin.has_value(),
                 +std::format("Failed to load LTO library cubin from disk at ({})", path),
                 std::runtime_error);
    fragment_t::load_params load_params{.binary = std::make_shared<blob_t>(std::move(*cubin)),
                                        .type   = binary_type::FATBIN};
    return fragment_t::load(load_params);
  };

  auto fut = cache.query_or_insert_fragment(cache_key_sha256, binary_type::FATBIN, compile);

  lto_library_ = fut.get();
}

std::string jit_bundle::get_hash() const
{
  auto str = sha256_hex_string::make(
    std::span{cudf_jit_embed_hash.data, static_cast<size_t>(cudf_jit_embed_hash.size)});
  return std::string{str.view()};
}

std::string jit_bundle::get_directory() const
{
  auto hash = get_hash();
  return std::format("{}/{}", install_dir_, hash);
}

fragment jit_bundle::get_lto_library() const { return lto_library_; }

std::vector<std::string> jit_bundle::get_include_directories() const
{
  std::vector<std::string> directories;
  auto base_dir = get_directory();

  auto include_directories_data =
    reinterpret_cast<char const*>(cudf_jit_embed_sources_include_directories.bytes.data);

  for (size_t i = 0; i < cudf_jit_embed_sources_include_directories.num_ranges; i++) {
    auto range                  = cudf_jit_embed_sources_include_directories.ranges[i];
    auto dest_include_directory = include_directories_data + range.offset;
    directories.emplace_back(std::format("{}/{}", base_dir, dest_include_directory));
  }

  return directories;
}

std::vector<std::string> jit_bundle::get_compile_options() const
{
  std::vector<std::string> options;

  auto embed_options_data = reinterpret_cast<const char*>(cudf_jit_embed_options.bytes.data);

  for (size_t i = 0; i < cudf_jit_embed_options.num_ranges; i++) {
    auto range  = cudf_jit_embed_options.ranges[i];
    auto option = embed_options_data + range.offset;
    options.emplace_back(option);
  }

  return options;
}

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

}  // namespace

fragment get_or_compile_fragment(char const* name, char const* source_code_cstr, char const* key)
{
  CUDF_FUNC_RANGE();

  auto& bundle = get_bundle();
  auto& cache  = get_rtc_cache();

  auto runtime     = get_runtime_version();
  auto driver      = get_driver_version();
  auto sm          = get_current_device_physical_model();
  auto bundle_hash = bundle.get_hash();

  auto const cache_key = std::format(R"***(
      fragment_type=LTO_IR,
      key={},
      cuda_runtime={},
      cuda_driver={},
      arch={},
      bundle={}
      )***",
                                     key,
                                     runtime,
                                     driver,
                                     sm,
                                     bundle_hash);

  auto const cache_key_sha256 = hash_string(cache_key);

  // TODO: add time function in cache

  auto compile = [&] {
    auto begin       = std::chrono::high_resolution_clock::now();
    auto cache_dir   = cache.get_cache_dir();
    auto install_dir = std::format("{}/jit-install", cache_dir);

    auto include_dirs    = bundle.get_include_directories();
    auto compile_options = bundle.get_compile_options();

    std::vector<std::string> options;

    for (auto const& include_dir : include_dirs) {
      options.emplace_back(std::format("-I{}", include_dir));
    }

    for (auto const& compile_option : compile_options) {
      options.emplace_back(compile_option);
    }

    options.emplace_back(std::format("--gpu-architecture=sm_{}", sm));
    options.emplace_back("--dlink-time-opt");
    options.emplace_back("--relocatable-device-code=true");
    options.emplace_back("--device-as-default-execution-space");

    // TODO: experiment with:
    // --split-compile=0
    // --fdevice-time-trace=jit_comp_trace.json
    // --minimal
    // --time=compile_trace.json
    // -time
    // --fast-compile
    // --pch
    // --pch-dir=/tmp/cudf-rtc-pch

    std::vector<char const*> options_cstr;
    for (auto const& option : options) {
      options_cstr.emplace_back(option.c_str());
    }

    auto const params = fragment_t::compile_params{.name        = name,
                                                   .source      = source_code_cstr,
                                                   .headers     = {},
                                                   .options     = options_cstr,
                                                   .target_type = binary_type::LTO_IR};

    auto frag = fragment_t::compile(params);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = end - begin;

    CUDF_LOG_INFO(
      "Compiled fragment `{}` in {} ms",
      name,
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(duration).count());

    return frag;
  };

  auto fut = cache.query_or_insert_fragment(cache_key_sha256, binary_type::LTO_IR, compile);

  return fut.get();
}

jit_bundle& get_bundle()
{
  auto& context = cudf::get_context();
  return context.jit_bundle();
}

library compile_and_link_udf(char const* name,
                             char const* udf_code,
                             char const* udf_key,
                             char const* kernel_symbol)
{
  CUDF_FUNC_RANGE();

  auto& cache  = get_rtc_cache();
  auto& bundle = get_bundle();

  auto runtime     = get_runtime_version();
  auto driver      = get_driver_version();
  auto sm          = get_current_device_physical_model();
  auto bundle_hash = bundle.get_hash();

  auto compile = [&] {
    auto begin    = std::chrono::high_resolution_clock::now();
    auto library  = bundle.get_lto_library();
    auto fragment = get_or_compile_fragment(name, udf_code, udf_key);

    // TODO: experiment with:
    // optimization flags
    // split compile
    // split-compile-extended
    // lineinfo and debug info options
    // -kernels-used=
    // sass dump
    // time dump
    // env variable to control options
    // fma
    // variables-used
    // -optimize-unused-variables
    // -nocache
    // -device-stack-protector

    std::vector<std::string> options;

    options.emplace_back("-lto");
    options.emplace_back(std::format("-arch=sm_{}", sm));
    options.emplace_back(std::format("-kernels-used={}", kernel_symbol));

    std::vector<char const*> options_cstr;
    for (auto const& option : options) {
      options_cstr.emplace_back(option.c_str());
    }

    blob_view const link_fragments[] = {library->get(binary_type::LTO_IR)->view(),
                                        fragment->get(binary_type::LTO_IR)->view()};

    binary_type const fragment_binary_types[] = {binary_type::LTO_IR, binary_type::LTO_IR};

    char const* const fragment_names[] = {"cudf_lto_library", "cudf_udf_fragment"};

    auto const params = library_t::link_params{.name                  = name,
                                               .output_type           = binary_type::CUBIN,
                                               .fragments             = link_fragments,
                                               .fragment_binary_types = fragment_binary_types,
                                               .fragment_names        = fragment_names,
                                               .link_options          = options_cstr};

    auto blob = library_t::link_as_blob(params);

    auto load_params = library_t::load_params{.binary = blob->view(), .type = binary_type::CUBIN};

    auto linked_library = library_t::load(load_params);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = end - begin;

    CUDF_LOG_INFO(
      "Compiled fragment `{}` in {} ms",
      name,
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(duration).count());

    return std::make_tuple(linked_library, blob);
  };

  auto library_cache_key              = std::format(R"***(
      library_type=CUBIN,
      kernels={},
      udf={},
      cuda_runtime={},
      cuda_driver={},
      arch={})***",
                                       kernel_symbol,
                                       udf_key,
                                       runtime,
                                       driver,
                                       sm);
  auto const library_cache_key_sha256 = hash_string(library_cache_key);

  auto library =
    cache.query_or_insert_library(library_cache_key_sha256, binary_type::CUBIN, compile);

  return library.get();
}

}  // namespace rtc
}  // namespace cudf
