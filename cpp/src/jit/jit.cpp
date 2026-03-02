
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
#include <jit/jit.hpp>
#include <jit/rtc/cache.hpp>
#include <jit/rtc/rtc.hpp>
#include <jit/rtc/sha256.hpp>
#include <jit_embed/cudf_jit_embed/embed.hpp>
#include <lz4.h>
#include <runtime/context.hpp>
#include <sys/file.h>
#include <sys/stat.h>
#include <zstd.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <format>
#include <future>

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

namespace CUDF_EXPORT cudf {

namespace {

rtc::sha256_hash hash_string(std::span<char const> input)
{
  rtc::sha256_context ctx;
  ctx.update(std::span{reinterpret_cast<uint8_t const*>(input.data()), input.size()});
  return ctx.finalize();
}

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

std::vector<unsigned char> decompress_blob(jit_bytes_t const& compressed_binary,
                                           size_t uncompressed_size,
                                           char const* compression)
{
  std::vector<unsigned char> decompressed;
  decompressed.resize(uncompressed_size);

  if (std::string_view{compression} == "lz4") {
    int errc = LZ4_decompress_safe(reinterpret_cast<char const*>(compressed_binary.data),
                                   reinterpret_cast<char*>(decompressed.data()),
                                   compressed_binary.size,
                                   uncompressed_size);

    CUDF_EXPECTS(
      errc == static_cast<int64_t>(uncompressed_size),
      +std::format("Failed to decompress embedded RTC source files with LZ4, error code {}", errc),
      std::runtime_error);

  } else if (std::string_view{compression} == "zstd") {
    size_t const errc = ZSTD_decompress(
      decompressed.data(), uncompressed_size, compressed_binary.data, compressed_binary.size);

    CUDF_EXPECTS(
      !ZSTD_isError(errc) && errc == uncompressed_size,
      +std::format("Failed to decompress embedded RTC source files with ZSTD, error code {} : ",
                   errc,
                   ZSTD_getErrorName(errc)),
      std::runtime_error);
  } else {
    // compression is "none", so just copy the data
    std::copy(
      compressed_binary.data, compressed_binary.data + compressed_binary.size, decompressed.data());
  }

  return decompressed;
}

void install_file_set(char const* target_dir,
                      jit_bytes_t const& compressed_binary,
                      size_t uncompressed_size,
                      std::span<jit_byte_range_t const> file_ranges,
                      jit_bytes_array_t const& dst,
                      char const* compression)
{
  CUDF_EXPECTS(compression != nullptr, "Compression type must be specified", std::runtime_error);
  CUDF_EXPECTS(compression == std::string_view{"none"} || compression == std::string_view{"lz4"} ||
                 compression == std::string_view{"zstd"},
               +std::format("Unsupported compression type specified: {}", compression),
               std::runtime_error);

  auto decompressed = decompress_blob(compressed_binary, uncompressed_size, compression);
  auto files_data   = decompressed.data();

  for (size_t i = 0; i < file_ranges.size(); ++i) {
    auto file_data_range = file_ranges[i];
    auto dst_range       = dst.ranges[i];
    auto file_data       = std::span{files_data + file_data_range.offset, file_data_range.size};
    auto dst_path        = std::string_view{
      reinterpret_cast<char const*>(dst.bytes.data) + dst_range.offset, dst_range.size};

    auto target_path = std::format("{}/{}", target_dir, dst_path);

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
                   cudf_jit_embed_blobs_file_destinations,
                   cudf_jit_embed_blobs_compression);

  install_file_set(target_dir,
                   cudf_jit_embed_sources_binary,
                   cudf_jit_embed_sources_uncompressed_size,
                   cudf_jit_embed_sources_ranges,
                   cudf_jit_embed_sources_file_destinations,
                   cudf_jit_embed_sources_compression);
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

jit_bundle_t::jit_bundle_t(std::string install_dir, rtc::cache_t& cache)
  : install_dir_{std::move(install_dir)}, cache_{&cache}
{
  ensure_installed();
  preload_lto_library();
  // TODO: fix cmake tracking of the scripts and embedded files
}

void jit_bundle_t::ensure_installed() const
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
      CUDF_LOG_INFO("Creating JIT install directory at ({})", expected_path);
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

void jit_bundle_t::preload_lto_library()
{
  auto& cache = *cache_;

  auto bundle_hash = get_hash();

  auto cache_key = std::format(R"***(fragment_type=FATBIN
key={}
bundle={})***",
                               "cudf_lto_library",
                               bundle_hash);

  auto cache_key_sha256 = hash_string(cache_key);

  auto compile = [&] {
    auto path  = std::format("{}/{}", get_directory(), "cudf_lto_library.fatbin");
    auto cubin = rtc::blob_t::from_file(path.c_str());
    CUDF_EXPECTS(cubin.has_value(),
                 +std::format("Failed to load LTO library cubin from disk at ({})", path),
                 std::runtime_error);
    rtc::fragment_t::load_params load_params{
      .binary = std::make_shared<rtc::blob_t>(std::move(*cubin)), .type = rtc::binary_type::FATBIN};
    return rtc::fragment_t::load(load_params);
  };

  auto fut =
    cache.query_or_insert_fragment(cache_key_sha256,
                                   rtc::binary_type::FATBIN,
                                   rtc::fragment_compile_function_t::from_functor(compile));

  lto_library_ = fut.get();
}

std::string jit_bundle_t::get_hash() const
{
  auto str = rtc::sha256_hex_string::make(
    std::span{cudf_jit_embed_hash.data, static_cast<size_t>(cudf_jit_embed_hash.size)});
  return std::string{str.view()};
}

std::string jit_bundle_t::get_directory() const
{
  return std::format("{}/{}", install_dir_, get_hash());
}

rtc::fragment jit_bundle_t::get_lto_library() const { return lto_library_; }

std::vector<std::string> jit_bundle_t::get_include_directories() const
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

std::vector<std::string> jit_bundle_t::get_compile_options() const
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

rtc::fragment compile_udf_uncached(char const* name,
                                   char const* cuda_code,
                                   bool use_pch,
                                   bool log_pch)
{
  CUDF_FUNC_RANGE();

  auto& bundle = cudf::get_context().jit_bundle();
  auto begin   = std::chrono::steady_clock::now();
  auto sm      = get_current_device_physical_model();

  auto include_dirs    = bundle.get_include_directories();
  auto compile_options = bundle.get_compile_options();
  auto pch_dir         = cudf::get_context().get_jit_pch_dir();

  std::vector<std::string> options;

  for (auto const& include_dir : include_dirs) {
    options.emplace_back(std::format("-I{}", include_dir));
  }

  for (auto const& compile_option : compile_options) {
    options.emplace_back(compile_option);
  }

  // TODO: experiment with:
  // --fdevice-time-trace=jit_comp_trace.json
  // --time=compile_trace.json
  // -time

  options.emplace_back(std::format("--gpu-architecture=sm_{}", sm));
  options.emplace_back("--dlink-time-opt");
  options.emplace_back("--relocatable-device-code=true");
  options.emplace_back("--device-as-default-execution-space");
  options.emplace_back("--restrict");
  options.emplace_back("--minimal");
  options.emplace_back("--split-compile=0");

  if (use_pch) {
    options.emplace_back("--pch");
    options.emplace_back(std::format("--pch-dir={}", pch_dir));

    if (log_pch) {
      options.emplace_back("--pch-verbose=true");
      options.emplace_back("--pch-messages=true");
    }
  }

  std::vector<char const*> options_cstr;
  for (auto const& option : options) {
    options_cstr.emplace_back(option.c_str());
  }

  auto params = rtc::fragment_t::compile_params{.name        = name,
                                                .source      = cuda_code,
                                                .headers     = {},
                                                .options     = options_cstr,
                                                .target_type = rtc::binary_type::LTO_IR};

  auto frag = rtc::fragment_t::compile(params);

  auto end = std::chrono::steady_clock::now();

  auto duration = end - begin;

  CUDF_LOG_WARN(
    "Compiled fragment `%s` in %f ms",
    name,
    std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(duration).count());

  return frag;
}

std::tuple<rtc::library, rtc::blob> link_udf_uncached(char const* name,
                                                      rtc::fragment const& fragment,
                                                      char const* kernel_symbol)
{
  CUDF_FUNC_RANGE();

  auto sm      = get_current_device_physical_model();
  auto& bundle = cudf::get_context().jit_bundle();

  auto begin   = std::chrono::steady_clock::now();
  auto library = bundle.get_lto_library();

  // TODO: sass dump
  // TODO: time trace dump
  // TODO: lineinfo and debug info options
  // TODO: -nocache

  std::vector<std::string> options;

  options.emplace_back("-O3");
  options.emplace_back("-lto");
  options.emplace_back(std::format("-arch=sm_{}", sm));
  options.emplace_back(std::format("-kernels-used={}", kernel_symbol));
  options.emplace_back("-optimize-unused-variables");
  options.emplace_back("-split-compile=0");

  std::vector<char const*> options_cstr;
  for (auto const& option : options) {
    options_cstr.emplace_back(option.c_str());
  }

  rtc::blob_view link_fragments[] = {library->get(rtc::binary_type::FATBIN)->view(),
                                     fragment->get(rtc::binary_type::LTO_IR)->view()};

  rtc::binary_type fragment_binary_types[] = {rtc::binary_type::FATBIN, rtc::binary_type::LTO_IR};

  char const* fragment_names[] = {"cudf_lto_library", name};

  auto params = rtc::library_t::link_params{.name                  = name,
                                            .output_type           = rtc::binary_type::CUBIN,
                                            .fragments             = link_fragments,
                                            .fragment_binary_types = fragment_binary_types,
                                            .fragment_names        = fragment_names,
                                            .link_options          = options_cstr};

  auto blob = rtc::library_t::link_as_blob(params);

  auto load_params =
    rtc::library_t::load_params{.binary = blob->view(), .type = rtc::binary_type::CUBIN};

  auto linked_library = rtc::library_t::load(load_params);

  auto end = std::chrono::steady_clock::now();

  auto duration = end - begin;

  CUDF_LOG_WARN(
    "Linked fragment `%s` in %f ms",
    name,
    std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(duration).count());

  return std::make_tuple(linked_library, blob);
}

rtc::fragment compile_udf(char const* name,
                          char const* key,
                          char const* cuda_udf,
                          bool use_cache,
                          bool use_pch,
                          bool log_pch)
{
  CUDF_FUNC_RANGE();

  auto& cache  = cudf::get_context().rtc_cache();
  auto& bundle = cudf::get_context().jit_bundle();

  auto runtime     = get_runtime_version();
  auto driver      = get_driver_version();
  auto sm          = get_current_device_physical_model();
  auto bundle_hash = bundle.get_hash();

  auto cache_key = std::format(R"***(fragment_type=LTO_IR
key={}
cuda_runtime={}
cuda_driver={}
arch={}
bundle={})***",
                               key,
                               runtime,
                               driver,
                               sm,
                               bundle_hash);

  auto cache_key_sha256 = hash_string(cache_key);

  auto compile = [&] { return compile_udf_uncached(name, cuda_udf, use_pch, log_pch); };

  if (!use_cache) { return compile(); }

  auto fut =
    cache.query_or_insert_fragment(cache_key_sha256,
                                   rtc::binary_type::LTO_IR,
                                   rtc::fragment_compile_function_t::from_functor(compile));

  return fut.get();
}

rtc::library link_udf(char const* name,
                      char const* key,
                      rtc::fragment const& fragment,
                      char const* kernel_symbol,
                      bool use_cache)
{
  CUDF_FUNC_RANGE();

  auto& cache  = cudf::get_context().rtc_cache();
  auto& bundle = cudf::get_context().jit_bundle();

  auto runtime     = get_runtime_version();
  auto driver      = get_driver_version();
  auto sm          = get_current_device_physical_model();
  auto bundle_hash = bundle.get_hash();

  auto cache_key = std::format(R"***(library_type=CUBIN
key={}
kernel={}
cuda_runtime={}
cuda_driver={}
arch={},
bundle={})***",
                               key,
                               kernel_symbol,
                               runtime,
                               driver,
                               sm,
                               bundle_hash);

  auto cache_key_sha256 = hash_string(cache_key);

  auto link = [&] { return link_udf_uncached(name, fragment, kernel_symbol); };

  if (!use_cache) {
    auto [lib, blob] = link();
    return lib;
  }

  auto fut = cache.query_or_insert_library(
    cache_key_sha256, rtc::binary_type::CUBIN, rtc::library_compile_function_t::from_functor(link));

  return fut.get();
}

rtc::library compile_cuda_library(char const* name,
                                  char const* key,
                                  char const* cuda_udf,
                                  char const* kernel_symbol,
                                  bool use_cache,
                                  bool use_pch,
                                  bool log_pch)
{
  CUDF_FUNC_RANGE();
  auto fragment = compile_udf(name, key, cuda_udf, use_cache, use_pch, log_pch);
  auto library  = link_udf(name, key, fragment, kernel_symbol, use_cache);
  return library;
}

}  // namespace

rtc::library compile_kernel(std::string const& name,
                            std::string const& key,
                            std::string const& cuda_udf,
                            std::string const& kernel_symbol,
                            bool use_cache,
                            bool use_pch,
                            bool log_pch)
{
  return compile_cuda_library(name.c_str(),
                              key.c_str(),
                              cuda_udf.c_str(),
                              kernel_symbol.c_str(),
                              use_cache,
                              use_pch,
                              log_pch);
}

rtc::library compile_lto_ir_kernel(std::string const& name,
                                   std::string const& key,
                                   std::span<uint8_t const> lto_ir_binary,
                                   std::string const& kernel_symbol,
                                   bool use_cache)
{
  auto blob     = std::make_shared<rtc::blob_t>(rtc::blob_t::from_static_data(lto_ir_binary));
  auto fragment = rtc::fragment_t::load(
    rtc::fragment_t::load_params{.binary = blob, .type = rtc::binary_type::LTO_IR});
  return link_udf(name.c_str(), key.c_str(), fragment, kernel_symbol.c_str(), use_cache);
}

}  // namespace CUDF_EXPORT cudf
