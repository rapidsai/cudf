
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime.h>

#include <cudf_jit_embed.hpp>
#include <fcntl.h>
#include <jit/jit.hpp>
#include <librtcx/rtcx.hpp>
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

rtcx::sha256 hash_string(std::span<char const> input)
{
  rtcx::sha256_context ctx;
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

  RTCX_DEFER([&] {
    if (close(dst_file) != 0) {
      throw_posix(std::format("Failed to close file ({})", dst_path), "close");
    }
  });

  if (write(dst_file, contents.data(), contents.size()) == -1) {
    throw_posix(std::format("Failed to write file ({})", dst_path), "write");
  }
}

std::vector<unsigned char> decompress_blob(std::span<uint8_t const> compressed_binary,
                                           size_t uncompressed_size,
                                           char const* compression)
{
  std::vector<unsigned char> decompressed;
  decompressed.resize(uncompressed_size);

  if (std::string_view{compression} == "lz4") {
    int errc = LZ4_decompress_safe(reinterpret_cast<char const*>(compressed_binary.data()),
                                   reinterpret_cast<char*>(decompressed.data()),
                                   compressed_binary.size(),
                                   uncompressed_size);

    CUDF_EXPECTS(
      errc == static_cast<int64_t>(uncompressed_size),
      +std::format("Failed to decompress embedded RTC source files with LZ4, error code {}", errc),
      std::runtime_error);

  } else if (std::string_view{compression} == "zstd") {
    size_t const errc = ZSTD_decompress(
      decompressed.data(), uncompressed_size, compressed_binary.data(), compressed_binary.size());

    CUDF_EXPECTS(
      !ZSTD_isError(errc) && errc == uncompressed_size,
      +std::format("Failed to decompress embedded RTC source files with ZSTD, error code {} : ",
                   errc,
                   ZSTD_getErrorName(errc)),
      std::runtime_error);
  } else {
    // compression is "none", so just copy the data
    std::copy(compressed_binary.data(),
              compressed_binary.data() + compressed_binary.size(),
              decompressed.data());
  }

  return decompressed;
}

void install_file_set(char const* target_dir,
                      std::span<uint8_t const> compressed_binary,
                      size_t uncompressed_size,
                      std::span<rtcx_embed::range const> file_ranges,
                      std::span<char const* const> destinations,
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
    auto file_data       = std::span{files_data + file_data_range.offset, file_data_range.size};
    auto dst_path        = destinations[i];

    auto target_path = std::format("{}/{}", target_dir, dst_path);

    std::filesystem::create_directories(std::filesystem::path{target_path}.parent_path());
    install_file(target_path.c_str(), file_data);
  }
}

void install_cudf_jit_files(char const* target_dir)
{
  // directory does not exist, so create it
  char tmp_dir_[] = "/tmp/cudf-jit-tmpdir_XXXXXX";
  char* tmp_dir   = mkdtemp(tmp_dir_);
  if (tmp_dir == nullptr) {
    throw_posix(
      std::format("Failed to create temporary JIT install directory for ({})", target_dir),
      "mkdtemp");
  }

  install_file_set(target_dir,
                   rtcx_embed::cudf_jit_embed_files,
                   rtcx_embed::cudf_jit_embed_files_uncompressed_size,
                   rtcx_embed::cudf_jit_embed_file_ranges,
                   rtcx_embed::cudf_jit_embed_file_destinations,
                   rtcx_embed::cudf_jit_embed_files_compression);

  // rename the temporary directory to the target install directory
  if (rename(tmp_dir, target_dir) == -1) {
    throw_posix(std::format("Failed to rename temporary JIT install directory to ({})", target_dir),
                "rename");
  }
}

}  // namespace

jit_bundle_t::jit_bundle_t(std::string install_dir, rtcx::cache_t& cache)
  : install_dir_{std::move(install_dir)}, cache_{&cache}
{
  ensure_installed();
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
      install_cudf_jit_files(expected_path.c_str());
    }
  } else {
    // directory exists, perform minor sanity check
    CUDF_EXPECTS(S_ISDIR(path_info.st_mode),
                 +std::format("JIT install path ({}) exists but is not a directory", expected_path),
                 std::runtime_error);
  }
}

std::string jit_bundle_t::get_hash() const
{
  auto str = rtcx::sha256_hex_string::make(rtcx_embed::cudf_jit_embed_hash);
  return std::string{str.view()};
}

std::string jit_bundle_t::get_directory() const
{
  return std::format("{}/{}", install_dir_, get_hash());
}

std::vector<std::string> jit_bundle_t::get_include_directories() const
{
  std::vector<std::string> directories;
  auto base_dir = get_directory();

  for (auto dir : rtcx_embed::cudf_jit_embed_include_directories) {
    directories.emplace_back(std::format("{}/{}", base_dir, dir));
  }

  return directories;
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

std::tuple<rtcx::library, rtcx::blob> compile_library_uncached(char const* name,
                                                               char const* cuda_code,
                                                               bool use_pch,
                                                               bool log_pch)
{
  CUDF_FUNC_RANGE();

  auto& bundle = cudf::get_context().jit_bundle();
  auto begin   = std::chrono::steady_clock::now();
  auto sm      = get_current_device_physical_model();

  auto include_dirs = bundle.get_include_directories();
  auto pch_dir      = cudf::get_context().get_jit_pch_dir();

  std::vector<std::string> options;

  for (auto const& include_dir : include_dirs) {
    options.emplace_back(std::format("-I{}", include_dir));
  }

  // TODO: experiment with:
  // --fdevice-time-trace=jit_comp_trace.json
  // --time=compile_trace.json
  // -time
  // --restrict

  options.emplace_back(std::format("--gpu-architecture=sm_{}", sm));
  options.emplace_back("--minimal");
  options.emplace_back("-D__CUDACC_RTC__");
  options.emplace_back("-DCUDF_RUNTIME_JIT");
  options.emplace_back("--diag-suppress=47");
  options.emplace_back("--device-int128");

  if (use_pch) {
    options.emplace_back("--pch");

    if (log_pch) {
      options.emplace_back("--pch-verbose=true");
      options.emplace_back("--pch-messages=true");
    }
  }

  std::vector<char const*> options_cstr;
  for (auto const& option : options) {
    options_cstr.emplace_back(option.c_str());
  }

  auto params = rtcx::compile_params{.name        = name,
                                     .source      = cuda_code,
                                     .headers     = {},
                                     .options     = options_cstr,
                                     .target_type = rtcx::binary_type::CUBIN};

  auto cubin = rtcx::compile(params);

  auto end = std::chrono::steady_clock::now();

  auto duration = end - begin;

  CUDF_LOG_INFO(
    "Compiled fragment `%s` in %f ms",
    name,
    std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(duration).count());

  auto library = rtcx::load_library(cubin, rtcx::binary_type::CUBIN);

  auto blob = rtcx::blob_t::from_vector(std::move(cubin));

  return std::make_tuple(library, std::make_shared<rtcx::blob_t>(std::move(blob)));
}

}  // namespace

[[nodiscard]] rtcx::library get_library(std::string const& name,
                                        std::string const& key,
                                        std::string const& cuda_udf,
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

  auto cache_key = std::format(R"***(binary_type=CUBIN
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

  auto compile = [&] {
    return compile_library_uncached(name.c_str(), cuda_udf.c_str(), use_pch, log_pch);
  };

  if (!use_cache) {
    auto [lib, blob] = compile();
    return lib;
  }

  auto fut = cache.get_or_add_library(
    cache_key_sha256, rtcx::binary_type::CUBIN, rtcx::library_compile_func::from_functor(compile));

  return fut.get();
}

}  // namespace CUDF_EXPORT cudf
