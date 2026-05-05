/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime.h>

#include <cudf_jit_embed.hpp>
#include <jit/jit.hpp>
#include <rtcx.hpp>
#include <runtime/context.hpp>

#include <filesystem>
#include <format>
#include <fstream>
#include <future>

namespace CUDF_EXPORT cudf {

namespace {

rtcx::sha256 hash(char const* input)
{
  rtcx::sha256_context ctx;
  ctx.update(std::span{reinterpret_cast<uint8_t const*>(input), std::strlen(input)});
  return ctx.finalize();
}

rtcx::sha256 hash(std::span<char const> input)
{
  rtcx::sha256_context ctx;
  ctx.update(std::span{reinterpret_cast<uint8_t const*>(input.data()), input.size()});
  return ctx.finalize();
}

rtcx::sha256 hash(std::span<char const* const> inputs)
{
  rtcx::sha256_context ctx;
  for (auto const* input : inputs) {
    ctx.update(std::span{reinterpret_cast<uint8_t const*>(input), std::strlen(input)});
  }
  return ctx.finalize();
}

[[noreturn]] void throw_posix(std::string_view message, std::string_view syscall_name)
{
  auto error_code = errno;
  auto error_str  = std::format(
    "{}. `{}` failed with {} ({})", message, syscall_name, error_code, std::strerror(error_code));
  CUDF_FAIL(error_str, std::runtime_error);
}

void install_file(char const* dst_path, std::span<std::uint8_t const> contents)
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

/**
 * @brief Reads the contents of a file into a byte buffer and null-terminates it to allow for safe
 * usage as a C-string.
 */
rtcx::byte_buffer read_blob_cstring(char const* path)
{
  int32_t fd = open(path, O_RDONLY);
  if (fd == -1) { throw_posix(std::format("Failed to open file ({})", path), "open"); }

  RTCX_DEFER([&] {
    if (close(fd) == -1) { throw_posix(std::format("Failed to close file ({})", path), "close"); }
  });

  auto file_size = lseek(fd, 0, SEEK_END);
  if (file_size == -1) {
    throw_posix(std::format("Failed to determine size of file ({})", path), "lseek");
  }
  // TODO: make all read/write syscalls call read/write in a loop

  if (lseek(fd, 0, SEEK_SET) == -1) {
    throw_posix(std::format("Failed to reset file offset for file ({})", path), "lseek");
  }

  auto contents = rtcx::byte_buffer::make(file_size + 1U);  // +1 for null terminator

  if (read(fd, contents.data(), file_size) == -1) {
    throw_posix(std::format("Failed to read file ({})", path), "read");
  }

  contents.data()[file_size] = '\0';  // null-terminate the buffer

  return contents;
}

rtcx::byte_buffer decompress_blob(std::span<uint8_t const> compressed_binary,
                                  size_t uncompressed_size,
                                  std::string_view compression)
{
  CUDF_EXPECTS(compression == "none" || compression == "zstd",
               std::format("Unsupported compression type specified: {}", compression),
               std::runtime_error);
  auto decompressed = rtcx::byte_buffer::make(uncompressed_size);

  if (compression == "zstd") {
    size_t errc = ZSTD_decompress(
      decompressed.data(), uncompressed_size, compressed_binary.data(), compressed_binary.size());

    CUDF_EXPECTS(
      !ZSTD_isError(errc) && errc == uncompressed_size,
      std::format("Failed to decompress embedded RTC source files with ZSTD, error code {} : ",
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

void install_file_set(std::string_view target_dir,
                      std::span<uint8_t const> compressed_binary,
                      size_t uncompressed_size,
                      std::span<rtcx_embed::range const> file_ranges,
                      std::span<char const* const> destinations,
                      std::string_view compression)
{
  auto decompressed = rtcx::decompress_blob(compressed_binary, uncompressed_size, compression);
  for (size_t i = 0; i < file_ranges.size(); ++i) {
    auto file_data_range = file_ranges[i];
    auto file_data = std::span{decompressed.data() + file_data_range.offset, file_data_range.size};
    auto dst_path  = destinations[i];
    auto target_path = std::format("{}/{}", target_dir, dst_path);

    std::filesystem::create_directories(std::filesystem::path{target_path}.parent_path());

    std::ofstream file(target_path, std::ios::binary);
    if (!file) {
      throw std::runtime_error(
        std::format("Failed to open file for writing at path: {}", target_path));
    }

    file.write(reinterpret_cast<char const*>(file_data.data()), file_data.size());
    if (!file) {
      throw std::runtime_error(std::format("Failed to write file at path: {}", target_path));
    }
  }
}

std::string read_file_string(char const* path)
{
  std::ifstream file(std::string{path}, std::ios::binary | std::ios::ate);
  if (!file) { throw std::runtime_error(std::format("Failed to open file at path: {}", path)); }

  auto size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string contents(size, '\0');
  if (!file.read(contents.data(), size)) {
    throw std::runtime_error(std::format("Failed to read file at path: {}", path));
  }

  return contents;
}

void install_cudf_jit_files(std::string const& target_dir, std::string const& tmp_dir)
{
  // directory does not exist, so create it
  auto tmp_dir_path_str = std::format("{}/cudf-jit-tmpdir_XXXXXX", tmp_dir);
  char* tmp_dir_path    = ::mkdtemp(tmp_dir_path_str.data());
  CUDF_EXPECTS(
    tmp_dir_path != nullptr,
    std::format("Failed to create temporary directory for JIT file installation in tmp dir: {}",
                tmp_dir),
    std::runtime_error);

  install_file_set(tmp_dir_path,
                   rtcx_embed::cudf_jit_embed_files,
                   rtcx_embed::cudf_jit_embed_files_uncompressed_size,
                   rtcx_embed::cudf_jit_embed_file_ranges,
                   rtcx_embed::cudf_jit_embed_file_destinations,
                   rtcx_embed::cudf_jit_embed_files_compression);

  // rename the temporary directory to the target install directory
  if (::rename(tmp_dir_path, target_dir.c_str()) == -1) {
    auto errc = errno;
    // another process created it
    if (errc == ENOTEMPTY || errc == EEXIST) {
      std::filesystem::remove_all(tmp_dir_path);
    } else {
      CUDF_FAIL(
        std::format("Failed to install JIT files to target directory: {} with error ({}): {}",
                    target_dir,
                    errc,
                    std::strerror(errc)),
        std::runtime_error);
    }
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

  if (!std::filesystem::exists(expected_path)) {
    // ensure base install directory exists
    std::filesystem::create_directories(install_dir_);
    install_cudf_jit_files(expected_path.c_str(), cache_->get_tmp_dir());
  } else {
    // directory exists, perform minor sanity check
    CUDF_EXPECTS(std::filesystem::is_directory(expected_path),  // throws if path does not exist
                 std::format("JIT install path ({}) exists but is not a directory", expected_path),
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
  CUDF_CUDA_TRY(cudaDriverGetVersion(&driver_version));
  return driver_version;
}

int32_t get_runtime_version()
{
  int32_t runtime_version;
  CUDF_CUDA_TRY(cudaRuntimeGetVersion(&runtime_version));
  return runtime_version;
}

int32_t get_current_device_physical_model()
{
  int32_t device;
  CUDF_CUDA_TRY(cudaGetDevice(&device));

  cudaDeviceProp props;
  CUDF_CUDA_TRY(cudaGetDeviceProperties(&props, device));

  return props.major * 10 + props.minor;
}

std::tuple<rtcx::library, rtcx::blob> compile_library_uncached(
  char const* name,
  char const* cuda_code,
  std::span<char const* const> extra_header_include_names,
  std::span<char const* const> extra_headers,
  std::span<char const* const> extra_options,
  std::span<char const* const> name_expressions,
  bool use_pch,
  bool use_minimal,
  bool log_pch)
{
  CUDF_FUNC_RANGE();

  auto& bundle = cudf::get_context().jit_bundle();
  auto sm      = get_current_device_physical_model();

  auto include_dirs = bundle.get_include_directories();
  auto pch_dir      = cudf::get_context().get_jit_pch_dir();

  std::vector<std::string> options;

  for (auto const& include_dir : include_dirs) {
    options.emplace_back(std::format("-I{}", include_dir));
  }

  options.emplace_back(std::format("--gpu-architecture=sm_{}", sm));

  options.emplace_back("--diag-suppress=47");
  options.emplace_back("--device-int128");

  if (sm >= 100) { options.emplace_back("--device-float128"); }

  options.emplace_back("-std=c++20");
  options.emplace_back("--device-as-default-execution-space");
  options.emplace_back("--generate-line-info");
  options.emplace_back("--dopt=on");

  if (use_minimal) { options.emplace_back("--minimal"); }

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
  for (auto* option : extra_options) {
    options_cstr.emplace_back(option);
  }

  auto params = rtcx::compile_params{.name                 = name,
                                     .source               = cuda_code,
                                     .header_include_names = extra_header_include_names,
                                     .headers              = extra_headers,
                                     .options              = options_cstr,
                                     .name_expressions     = name_expressions,
                                     .target_type          = rtcx::binary_type::CUBIN};

  auto cubin   = rtcx::compile(params);
  auto library = rtcx::load_library(cubin);
  auto blob    = rtcx::blob_t::from_buffer(std::move(cubin));

  return std::make_tuple(library, std::make_shared<rtcx::blob_t>(std::move(blob)));
}

rtcx::byte_buffer compile_fragment_uncached(char const* name,
                                            char const* cuda_code,
                                            std::span<char const* const> extra_header_include_names,
                                            std::span<char const* const> extra_headers,
                                            std::span<char const* const> extra_options,
                                            std::span<char const* const> name_expressions,
                                            bool use_pch,
                                            bool use_minimal,
                                            bool log_pch)
{
  CUDF_FUNC_RANGE();

  auto& bundle = cudf::get_context().jit_bundle();
  auto sm      = get_current_device_physical_model();

  auto include_dirs = bundle.get_include_directories();
  auto pch_dir      = cudf::get_context().get_jit_pch_dir();

  std::vector<std::string> options;

  for (auto const& include_dir : include_dirs) {
    options.emplace_back(std::format("-I{}", include_dir));
  }

  options.emplace_back(std::format("--gpu-architecture=sm_{}", sm));

  options.emplace_back("--diag-suppress=47");
  options.emplace_back("--device-int128");

  if (sm >= 100) { options.emplace_back("--device-float128"); }

  options.emplace_back("-std=c++20");
  options.emplace_back("--device-as-default-execution-space");
  options.emplace_back("--generate-line-info");
  options.emplace_back("--dopt=on");

  if (use_minimal) { options.emplace_back("--minimal"); }

  if (use_pch) {
    options.emplace_back("--pch");

    if (log_pch) {
      options.emplace_back("--pch-verbose=true");
      options.emplace_back("--pch-messages=true");
    }
  }

  options.emplace_back("--relocatable-device-code=true");
  options.emplace_back("--gen-opt-lto");
  options.emplace_back("--dlink-time-opt");

  std::vector<char const*> options_cstr;
  for (auto const& option : options) {
    options_cstr.emplace_back(option.c_str());
  }
  for (auto* option : extra_options) {
    options_cstr.emplace_back(option);
  }

  auto params = rtcx::compile_params{.name                 = name,
                                     .source               = cuda_code,
                                     .header_include_names = extra_header_include_names,
                                     .headers              = extra_headers,
                                     .options              = options_cstr,
                                     .name_expressions     = name_expressions,
                                     .target_type          = rtcx::binary_type::LTO_IR};

  return rtcx::compile(params);
}

std::variant<rtcx::byte_buffer, rtcx::blob> get_cuda_udf_fragment(
  char const* name,
  char const* cuda_code,
  std::span<char const* const> extra_header_include_names,
  std::span<char const* const> extra_headers,
  std::span<char const* const> extra_options,
  std::span<char const* const> name_expressions,
  bool use_pch,
  bool use_minimal,
  bool log_pch,
  bool use_cache)
{
  auto& cache  = cudf::get_context().rtcx_cache();
  auto& bundle = cudf::get_context().jit_bundle();

  auto runtime                   = get_runtime_version();
  auto driver                    = get_driver_version();
  auto sm                        = get_current_device_physical_model();
  auto header_include_names_hash = hash(extra_header_include_names).to_hex_string();
  auto headers_hash              = hash(extra_headers).to_hex_string();
  auto bundle_hash               = bundle.get_hash();
  auto source_hash               = hash(cuda_code).to_hex_string();

  auto cache_key = std::format(R"***(cudaFragment
binary_type=LTO_IR
name={}
cuda_runtime={}
cuda_driver={}
arch={}
header_include_names={},
headers={},
bundle={},
source_file={}
)***",
                               name,
                               runtime,
                               driver,
                               sm,
                               header_include_names_hash.view(),
                               headers_hash.view(),
                               bundle_hash,
                               source_hash.view());

  auto cache_key_sha256 = hash(cache_key);

  auto compile = [&] {
    return compile_fragment_uncached(name,
                                     cuda_code,
                                     extra_header_include_names,
                                     extra_headers,
                                     extra_options,
                                     name_expressions,
                                     use_pch,
                                     use_minimal,
                                     log_pch);
  };

  if (!use_cache) { return compile(); }

  auto fut =
    cache.get_or_add_blob(cache_key_sha256, rtcx::blob_compile_func::from_functor(compile));
  return fut.get();
}

std::tuple<rtcx::library, rtcx::blob> link_library_uncached(
  char const* name,
  std::span<rtcx::file_fragment const> file_fragments,
  std::span<rtcx::memory_fragment const> memory_fragments,
  std::span<char const* const> extra_options)
{
  CUDF_FUNC_RANGE();

  auto sm = get_current_device_physical_model();

  std::vector<std::string> options;

  options.emplace_back(std::format("--gpu-architecture=sm_{}", sm));
  options.emplace_back("-lto");

  std::vector<char const*> options_cstr;

  for (auto const& option : options) {
    options_cstr.emplace_back(option.c_str());
  }

  for (auto* option : extra_options) {
    options_cstr.emplace_back(option);
  }

  auto params = rtcx::link_params{.name             = name,
                                  .output_type      = rtcx::binary_type::CUBIN,
                                  .file_fragments   = file_fragments,
                                  .memory_fragments = memory_fragments,
                                  .link_options     = options_cstr};

  auto cubin   = rtcx::link_library(params);
  auto library = rtcx::load_library(cubin);
  auto blob    = rtcx::blob_t::from_buffer(std::move(cubin));

  return std::make_tuple(library, std::make_shared<rtcx::blob_t>(std::move(blob)));
}

}  // namespace

kernel get_kernel(std::string const& name,
                  std::string const& source_file_id,
                  std::span<char const* const> header_include_names,
                  std::span<char const* const> headers,
                  std::string const& kernel_instance,
                  bool use_cache,
                  bool use_pch,
                  bool use_minimal,
                  bool log_pch,
                  std::span<std::string const> extra_options)
{
  CUDF_FUNC_RANGE();

  auto& cache  = cudf::get_context().rtcx_cache();
  auto& bundle = cudf::get_context().jit_bundle();

  auto runtime                   = get_runtime_version();
  auto driver                    = get_driver_version();
  auto sm                        = get_current_device_physical_model();
  auto header_include_names_hash = hash(header_include_names).to_hex_string();
  auto headers_hash              = hash(headers).to_hex_string();
  auto bundle_hash               = bundle.get_hash();
  auto source_file = std::format("{}/cudf/cpp/src/{}", bundle.get_directory(), source_file_id);

  auto cache_key = std::format(R"***(cuLibrary
name={}
binary_type=CUBIN
cuda_runtime={}
cuda_driver={}
arch={}
bundle={}
source_file={}
header_include_names={}
headers={}
kernel_instance={}
)***",
                               name,
                               runtime,
                               driver,
                               sm,
                               bundle_hash,
                               source_file,
                               header_include_names_hash.view(),
                               headers_hash.view(),
                               kernel_instance);

  auto cache_key_sha256 = hash(cache_key);

  auto compile = [&] {
    auto bundle_dir = cudf::get_context().jit_bundle().get_directory();
    auto source     = read_file_string(source_file.c_str());
    std::vector<char const*> extra_options_cstr;
    for (auto const& option : extra_options) {
      extra_options_cstr.emplace_back(option.c_str());
    }

    return compile_library_uncached(name.c_str(),
                                    source.c_str(),
                                    header_include_names,
                                    headers,
                                    extra_options_cstr,
                                    {},
                                    use_pch,
                                    use_minimal,
                                    log_pch);
  };

  if (!use_cache) {
    auto [lib, blob] = compile();
    return kernel{lib, lib->get_kernel("cudf_kernel")};
  }

  auto fut =
    cache.get_or_add_library(cache_key_sha256, rtcx::library_compile_func::from_functor(compile));

  auto lib = fut.get();
  return kernel{lib, lib->get_kernel("cudf_kernel")};
}

kernel get_linked_kernel(std::string const& name,
                         std::string const& fragment_file_id,
                         std::span<rtcx::file_fragment const> file_fragments,
                         std::span<rtcx::memory_fragment const> memory_fragments,
                         bool use_cache,
                         std::span<std::string const> extra_options)
{
  CUDF_FUNC_RANGE();

  auto& cache  = cudf::get_context().rtcx_cache();
  auto& bundle = cudf::get_context().jit_bundle();

  auto runtime               = get_runtime_version();
  auto driver                = get_driver_version();
  auto sm                    = get_current_device_physical_model();
  auto file_fragments_hash   = hash(file_fragments).to_hex_string();
  auto memory_fragments_hash = hash(memory_fragments).to_hex_string();
  auto bundle_hash           = bundle.get_hash();
  auto fragment_file = std::format("{}/fragments/{}", bundle.get_directory(), fragment_file_id);
  // TODO: resolve file fragment IDs

  auto cache_key = std::format(R"***(cuLibrary
name={}
binary_type=CUBIN
cuda_runtime={}
cuda_driver={}
arch={}
bundle={}
fragment_file={}
file_fragments={}
memory_fragments={}
)***",
                               name,
                               runtime,
                               driver,
                               sm,
                               bundle_hash,
                               fragment_file,
                               file_fragments_hash.view(),
                               memory_fragments_hash.view());

  auto cache_key_sha256 = hash(cache_key);

  auto compile = [&] {
    std::vector<char const*> extra_options_cstr;
    for (auto const& option : extra_options) {
      extra_options_cstr.emplace_back(option.c_str());
    }

    return link_library_uncached(
      name.c_str(), file_fragments, memory_fragments, extra_options_cstr);
  };

  if (!use_cache) {
    auto [lib, blob] = compile();
    return kernel{lib, lib->get_kernel("kernel")};
  }

  auto fut =
    cache.get_or_add_library(cache_key_sha256, rtcx::library_compile_func::from_functor(compile));

  auto lib = fut.get();
  return kernel{lib, lib->get_kernel("kernel")};
}

kernel get_cuda_linked_kernel(std::string const& kernel_name,
                              std::string const& kernel_fragment_file_id,
                              std::string const& cuda_udf_name,
                              std::string const& cuda_udf_source,
                              std::span<char const* const> header_include_names,
                              std::span<char const* const> headers,
                              std::span<rtcx::file_fragment const> file_fragments,
                              std::span<rtcx::memory_fragment const> memory_fragments,
                              bool use_cache,
                              bool use_pch,
                              bool use_minimal,
                              bool log_pch,
                              std::span<std::string const> extra_compile_options,
                              std::span<std::string const> extra_link_options)
{
  CUDF_FUNC_RANGE();

  std::vector<char const*> extra_compile_options_cstr;
  for (auto const& option : extra_compile_options) {
    extra_compile_options_cstr.emplace_back(option.c_str());
  }

  std::vector<char const*> extra_link_options_cstr;
  for (auto const& option : extra_link_options) {
    extra_link_options_cstr.emplace_back(option.c_str());
  }

  auto udf_fragment = get_cuda_udf_fragment(cuda_udf_name.c_str(),
                                            cuda_udf_source.c_str(),
                                            header_include_names,
                                            headers,
                                            extra_compile_options_cstr,
                                            {},
                                            use_pch,
                                            use_minimal,
                                            log_pch,
                                            use_cache);

  auto fragment_name = std::format("{}_udf_fragment", cuda_udf_name);

  std::vector<rtcx::memory_fragment> all_fragments;
  all_fragments.insert(all_fragments.end(), memory_fragments.begin(), memory_fragments.end());

  if (std::holds_alternative<rtcx::blob>(udf_fragment)) {
    auto& udf_blob = std::get<rtcx::blob>(udf_fragment);
    all_fragments.push_back(rtcx::memory_fragment{
      .data = udf_blob->view(), .type = rtcx::binary_type::LTO_IR, .name = fragment_name.c_str()});
  } else {
    auto& udf_buffer = std::get<rtcx::byte_buffer>(udf_fragment);
    all_fragments.push_back(rtcx::memory_fragment{
      .data = udf_buffer, .type = rtcx::binary_type::LTO_IR, .name = fragment_name.c_str()});
  }

  return get_linked_kernel(kernel_name,
                           kernel_fragment_file_id,
                           file_fragments,
                           all_fragments,
                           use_cache,
                           extra_link_options);
}

}  // namespace CUDF_EXPORT cudf
