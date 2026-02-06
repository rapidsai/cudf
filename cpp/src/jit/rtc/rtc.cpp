
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/logger.hpp>
#include <cudf/logger_macros.hpp>
#include <cudf/utilities/defer.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cxxabi.h>
#include <jit/rtc/rtc.hpp>
#include <nvJitLink.h>
#include <nvrtc.h>

#include <format>

#define CUDFRTC_CONCATENATE_DETAIL(x, y) x##y
#define CUDFRTC_CONCATENATE(x, y)        CUDFRTC_CONCATENATE_DETAIL(x, y)

#define CUDFRTC_CHECK_CUDA(...)                                                   \
  do {                                                                            \
    ::CUresult __result = (__VA_ARGS__);                                          \
    if (__result != ::CUDA_SUCCESS) {                                             \
      char const* __enum_str;                                                     \
      CUDF_EXPECTS(::cuGetErrorString(__result, &__enum_str) == ::CUDA_SUCCESS,   \
                   "Unable to get CUDA error string");                            \
      auto __errstr = ::std::format("(cuda) Call {} failed, with error ({}): {}", \
                                    #__VA_ARGS__,                                 \
                                    static_cast<::int64_t>(__result),             \
                                    __enum_str);                                  \
      CUDF_FAIL(+__errstr, ::std::runtime_error);                                 \
    }                                                                             \
  } while (0)

#define CUDFRTC_CHECK_NVRTC(params, program, ...)                                  \
  do {                                                                             \
    ::nvrtcResult __result = (__VA_ARGS__);                                        \
    ::cudf::rtc::log_nvrtc_result(params, program, __result);                      \
    if (__result != ::NVRTC_SUCCESS) {                                             \
      auto __errstr = ::std::format("(nvrtc) Call {} failed, with error ({}): {}", \
                                    #__VA_ARGS__,                                  \
                                    static_cast<::int64_t>(__result),              \
                                    ::nvrtcGetErrorString(__result));              \
      CUDF_FAIL(+__errstr, ::std::runtime_error);                                  \
    }                                                                              \
  } while (0)

#define CUDFRTC_CHECK_NVJITLINK(params, handle, ...)                                   \
  do {                                                                                 \
    ::nvJitLinkResult __result = (__VA_ARGS__);                                        \
    ::cudf::rtc::log_nvJitLink_result(params, handle, __result);                       \
    if (__result != ::NVJITLINK_SUCCESS) {                                             \
      auto __errstr = ::std::format("(nvJitLink) Call {} failed, with error ({}): {}", \
                                    #__VA_ARGS__,                                      \
                                    static_cast<::int64_t>(__result),                  \
                                    ::cudf::rtc::get_nvJitLinkResultString(__result)); \
      CUDF_FAIL(+__errstr, ::std::runtime_error);                                      \
    }                                                                                  \
  } while (0)

namespace cudf {
namespace rtc {

namespace {
char const* get_nvJitLinkResultString(nvJitLinkResult result)
{
  switch (result) {
    case NVJITLINK_SUCCESS: return "NVJITLINK_SUCCESS";
    case NVJITLINK_ERROR_UNRECOGNIZED_OPTION: return "NVJITLINK_ERROR_UNRECOGNIZED_OPTION";
    case NVJITLINK_ERROR_MISSING_ARCH: return "NVJITLINK_ERROR_MISSING_ARCH";
    case NVJITLINK_ERROR_INVALID_INPUT: return "NVJITLINK_ERROR_INVALID_INPUT";
    case NVJITLINK_ERROR_PTX_COMPILE: return "NVJITLINK_ERROR_PTX_COMPILE";
    case NVJITLINK_ERROR_NVVM_COMPILE: return "NVJITLINK_ERROR_NVVM_COMPILE";
    case NVJITLINK_ERROR_INTERNAL: return "NVJITLINK_ERROR_INTERNAL";
    case NVJITLINK_ERROR_THREADPOOL: return "NVJITLINK_ERROR_THREADPOOL";
    case NVJITLINK_ERROR_UNRECOGNIZED_INPUT: return "NVJITLINK_ERROR_UNRECOGNIZED_INPUT";
    case NVJITLINK_ERROR_FINALIZE: return "NVJITLINK_ERROR_FINALIZE";
    case NVJITLINK_ERROR_NULL_INPUT: return "NVJITLINK_ERROR_NULL_INPUT";
    case NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS: return "NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS";
    case NVJITLINK_ERROR_INCORRECT_INPUT_TYPE: return "NVJITLINK_ERROR_INCORRECT_INPUT_TYPE";
    case NVJITLINK_ERROR_ARCH_MISMATCH: return "NVJITLINK_ERROR_ARCH_MISMATCH";
    case NVJITLINK_ERROR_OUTDATED_LIBRARY: return "NVJITLINK_ERROR_OUTDATED_LIBRARY";
    case NVJITLINK_ERROR_MISSING_FATBIN: return "NVJITLINK_ERROR_MISSING_FATBIN";
    case NVJITLINK_ERROR_UNRECOGNIZED_ARCH: return "NVJITLINK_ERROR_UNRECOGNIZED_ARCH";
    case NVJITLINK_ERROR_UNSUPPORTED_ARCH: return "NVJITLINK_ERROR_UNSUPPORTED_ARCH";
    case NVJITLINK_ERROR_LTO_NOT_ENABLED: return "NVJITLINK_ERROR_LTO_NOT_ENABLED";
    default:
      CUDF_FAIL(
        +std::format("Unrecognized nvJitLinkResult type: ({})", static_cast<int64_t>(result)),
        std::runtime_error);
  }
}

char const* binary_type_string(binary_type type)
{
  switch (type) {
    case binary_type::LTO_IR: return "LTO_IR";
    case binary_type::CUBIN: return "CUBIN";
    case binary_type::FATBIN: return "FATBIN";
    case binary_type::PTX: return "PTX";
    default:
      CUDF_FAIL(+std::format("Unrecognized binary_type: ({})", static_cast<int64_t>(type)),
                std::runtime_error);
  }
}

nvJitLinkInputType to_nvjitlink_input_type(binary_type bin_type)
{
  switch (bin_type) {
    case binary_type::LTO_IR: return NVJITLINK_INPUT_LTOIR;
    case binary_type::CUBIN: return NVJITLINK_INPUT_CUBIN;
    case binary_type::FATBIN: return NVJITLINK_INPUT_FATBIN;
    case binary_type::PTX: return NVJITLINK_INPUT_PTX;
    default:
      CUDF_FAIL(
        +std::format("Unrecognized binary type for linking: ({}) ", static_cast<int64_t>(bin_type)),
        std::logic_error);
  }
}

void log_nvrtc_result(fragment_t::compile_params const& params,
                      nvrtcProgram program,
                      nvrtcResult compile_result)
{
  if (program == nullptr) { return; }

  size_t log_size;
  if (auto errc = nvrtcGetProgramLogSize(program, &log_size); errc != NVRTC_SUCCESS) {
    CUDF_FAIL(+std::format("Failed to get NVRTC program log size with error ({}): {}",
                           static_cast<int64_t>(errc),
                           nvrtcGetErrorString(errc)),
              std::runtime_error);
  }

  if (log_size <= 1) { return; }

  std::vector<char> log;
  log.resize(log_size);

  if (auto errc = nvrtcGetProgramLog(program, log.data()); errc != NVRTC_SUCCESS) {
    CUDF_FAIL(+std::format("Failed to get NVRTC program log with error ({}): {}",
                           static_cast<int64_t>(errc),
                           nvrtcGetErrorString(errc)),
              std::runtime_error);
  }

  log.resize(log_size == 0 ? 0 : (log_size - 1));

  auto status_str = (compile_result == NVRTC_SUCCESS && !log.empty()) ? "completed with warning"
                                                                      : "failed with error";

  std::string headers_str;
  for (auto const& header : params.headers.include_names) {
    headers_str = std::format("{}\t{}\n", headers_str, header);
  }

  std::string options_str;
  for (auto const& option : params.options) {
    options_str = std::format("{}\t{}\n", options_str, option);
  }

  auto msg = std::format(
    "NVRTC Compilation for `{}` {} ({}): {}.\nHeaders:\n{}\n\nOptions:\n{}\n\nLog:\n\t{}",
    params.name == nullptr ? "<unnamed>" : params.name,
    status_str,
    static_cast<int64_t>(compile_result),
    nvrtcGetErrorString(compile_result),
    headers_str,
    options_str,
    std::string_view{log.data(), log.size()});

  if (compile_result != NVRTC_SUCCESS) {
    CUDF_LOG_ERROR(msg);
  } else {
    CUDF_LOG_WARN(msg);
  }
}

void log_nvJitLink_result(library_t::link_params const& params,
                          nvJitLinkHandle handle,
                          nvJitLinkResult link_result)
{
  if (handle == nullptr) { return; }

  size_t info_log_size;
  if (auto errc = nvJitLinkGetInfoLogSize(handle, &info_log_size); errc != NVJITLINK_SUCCESS) {
    CUDF_FAIL(+std::format("Failed to get nvJitLink info log size with error ({}): {}",
                           static_cast<int64_t>(errc),
                           get_nvJitLinkResultString(errc)),
              std::runtime_error);
  }

  std::vector<char> info_log;
  if (info_log_size > 1) {
    info_log.resize(info_log_size);
    if (auto errc = nvJitLinkGetInfoLog(handle, info_log.data()); errc != NVJITLINK_SUCCESS) {
      CUDF_FAIL(+std::format("Failed to get nvJitLink info log with error ({}): {}",
                             static_cast<int64_t>(errc),
                             get_nvJitLinkResultString(errc)),
                std::runtime_error);
    }
  }
  info_log.resize(info_log_size == 0 ? 0 : (info_log_size - 1));

  size_t error_log_size;
  if (auto errc = nvJitLinkGetErrorLogSize(handle, &error_log_size); errc != NVJITLINK_SUCCESS) {
    CUDF_FAIL(+std::format("Failed to get nvJitLink error log size with error ({}): {}",
                           static_cast<int64_t>(errc),
                           get_nvJitLinkResultString(errc)),
              std::runtime_error);
  }

  std::vector<char> error_log;

  if (error_log_size > 1) {
    error_log.resize(error_log_size);
    if (auto errc = nvJitLinkGetErrorLog(handle, error_log.data()); errc != NVJITLINK_SUCCESS) {
      CUDF_FAIL(+std::format("Failed to get nvJitLink error log with error ({}): {}",
                             static_cast<int64_t>(errc),
                             get_nvJitLinkResultString(errc)),
                std::runtime_error);
    }
  }
  error_log.resize(error_log_size == 0 ? 0 : (error_log_size - 1));

  if (info_log.empty() && error_log.empty()) { return; }

  std::string fragments_str;
  for (auto const& fragment_name : params.fragment_names) {
    fragments_str = std::format("{}\t{}\n", fragments_str, fragment_name);
  }

  std::string link_options_str;
  for (auto const& option : params.link_options) {
    link_options_str = std::format("{}\t{}\n", link_options_str, option);
  }

  char const* binary_type_str = binary_type_string(params.output_type);

  auto status_str = error_log.empty() ? "completed with warnings" : "failed with errors";

  auto msg = std::format(
    "(nvJitLink) Linking for `{}` ({}) {}, error code ({}): {}.\nFragments: \n{}\n"
    "Link Options: \n{}\n\nInfo Log:\n\t{}\n\nError Log:\n\t{}\n\n",
    params.name == nullptr ? "<unnamed>" : params.name,
    binary_type_str,
    status_str,
    static_cast<int64_t>(link_result),
    get_nvJitLinkResultString(link_result),
    fragments_str,
    link_options_str,
    std::string_view{info_log.data(), info_log.size()},
    std::string_view{error_log.data(), error_log.size()});

  if (!error_log.empty()) {
    CUDF_LOG_ERROR(msg);
  } else {
    CUDF_LOG_WARN(msg);
  }
}
}  // namespace

blob_t blob_t::from_vector(std::vector<uint8_t>&& data)
{
  auto ptr = new std::vector<uint8_t>(std::move(data));
  return blob_t::from_parts(
    ptr->data(), ptr->size(), ptr, [](void* user_data, uint8_t const*, size_t) {
      delete reinterpret_cast<std::vector<uint8_t>*>(user_data);
    });
}

blob_t blob_t::from_static_data(std::span<uint8_t const> data)
{
  return blob_t::from_parts(
    data.data(), data.size(), nullptr, [](void*, uint8_t const*, size_t) {});
}

fragment fragment_t::load(load_params const& params)
{
  CUDF_FUNC_RANGE();
  // TODO: validate parameters

  return std::make_shared<fragment_t>(params.binary, params.type);
}

fragment fragment_t::compile(compile_params const& params)
{
  // TODO: check
  CUDF_FUNC_RANGE();

  nvrtcProgram program = nullptr;

  CUDFRTC_CHECK_NVRTC(params,
                      program,
                      nvrtcCreateProgram(&program,
                                         params.source,
                                         params.name,
                                         static_cast<int>(params.headers.headers.size()),
                                         params.headers.headers.data(),
                                         params.headers.include_names.data()));

  CUDF_DEFER([&] { nvrtcDestroyProgram(&program); });

  CUDFRTC_CHECK_NVRTC(
    params,
    program,
    nvrtcCompileProgram(program, static_cast<int>(params.options.size()), params.options.data()));

  switch (params.target_type) {
    case binary_type::LTO_IR: {
      size_t lto_ir_size;
      CUDFRTC_CHECK_NVRTC(params, program, nvrtcGetLTOIRSize(program, &lto_ir_size));

      std::vector<unsigned char> lto_ir;
      lto_ir.resize(lto_ir_size);

      CUDFRTC_CHECK_NVRTC(
        params, program, nvrtcGetLTOIR(program, reinterpret_cast<char*>(lto_ir.data())));

      auto shared_blob = std::make_shared<blob_t>(blob_t::from_vector(std::move(lto_ir)));

      return std::make_shared<fragment_t>(std::move(shared_blob), binary_type::LTO_IR);

    } break;
    case binary_type::CUBIN: {
      size_t cubin_size;
      CUDFRTC_CHECK_NVRTC(params, program, nvrtcGetCUBINSize(program, &cubin_size));

      std::vector<unsigned char> cubin;
      cubin.resize(cubin_size);
      CUDFRTC_CHECK_NVRTC(
        params, program, nvrtcGetCUBIN(program, reinterpret_cast<char*>(cubin.data())));

      auto shared_blob = std::make_shared<blob_t>(blob_t::from_vector(std::move(cubin)));

      return std::make_shared<fragment_t>(std::move(shared_blob), binary_type::CUBIN);

    } break;
    default: CUDF_FAIL("Unsupported binary type for compiling fragment");
  }
}

blob const& fragment_t::get(binary_type type) const
{
  CUDF_EXPECTS(type_ == type, "Fragment does not contain expected binary type");
  return blob_;
}

std::tuple<int32_t, int32_t> kernel_ref::max_occupancy_config(size_t dynamic_shared_memory_bytes,
                                                              int32_t block_size_limit) const
{
  int32_t min_grid_size;
  int32_t block_size;
  CUDFRTC_CHECK_CUDA(cuOccupancyMaxPotentialBlockSize(&min_grid_size,
                                                      &block_size,
                                                      reinterpret_cast<CUfunction>(handle_),
                                                      nullptr,
                                                      dynamic_shared_memory_bytes,
                                                      block_size_limit));
  return {min_grid_size, block_size};
}

void kernel_ref::launch(uint32_t grid_dim_x,
                        uint32_t grid_dim_y,
                        uint32_t grid_dim_z,
                        uint32_t block_dim_x,
                        uint32_t block_dim_y,
                        uint32_t block_dim_z,
                        uint32_t shared_mem_bytes,
                        CUstream stream,
                        void** kernel_params)
{
  CUDF_FUNC_RANGE();

  CUlaunchConfig cfg{.gridDimX       = grid_dim_x,
                     .gridDimY       = grid_dim_y,
                     .gridDimZ       = grid_dim_z,
                     .blockDimX      = block_dim_x,
                     .blockDimY      = block_dim_y,
                     .blockDimZ      = block_dim_z,
                     .sharedMemBytes = shared_mem_bytes,
                     .hStream        = stream,
                     .attrs          = nullptr,
                     .numAttrs       = 0};
  CUDFRTC_CHECK_CUDA(
    cuLaunchKernelEx(&cfg, reinterpret_cast<CUfunction>(handle_), kernel_params, nullptr));
}

std::string_view kernel_ref::get_name() const
{
  char const* name;
  CUDFRTC_CHECK_CUDA(cuKernelGetName(&name, handle_));
  return std::string_view{name == nullptr ? "" : name};
}

library_t::~library_t()
{
  if (handle_ != nullptr) {
    if (cuLibraryUnload(handle_) != CUDA_SUCCESS) { std::terminate(); }
  }
}

library library_t::load(load_params const& params)
{
  // TODO: check
  CUDF_FUNC_RANGE();

  CUlibrary handle;

  CUDFRTC_CHECK_CUDA(
    cuLibraryLoadData(&handle, params.binary.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  CUDF_DEFER([&] {
    if (handle != nullptr) { CUDFRTC_CHECK_CUDA(cuLibraryUnload(handle)); }
  });

  auto module = std::make_shared<library_t>(handle);

  handle = nullptr;

  return module;
}

blob library_t::link_as_blob(link_params const& params)
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(params.output_type == binary_type::CUBIN || params.output_type == binary_type::PTX,
               "Only CUBIN and PTX output types are supported for linking modules",
               std::logic_error);
  CUDF_EXPECTS(params.fragments.size() == params.fragment_binary_types.size(),
               "Mismatched number of fragments and fragment binary types",
               std::logic_error);
  CUDF_EXPECTS(params.fragments.size() == params.fragment_names.size(),
               "Mismatched number of fragments and fragment names",
               std::logic_error);
  CUDF_EXPECTS(params.fragments.size() > 0, "No fragments provided for linking", std::logic_error);

  for (auto& frag : params.fragments) {
    CUDF_EXPECTS(frag.size_bytes() > 0, "Fragment binary data must be non-empty", std::logic_error);
  }

  nvJitLinkHandle handle = nullptr;

  CUDFRTC_CHECK_NVJITLINK(params,
                          handle,
                          nvJitLinkCreate(&handle,
                                          static_cast<uint32_t>(params.link_options.size()),
                                          const_cast<char const**>(params.link_options.data())));

  CUDF_DEFER([&] { nvJitLinkDestroy(&handle); });

  for (size_t i = 0; i < params.fragments.size(); i++) {
    auto name                  = params.fragment_names[i];
    auto fragment              = params.fragments[i];
    auto bin_type              = params.fragment_binary_types[i];
    nvJitLinkInputType nv_type = to_nvjitlink_input_type(bin_type);

    CUDFRTC_CHECK_NVJITLINK(
      params,
      handle,
      nvJitLinkAddData(handle, nv_type, fragment.data(), fragment.size_bytes(), name));
  }

  CUDFRTC_CHECK_NVJITLINK(params, handle, nvJitLinkComplete(handle));

  switch (params.output_type) {
    case binary_type::CUBIN: {
      size_t cubin_size;
      CUDFRTC_CHECK_NVJITLINK(params, handle, nvJitLinkGetLinkedCubinSize(handle, &cubin_size));
      std::vector<unsigned char> cubin;
      cubin.resize(cubin_size);
      CUDFRTC_CHECK_NVJITLINK(params, handle, nvJitLinkGetLinkedCubin(handle, cubin.data()));

      return std::make_shared<blob_t>(blob_t::from_vector(std::move(cubin)));
    } break;

    case binary_type::PTX: {
      size_t ptx_size;

      CUDFRTC_CHECK_NVJITLINK(params, handle, nvJitLinkGetLinkedPtxSize(handle, &ptx_size));
      std::vector<unsigned char> ptx;
      ptx.resize(ptx_size);

      CUDFRTC_CHECK_NVJITLINK(
        params, handle, nvJitLinkGetLinkedPtx(handle, reinterpret_cast<char*>(ptx.data())));

      return std::make_shared<blob_t>(blob_t::from_vector(std::move(ptx)));
    } break;

    default:
      CUDF_FAIL(+std::format("Unsupported output binary type for linking CUDA libraries: ({})",
                             static_cast<int64_t>(params.output_type)),
                std::runtime_error);
  }
}

library library_t::link(link_params const& params)
{
  CUDF_FUNC_RANGE();

  auto blob = link_as_blob(params);
  return load(load_params{blob->view(), params.output_type});
}

kernel_ref library_t::get_kernel(char const* name) const
{
  CUkernel kernel;
  CUDFRTC_CHECK_CUDA(cuLibraryGetKernel(&kernel, handle_, name));
  return kernel_ref{kernel};
}

std::vector<kernel_ref> library_t::enumerate_kernels() const
{
  uint32_t num_kernels;
  CUDFRTC_CHECK_CUDA(cuLibraryGetKernelCount(&num_kernels, handle_));

  std::vector<CUkernel> kernels;
  kernels.resize(num_kernels);

  CUDFRTC_CHECK_CUDA(cuLibraryEnumerateKernels(kernels.data(), num_kernels, handle_));

  std::vector<kernel_ref> result;
  for (CUkernel k : kernels) {
    result.emplace_back(k);
  }

  return result;
}

}  // namespace rtc

std::string rtc::demangle_cuda_symbol(char const* mangled_name)
{
  int status;
  size_t length;

  char* demangled_name = abi::__cxa_demangle(mangled_name, nullptr, &length, &status);

  CUDF_EXPECTS(status == 0, "Demangling CUDA symbol name failed");
  CUDF_EXPECTS(demangled_name != nullptr, "Demangling CUDA symbol name failed");

  CUDF_DEFER([&] {
    if (demangled_name != nullptr) free(demangled_name);
  });

  std::string result{demangled_name};

  return result;
}

}  // namespace cudf
