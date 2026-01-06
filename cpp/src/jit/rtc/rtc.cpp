
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

#define CUDFRTC_CHECK_CUDA(...)                                             \
  do {                                                                      \
    CUresult result = (__VA_ARGS__);                                        \
    if (result != CUDA_SUCCESS) {                                           \
      char const* enum_str;                                                 \
      CUDF_EXPECTS(cuGetErrorString(result, &enum_str) == CUDA_SUCCESS,     \
                   "Unable to get CUDA error string");                      \
      auto errstr = std::format("CUDA Call {} failed, with error ({}): {}", \
                                #__VA_ARGS__,                               \
                                static_cast<int64_t>(result),               \
                                enum_str);                                  \
      CUDF_FAIL(+errstr, std::runtime_error);                               \
    }                                                                       \
  } while (0)

#define CUDFRTC_CHECK_NVRTC(...)                                             \
  do {                                                                       \
    nvrtcResult result = (__VA_ARGS__);                                      \
    if (result != NVRTC_SUCCESS) {                                           \
      auto errstr = std::format("NVRTC Call {} failed, with error ({}): {}", \
                                #__VA_ARGS__,                                \
                                static_cast<int64_t>(result),                \
                                nvrtcGetErrorString(result));                \
      CUDF_FAIL(+errstr, std::runtime_error);                                \
    }                                                                        \
  } while (0)

#define CUDFRTC_CHECK_NVJITLINK(...)                                             \
  do {                                                                           \
    nvJitLinkResult result = (__VA_ARGS__);                                      \
    if (result != NVJITLINK_SUCCESS) {                                           \
      auto errstr = std::format("nvJitLink Call {} failed, with error ({}): {}", \
                                #__VA_ARGS__,                                    \
                                static_cast<int64_t>(result),                    \
                                cudf_nvJitLinkResultString(result));             \
      CUDF_FAIL(+errstr, std::runtime_error);                                    \
    }                                                                            \
  } while (0)

namespace cudf {
namespace rtc {

char const* cudf_nvJitLinkResultString(nvJitLinkResult result)
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
    default: CUDF_FAIL("Unrecognized nvJitLinkResult type", std::runtime_error);
  }
}

char const* binary_type_string(binary_type type)
{
  switch (type) {
    case binary_type::LTO_IR: return "LTO_IR";
    case binary_type::CUBIN: return "CUBIN";
    case binary_type::FATBIN: return "FATBIN";
    case binary_type::PTX: return "PTX";
    default: CUDF_FAIL("Unrecognized binary_type", std::runtime_error);
  }
}

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

  return std::make_shared<fragment_t>(params.binary, params.type);
}

void log_nvrtc_compile_result(fragment_t::compile_params const& params,
                              nvrtcProgram program,
                              nvrtcResult compile_result)
{
  size_t log_size;
  CUDFRTC_CHECK_NVRTC(nvrtcGetProgramLogSize(program, &log_size));

  std::vector<char> log;
  log.resize(log_size);
  CUDFRTC_CHECK_NVRTC(nvrtcGetProgramLog(program, log.data()));

  auto status_str =
    (compile_result == NVRTC_SUCCESS) ? "completed with warning" : "failed with error";

  std::string headers_str;
  for (auto const& header : params.headers.include_names) {
    headers_str += std::format("\t{}\n", header);
  }

  std::string options_str;
  for (auto const& option : params.options) {
    options_str += std::format("\t{}\n", option);
  }

  auto str = std::format("NCRTC Compilation for {} {} ({}): {}.\nHeaders: {}\nOptions: {}\n\n{}",
                         params.name == nullptr ? "<unnamed>" : params.name,
                         status_str,
                         static_cast<int64_t>(compile_result),
                         nvrtcGetErrorString(compile_result),
                         headers_str,
                         options_str,
                         log.data());

  if (compile_result != NVRTC_SUCCESS) {
    CUDF_FAIL(+str, std::runtime_error);
  } else if (!log.empty()) {
    CUDF_LOG_WARN(str);
  }
}

void log_nvJitLink_link_result(library_t::link_params const& params,
                               nvJitLinkHandle handle,
                               nvJitLinkResult link_result)
{
  size_t info_log_size;
  CUDFRTC_CHECK_NVJITLINK(nvJitLinkGetInfoLogSize(handle, &info_log_size));

  std::vector<char> info_log;
  info_log.resize(info_log_size);
  CUDFRTC_CHECK_NVJITLINK(nvJitLinkGetInfoLog(handle, info_log.data()));

  size_t error_log_size;
  CUDFRTC_CHECK_NVJITLINK(nvJitLinkGetErrorLogSize(handle, &error_log_size));

  std::vector<char> error_log;
  error_log.resize(error_log_size);
  CUDFRTC_CHECK_NVJITLINK(nvJitLinkGetErrorLog(handle, error_log.data()));

  std::string fragments_str;
  for (auto const& fragment : params.names) {
    fragments_str += std::format("\t{}\n", fragment);
  }

  std::string link_options_str;
  for (auto const& option : params.link_options) {
    link_options_str += std::format("\t{}\n", option);
  }

  char const* binary_type_str = binary_type_string(params.output_type);

  auto status_str =
    (link_result == NVJITLINK_SUCCESS) ? "completed successfully" : "failed with error";

  auto str = std::format(
    "nvJitLink Linking for {} ({}) {} ({}): {}.\nFragments: {}\n"
    "Link Options: {}\n\nInfo Log:\n{}\n\nError Log:\n{}",
    params.name == nullptr ? "<unnamed>" : params.name,
    binary_type_str,
    status_str,
    static_cast<int64_t>(link_result),
    cudf_nvJitLinkResultString(link_result),
    fragments_str,
    link_options_str,
    info_log.data(),
    error_log.data());

  if (link_result != NVJITLINK_SUCCESS) {
    CUDF_FAIL(+str, std::runtime_error);
  } else if (!info_log.empty() || !error_log.empty()) {
    CUDF_LOG_WARN(str);
  }
}

fragment fragment_t::compile(compile_params const& params)
{
  CUDF_FUNC_RANGE();

  nvrtcProgram program;
  CUDFRTC_CHECK_NVRTC(nvrtcCreateProgram(&program,
                                         params.source,
                                         params.name,
                                         static_cast<int>(params.headers.headers.size()),
                                         params.headers.headers.data(),
                                         params.headers.include_names.data()));

  CUDF_DEFER([&] { nvrtcDestroyProgram(&program); });

  auto compile_result =
    nvrtcCompileProgram(program, static_cast<int>(params.options.size()), params.options.data());

  log_nvrtc_compile_result(params, program, compile_result);

  switch (params.target_type) {
    case binary_type::LTO_IR: {
      size_t lto_ir_size;
      CUDFRTC_CHECK_NVRTC(nvrtcGetLTOIRSize(program, &lto_ir_size));

      std::vector<unsigned char> lto_ir;
      lto_ir.resize(lto_ir_size);
      CUDFRTC_CHECK_NVRTC(nvrtcGetLTOIR(program, reinterpret_cast<char*>(lto_ir.data())));

      auto shared_blob = std::make_shared<blob_t>(blob_t::from_vector(std::move(lto_ir)));

      return std::make_shared<fragment_t>(std::move(shared_blob), binary_type::LTO_IR);

    } break;
    case binary_type::CUBIN: {
      size_t cubin_size;
      CUDFRTC_CHECK_NVRTC(nvrtcGetCUBINSize(program, &cubin_size));

      std::vector<unsigned char> cubin;
      cubin.resize(cubin_size);
      CUDFRTC_CHECK_NVRTC(nvrtcGetCUBIN(program, reinterpret_cast<char*>(cubin.data())));

      auto shared_blob = std::make_shared<blob_t>(blob_t::from_vector(std::move(cubin)));

      return std::make_shared<fragment_t>(std::move(shared_blob), binary_type::CUBIN);

    } break;
    default: CUDF_FAIL("Unsupported binary type for compiling fragment");
  }
}

blob const& fragment_t::get_lto_ir() const
{
  CUDF_EXPECTS(type_ == binary_type::LTO_IR, "Fragment does not contain LTO IR");
  return blob_;
}

blob const& fragment_t::get_cubin() const
{
  CUDF_EXPECTS(type_ == binary_type::CUBIN, "Fragment does not contain CUBIN");
  return blob_;
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

  nvJitLinkHandle handle;

  CUDFRTC_CHECK_NVJITLINK(nvJitLinkCreate(&handle,
                                          static_cast<uint32_t>(params.link_options.size()),
                                          const_cast<char const**>(params.link_options.data())));

  CUDF_DEFER([&] { nvJitLinkDestroy(&handle); });

  for (size_t i = 0; i < params.fragments.size(); i++) {
    auto name     = params.names[i];
    auto fragment = params.fragments[i];
    auto bin_type = params.binary_types[i];

    nvJitLinkInputType nv_type;

    switch (bin_type) {
      case binary_type::LTO_IR: nv_type = NVJITLINK_INPUT_LTOIR; break;
      case binary_type::CUBIN: nv_type = NVJITLINK_INPUT_CUBIN; break;
      case binary_type::FATBIN: nv_type = NVJITLINK_INPUT_FATBIN; break;
      case binary_type::PTX: nv_type = NVJITLINK_INPUT_PTX; break;
      default: CUDF_FAIL("Unsupported binary type for loading fragment", std::logic_error);
    }

    CUDFRTC_CHECK_NVJITLINK(
      nvJitLinkAddData(handle, nv_type, fragment.data(), fragment.size_bytes(), name));
  }

  nvJitLinkResult link_result = nvJitLinkComplete(handle);

  log_nvJitLink_link_result(params, handle, link_result);

  switch (params.output_type) {
    case binary_type::CUBIN: {
      size_t cubin_size;
      CUDFRTC_CHECK_NVJITLINK(nvJitLinkGetLinkedCubinSize(handle, &cubin_size));
      std::vector<unsigned char> cubin;
      cubin.resize(cubin_size);
      CUDFRTC_CHECK_NVJITLINK(nvJitLinkGetLinkedCubin(handle, cubin.data()));
      return std::make_shared<blob_t>(blob_t::from_vector(std::move(cubin)));
    } break;

    case binary_type::PTX: {
      size_t ptx_size;
      CUDFRTC_CHECK_NVJITLINK(nvJitLinkGetLinkedPtxSize(handle, &ptx_size));
      std::vector<unsigned char> ptx;
      ptx.resize(ptx_size);
      CUDFRTC_CHECK_NVJITLINK(nvJitLinkGetLinkedPtx(handle, reinterpret_cast<char*>(ptx.data())));
      return std::make_shared<blob_t>(blob_t::from_vector(std::move(ptx)));
    } break;

    default:
      CUDF_FAIL("Unsupported output binary type for linking CUDA libraries", std::runtime_error);
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
