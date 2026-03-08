
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cxxabi.h>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <librtcx/rtcx.hpp>
#include <nvJitLink.h>
#include <nvrtc.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <syscall.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <format>
#include <numeric>
#include <source_location>

extern "C" {
#include <openssl/evp.h>
}

#define RTCX_EXPECTS(_condition, _reason, _exception_type)                               \
  do {                                                                                   \
    if (!(_condition)) {                                                                 \
      throw _exception_type{::std::format("RTCX failure at: {}:{}: {}",                  \
                                          ::std::source_location::current().file_name(), \
                                          ::std::source_location::current().line(),      \
                                          (_reason))};                                   \
    }                                                                                    \
  } while (0)

#define RTCX_FAIL(_reason, _exception_type)                                            \
  do {                                                                                 \
    throw _exception_type{::std::format("RTCX failure at: {}:{}: {}",                  \
                                        ::std::source_location::current().file_name(), \
                                        ::std::source_location::current().line(),      \
                                        (_reason))};                                   \
  } while (0)

#define RTCX_CHECK_CUDA(...)                                                              \
  do {                                                                                    \
    ::CUresult __result = (__VA_ARGS__);                                                  \
    if (__result != ::CUDA_SUCCESS) {                                                     \
      char const* __enum_str;                                                             \
      RTCX_EXPECTS(::rtcx::cuda->GetErrorString(__result, &__enum_str) == ::CUDA_SUCCESS, \
                   "Unable to get CUDA error string",                                     \
                   std::runtime_error);                                                   \
      auto __errstr = ::std::format("(cuda) expression `{}` failed, with error ({}): {}", \
                                    #__VA_ARGS__,                                         \
                                    static_cast<::rtcx::i64>(__result),                   \
                                    __enum_str);                                          \
      RTCX_FAIL(__errstr, ::std::runtime_error);                                          \
    }                                                                                     \
  } while (0)

#define RTCX_CHECK_CUDART(...)                                                                  \
  do {                                                                                          \
    ::cudaError_t __result = (__VA_ARGS__);                                                     \
    if (__result != ::cudaSuccess) {                                                            \
      char const* __enum_name = ::cudaGetErrorName(__result);                                   \
      char const* __enum_msg  = ::cudaGetErrorString(__result);                                 \
      auto __errstr = ::std::format("(cudart) expression `{}` failed, with error ({}: {}): {}", \
                                    #__VA_ARGS__,                                               \
                                    static_cast<::rtcx::i64>(__result),                         \
                                    __enum_name,                                                \
                                    __enum_msg);                                                \
      RTCX_FAIL(__errstr, ::std::runtime_error);                                                \
    }                                                                                           \
  } while (0)

#define RTCX_CHECK_NVRTC(params, program, ...)                                             \
  do {                                                                                     \
    ::nvrtcResult __result = (__VA_ARGS__);                                                \
    ::rtcx::log_nvrtc_result(params, program, __result);                                   \
    if (__result != ::NVRTC_SUCCESS) {                                                     \
      auto __errstr = ::std::format("(nvrtc) expression `{}` failed, with error ({}): {}", \
                                    #__VA_ARGS__,                                          \
                                    static_cast<::rtcx::i64>(__result),                    \
                                    ::rtcx::nvrtc->GetErrorString(__result));              \
      RTCX_FAIL(__errstr, ::std::runtime_error);                                           \
    }                                                                                      \
  } while (0)

#define RTCX_CHECK_NVJITLINK(params, handle, ...)                                              \
  do {                                                                                         \
    ::nvJitLinkResult __result = (__VA_ARGS__);                                                \
    ::rtcx::log_nvJitLink_result(params, handle, __result);                                    \
    if (__result != ::NVJITLINK_SUCCESS) {                                                     \
      auto __errstr = ::std::format("(nvJitLink) expression `{}` failed, with error ({}): {}", \
                                    #__VA_ARGS__,                                              \
                                    static_cast<::rtcx::i64>(__result),                        \
                                    ::rtcx::get_nvJitLinkResultString(__result));              \
      RTCX_FAIL(__errstr, ::std::runtime_error);                                               \
    }                                                                                          \
  } while (0)

namespace RTCX_EXPORT rtcx {

void log_warning(std::string_view message)
{
  std::fprintf(stdout, "RTCX WARNING: %.*s\n", static_cast<i32>(message.size()), message.data());
}

void log_error(std::string_view message)
{
  std::fprintf(stderr, "RTCX ERROR: %.*s\n", static_cast<i32>(message.size()), message.data());
}

sha256_context::sha256_context() : ectx_(nullptr)
{
  const EVP_MD* type = EVP_sha256();
  ectx_              = EVP_MD_CTX_new();
  RTCX_EXPECTS(ectx_ != nullptr, "EVP_MD_CTX_new failed", std::runtime_error);
  RTCX_EXPECTS(
    EVP_DigestInit_ex(ectx_, type, nullptr) == 1, "EVP_DigestInit_ex failed", std::runtime_error);
}

sha256_context::~sha256_context()
{
  if (ectx_ != nullptr) { EVP_MD_CTX_free(ectx_); }
}

void sha256_context::update(std::span<u8 const> data)
{
  RTCX_EXPECTS(EVP_DigestUpdate(ectx_, data.data(), data.size()) == 1,
               "EVP_DigestUpdate failed",
               std::runtime_error);
}

sha256 sha256_context::finalize()
{
  sha256 hash;
  u32 length = 0;
  RTCX_EXPECTS(EVP_DigestFinal_ex(ectx_, hash.data_, &length) == 1,
               "EVP_DigestFinal_ex failed",
               std::runtime_error);
  RTCX_EXPECTS(length == sizeof(sha256::data_), "Unexpected SHA256 length", std::runtime_error);
  EVP_MD const* type = EVP_sha256();
  RTCX_EXPECTS(
    EVP_DigestInit_ex(ectx_, type, nullptr) == 1, "EVP_DigestInit_ex failed", std::runtime_error);
  return hash;
}

#define FOR_EACH_CUDA_FUNC(DO_IT)       \
  DO_IT(GetErrorString)                 \
  DO_IT(GetErrorName)                   \
  DO_IT(Init)                           \
  DO_IT(DeviceGet)                      \
  DO_IT(DeviceGetCount)                 \
  DO_IT(DeviceGetName)                  \
  DO_IT(OccupancyMaxPotentialBlockSize) \
  DO_IT(LaunchKernel)                   \
  DO_IT(LaunchKernelEx)                 \
  DO_IT(KernelGetName)                  \
  DO_IT(LibraryLoadData)                \
  DO_IT(LibraryGetKernel)               \
  DO_IT(LibraryGetKernelCount)          \
  DO_IT(LibraryEnumerateKernels)        \
  DO_IT(LibraryUnload)

#define FOR_EACH_NVRTC_FUNC(DO_IT) \
  DO_IT(GetErrorString)            \
  DO_IT(CreateProgram)             \
  DO_IT(DestroyProgram)            \
  DO_IT(CompileProgram)            \
  DO_IT(GetPTXSize)                \
  DO_IT(GetPTX)                    \
  DO_IT(GetCUBINSize)              \
  DO_IT(GetCUBIN)                  \
  DO_IT(GetLTOIRSize)              \
  DO_IT(GetLTOIR)                  \
  DO_IT(GetProgramLogSize)         \
  DO_IT(GetProgramLog)             \
  DO_IT(AddNameExpression)         \
  DO_IT(GetLoweredName)            \
  DO_IT(GetPCHHeapSize)            \
  DO_IT(SetPCHHeapSize)            \
  DO_IT(GetPCHCreateStatus)        \
  DO_IT(GetPCHHeapSizeRequired)    \
  DO_IT(SetFlowCallback)

#define FOR_EACH_NVJITLINK_FUNC(DO_IT) \
  DO_IT(Create)                        \
  DO_IT(Destroy)                       \
  DO_IT(AddData)                       \
  DO_IT(AddFile)                       \
  DO_IT(Complete)                      \
  DO_IT(GetLinkedCubinSize)            \
  DO_IT(GetLinkedCubin)                \
  DO_IT(GetLinkedPtxSize)              \
  DO_IT(GetLinkedPtx)                  \
  DO_IT(GetErrorLogSize)               \
  DO_IT(GetErrorLog)                   \
  DO_IT(GetInfoLog)                    \
  DO_IT(GetInfoLogSize)

namespace {

void* load_dll(std::string_view base_name, std::span<std::string const> names)
{
  for (auto const& name : names) {
    void* handle = ::dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle != nullptr) { return handle; }
  }

  std::string tried_names = std::accumulate(
    names.begin(), names.end(), std::string{}, [&](std::string acc, std::string const& name) {
      if (!acc.empty()) { acc += ", "; }
      acc += name;
      return acc;
    });

  RTCX_FAIL(std::format("Failed to load dynamic library `{}` (tried: {})", base_name, tried_names),
            std::runtime_error);
}

void* get_symbol(char const* lib_name, void* handle, char const* sym_name)
{
  void* sym = ::dlsym(handle, sym_name);
  if (sym == nullptr) {
    RTCX_FAIL(std::format(
                "Failed to load symbol `{}` from `{}`, error: `{}`", sym_name, lib_name, dlerror()),
              std::runtime_error);
  }
  return sym;
}

inline constexpr i32 major_version(i32 version) { return version / 1000; }

inline constexpr i32 minor_version(i32 version) { return (version % 1000) / 10; }

struct LibCuda {
  void* _handle = nullptr;

#define DO_IT(func) decltype(::cu##func)* func = nullptr;
  FOR_EACH_CUDA_FUNC(DO_IT)
#undef DO_IT

  explicit LibCuda(void* handle) : _handle(handle) { _load_symbols(); }
  LibCuda(LibCuda const&)            = delete;
  LibCuda(LibCuda&&)                 = delete;
  LibCuda& operator=(LibCuda const&) = delete;
  LibCuda& operator=(LibCuda&&)      = delete;
  ~LibCuda() { dlclose(_handle); }

  static void* _load()
  {
    std::string lib_names[] = {"libcuda.so"};
    return load_dll("libcuda.so", lib_names);
  }

 private:
  void _load_symbols()
  {
#define DO_IT(func) \
  this->func = reinterpret_cast<decltype(cu##func)*>(get_symbol("libcuda", _handle, "cu" #func));
    FOR_EACH_CUDA_FUNC(DO_IT)
#undef DO_IT
  }
};

struct LibNVRTC {
  void* _handle = nullptr;

#define DO_IT(func) decltype(::nvrtc##func)* func = nullptr;
  FOR_EACH_NVRTC_FUNC(DO_IT)
#undef DO_IT

  explicit LibNVRTC(void* handle) : _handle(handle) { _load_symbols(); }
  LibNVRTC(LibNVRTC const&)            = delete;
  LibNVRTC(LibNVRTC&&)                 = delete;
  LibNVRTC& operator=(LibNVRTC const&) = delete;
  LibNVRTC& operator=(LibNVRTC&&)      = delete;
  ~LibNVRTC() { dlclose(_handle); }

  static void* _load()
  {
    i32 cuda_version;
    RTCX_CHECK_CUDART(cudaRuntimeGetVersion(&cuda_version));

    i32 major = major_version(cuda_version);
    i32 minor = minor_version(cuda_version);

    std::string lib_names[] = {std::format("libnvrtc.so.{}.{}", major, minor),
                               std::format("libnvrtc.so.{}", major),
                               "libnvrtc.so"};

    return load_dll("libnvrtc.so", lib_names);
  }

 private:
  void _load_symbols()
  {
#define DO_IT(func) \
  this->func =      \
    reinterpret_cast<decltype(nvrtc##func)*>(get_symbol("libnvrtc", _handle, "nvrtc" #func));
    FOR_EACH_NVRTC_FUNC(DO_IT)
#undef DO_IT
  }
};

struct LibNVJitLink {
  void* _handle = nullptr;

#define DO_IT(func) decltype(::nvJitLink##func)* func = nullptr;
  FOR_EACH_NVJITLINK_FUNC(DO_IT)
#undef DO_IT

  explicit LibNVJitLink(void* handle) : _handle(handle) { _load_symbols(); }
  LibNVJitLink(LibNVJitLink const&)            = delete;
  LibNVJitLink(LibNVJitLink&&)                 = delete;
  LibNVJitLink& operator=(LibNVJitLink const&) = delete;
  LibNVJitLink& operator=(LibNVJitLink&&)      = delete;
  ~LibNVJitLink() { dlclose(_handle); }

  static void* _load()
  {
    i32 cuda_version;
    RTCX_CHECK_CUDART(cudaRuntimeGetVersion(&cuda_version));

    i32 major = major_version(cuda_version);
    i32 minor = minor_version(cuda_version);

    std::string lib_names[] = {std::format("libnvJitLink.so.{}.{}", major, minor),
                               std::format("libnvJitLink.so.{}", major),
                               "libnvJitLink.so"};

    return load_dll("libnvJitLink.so", lib_names);
  }

 private:
  void _load_symbols()
  {
#define DO_IT(func)                                          \
  this->func = reinterpret_cast<decltype(nvJitLink##func)*>( \
    get_symbol("libnvJitLink", _handle, "nvJitLink" #func));
    FOR_EACH_NVJITLINK_FUNC(DO_IT)
#undef DO_IT
  }
};

static std::optional<LibCuda> cuda;
static std::optional<LibNVRTC> nvrtc;
static std::optional<LibNVJitLink> nvjitlink;
static std::optional<std::once_flag> init_libraries_flag{std::in_place};
static std::optional<std::once_flag> teardown_libraries_flag{std::in_place};

}  // namespace

void initialize()
{
  std::call_once(*init_libraries_flag, [] {
    cuda.emplace(LibCuda::_load());
    nvrtc.emplace(LibNVRTC::_load());
    nvjitlink.emplace(LibNVJitLink::_load());
  });
}

void teardown()
{
  std::call_once(*teardown_libraries_flag, [] {
    cuda.reset();
    nvrtc.reset();
    nvjitlink.reset();
    init_libraries_flag.reset();
    teardown_libraries_flag.reset();
  });
}

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
#if CUDA_VERSION >= 13000
    case NVJITLINK_ERROR_NULL_INPUT: return "NVJITLINK_ERROR_NULL_INPUT";
    case NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS: return "NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS";
    case NVJITLINK_ERROR_INCORRECT_INPUT_TYPE: return "NVJITLINK_ERROR_INCORRECT_INPUT_TYPE";
    case NVJITLINK_ERROR_ARCH_MISMATCH: return "NVJITLINK_ERROR_ARCH_MISMATCH";
    case NVJITLINK_ERROR_OUTDATED_LIBRARY: return "NVJITLINK_ERROR_OUTDATED_LIBRARY";
    case NVJITLINK_ERROR_MISSING_FATBIN: return "NVJITLINK_ERROR_MISSING_FATBIN";
    case NVJITLINK_ERROR_UNRECOGNIZED_ARCH: return "NVJITLINK_ERROR_UNRECOGNIZED_ARCH";
    case NVJITLINK_ERROR_UNSUPPORTED_ARCH: return "NVJITLINK_ERROR_UNSUPPORTED_ARCH";
    case NVJITLINK_ERROR_LTO_NOT_ENABLED: return "NVJITLINK_ERROR_LTO_NOT_ENABLED";
#endif
    default:
      RTCX_FAIL(std::format("Unrecognized nvJitLinkResult type: ({})", static_cast<i64>(result)),
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
      RTCX_FAIL(std::format("Unrecognized binary_type: ({})", static_cast<i64>(type)),
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
      RTCX_FAIL(
        std::format("Unrecognized binary type for linking: ({}) ", static_cast<i64>(bin_type)),
        std::logic_error);
  }
}

void log_nvrtc_result(compile_params const& params,
                      nvrtcProgram program,
                      nvrtcResult compile_result)
{
  if (program == nullptr) { return; }

  usize log_size;
  if (auto errc = nvrtc->GetProgramLogSize(program, &log_size); errc != NVRTC_SUCCESS) {
    RTCX_FAIL(std::format("Failed to get NVRTC program log size with error ({}): {}",
                          static_cast<i64>(errc),
                          nvrtc->GetErrorString(errc)),
              std::runtime_error);
  }

  if (log_size <= 1) { return; }

  std::vector<char> log;
  log.resize(log_size);

  if (auto errc = nvrtc->GetProgramLog(program, log.data()); errc != NVRTC_SUCCESS) {
    RTCX_FAIL(std::format("Failed to get NVRTC program log with error ({}): {}",
                          static_cast<i64>(errc),
                          nvrtc->GetErrorString(errc)),
              std::runtime_error);
  }

  log.resize(log_size == 0 ? 0 : (log_size - 1));

  auto status_str = (compile_result == NVRTC_SUCCESS && !log.empty()) ? "completed with warning"
                                                                      : "failed with error";

  std::string headers_str;
  for (auto const& header : params.header_include_names) {
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
    static_cast<i64>(compile_result),
    nvrtc->GetErrorString(compile_result),
    headers_str,
    options_str,
    std::string_view{log.data(), log.size()});

  if (compile_result != NVRTC_SUCCESS) {
    log_error(msg);
  } else {
    log_warning(msg);
  }
}

void log_nvJitLink_result(link_params const& params,
                          nvJitLinkHandle handle,
                          nvJitLinkResult link_result)
{
  if (handle == nullptr) { return; }

  usize info_log_size;
  if (auto errc = nvjitlink->GetInfoLogSize(handle, &info_log_size); errc != NVJITLINK_SUCCESS) {
    RTCX_FAIL(std::format("Failed to get nvJitLink info log size with error ({}): {}",
                          static_cast<i64>(errc),
                          get_nvJitLinkResultString(errc)),
              std::runtime_error);
  }

  std::vector<char> info_log;
  if (info_log_size > 1) {
    info_log.resize(info_log_size);
    if (auto errc = nvjitlink->GetInfoLog(handle, info_log.data()); errc != NVJITLINK_SUCCESS) {
      RTCX_FAIL(std::format("Failed to get nvJitLink info log with error ({}): {}",
                            static_cast<i64>(errc),
                            get_nvJitLinkResultString(errc)),
                std::runtime_error);
    }
  }
  info_log.resize(info_log_size == 0 ? 0 : (info_log_size - 1));

  usize error_log_size;
  if (auto errc = nvjitlink->GetErrorLogSize(handle, &error_log_size); errc != NVJITLINK_SUCCESS) {
    RTCX_FAIL(std::format("Failed to get nvJitLink error log size with error ({}): {}",
                          static_cast<i64>(errc),
                          get_nvJitLinkResultString(errc)),
              std::runtime_error);
  }

  std::vector<char> error_log;

  if (error_log_size > 1) {
    error_log.resize(error_log_size);
    if (auto errc = nvjitlink->GetErrorLog(handle, error_log.data()); errc != NVJITLINK_SUCCESS) {
      RTCX_FAIL(std::format("Failed to get nvJitLink error log with error ({}): {}",
                            static_cast<i64>(errc),
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
    static_cast<i64>(link_result),
    get_nvJitLinkResultString(link_result),
    fragments_str,
    link_options_str,
    std::string_view{info_log.data(), info_log.size()},
    std::string_view{error_log.data(), error_log.size()});

  if (!error_log.empty()) {
    log_error(msg);
  } else {
    log_warning(msg);
  }
}

}  // namespace

blob_t blob_t::from_vector(std::vector<u8>&& data)
{
  auto ptr = new std::vector<u8>(std::move(data));
  return blob_t::from_parts(
    ptr->data(), ptr->size(), blob_t::deallocator{ptr, [](void* user_data, u8 const*, usize) {
                                                    delete reinterpret_cast<std::vector<u8>*>(
                                                      user_data);
                                                  }});
}

blob_t blob_t::from_static_data(std::span<u8 const> data)
{
  return blob_t::from_parts(data.data(), data.size(), blob_t::noop_deallocator);
}

std::vector<unsigned char> compile(compile_params const& params)
{
  RTCX_EXPECTS(params.name != nullptr, "Fragment name must not be null", std::logic_error);
  RTCX_EXPECTS(params.source != nullptr, "Fragment source must not be null", std::logic_error);

  nvrtcProgram program = nullptr;

  RTCX_CHECK_NVRTC(params,
                   program,
                   nvrtc->CreateProgram(&program,
                                        params.source,
                                        params.name,
                                        static_cast<i32>(params.headers.size()),
                                        params.headers.data(),
                                        params.header_include_names.data()));

  RTCX_DEFER([&] { nvrtc->DestroyProgram(&program); });

  for (auto* name_expr : params.name_expressions) {
    RTCX_CHECK_NVRTC(params, program, nvrtc->AddNameExpression(program, name_expr));
  }

  // TODO: log is printed twice when warnings are raised
  RTCX_CHECK_NVRTC(
    params,
    program,
    nvrtc->CompileProgram(program, static_cast<i32>(params.options.size()), params.options.data()));

  switch (params.target_type) {
    case binary_type::LTO_IR: {
      usize lto_ir_size;
      RTCX_CHECK_NVRTC(params, program, nvrtc->GetLTOIRSize(program, &lto_ir_size));

      std::vector<unsigned char> lto_ir;
      lto_ir.resize(lto_ir_size);

      RTCX_CHECK_NVRTC(
        params, program, nvrtc->GetLTOIR(program, reinterpret_cast<char*>(lto_ir.data())));

      return lto_ir;

    } break;
    case binary_type::CUBIN: {
      usize cubin_size;
      RTCX_CHECK_NVRTC(params, program, nvrtc->GetCUBINSize(program, &cubin_size));

      std::vector<unsigned char> cubin;
      cubin.resize(cubin_size);
      RTCX_CHECK_NVRTC(
        params, program, nvrtc->GetCUBIN(program, reinterpret_cast<char*>(cubin.data())));

      return cubin;

    } break;
    default: RTCX_FAIL("Unsupported binary type for compiling fragment", std::logic_error);
  }
}

kernel_occupancy_config kernel_ref::max_occupancy_config(usize dynamic_shared_memory_bytes,
                                                         i32 block_size_limit) const
{
  i32 min_grid_size;
  i32 block_size;
  RTCX_CHECK_CUDA(cuda->OccupancyMaxPotentialBlockSize(&min_grid_size,
                                                       &block_size,
                                                       reinterpret_cast<CUfunction>(handle_),
                                                       nullptr,
                                                       dynamic_shared_memory_bytes,
                                                       block_size_limit));

  return kernel_occupancy_config{.min_grid_size = min_grid_size, .block_size = block_size};
}

void kernel_ref::launch(u32 grid_dim_x,
                        u32 grid_dim_y,
                        u32 grid_dim_z,
                        u32 block_dim_x,
                        u32 block_dim_y,
                        u32 block_dim_z,
                        u32 shared_mem_bytes,
                        CUstream stream,
                        void** kernel_params) const
{
  RTCX_EXPECTS(grid_dim_x > 0 && grid_dim_y > 0 && grid_dim_z > 0,
               "Grid dimensions must be greater than zero",
               std::logic_error);
  RTCX_EXPECTS(block_dim_x > 0 && block_dim_y > 0 && block_dim_z > 0,
               "Block dimensions must be greater than zero",
               std::logic_error);
  RTCX_EXPECTS(
    kernel_params != nullptr, "Kernel parameters pointer must not be null", std::logic_error);

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

  RTCX_CHECK_CUDA(
    cuda->LaunchKernelEx(&cfg, reinterpret_cast<CUfunction>(handle_), kernel_params, nullptr));
}

std::string_view kernel_ref::get_name() const
{
  char const* name;
  RTCX_CHECK_CUDA(cuda->KernelGetName(&name, handle_));
  return std::string_view{name == nullptr ? "" : name};
}

library_t::~library_t()
{
  if (handle_ != nullptr) {
    if (cuda->LibraryUnload(handle_) != CUDA_SUCCESS) { std::terminate(); }
  }
}

library load_library(std::span<u8 const> binary, binary_type type)
{
  CUlibrary handle;

  RTCX_CHECK_CUDA(
    cuda->LibraryLoadData(&handle, binary.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  RTCX_DEFER([&] {
    if (handle != nullptr) { RTCX_CHECK_CUDA(cuda->LibraryUnload(handle)); }
  });

  auto library = std::make_shared<library_t>(handle);

  handle = nullptr;

  return library;
}

std::vector<u8> link_library(link_params const& params)
{
  RTCX_EXPECTS(params.name != nullptr, "Link output name must not be null", std::logic_error);
  RTCX_EXPECTS(params.output_type == binary_type::CUBIN || params.output_type == binary_type::PTX,
               "Only CUBIN and PTX output types are supported for linking modules",
               std::logic_error);
  RTCX_EXPECTS(params.fragments.size() == params.fragment_binary_types.size(),
               "Mismatched number of fragments and fragment binary types",
               std::logic_error);
  RTCX_EXPECTS(params.fragments.size() == params.fragment_names.size(),
               "Mismatched number of fragments and fragment names",
               std::logic_error);
  RTCX_EXPECTS(params.fragments.size() > 0, "No fragments provided for linking", std::logic_error);

  for (auto& frag : params.fragments) {
    RTCX_EXPECTS(frag.size_bytes() > 0, "Fragment binary data must be non-empty", std::logic_error);
  }

  nvJitLinkHandle handle = nullptr;

  RTCX_CHECK_NVJITLINK(params,
                       handle,
                       nvjitlink->Create(&handle,
                                         static_cast<u32>(params.link_options.size()),
                                         const_cast<char const**>(params.link_options.data())));

  RTCX_DEFER([&] { nvjitlink->Destroy(&handle); });

  for (usize i = 0; i < params.fragments.size(); i++) {
    auto name                  = params.fragment_names[i];
    auto fragment              = params.fragments[i];
    auto bin_type              = params.fragment_binary_types[i];
    nvJitLinkInputType nv_type = to_nvjitlink_input_type(bin_type);

    RTCX_CHECK_NVJITLINK(
      params,
      handle,
      nvjitlink->AddData(handle, nv_type, fragment.data(), fragment.size_bytes(), name));
  }

  RTCX_CHECK_NVJITLINK(params, handle, nvjitlink->Complete(handle));

  switch (params.output_type) {
    case binary_type::CUBIN: {
      usize cubin_size;
      RTCX_CHECK_NVJITLINK(params, handle, nvjitlink->GetLinkedCubinSize(handle, &cubin_size));
      std::vector<unsigned char> cubin;
      cubin.resize(cubin_size);
      RTCX_CHECK_NVJITLINK(params, handle, nvjitlink->GetLinkedCubin(handle, cubin.data()));
      return cubin;
    } break;

    case binary_type::PTX: {
      usize ptx_size;

      RTCX_CHECK_NVJITLINK(params, handle, nvjitlink->GetLinkedPtxSize(handle, &ptx_size));
      std::vector<unsigned char> ptx;
      ptx.resize(ptx_size);

      RTCX_CHECK_NVJITLINK(
        params, handle, nvjitlink->GetLinkedPtx(handle, reinterpret_cast<char*>(ptx.data())));

      return ptx;
    } break;

    default:
      RTCX_FAIL(std::format("Unsupported output binary type for linking CUDA libraries: ({})",
                            static_cast<i64>(params.output_type)),
                std::runtime_error);
  }
}

kernel_ref library_t::get_kernel(char const* name) const
{
  CUkernel kernel;
  RTCX_CHECK_CUDA(cuda->LibraryGetKernel(&kernel, handle_, name));
  return kernel_ref{kernel};
}

std::vector<kernel_ref> library_t::enumerate_kernels() const
{
  u32 num_kernels;
  RTCX_CHECK_CUDA(cuda->LibraryGetKernelCount(&num_kernels, handle_));

  std::vector<CUkernel> kernels;
  kernels.resize(num_kernels);

  RTCX_CHECK_CUDA(cuda->LibraryEnumerateKernels(kernels.data(), num_kernels, handle_));

  std::vector<kernel_ref> result;
  for (CUkernel k : kernels) {
    result.emplace_back(k);
  }

  return result;
}

std::string demangle_cuda_symbol(char const* mangled_name)
{
  i32 status;
  usize length;

  char* demangled_name = abi::__cxa_demangle(mangled_name, nullptr, &length, &status);

  RTCX_EXPECTS(status == 0, "Demangling CUDA symbol name failed", std::runtime_error);
  RTCX_EXPECTS(demangled_name != nullptr, "Demangling CUDA symbol name failed", std::runtime_error);

  RTCX_DEFER([&] {
    if (demangled_name != nullptr) free(demangled_name);
  });

  std::string result{demangled_name};

  return result;
}

namespace {

[[noreturn]] void throw_posix(std::string_view message, std::string_view syscall_name)
{
  auto errc = errno;
  RTCX_FAIL(
    std::format("{}. `{}` failed with {} ({})", message, syscall_name, errc, std::strerror(errc)),
    std::runtime_error);
}

}  // namespace

cache_t::cache_t(std::string cache_dir, cache_limits const& limits)
  : cache_dir_{std::move(cache_dir)},
    limits_{limits},
    blobs_cache_{limits.num_mem_blobs},
    libraries_cache_{limits.num_mem_libraries},
    tick_{0}
{
}

std::string const& cache_t::get_cache_dir() { return cache_dir_; }

std::optional<blob_t> blob_t::from_file(char const* path)
{
  i32 fd = open(path, O_RDONLY);

  if (fd == -1) {
    if (errno == ENOENT) {
      return std::nullopt;
    } else {
      throw_posix("Failed to open RTCX cache file from disk", "open");
    }
  }

  auto file_size = lseek(fd, 0, SEEK_END);
  if (file_size == -1) { throw_posix("Failed to determine size of RTCX cache file", "lseek"); }

  void* map = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);

  if (map == MAP_FAILED) { throw_posix("Failed to memory-map RTCX cache file", "mmap"); }

  if (close(fd) == -1) {
    throw_posix("Failed to close RTCX cache file after memory-mapping", "close");
  }

  auto deleter = +[](u8 const* buffer, usize size) {
    if (munmap(static_cast<void*>(const_cast<u8*>(buffer)), size) == -1) {
      throw_posix("Failed to unmap RTCX cache file from memory", "munmap");
    }
  };

  return blob_t::from_parts(static_cast<u8 const*>(map), file_size, deleter);
}

namespace {

/// @brief retrieves a blob from disk based on the given sha256 hash and object type (e.g. "blob",
/// "fragment", "library"). Returns nullopt if the file doesn't exist on disk, and throws if any
/// other error occurs.
std::optional<blob> get_disk_blob(std::string const& cache_dir,
                                  std::string const& object_type,
                                  sha256 const& sha)
{
  auto hex  = sha.to_hex_string();
  auto path = std::format("{}/{}.{}.bin", cache_dir, hex.view(), object_type);

  auto blob = blob_t::from_file(path.c_str());

  if (!blob.has_value()) { return std::nullopt; }
  {
    return std::make_shared<blob_t>(std::move(*blob));
  }
}

void evict_disk_entries(std::string const& cache_dir, u32 limit)
{
  i32 dir = open(cache_dir.c_str(), O_RDONLY | O_DIRECTORY);

  if (dir == -1) { throw_posix("Failed to open RTCX cache directory for evicting", "open"); }

  RTCX_DEFER([&] { close(dir); });

  std::vector<char> buffer;
  buffer.resize(8192);

  std::vector<std::string> paths;
  std::vector<std::chrono::nanoseconds> access_times;

  isize num_read = 0;

  while ((num_read = syscall(SYS_getdents64, dir, buffer.data(), buffer.size())) > 0) {
    isize byte_pos = 0;

    while (byte_pos < num_read) {
      auto* ent = reinterpret_cast<struct dirent64 const*>(buffer.data() + byte_pos);

      if (memcmp(ent->d_name, ".", 2) != 0 && memcmp(ent->d_name, "..", 3) != 0) {
        RTCX_EXPECTS(ent->d_type != DT_UNKNOWN,
                     "Found unknown directory entry type in RTCX cache dir",
                     std::runtime_error);

        if (ent->d_type == DT_REG) {
          auto path = std::format("{}/{}", cache_dir, ent->d_name);
          struct stat st;
          if (stat(path.c_str(), &st) == -1 && errno != ENOENT) {
            throw_posix("Failed to get RTCX cache file stats", "stat");
          }

          auto access_time =
            std::chrono::seconds{st.st_atim.tv_sec} + std::chrono::nanoseconds{st.st_atim.tv_nsec};

          paths.emplace_back(std::move(path));
          access_times.emplace_back(access_time);
        }
      }

      byte_pos += ent->d_reclen;
    }
  }

  if (num_read == -1) {
    throw_posix("Failed to read RTCX cache directory for clearing", "getdents64");
  }

  if (paths.size() < limit) { return; }

  std::vector<u32> ranking_indices;
  ranking_indices.resize(paths.size());

  std::iota(ranking_indices.begin(), ranking_indices.end(), 0);

  std::sort(ranking_indices.begin(), ranking_indices.end(), [&](i32 a, i32 b) {
    return access_times[a] < access_times[b];
  });

  // evict half of the least recently accessed
  auto num_evict = (limit == 0) ? paths.size() : ((limit + 1) / 2);

  for (auto index : std::span{ranking_indices}.subspan(0, num_evict)) {
    if (unlink(paths[index].c_str()) == -1 && errno != ENOENT) {
      throw_posix("Failed to evict RTCX cache file", "unlink");
    }
  }
}

/// @brief atomically writes a blob to disk by first writing to a temporary file and then renaming
/// it to the final path.
void cache_blob_to_disk(std::string const& cache_dir,
                        std::string const& object_type,
                        sha256 const& sha,
                        std::span<u8 const> binary,
                        u32 limit)
{
  if (limit > 0) {
    char temp_path[] = "/tmp/rtcx-bin-XXXXXX";

    {
      i32 fd = mkstemp(temp_path);
      if (fd == -1) { throw_posix("Failed to create temporary file for RTCX cache", "mkstemp"); }

      RTCX_DEFER([&] {
        if (close(fd) == -1) { throw_posix("Failed to close temporary RTCX cache file", "close"); }
      });

      if (write(fd, binary.data(), binary.size()) == -1) {
        throw_posix("Failed to write RTCX cache to temporary file", "write");
      }
    }

    auto hex        = sha.to_hex_string();
    auto final_path = std::format("{}/{}.{}.bin", cache_dir, hex.view(), object_type);

    std::filesystem::create_directories(std::filesystem::path{final_path}.parent_path());

    // rename is atomic, even if another process is performing the same operation
    if (rename(temp_path, final_path.c_str()) == -1) {
      if (errno == EEXIST) {
        // another process has already created the file, so just remove our temp file
        if (remove(temp_path) == -1) {
          throw_posix("Failed to remove temporary RTCX cache file", "remove");
        }
        return;
      } else {
        throw_posix(std::format("Failed to move temporary RTCX cache file to final location ({})",
                                final_path),
                    "rename");
      }
    }
  }

  evict_disk_entries(cache_dir, limit);
}

}  // namespace

std::shared_future<blob> cache_t::get_or_add_blob(sha256 const& sha, blob_compile_func compile)
{
  std::atomic_ref tick{tick_};
  auto current_tick = tick.fetch_add(1, std::memory_order_relaxed);

  bool unlocked = false;
  lock_.lock();

  RTCX_DEFER([&] {
    if (!unlocked) { lock_.unlock(); }
  });

  // check memory cache
  if (auto it = blobs_cache_.entries_.find(sha); it != blobs_cache_.entries_.end()) {
    counter_.blob_mem_hits.incr();

    // update LRU tick
    it->second.hit(current_tick);

    return it->second.value;

  } else {
    counter_.blob_mem_misses.incr();

    // check disk cache
    auto disk_blob = get_disk_blob(cache_dir_, "blob", sha);

    std::promise<blob> promise;
    auto fut       = promise.get_future().share();
    auto cache_fut = fut;
    auto ret_fut   = fut;

    if (disk_blob.has_value()) {
      counter_.blob_disk_hits.incr();

      promise.set_value(std::move(*disk_blob));

      // insert into cache
      blobs_cache_.insert(sha, std::move(cache_fut), current_tick);

      return ret_fut;

    } else {
      counter_.blob_disk_misses.incr();

      blobs_cache_.insert(sha, std::move(cache_fut), current_tick);

      // we can release the lock while calling the maker function since it may be expensive and we
      // have already reserved a spot in the cache for this sha
      lock_.unlock();
      unlocked = true;

      auto result = compile();
      promise.set_value(result);

      // store result to disk
      cache_blob_to_disk(cache_dir_, "blob", sha, result->view(), limits_.num_disk_entries);

      return ret_fut;
    }
  }
}

std::shared_future<library> cache_t::get_or_add_library(sha256 const& sha,
                                                        binary_type type,
                                                        library_compile_func compile)
{
  std::atomic_ref tick{tick_};
  auto current_tick = tick.fetch_add(1, std::memory_order_relaxed);

  bool unlocked = false;
  lock_.lock();

  RTCX_DEFER([&] {
    if (!unlocked) { lock_.unlock(); }
  });

  // check memory cache
  if (auto it = libraries_cache_.entries_.find(sha); it != libraries_cache_.entries_.end()) {
    counter_.library_mem_hits.incr();

    // update LRU tick
    it->second.hit(current_tick);

    return it->second.value;

  } else {
    counter_.library_mem_misses.incr();

    // check disk cache
    auto disk_blob = get_disk_blob(cache_dir_, "library", sha);

    std::promise<library> promise;
    auto fut       = promise.get_future().share();
    auto cache_fut = fut;
    auto ret_fut   = fut;

    if (disk_blob.has_value()) {
      counter_.library_disk_hits.incr();

      libraries_cache_.insert(sha, std::move(cache_fut), current_tick);

      // we can release the lock while calling the maker function since it may be expensive and we
      // have already reserved a spot in the cache for this sha
      lock_.unlock();
      unlocked = true;

      auto lib = load_library((*disk_blob)->view(), type);
      promise.set_value(std::move(lib));

      return ret_fut;

    } else {
      counter_.library_disk_misses.incr();

      libraries_cache_.insert(sha, std::move(cache_fut), current_tick);

      // we can release the lock while calling the maker function since it may be expensive and we
      // have already reserved a spot in the cache for this sha
      lock_.unlock();
      unlocked = true;

      auto [library, blob] = compile();
      promise.set_value(library);

      // store result to disk
      cache_blob_to_disk(cache_dir_, "library", sha, blob->view(), limits_.num_disk_entries);

      return ret_fut;
    }
  }
}

cache_stats cache_t::get_stats()
{
  return cache_stats{.blob_mem_hits       = counter_.blob_mem_hits.get(),
                     .blob_mem_misses     = counter_.blob_mem_misses.get(),
                     .blob_disk_hits      = counter_.blob_disk_hits.get(),
                     .blob_disk_misses    = counter_.blob_disk_misses.get(),
                     .library_mem_hits    = counter_.library_mem_hits.get(),
                     .library_mem_misses  = counter_.library_mem_misses.get(),
                     .library_disk_hits   = counter_.library_disk_hits.get(),
                     .library_disk_misses = counter_.library_disk_misses.get()};
}

void cache_t::clear_stats()
{
  counter_.blob_mem_hits.reset();
  counter_.blob_mem_misses.reset();
  counter_.blob_disk_hits.reset();
  counter_.blob_disk_misses.reset();
  counter_.library_mem_hits.reset();
  counter_.library_mem_misses.reset();
  counter_.library_disk_hits.reset();
  counter_.library_disk_misses.reset();
}

cache_limits cache_t::get_limits() { return limits_; }

usize cache_t::get_blob_count()
{
  std::lock_guard guard{lock_};
  return blobs_cache_.entries_.size();
}

usize cache_t::get_library_count()
{
  std::lock_guard guard{lock_};
  return libraries_cache_.entries_.size();
}

void cache_t::clear_memory_store()
{
  std::lock_guard guard{lock_};

  blobs_cache_.entries_.clear();
  libraries_cache_.entries_.clear();
}

void cache_t::clear_disk_store()
{
  i32 dir = open(cache_dir_.c_str(), O_RDONLY | O_DIRECTORY);

  if (dir == -1) { throw_posix("Failed to open RTCX cache directory for clearing", "opendir"); }

  RTCX_DEFER([&] { close(dir); });

  std::vector<char> buffer;
  buffer.resize(8192);
  std::vector<char> entry_path;
  entry_path.resize(4096);

  isize num_read = 0;

  while ((num_read = syscall(SYS_getdents64, dir, buffer.data(), buffer.size())) > 0) {
    isize byte_pos = 0;

    while (byte_pos < num_read) {
      auto* ent = reinterpret_cast<struct dirent64 const*>(buffer.data() + byte_pos);

      if (memcmp(ent->d_name, ".", 2) != 0 && memcmp(ent->d_name, "..", 3) != 0) {
        RTCX_EXPECTS(ent->d_type != DT_UNKNOWN,
                     "Found unknown directory entry type in RTCX cache dir",
                     std::runtime_error);

        if (ent->d_type == DT_REG) {
          snprintf(entry_path.data(), entry_path.size(), "%s/%s", cache_dir_.c_str(), ent->d_name);

          if (unlink(entry_path.data()) == -1 && errno != ENOENT) {
            throw_posix("Failed to unlink RTCX cache file during clearing", "unlink");
          }
        }
      }

      byte_pos += ent->d_reclen;
    }
  }

  if (num_read == -1) {
    throw_posix("Failed to read RTCX cache directory for clearing", "getdents64");
  }
}

}  // namespace RTCX_EXPORT rtcx
