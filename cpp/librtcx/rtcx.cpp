
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

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <format>
#include <numeric>
#include <source_location>

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
      RTCX_EXPECTS(::rtcx::cu->GetErrorString(__result, &__enum_str) == ::CUDA_SUCCESS,   \
                   "Unable to get CUDA error string",                                     \
                   std::runtime_error);                                                   \
      auto __errstr = ::std::format("(cuda) expression `{}` failed, with error ({}): {}", \
                                    #__VA_ARGS__,                                         \
                                    static_cast<::std::int64_t>(__result),                \
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
                                    static_cast<::std::int64_t>(__result),                      \
                                    __enum_name,                                                \
                                    __enum_msg);                                                \
      RTCX_FAIL(__errstr, ::std::runtime_error);                                                \
    }                                                                                           \
  } while (0)

#define RTCX_CHECK_NVRTC(...)                                                              \
  do {                                                                                     \
    ::nvrtcResult __result = (__VA_ARGS__);                                                \
    if (__result != ::NVRTC_SUCCESS) {                                                     \
      auto __errstr = ::std::format("(nvrtc) expression `{}` failed, with error ({}): {}", \
                                    #__VA_ARGS__,                                          \
                                    static_cast<::std::int64_t>(__result),                 \
                                    ::rtcx::nvrtc->GetErrorString(__result));              \
      RTCX_FAIL(__errstr, ::std::runtime_error);                                           \
    }                                                                                      \
  } while (0)

#define RTCX_CHECK_NVJITLINK(...)                                                              \
  do {                                                                                         \
    ::nvJitLinkResult __result = (__VA_ARGS__);                                                \
    if (__result != ::NVJITLINK_SUCCESS) {                                                     \
      auto __errstr = ::std::format("(nvJitLink) expression `{}` failed, with error ({}): {}", \
                                    #__VA_ARGS__,                                              \
                                    static_cast<::std::int64_t>(__result),                     \
                                    ::rtcx::get_nvJitLinkResultString(__result));              \
      RTCX_FAIL(__errstr, ::std::runtime_error);                                               \
    }                                                                                          \
  } while (0)

namespace rtcx {

namespace {

template <typename StringType>
std::string join_strings(std::span<StringType> strings, std::string_view separator)
{
  if (strings.empty()) { return {}; }

  if (strings.size() == 1) { return std::string{strings[0].begin(), strings[0].end()}; }

  auto total_size = std::transform_reduce(
    strings.begin(),
    strings.end(),
    size_t{0},
    [](size_t total, size_t str_size) { return total + str_size; },
    [](auto& str) { return str.size(); });

  auto separator_size = separator.size() * (strings.size() - 1);

  std::string result;
  result.reserve(total_size + separator_size);

  for (size_t i = 0; i < strings.size(); ++i) {
    result.append(strings[i].begin(), strings[i].end());
    if (i != (strings.size() - 1)) { result.append(separator); }
  }

  return result;
}

}  // namespace

void log_warning(std::string_view message)
{
  ::fprintf(
    stdout, "[RTCX WARNING] %.*s\n", static_cast<std::int32_t>(message.size()), message.data());
}

void log_error(std::string_view message)
{
  ::fprintf(
    stderr, "[RTCX ERROR] %.*s\n", static_cast<std::int32_t>(message.size()), message.data());
}

void log_trace(std::string_view message)
{
  ::fprintf(
    stdout, "[RTCX TRACE] %.*s\n", static_cast<std::int32_t>(message.size()), message.data());
}

#define FOR_EACH_CUDA_FUNC(DO_IT)       \
  DO_IT(GetErrorString)                 \
  DO_IT(GetErrorName)                   \
  DO_IT(Init)                           \
  DO_IT(OccupancyMaxPotentialBlockSize) \
  DO_IT(LaunchKernel)                   \
  DO_IT(LaunchKernelEx)                 \
  DO_IT(LaunchCooperativeKernel)        \
  DO_IT(KernelGetFunction)              \
  DO_IT(LibraryLoadData)                \
  DO_IT(LibraryLoadFromFile)            \
  DO_IT(LibraryGetKernel)               \
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
  DO_IT(GetLoweredName)

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
      RTCX_FAIL(
        std::format("Unrecognized nvJitLinkResult type: ({})", static_cast<std::int64_t>(result)),
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
      RTCX_FAIL(std::format("Unrecognized binary_type: ({})", static_cast<std::int64_t>(type)),
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
      RTCX_FAIL(std::format("Unrecognized binary type for linking: ({}) ",
                            static_cast<std::int64_t>(bin_type)),
                std::logic_error);
  }
}

void* load_dll(std::string_view base_name, std::span<std::string const> names)
{
  for (auto& name : names) {
    void* handle = ::dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle != nullptr) { return handle; }
  }

  RTCX_FAIL(
    std::format(
      "Failed to load dynamic library `{}` (tried: {})", base_name, join_strings(names, ", ")),
    std::runtime_error);
}

void* get_symbol(char const* lib_name, void* handle, char const* sym_name)
{
  void* sym = ::dlsym(handle, sym_name);
  if (sym == nullptr) {
    RTCX_FAIL(
      std::format(
        "Failed to load symbol `{}` from `{}`, error: `{}`", sym_name, lib_name, ::dlerror()),
      std::runtime_error);
  }
  return sym;
}

inline constexpr std::int32_t major_version(std::int32_t version) { return version / 1000; }

inline constexpr std::int32_t minor_version(std::int32_t version) { return (version % 1000) / 10; }

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
  ~LibCuda() { ::dlclose(_handle); }

  static void* _load()
  {
    std::string lib_names[] = {"libcuda.so"};  // NOLINT(modernize-avoid-c-arrays)
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
    std::int32_t cuda_version;
    RTCX_CHECK_CUDART(cudaRuntimeGetVersion(&cuda_version));
    std::int32_t major = major_version(cuda_version);
    std::int32_t minor = minor_version(cuda_version);

    std::string lib_names[] =  // NOLINT(modernize-avoid-c-arrays)
      {std::format("libnvrtc.so.{}.{}", major, minor),
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
  ~LibNVJitLink() { ::dlclose(_handle); }

  static void* _load()
  {
    std::int32_t cuda_version;
    RTCX_CHECK_CUDART(cudaRuntimeGetVersion(&cuda_version));
    std::int32_t major = major_version(cuda_version);
    std::int32_t minor = minor_version(cuda_version);

    std::string lib_names[] =  // NOLINT(modernize-avoid-c-arrays)
      {std::format("libnvJitLink.so.{}.{}", major, minor),
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

static std::optional<LibCuda> cu;
static std::optional<LibNVRTC> nvrtc;
static std::optional<LibNVJitLink> nvjitlink;
static std::optional<std::once_flag> init_libraries_flag{std::in_place};
static std::optional<std::once_flag> teardown_libraries_flag{std::in_place};

}  // namespace

void initialize()
{
  std::call_once(*init_libraries_flag, [] {
    cu.emplace(LibCuda::_load());
    RTCX_EXPECTS(
      cu->Init(0) == CUDA_SUCCESS, "Failed to initialize CUDA driver API", std::runtime_error);
    nvrtc.emplace(LibNVRTC::_load());
    nvjitlink.emplace(LibNVJitLink::_load());
  });
}

void teardown()
{
  std::call_once(*teardown_libraries_flag, [] {
    nvjitlink.reset();
    nvrtc.reset();
    cu.reset();
    init_libraries_flag.reset();
    teardown_libraries_flag.reset();
  });
}

blob_t blob_t::from_buffer(byte_buffer&& buffer)
{
  auto size = buffer.size();
  auto data = buffer.release();
  return blob_t::from_parts(
    data, size, +[](std::uint8_t const* data, std::size_t) {
      ::free(const_cast<std::uint8_t*>(data));
    });
}

blob_t blob_t::from_static_data(std::span<std::uint8_t const> data)
{
  return blob_t::from_parts(data.data(), data.size(), blob_t::noop_deallocator);
}

namespace {
void log_nvrtc_result(compile_params const& params,
                      nvrtcProgram program,
                      nvrtcResult compile_result)
{
  if (program == nullptr || compile_result == NVRTC_SUCCESS) { return; }

  std::size_t log_size;
  if (auto errc = nvrtc->GetProgramLogSize(program, &log_size); errc != NVRTC_SUCCESS) {
    RTCX_FAIL(std::format("Failed to get NVRTC program log size with error ({}): {}",
                          static_cast<std::int64_t>(errc),
                          nvrtc->GetErrorString(errc)),
              std::runtime_error);
  }

  if (log_size <= 1) { return; }

  std::vector<char> log;
  log.resize(log_size);

  if (auto errc = nvrtc->GetProgramLog(program, log.data()); errc != NVRTC_SUCCESS) {
    RTCX_FAIL(std::format("Failed to get NVRTC program log with error ({}): {}",
                          static_cast<std::int64_t>(errc),
                          nvrtc->GetErrorString(errc)),
              std::runtime_error);
  }

  log.resize(log_size == 0 ? 0 : (log_size - 1));

  auto status_str = (compile_result == NVRTC_SUCCESS && !log.empty()) ? "completed with warning"
                                                                      : "failed with error";

  std::string headers_str;
  for (auto& header : params.header_include_names) {
    headers_str = std::format("{}\t{}\n", headers_str, header);
  }

  std::string options_str;
  for (auto& option : params.options) {
    options_str = std::format("{}\t{}\n", options_str, option);
  }

  auto msg = std::format(
    "NVRTC Compilation for `{}` {} ({}): {}.\nHeaders:\n{}\n\nOptions:\n{}\n\nLog:\n\t{}",
    params.name == nullptr ? "<unnamed>" : params.name,
    status_str,
    static_cast<std::int64_t>(compile_result),
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
  if (handle == nullptr || link_result == NVJITLINK_SUCCESS) { return; }

  std::size_t info_log_size;
  if (auto errc = nvjitlink->GetInfoLogSize(handle, &info_log_size); errc != NVJITLINK_SUCCESS) {
    RTCX_FAIL(std::format("Failed to get nvJitLink info log size with error ({}): {}",
                          static_cast<std::int64_t>(errc),
                          get_nvJitLinkResultString(errc)),
              std::runtime_error);
  }

  std::vector<char> info_log;
  if (info_log_size > 1) {
    info_log.resize(info_log_size);
    if (auto errc = nvjitlink->GetInfoLog(handle, info_log.data()); errc != NVJITLINK_SUCCESS) {
      RTCX_FAIL(std::format("Failed to get nvJitLink info log with error ({}): {}",
                            static_cast<std::int64_t>(errc),
                            get_nvJitLinkResultString(errc)),
                std::runtime_error);
    }
  }
  info_log.resize(info_log_size == 0 ? 0 : (info_log_size - 1));

  std::size_t error_log_size;
  if (auto errc = nvjitlink->GetErrorLogSize(handle, &error_log_size); errc != NVJITLINK_SUCCESS) {
    RTCX_FAIL(std::format("Failed to get nvJitLink error log size with error ({}): {}",
                          static_cast<std::int64_t>(errc),
                          get_nvJitLinkResultString(errc)),
              std::runtime_error);
  }

  std::vector<char> error_log;

  if (error_log_size > 1) {
    error_log.resize(error_log_size);
    if (auto errc = nvjitlink->GetErrorLog(handle, error_log.data()); errc != NVJITLINK_SUCCESS) {
      RTCX_FAIL(std::format("Failed to get nvJitLink error log with error ({}): {}",
                            static_cast<std::int64_t>(errc),
                            get_nvJitLinkResultString(errc)),
                std::runtime_error);
    }
  }
  error_log.resize(error_log_size == 0 ? 0 : (error_log_size - 1));

  if (info_log.empty() && error_log.empty()) { return; }

  std::string fragments_str;
  for (auto& frag : params.file_fragments) {
    fragments_str = std::format("{}\t{}\n", fragments_str, frag.path);
  }

  for (auto& frag : params.memory_fragments) {
    fragments_str = std::format("{}\t{}\n", fragments_str, frag.name);
  }

  std::string link_options_str;
  for (auto& option : params.link_options) {
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
    static_cast<std::int64_t>(link_result),
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

byte_buffer compile(compile_params const& params)
{
  RTCX_EXPECTS(params.name != nullptr, "Fragment name must not be null", std::logic_error);
  RTCX_EXPECTS(params.source != nullptr, "Fragment source must not be null", std::logic_error);

  nvrtcProgram program = nullptr;
  RTCX_CHECK_NVRTC(nvrtc->CreateProgram(&program,
                                        params.source,
                                        params.name,
                                        static_cast<std::int32_t>(params.headers.size()),
                                        params.headers.data(),
                                        params.header_include_names.data()));

  RTCX_DEFER([&] { nvrtc->DestroyProgram(&program); });

  for (auto* name_expr : params.name_expressions) {
    RTCX_CHECK_NVRTC(nvrtc->AddNameExpression(program, name_expr));
  }

  auto compile_result = nvrtc->CompileProgram(
    program, static_cast<std::int32_t>(params.options.size()), params.options.data());
  log_nvrtc_result(params, program, compile_result);
  RTCX_CHECK_NVRTC(compile_result);

  switch (params.target_type) {
    case binary_type::CUBIN: {
      std::size_t cubin_size;
      RTCX_CHECK_NVRTC(nvrtc->GetCUBINSize(program, &cubin_size));
      auto cubin = byte_buffer::make(cubin_size);
      RTCX_CHECK_NVRTC(nvrtc->GetCUBIN(program, reinterpret_cast<char*>(cubin.data())));
      return cubin;
    } break;
    case binary_type::LTO_IR: {
      std::size_t lto_ir_size;
      RTCX_CHECK_NVRTC(nvrtc->GetLTOIRSize(program, &lto_ir_size));
      auto lto_ir = byte_buffer::make(lto_ir_size);
      RTCX_CHECK_NVRTC(nvrtc->GetLTOIR(program, reinterpret_cast<char*>(lto_ir.data())));
      return lto_ir;
    } break;
    case binary_type::PTX: {
      std::size_t ptx_size;
      RTCX_CHECK_NVRTC(nvrtc->GetPTXSize(program, &ptx_size));
      auto ptx = byte_buffer::make(ptx_size);
      RTCX_CHECK_NVRTC(nvrtc->GetPTX(program, reinterpret_cast<char*>(ptx.data())));
      return ptx;
    } break;
    default:
      RTCX_FAIL(std::format("Unsupported binary type for compiling fragment: {}",
                            binary_type_string(params.target_type)),
                std::logic_error);
  }
}

kernel_occupancy_config kernel_ref::max_occupancy_config(std::size_t dynamic_shared_memory_bytes,
                                                         std::int32_t block_size_limit) const
{
  std::int32_t min_grid_size;
  std::int32_t block_size;
  RTCX_CHECK_CUDA(cu->OccupancyMaxPotentialBlockSize(&min_grid_size,
                                                     &block_size,
                                                     reinterpret_cast<CUfunction>(handle_),
                                                     nullptr,
                                                     dynamic_shared_memory_bytes,
                                                     block_size_limit));

  return kernel_occupancy_config{.min_grid_size = static_cast<std::uint32_t>(min_grid_size),
                                 .block_size    = static_cast<std::uint32_t>(block_size)};
}

void kernel_ref::launch(cuda_dim3 grid_dim,
                        cuda_dim3 block_dim,
                        std::uint32_t shared_mem_bytes,
                        CUstream stream,
                        void** kernel_params) const
{
  RTCX_EXPECTS(grid_dim.is_valid(), "Grid dimensions must be greater than zero", std::logic_error);
  RTCX_EXPECTS(
    block_dim.is_valid(), "Block dimensions must be greater than zero", std::logic_error);
  RTCX_EXPECTS(
    kernel_params != nullptr, "Kernel parameters pointer must not be null", std::logic_error);

  CUlaunchConfig cfg{.gridDimX       = grid_dim.x,
                     .gridDimY       = grid_dim.y,
                     .gridDimZ       = grid_dim.z,
                     .blockDimX      = block_dim.x,
                     .blockDimY      = block_dim.y,
                     .blockDimZ      = block_dim.z,
                     .sharedMemBytes = shared_mem_bytes,
                     .hStream        = stream,
                     .attrs          = nullptr,
                     .numAttrs       = 0};

  RTCX_CHECK_CUDA(
    cu->LaunchKernelEx(&cfg, reinterpret_cast<CUfunction>(handle_), kernel_params, nullptr));
}

void kernel_ref::launch_cooperative(cuda_dim3 grid_dim,
                                    cuda_dim3 block_dim,
                                    std::uint32_t shared_mem_bytes,
                                    CUstream stream,
                                    void** kernel_params) const
{
  RTCX_EXPECTS(grid_dim.is_valid(), "Grid dimensions must be greater than zero", std::logic_error);
  RTCX_EXPECTS(
    block_dim.is_valid(), "Block dimensions must be greater than zero", std::logic_error);
  RTCX_EXPECTS(
    kernel_params != nullptr, "Kernel parameters pointer must not be null", std::logic_error);

  RTCX_CHECK_CUDA(cu->LaunchCooperativeKernel(reinterpret_cast<CUfunction>(handle_),
                                              grid_dim.x,
                                              grid_dim.y,
                                              grid_dim.z,
                                              block_dim.x,
                                              block_dim.y,
                                              block_dim.z,
                                              shared_mem_bytes,
                                              stream,
                                              kernel_params));
}

library_t::~library_t()
{
  if (handle_ != nullptr) { cu->LibraryUnload(handle_); }
}

library load_library(std::span<std::uint8_t const> binary)
{
  CUlibrary handle;

  RTCX_CHECK_CUDA(
    cu->LibraryLoadData(&handle, binary.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  RTCX_DEFER([&] {
    if (handle != nullptr) { RTCX_CHECK_CUDA(cu->LibraryUnload(handle)); }
  });

  auto library = std::make_shared<library_t>(handle);

  handle = nullptr;

  return library;
}

library load_library_from_file(char const* path)
{
  RTCX_EXPECTS(path != nullptr, "Library path must not be null", std::logic_error);

  CUlibrary handle;

  RTCX_CHECK_CUDA(cu->LibraryLoadFromFile(&handle, path, nullptr, nullptr, 0, nullptr, nullptr, 0));

  RTCX_DEFER([&] {
    if (handle != nullptr) { RTCX_CHECK_CUDA(cu->LibraryUnload(handle)); }
  });

  auto library = std::make_shared<library_t>(handle);

  handle = nullptr;

  return library;
}

byte_buffer link_library(link_params const& params)
{
  RTCX_EXPECTS(params.name != nullptr, "Link output name must not be null", std::logic_error);
  RTCX_EXPECTS(params.output_type == binary_type::CUBIN || params.output_type == binary_type::PTX,
               "Only CUBIN and PTX output types are supported for linking modules",
               std::logic_error);
  RTCX_EXPECTS(params.file_fragments.size() != 0 || params.memory_fragments.size() != 0,
               "At least one fragment must be provided for linking",
               std::logic_error);

  for (auto& frag : params.file_fragments) {
    RTCX_EXPECTS(frag.path != nullptr, "Fragment file path must not be empty", std::logic_error);
  }

  for (auto& frag : params.memory_fragments) {
    RTCX_EXPECTS(
      frag.data.size_bytes() > 0, "Fragment binary data must be non-empty", std::logic_error);
  }

  nvJitLinkHandle handle = nullptr;
  RTCX_CHECK_NVJITLINK(nvjitlink->Create(&handle,
                                         static_cast<std::uint32_t>(params.link_options.size()),
                                         const_cast<char const**>(params.link_options.data())));

  RTCX_DEFER([&] { nvjitlink->Destroy(&handle); });

  for (auto& frag : params.file_fragments) {
    RTCX_CHECK_NVJITLINK(nvjitlink->AddFile(handle, to_nvjitlink_input_type(frag.type), frag.path));
  }

  for (auto& frag : params.memory_fragments) {
    RTCX_CHECK_NVJITLINK(nvjitlink->AddData(handle,
                                            to_nvjitlink_input_type(frag.type),
                                            frag.data.data(),
                                            frag.data.size_bytes(),
                                            frag.name));
  }

  auto link_result = nvjitlink->Complete(handle);
  log_nvJitLink_result(params, handle, link_result);
  RTCX_CHECK_NVJITLINK(link_result);

  switch (params.output_type) {
    case binary_type::CUBIN: {
      std::size_t cubin_size;
      RTCX_CHECK_NVJITLINK(nvjitlink->GetLinkedCubinSize(handle, &cubin_size));
      auto cubin = byte_buffer::make(cubin_size);
      RTCX_CHECK_NVJITLINK(nvjitlink->GetLinkedCubin(handle, cubin.data()));
      return cubin;
    } break;
    case binary_type::PTX: {
      std::size_t ptx_size;
      RTCX_CHECK_NVJITLINK(nvjitlink->GetLinkedPtxSize(handle, &ptx_size));
      auto ptx = byte_buffer::make(ptx_size);
      RTCX_CHECK_NVJITLINK(nvjitlink->GetLinkedPtx(handle, reinterpret_cast<char*>(ptx.data())));
      return ptx;
    } break;
    default:
      RTCX_FAIL(std::format("Unsupported output binary type for linking CUDA libraries: ({})",
                            binary_type_string(params.output_type)),
                std::runtime_error);
  }
}

kernel_ref library_t::get_kernel(char const* name) const
{
  CUkernel kernel;
  RTCX_CHECK_CUDA(cu->LibraryGetKernel(&kernel, handle_, name));
  return kernel_ref{kernel};
}

std::string demangle_cuda_symbol(char const* mangled_name)
{
  std::int32_t status;
  std::size_t length;
  char* demangled_name = abi::__cxa_demangle(mangled_name, nullptr, &length, &status);

  RTCX_EXPECTS(status == 0, "Demangling CUDA symbol name failed", std::runtime_error);
  RTCX_EXPECTS(demangled_name != nullptr, "Demangling CUDA symbol name failed", std::runtime_error);

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

cache_t::cache_t(std::string cache_dir,
                 std::string tmp_dir,
                 cache_limits const& limits,
                 bool preload,
                 bool disable)
  : enabled_{!disable},
    cache_dir_{std::move(cache_dir)},
    tmp_dir_{std::move(tmp_dir)},
    limits_{limits},
    lock_{},
    blobs_cache_{limits.num_mem_blobs},
    libraries_cache_{limits.num_mem_libraries},
    tick_{0}
{
  if (preload) { preload_from_disk(); }
}

std::string const& cache_t::get_cache_dir() { return cache_dir_; }

std::string const& cache_t::get_tmp_dir() { return tmp_dir_; }

std::optional<blob_t> blob_t::from_file(char const* path)
{
  std::int32_t fd = ::open(path, O_RDONLY);

  if (fd == -1) {
    if (errno == ENOENT) {
      return std::nullopt;
    } else {
      throw_posix("Failed to open RTCX cache file from disk", "open");
    }
  }

  RTCX_DEFER([&] {
    if (::close(fd) == -1) {
      throw_posix("Failed to close RTCX cache file after memory-mapping", "close");
    }
  });

  auto file_size = ::lseek(fd, 0, SEEK_END);
  if (file_size == -1) { throw_posix("Failed to determine size of RTCX cache file", "lseek"); }

  void* map = ::mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);

  if (map == MAP_FAILED) { throw_posix("Failed to memory-map RTCX cache file", "mmap"); }

  auto deleter = +[](std::uint8_t const* buffer, std::size_t size) {
    if (::munmap(static_cast<void*>(const_cast<std::uint8_t*>(buffer)), size) == -1) {
      throw_posix("Failed to unmap RTCX cache file from memory", "munmap");
    }
  };

  return blob_t::from_parts(static_cast<std::uint8_t const*>(map), file_size, deleter);
}

namespace {

/// @brief retrieves a blob from disk based on the given sha256 hash and object type (e.g. "blob",
/// "cuLibrary"). Returns nullopt if the file doesn't exist on disk, and throws if any other error
/// occurs.
std::optional<blob> get_disk_blob(std::string const& cache_dir,
                                  std::string const& object_type,
                                  sha256 const& sha)
{
  auto hex  = sha.to_hex_string();
  auto path = std::format("{}/{}.{}.bin", cache_dir, hex.view(), object_type);

  auto blob = blob_t::from_file(path.c_str());

  if (!blob.has_value()) { return std::nullopt; }
  return std::make_shared<blob_t>(std::move(*blob));
}

std::optional<library> get_disk_library(std::string const& cache_dir, sha256 const& sha)
{
  auto hex  = sha.to_hex_string();
  auto path = std::format("{}/{}.cuLibrary.bin", cache_dir, hex.view());

  CUlibrary handle;
  auto errc =
    cu->LibraryLoadFromFile(&handle, path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0);

  if (errc == CUDA_ERROR_FILE_NOT_FOUND) { return std::nullopt; }

  RTCX_EXPECTS(errc == CUDA_SUCCESS,
               std::format("Failed to load library `{}` from RTCX cache file", path),
               std::runtime_error);

  return std::make_shared<library_t>(handle);
}

std::pair<std::vector<std::string>, std::vector<std::chrono::nanoseconds>> get_disk_entries(
  std::string const& cache_dir)
{
  std::int32_t dir = ::open(cache_dir.c_str(), O_RDONLY | O_DIRECTORY);

  if (dir == -1) { throw_posix("Failed to open RTCX cache directory for evicting", "open"); }

  RTCX_DEFER([&] { ::close(dir); });

  std::vector<char> buffer;
  buffer.resize(8192);

  std::vector<std::string> paths;
  std::vector<std::chrono::nanoseconds> access_times;

  std::ptrdiff_t num_read = 0;

  while ((num_read = ::syscall(SYS_getdents64, dir, buffer.data(), buffer.size())) > 0) {
    std::ptrdiff_t byte_pos = 0;

    while (byte_pos < num_read) {
      auto* ent = reinterpret_cast<struct dirent64 const*>(buffer.data() + byte_pos);

      if (::memcmp(ent->d_name, ".", 2) != 0 && ::memcmp(ent->d_name, "..", 3) != 0) {
        RTCX_EXPECTS(ent->d_type != DT_UNKNOWN,
                     "Found unknown directory entry type in RTCX cache dir",
                     std::runtime_error);

        if (ent->d_type == DT_REG) {
          auto path = std::format("{}/{}", cache_dir, ent->d_name);
          struct stat st;
          if (::stat(path.c_str(), &st) == -1) {
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

  return {std::move(paths), std::move(access_times)};
}

void evict_disk_entries(std::string const& cache_dir, std::uint32_t limit)
{
  auto [paths, access_times] = get_disk_entries(cache_dir);

  if (paths.size() < limit) { return; }

  std::vector<std::size_t> ranking_indices;
  ranking_indices.resize(paths.size());

  std::iota(ranking_indices.begin(), ranking_indices.end(), 0);

  std::sort(ranking_indices.begin(), ranking_indices.end(), [&](auto a, auto b) {
    return access_times[a] < access_times[b];  // NOLINT(clang-analyzer-core.CallAndMessage)
  });

  // evict half of the least recently accessed
  auto num_evict = (limit == 0) ? paths.size() : ((limit + 1) / 2);

  for (auto index : std::span{ranking_indices}.subspan(0, num_evict)) {
    if (::unlink(paths[index].c_str()) == -1 && errno != ENOENT) {
      throw_posix("Failed to evict RTCX cache file", "unlink");
    }
  }
}

/// @brief atomically writes a blob to disk by first writing to a temporary file and then renaming
/// it to the final path.
void cache_blob_to_disk(std::string const& cache_dir,
                        std::string const& tmp_dir,
                        std::string const& object_type,
                        sha256 const& sha,
                        std::span<std::uint8_t const> binary,
                        std::uint32_t limit)
{
  // TODO: add cuda driver and runtime version to log
  if (limit > 0) {
    auto tmp_path = std::format("{}/rtcx-bin-XXXXXX", tmp_dir);
    (void)tmp_path.c_str();  // to ensure null-termination for mkstemp

    {
      std::int32_t fd = ::mkstemp(tmp_path.data());
      if (fd == -1) { throw_posix("Failed to create temporary file for RTCX cache", "mkstemp"); }

      RTCX_DEFER([&] {
        if (::close(fd) == -1) {
          throw_posix("Failed to close temporary RTCX cache file", "close");
        }
      });

      if (::write(fd, binary.data(), binary.size()) == -1) {
        throw_posix("Failed to write RTCX cache to temporary file", "write");
      }
    }

    auto hex        = sha.to_hex_string();
    auto final_path = std::format("{}/{}.{}.bin", cache_dir, hex.view(), object_type);

    std::filesystem::create_directories(std::filesystem::path{final_path}.parent_path());

    // rename is atomic, even if another process is performing the same operation
    if (::rename(tmp_path.c_str(), final_path.c_str()) == -1) {
      if (errno == EEXIST) {
        // another process has already created the file, so just remove our temp file
        if (::remove(tmp_path.c_str()) == -1) {
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
  if (auto it = enabled_ ? blobs_cache_.entries_.find(sha) : blobs_cache_.entries_.end();
      it != blobs_cache_.entries_.end()) {
    counter_.blob_mem_hits.incr();

    // update LRU tick
    it->second.hit(current_tick);

    return it->second.value;

  } else {
    counter_.blob_mem_misses.incr();

    // check disk cache
    std::optional<blob> disk_blob = std::nullopt;
    if (enabled_) { disk_blob = get_disk_blob(cache_dir_, "blob", sha); }

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
      cache_blob_to_disk(
        cache_dir_, tmp_dir_, "blob", sha, result->view(), limits_.num_disk_entries);

      return ret_fut;
    }
  }
}

std::shared_future<library> cache_t::get_or_add_library(sha256 const& sha,
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
  if (auto it = enabled_ ? libraries_cache_.entries_.find(sha) : libraries_cache_.entries_.end();
      it != libraries_cache_.entries_.end()) {
    counter_.library_mem_hits.incr();

    // update LRU tick
    it->second.hit(current_tick);

    return it->second.value;

  } else {
    counter_.library_mem_misses.incr();

    // check disk cache
    std::optional<library> disk_library = std::nullopt;
    if (enabled_) { disk_library = get_disk_library(cache_dir_, sha); }

    std::promise<library> promise;
    auto fut       = promise.get_future().share();
    auto cache_fut = fut;
    auto ret_fut   = fut;

    if (disk_library.has_value()) {
      counter_.library_disk_hits.incr();

      libraries_cache_.insert(sha, std::move(cache_fut), current_tick);

      // we can release the lock while calling the maker function since it may be expensive and we
      // have already reserved a spot in the cache for this sha
      lock_.unlock();
      unlocked = true;

      promise.set_value(std::move(*disk_library));

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
      cache_blob_to_disk(
        cache_dir_, tmp_dir_, "cuLibrary", sha, blob->view(), limits_.num_disk_entries);

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

std::size_t cache_t::get_blob_count()
{
  std::lock_guard guard{lock_};
  return blobs_cache_.entries_.size();
}

std::size_t cache_t::get_library_count()
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

void cache_t::clear_disk_store() { evict_disk_entries(cache_dir_, 0); }

void cache_t::preload_from_disk()
{
  auto [paths, access_times] = get_disk_entries(cache_dir_);

  std::vector<std::size_t> ranking_indices;
  ranking_indices.resize(paths.size());
  std::iota(ranking_indices.begin(), ranking_indices.end(), 0);
  std::sort(ranking_indices.begin(), ranking_indices.end(), [&](auto a, auto b) {
    return access_times[a] > access_times[b];  // NOLINT(clang-analyzer-core.CallAndMessage)
  });

  auto load_count = std::min<std::size_t>(ranking_indices.size(),
                                          limits_.num_mem_blobs + limits_.num_mem_libraries);

  ranking_indices.resize(load_count);

  {
    std::lock_guard guard{lock_};
    tick_++;

    for (auto index : ranking_indices) {
      auto path = paths[index];
      try {
        auto file_name = std::filesystem::path{path}.filename().string();
        auto sha_str   = file_name.substr(0, file_name.find('.'));
        auto sha       = sha256::parse(sha_str);

        if (path.ends_with(".blob.bin")) {
          auto data = blob_t::from_file(path.c_str());
          if (!data.has_value()) { continue; }
          auto blob = std::make_shared<blob_t>(std::move(*data));
          std::promise<rtcx::blob> promise;
          auto fut = promise.get_future().share();
          promise.set_value(std::move(blob));
          blobs_cache_.insert(sha, std::move(fut), tick_);
        } else if (path.ends_with(".cuLibrary.bin")) {
          auto lib = get_disk_library(cache_dir_, sha);
          if (!lib.has_value()) { continue; }
          std::promise<library> promise;
          auto fut = promise.get_future().share();
          promise.set_value(std::move(*lib));
          libraries_cache_.insert(sha, std::move(fut), tick_);
        }
      } catch (std::exception const& e) {
        // ignore any errors during preload
        log_error(e.what());
      } catch (...) {
        log_error("Unknown error during preload");
      }
    }
  }
}

void cache_t::enable(bool enable)
{
  std::lock_guard guard{lock_};
  enabled_ = enable;
}

bool cache_t::is_enabled()
{
  std::lock_guard guard{lock_};
  return enabled_;
}

std::string reflect_bool(bool value) { return std::format("(bool){}", value); }

std::string reflect_int(std::uint8_t value) { return std::format("(unsigned char){}U", value); }

std::string reflect_int(std::uint16_t value) { return std::format("(unsigned short){}U", value); }

std::string reflect_int(std::uint32_t value) { return std::format("(unsigned int){}U", value); }

std::string reflect_int(std::uint64_t value)
{
  return std::format("(unsigned long long int){}ULL", value);
}

std::string reflect_int(std::int8_t value) { return std::format("(signed char){}", value); }

std::string reflect_int(std::int16_t value) { return std::format("(signed short){}", value); }

std::string reflect_int(std::int32_t value) { return std::format("(signed int){}", value); }

std::string reflect_int(std::int64_t value)
{
  return std::format("(signed long long int){}LL", value);
}

std::string reflect_float(float value) { return std::format("(float){}F", value); }

std::string reflect_float(double value) { return std::format("(double){}", value); }

std::string reflect_cast(std::string_view type, std::string_view value)
{
  return std::format("(({})({}))", type, value);
}

std::string reflect_template(std::string_view template_name,
                             std::span<std::string_view const> template_args)
{
  return std::format("{}<{}>", template_name, join_strings(template_args, ", "));
}

std::string reflect_template(std::string_view template_name,
                             std::span<std::string const> template_args)
{
  return std::format("{}<{}>", template_name, join_strings(template_args, ", "));
}

}  // namespace rtcx
