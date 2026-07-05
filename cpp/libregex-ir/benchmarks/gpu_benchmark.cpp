/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/extract.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/split/split_re.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda.h>
#include <cuda/std/utility>
#include <cuda_runtime_api.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <fmt/format.h>
#include <nvJitLink.h>
#include <nvbench/nvbench.cuh>
#include <nvvm.h>
#include <regex_ir.hpp>
#include <regex_ir_benchmark_contains.fatbin.inc>
#include <regex_ir_benchmark_count.fatbin.inc>
#include <regex_ir_benchmark_extract.fatbin.inc>
#include <regex_ir_benchmark_replace.fatbin.inc>
#include <regex_ir_benchmark_split.fatbin.inc>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct benchmark_pattern {
  std::string_view name;
  std::string_view expression;
  std::string_view matching_text;
};

// complex cases are adapted from github.com/mariomka/regex-benchmark
constexpr std::array benchmark_patterns{
  benchmark_pattern{"log", R"REGEX(error:[ ]+[0-9]+)REGEX", "error: 12345"},
  benchmark_pattern{"email", R"REGEX([\w\.+-]+@[\w\.-]+\.[\w\.-]+)REGEX", "a.b+c@d-e.co"},
  benchmark_pattern{
    "uri", R"REGEX([\w]+://[^/\s?#]+[^\s?#]+(?:\?[^\s#]*)?(?:#[^\s]*)?)REGEX", "https://a.co/x"},
  benchmark_pattern{
    "ipv4",
    R"REGEX((?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]))REGEX",
    "192.168.100.42"}};

constexpr std::array cudf_contains_patterns{
  benchmark_pattern{"0", R"REGEX(^\d+ [a-z]+)REGEX", "123 abc"},
  benchmark_pattern{"1", R"REGEX([A-Z ]+\d+ +\d+[A-Z]+\d+$)REGEX", "ABC 123  45XYZ6"},
  benchmark_pattern{"2", "^123 abc", "123 abc"},
  benchmark_pattern{"3", "0987 5W43$", "0987 5W43"},
  benchmark_pattern{"4", "0987 5W43", "0987 5W43"},
  benchmark_pattern{"5", R"REGEX(5[A-Z]\d+)REGEX", "5W43"},
  benchmark_pattern{"6", "5W43|X9Z8", "5W43"},
  benchmark_pattern{"7", "7 5W4{1,3}", "7 5W44"},
  benchmark_pattern{"8", "7 (?:5W){1,2}", "7 5W5W"},
  benchmark_pattern{"9", "7 5.4.", "7 5x4y"},
  benchmark_pattern{"10", ".+5W", "abc5W"},
};

constexpr std::array cudf_transform_patterns{
  benchmark_pattern{"0", R"REGEX(\d+)REGEX", "123"},
  benchmark_pattern{"1", " ", " "},
  benchmark_pattern{"2", "[a-z]+[A-Z]+", "abcXYZ"},
  benchmark_pattern{"3", "[a-f]+|[0-5]+", "abcdef"},
  benchmark_pattern{"4", "[a-z][0-9]{0,3}[A-Z]", "a12Z"},
  benchmark_pattern{"5", ".+[0-9]", "abc7"},
  benchmark_pattern{"6", "[a-z]+Z", "abcZ"},
};

std::vector<nvbench::int64_t> api_row_counts()
{
  // doubling gives evenly spaced samples on the logarithmic presentation axis.
  return {1'024,
          2'048,
          4'096,
          8'192,
          16'384,
          32'768,
          65'536,
          131'072,
          262'144,
          524'288,
          1'048'576,
          2'097'152,
          4'194'304,
          8'388'607};
}

void check_cuda(cudaError_t result, std::string_view operation)
{
  if (result != cudaSuccess) {
    throw std::runtime_error(fmt::format("{} failed: {}", operation, cudaGetErrorString(result)));
  }
}

void check_driver(CUresult result, std::string_view operation)
{
  if (result == CUDA_SUCCESS) return;
  char const* detail = nullptr;
  static_cast<void>(cuGetErrorString(result, &detail));
  throw std::runtime_error(
    fmt::format("{} failed: {}", operation, detail == nullptr ? "unknown" : detail));
}

std::size_t next_artifact_id()
{
  static std::size_t id = 0;
  return id++;
}

void dump_artifact(std::size_t id, std::string_view stage, std::span<char const> contents)
{
  char const* directory = std::getenv("REGEX_IR_BENCHMARK_DUMP_DIR");
  if (directory == nullptr || *directory == '\0') return;
  std::filesystem::create_directories(directory);
  auto path = std::filesystem::path{directory} / fmt::format("regex_ir_{}_{}", id, stage);
  std::ofstream output{path, std::ios::binary};
  if (!output) throw std::runtime_error(fmt::format("could not open {}", path.string()));
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!output) throw std::runtime_error(fmt::format("could not write {}", path.string()));
}

std::string nvvm_log(nvvmProgram program)
{
  std::size_t size = 0;
  if (nvvmGetProgramLogSize(program, &size) != NVVM_SUCCESS || size == 0) return {};
  std::string log(size, '\0');
  if (nvvmGetProgramLog(program, log.data()) != NVVM_SUCCESS) return {};
  return log;
}

void check_nvvm(nvvmResult result, nvvmProgram program, std::string_view operation)
{
  if (result != NVVM_SUCCESS) {
    throw std::runtime_error(fmt::format("{} failed: {}", operation, nvvm_log(program)));
  }
}

std::string compile_to_ptx(std::string const& nvvm_ir, std::string const& compute_arch)
{
  nvvmProgram program = nullptr;
  if (nvvmCreateProgram(&program) != NVVM_SUCCESS) {
    throw std::runtime_error("nvvmCreateProgram failed");
  }
  try {
    check_nvvm(nvvmAddModuleToProgram(
                 program, nvvm_ir.data(), nvvm_ir.size(), "generated_benchmark_regex.nvvm"),
               program,
               "nvvmAddModuleToProgram");
    auto arch_option      = fmt::format("-arch={}", compute_arch);
    char const* verify[]  = {arch_option.c_str()};
    char const* compile[] = {arch_option.c_str(), "-opt=3"};
    check_nvvm(nvvmVerifyProgram(program, 1, verify), program, "nvvmVerifyProgram");
    check_nvvm(nvvmCompileProgram(program, 2, compile), program, "nvvmCompileProgram");
    std::size_t size = 0;
    check_nvvm(nvvmGetCompiledResultSize(program, &size), program, "nvvmGetCompiledResultSize");
    std::string ptx(size, '\0');
    check_nvvm(nvvmGetCompiledResult(program, ptx.data()), program, "nvvmGetCompiledResult");
    nvvmDestroyProgram(&program);
    return ptx;
  } catch (...) {
    nvvmDestroyProgram(&program);
    throw;
  }
}

std::string jitlink_log(nvJitLinkHandle handle)
{
  std::size_t size = 0;
  if (nvJitLinkGetErrorLogSize(handle, &size) != NVJITLINK_SUCCESS || size == 0) return {};
  std::string log(size, '\0');
  if (nvJitLinkGetErrorLog(handle, log.data()) != NVJITLINK_SUCCESS) return {};
  return log;
}

void check_jitlink(nvJitLinkResult result, nvJitLinkHandle handle, std::string_view operation)
{
  if (result != NVJITLINK_SUCCESS) {
    throw std::runtime_error(fmt::format("{} failed: {}", operation, jitlink_log(handle)));
  }
}

std::vector<char> link_cubin(std::string const& ptx,
                             std::span<unsigned char const> kernel_fatbin,
                             std::string const& sm_arch)
{
  auto arch_option       = fmt::format("-arch={}", sm_arch);
  char const* options[]  = {arch_option.c_str(), "-lto", "-O3"};
  nvJitLinkHandle linker = nullptr;
  if (nvJitLinkCreate(&linker, 3, options) != NVJITLINK_SUCCESS) {
    throw std::runtime_error("nvJitLinkCreate failed");
  }
  try {
    check_jitlink(nvJitLinkAddData(linker,
                                   NVJITLINK_INPUT_FATBIN,
                                   const_cast<unsigned char*>(kernel_fatbin.data()),
                                   kernel_fatbin.size(),
                                   "benchmark_kernel.fatbin"),
                  linker,
                  "nvJitLinkAddData(kernel fatbin)");
    check_jitlink(
      nvJitLinkAddData(linker, NVJITLINK_INPUT_PTX, ptx.data(), ptx.size(), "generated_regex.ptx"),
      linker,
      "nvJitLinkAddData(PTX)");
    check_jitlink(nvJitLinkComplete(linker), linker, "nvJitLinkComplete");
    std::size_t size = 0;
    check_jitlink(
      nvJitLinkGetLinkedCubinSize(linker, &size), linker, "nvJitLinkGetLinkedCubinSize");
    std::vector<char> result(size);
    check_jitlink(
      nvJitLinkGetLinkedCubin(linker, result.data()), linker, "nvJitLinkGetLinkedCubin");
    nvJitLinkDestroy(&linker);
    return result;
  } catch (...) {
    nvJitLinkDestroy(&linker);
    throw;
  }
}

class loaded_kernel {
 public:
  explicit loaded_kernel(std::vector<char> const& cubin)
  {
    check_driver(cuModuleLoadData(&module_, cubin.data()), "cuModuleLoadData");
    try {
      check_driver(cuModuleGetFunction(&function_, module_, "regex_ir_benchmark_contains"),
                   "cuModuleGetFunction");
    } catch (...) {
      cuModuleUnload(module_);
      throw;
    }
  }

  loaded_kernel(loaded_kernel const&)            = delete;
  loaded_kernel& operator=(loaded_kernel const&) = delete;

  ~loaded_kernel()
  {
    if (module_ != nullptr) static_cast<void>(cuModuleUnload(module_));
  }

  void launch(char const* chars,
              std::int32_t const* offsets,
              std::int32_t rows,
              std::uint8_t* output,
              cudaStream_t stream) const
  {
    void* arguments[]             = {&chars, &offsets, &rows, &output};
    constexpr unsigned block_size = 256;
    auto grid_size =
      static_cast<unsigned>((static_cast<std::uint32_t>(rows) + block_size - 1) / block_size);
    check_driver(cuLaunchKernel(function_,
                                grid_size,
                                1,
                                1,
                                block_size,
                                1,
                                1,
                                0,
                                reinterpret_cast<CUstream>(stream),
                                arguments,
                                nullptr),
                 "cuLaunchKernel");
  }

 private:
  CUmodule module_     = nullptr;
  CUfunction function_ = nullptr;
};

class loaded_extract_kernel {
 public:
  explicit loaded_extract_kernel(std::vector<char> const& cubin)
  {
    check_driver(cuModuleLoadData(&module_, cubin.data()), "cuModuleLoadData");
    try {
      check_driver(cuModuleGetFunction(&function_, module_, "regex_ir_benchmark_extract_kernel"),
                   "cuModuleGetFunction");
    } catch (...) {
      static_cast<void>(cuModuleUnload(module_));
      throw;
    }
  }

  loaded_extract_kernel(loaded_extract_kernel const&)            = delete;
  loaded_extract_kernel& operator=(loaded_extract_kernel const&) = delete;

  ~loaded_extract_kernel()
  {
    if (module_ != nullptr) static_cast<void>(cuModuleUnload(module_));
  }

  void launch(char const* chars,
              std::int32_t const* offsets,
              std::int32_t rows,
              std::uint64_t* captures,
              void* pairs,
              std::uint32_t capture_count,
              cudaStream_t stream) const
  {
    void* arguments[]             = {&chars, &offsets, &rows, &captures, &pairs, &capture_count};
    constexpr unsigned block_size = 256;
    auto grid_size =
      static_cast<unsigned>((static_cast<std::uint32_t>(rows) + block_size - 1) / block_size);
    check_driver(cuLaunchKernel(function_,
                                grid_size,
                                1,
                                1,
                                block_size,
                                1,
                                1,
                                0,
                                reinterpret_cast<CUstream>(stream),
                                arguments,
                                nullptr),
                 "cuLaunchKernel(extract)");
  }

 private:
  CUmodule module_     = nullptr;
  CUfunction function_ = nullptr;
};

class loaded_count_kernel {
 public:
  explicit loaded_count_kernel(std::vector<char> const& cubin)
  {
    check_driver(cuModuleLoadData(&module_, cubin.data()), "cuModuleLoadData");
    try {
      check_driver(cuModuleGetFunction(&function_, module_, "regex_ir_benchmark_count_kernel"),
                   "cuModuleGetFunction");
    } catch (...) {
      static_cast<void>(cuModuleUnload(module_));
      throw;
    }
  }

  loaded_count_kernel(loaded_count_kernel const&)            = delete;
  loaded_count_kernel& operator=(loaded_count_kernel const&) = delete;

  ~loaded_count_kernel()
  {
    if (module_ != nullptr) static_cast<void>(cuModuleUnload(module_));
  }

  void launch(char const* chars,
              std::int32_t const* offsets,
              std::int32_t rows,
              std::int32_t* counts,
              cudaStream_t stream) const
  {
    void* arguments[]             = {&chars, &offsets, &rows, &counts};
    constexpr unsigned block_size = 256;
    auto grid_size =
      static_cast<unsigned>((static_cast<std::uint32_t>(rows) + block_size - 1) / block_size);
    check_driver(cuLaunchKernel(function_,
                                grid_size,
                                1,
                                1,
                                block_size,
                                1,
                                1,
                                0,
                                reinterpret_cast<CUstream>(stream),
                                arguments,
                                nullptr),
                 "cuLaunchKernel(count)");
  }

 private:
  CUmodule module_     = nullptr;
  CUfunction function_ = nullptr;
};

class loaded_materializing_kernel {
 public:
  loaded_materializing_kernel(std::vector<char> const& cubin,
                              char const* size_function,
                              char const* emit_function)
  {
    check_driver(cuModuleLoadData(&module_, cubin.data()), "cuModuleLoadData");
    try {
      check_driver(cuModuleGetFunction(&size_, module_, size_function),
                   "cuModuleGetFunction(size)");
      check_driver(cuModuleGetFunction(&emit_, module_, emit_function),
                   "cuModuleGetFunction(emit)");
    } catch (...) {
      static_cast<void>(cuModuleUnload(module_));
      throw;
    }
  }

  loaded_materializing_kernel(loaded_materializing_kernel const&)            = delete;
  loaded_materializing_kernel& operator=(loaded_materializing_kernel const&) = delete;

  ~loaded_materializing_kernel()
  {
    if (module_ != nullptr) static_cast<void>(cuModuleUnload(module_));
  }

  void launch_size(char const* chars,
                   std::int32_t const* offsets,
                   std::int32_t rows,
                   std::int32_t* sizes,
                   cudaStream_t stream) const
  {
    void* arguments[]             = {&chars, &offsets, &rows, &sizes};
    constexpr unsigned block_size = 256;
    auto work_items               = static_cast<std::uint32_t>(rows) + 1U;
    auto grid_size = static_cast<unsigned>((work_items + block_size - 1U) / block_size);
    check_driver(cuLaunchKernel(size_,
                                grid_size,
                                1,
                                1,
                                block_size,
                                1,
                                1,
                                0,
                                reinterpret_cast<CUstream>(stream),
                                arguments,
                                nullptr),
                 "cuLaunchKernel(transform size)");
  }

  void launch_emit(char const* chars,
                   std::int32_t const* offsets,
                   std::int32_t rows,
                   std::int32_t const* output_offsets,
                   void* output,
                   cudaStream_t stream) const
  {
    void* arguments[]             = {&chars, &offsets, &rows, &output_offsets, &output};
    constexpr unsigned block_size = 256;
    auto grid_size =
      static_cast<unsigned>((static_cast<std::uint32_t>(rows) + block_size - 1) / block_size);
    check_driver(cuLaunchKernel(emit_,
                                grid_size,
                                1,
                                1,
                                block_size,
                                1,
                                1,
                                0,
                                reinterpret_cast<CUstream>(stream),
                                arguments,
                                nullptr),
                 "cuLaunchKernel(transform emit)");
  }

  void launch_split_emit(char const* chars,
                         std::int32_t const* offsets,
                         std::int32_t rows,
                         std::int32_t const* output_offsets,
                         std::uint64_t* spans,
                         void* pairs,
                         cudaStream_t stream) const
  {
    void* arguments[]             = {&chars, &offsets, &rows, &output_offsets, &spans, &pairs};
    constexpr unsigned block_size = 256;
    auto grid_size =
      static_cast<unsigned>((static_cast<std::uint32_t>(rows) + block_size - 1) / block_size);
    check_driver(cuLaunchKernel(emit_,
                                grid_size,
                                1,
                                1,
                                block_size,
                                1,
                                1,
                                0,
                                reinterpret_cast<CUstream>(stream),
                                arguments,
                                nullptr),
                 "cuLaunchKernel(split emit)");
  }

 private:
  CUmodule module_ = nullptr;
  CUfunction size_ = nullptr;
  CUfunction emit_ = nullptr;
};

struct input_data {
  std::vector<std::int32_t> offsets;
  std::vector<char> chars;
  std::vector<std::uint8_t> expected;
};

input_data make_input(std::int32_t rows,
                      std::int32_t width,
                      std::string_view matching_text,
                      std::int32_t hit_rate = 50)
{
  input_data result;
  result.offsets.resize(static_cast<std::size_t>(rows) + 1);
  auto total_bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(width);
  result.chars.resize(total_bytes, ' ');
  result.expected.resize(static_cast<std::size_t>(rows));
  auto match_offset = (static_cast<std::size_t>(width) - matching_text.size()) / 2;
  for (std::int32_t row = 0; row < rows; ++row) {
    auto row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width);
    if (row % 100 < hit_rate) {
      std::copy(matching_text.begin(),
                matching_text.end(),
                result.chars.begin() + static_cast<std::ptrdiff_t>(row_offset + match_offset));
      result.expected[static_cast<std::size_t>(row)] = 1;
    }
    result.offsets[static_cast<std::size_t>(row) + 1] =
      static_cast<std::int32_t>(row_offset + static_cast<std::size_t>(width));
  }
  return result;
}

input_data make_cudf_contains_input(std::int32_t rows, std::int32_t width, std::int32_t hit_rate)
{
  constexpr std::array<std::string_view, 10> source_rows{"123 abc 4567890 DEFGHI 0987 5W43",
                                                         "012345 6789 01234 56789 0123 456",
                                                         "abc 4567890 DEFGHI 0987 Wxyz 123",
                                                         "abcdefghijklmnopqrstuvwxyz 01234",
                                                         "",
                                                         "AbcéDEFGHIJKLMNOPQRSTUVWXYZ 01",
                                                         "9876543210,abcdefghijklmnopqrstU",
                                                         "9876543210,abcdefghijklmnopqrstU",
                                                         "123 édf 4567890 DéFG 0987 X5",
                                                         "1"};
  input_data result;
  result.offsets.resize(static_cast<std::size_t>(rows) + 1U);
  result.expected.resize(static_cast<std::size_t>(rows));
  auto repetitions = std::max(width / 32, 1);
  for (std::int32_t row = 0; row < rows; ++row) {
    auto hit         = row % 100 < hit_rate;
    auto source      = hit ? 0U : 1U + static_cast<unsigned>(row % 9);
    auto source_text = source_rows[source];
    for (std::int32_t repetition = 0; repetition < repetitions; ++repetition) {
      result.chars.insert(result.chars.end(), source_text.begin(), source_text.end());
    }
    result.expected[static_cast<std::size_t>(row)] = hit ? 1U : 0U;
    result.offsets[static_cast<std::size_t>(row) + 1U] =
      static_cast<std::int32_t>(result.chars.size());
  }
  return result;
}

input_data make_random_string_input(std::int32_t rows, std::int32_t max_width)
{
  constexpr std::string_view alphabet =
    " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  std::mt19937 generator{1U};
  std::normal_distribution<double> width_distribution{static_cast<double>(max_width) / 2.0,
                                                      static_cast<double>(max_width) / 6.0};
  std::uniform_int_distribution<std::size_t> character_distribution{0U, alphabet.size() - 1U};
  input_data result;
  result.offsets.resize(static_cast<std::size_t>(rows) + 1U);
  result.expected.resize(static_cast<std::size_t>(rows));
  result.chars.reserve(static_cast<std::size_t>(rows) * static_cast<std::size_t>(max_width / 2));
  for (std::int32_t row = 0; row < rows; ++row) {
    auto sampled_width = static_cast<std::int32_t>(std::llround(width_distribution(generator)));
    auto width         = std::clamp(sampled_width, 0, max_width);
    for (std::int32_t index = 0; index < width; ++index) {
      result.chars.push_back(alphabet[character_distribution(generator)]);
    }
    result.offsets[static_cast<std::size_t>(row) + 1U] =
      static_cast<std::int32_t>(result.chars.size());
  }
  return result;
}

input_data make_extract_input(std::int32_t rows, std::int32_t row_width)
{
  std::default_random_engine generator;
  std::uniform_int_distribution<int> word_distribution{0, 999};
  std::array<std::string, 100> samples;
  for (auto& sample : samples) {
    while (static_cast<std::int32_t>(sample.size()) < row_width) {
      sample += std::to_string(word_distribution(generator)) + " ";
    }
    sample.resize(static_cast<std::size_t>(row_width));
  }
  input_data result;
  result.offsets.resize(static_cast<std::size_t>(rows) + 1U);
  result.expected.resize(static_cast<std::size_t>(rows));
  result.chars.reserve(static_cast<std::size_t>(rows) * static_cast<std::size_t>(row_width));
  for (std::int32_t row = 0; row < rows; ++row) {
    auto& sample = samples[static_cast<std::size_t>(row) % samples.size()];
    result.chars.insert(result.chars.end(), sample.begin(), sample.end());
    result.offsets[static_cast<std::size_t>(row) + 1U] =
      static_cast<std::int32_t>(result.chars.size());
  }
  return result;
}

std::unique_ptr<cudf::column> make_strings(input_data const& input, cudaStream_t stream)
{
  auto offsets = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                           static_cast<cudf::size_type>(input.offsets.size()));
  check_cuda(cudaMemcpyAsync(offsets->mutable_view().data<std::int32_t>(),
                             input.offsets.data(),
                             input.offsets.size() * sizeof(std::int32_t),
                             cudaMemcpyHostToDevice,
                             stream),
             "copy offsets");
  rmm::device_buffer chars(input.chars.size(), rmm::cuda_stream_view{stream});
  check_cuda(
    cudaMemcpyAsync(
      chars.data(), input.chars.data(), input.chars.size(), cudaMemcpyHostToDevice, stream),
    "copy chars");
  return cudf::make_strings_column(static_cast<cudf::size_type>(input.expected.size()),
                                   std::move(offsets),
                                   std::move(chars),
                                   0,
                                   rmm::device_buffer{});
}

std::int32_t get_axis(nvbench::state& state, std::string const& name)
{
  auto value = state.get_int64(name);
  if (value <= 0 || value > std::numeric_limits<std::int32_t>::max()) {
    throw std::invalid_argument(name + " must fit in a positive 32-bit integer");
  }
  return static_cast<std::int32_t>(value);
}

benchmark_pattern const& get_pattern(nvbench::state& state)
{
  auto name  = state.get_string("Pattern");
  auto match = std::find_if(benchmark_patterns.begin(), benchmark_patterns.end(), [&](auto& item) {
    return item.name == name;
  });
  if (match == benchmark_patterns.end()) {
    throw std::invalid_argument(fmt::format("unknown benchmark pattern: {}", name));
  }
  return *match;
}

template <std::size_t Size>
benchmark_pattern const& get_indexed_pattern(nvbench::state& state,
                                             std::array<benchmark_pattern, Size> const& patterns)
{
  std::int64_t index = state.get_int64("Pattern");
  if (index < 0 || static_cast<std::size_t>(index) >= patterns.size()) {
    throw std::invalid_argument("Pattern index is outside the registered range");
  }
  return patterns[static_cast<std::size_t>(index)];
}

bool skip_unsupported_workload(nvbench::state& state, benchmark_pattern const& pattern)
{
  auto rows        = get_axis(state, "Rows");
  auto string_size = get_axis(state, "StringBytes");
  if (static_cast<std::size_t>(string_size) < pattern.matching_text.size()) {
    state.skip("StringBytes is too small for the matching benchmark value");
    return true;
  }
  auto total_bytes = static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(string_size);
  if (total_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
    state.skip("Rows * StringBytes exceeds cuDF's 32-bit strings offset limit");
    return true;
  }
  return false;
}

class benchmark_workload {
 public:
  benchmark_workload(nvbench::state& state,
                     benchmark_pattern const& pattern,
                     std::int32_t hit_rate = 50)
    : benchmark_workload(
        state,
        make_input(
          get_axis(state, "Rows"), get_axis(state, "StringBytes"), pattern.matching_text, hit_rate))
  {
  }

  benchmark_workload(nvbench::state& state, input_data input)
    : rows_(get_axis(state, "Rows")),
      string_bytes_(get_axis(state, "StringBytes")),
      stream_(state.get_cuda_stream().get_stream()),
      input_(std::move(input)),
      strings_(make_strings(input_, stream_)),
      strings_view_(strings_->view())
  {
    check_cuda(cudaStreamSynchronize(stream_), "input synchronization");
  }

  [[nodiscard]] std::int32_t rows() const { return rows_; }
  [[nodiscard]] std::size_t bytes() const { return input_.chars.size(); }
  [[nodiscard]] cudaStream_t stream() const { return stream_; }
  [[nodiscard]] cudf::strings_column_view strings_view() const { return strings_view_; }

  [[nodiscard]] char const* chars() const
  {
    return strings_view_.chars_begin(rmm::cuda_stream_view{stream_});
  }

  [[nodiscard]] std::int32_t const* offsets() const
  {
    return strings_view_.offsets().data<std::int32_t>();
  }

  [[nodiscard]] std::vector<std::uint8_t> const& expected() const { return input_.expected; }

 private:
  std::int32_t rows_         = 0;
  std::int32_t string_bytes_ = 0;
  cudaStream_t stream_       = nullptr;
  input_data input_;
  std::unique_ptr<cudf::column> strings_;
  cudf::strings_column_view strings_view_;
};

using cudf_string_pair = cuda::std::pair<char const*, cudf::size_type>;
static_assert(sizeof(cudf_string_pair) == 16U);
static_assert(alignof(cudf_string_pair) == alignof(void*));

std::unique_ptr<cudf::column> make_regex_ir_boolean_column(benchmark_workload& workload,
                                                           loaded_kernel const& kernel,
                                                           cudaStream_t stream)
{
  auto output = cudf::make_numeric_column(cudf::data_type{cudf::type_id::BOOL8},
                                          workload.rows(),
                                          cudf::mask_state::UNALLOCATED,
                                          rmm::cuda_stream_view{stream});
  kernel.launch(workload.chars(),
                workload.offsets(),
                workload.rows(),
                reinterpret_cast<std::uint8_t*>(output->mutable_view().data<bool>()),
                stream);
  return output;
}

std::unique_ptr<cudf::column> make_regex_ir_count_column(benchmark_workload& workload,
                                                         loaded_count_kernel const& kernel,
                                                         cudaStream_t stream)
{
  auto output = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          workload.rows(),
                                          cudf::mask_state::UNALLOCATED,
                                          rmm::cuda_stream_view{stream});
  kernel.launch(workload.chars(),
                workload.offsets(),
                workload.rows(),
                output->mutable_view().data<std::int32_t>(),
                stream);
  return output;
}

std::unique_ptr<cudf::table> make_regex_ir_extract_table(benchmark_workload& workload,
                                                         loaded_extract_kernel const& kernel,
                                                         std::uint32_t capture_count,
                                                         cudaStream_t stream)
{
  auto rows = static_cast<std::size_t>(workload.rows());
  rmm::device_buffer captures(
    rows * static_cast<std::size_t>((capture_count + 1U) * 2U) * sizeof(std::uint64_t),
    rmm::cuda_stream_view{stream});
  rmm::device_buffer pairs(
    rows * static_cast<std::size_t>(capture_count) * sizeof(cudf_string_pair),
    rmm::cuda_stream_view{stream});
  kernel.launch(workload.chars(),
                workload.offsets(),
                workload.rows(),
                static_cast<std::uint64_t*>(captures.data()),
                pairs.data(),
                capture_count,
                stream);
  auto pair_data = static_cast<cudf_string_pair const*>(pairs.data());
  std::vector<cudf::device_span<cudf_string_pair const>> columns;
  columns.reserve(capture_count);
  for (std::uint32_t capture = 0; capture < capture_count; ++capture) {
    columns.emplace_back(pair_data + static_cast<std::size_t>(capture) * rows, rows);
  }
  auto output = cudf::make_strings_column_batch(columns, rmm::cuda_stream_view{stream});
  return std::make_unique<cudf::table>(std::move(output));
}

void scan_transform_sizes(std::int32_t* offsets, std::int32_t rows, cudaStream_t stream)
{
  thrust::exclusive_scan(
    thrust::cuda::par.on(stream), offsets, offsets + static_cast<std::size_t>(rows) + 1U, offsets);
}

std::int32_t copy_transform_size(std::int32_t const* offsets,
                                 std::int32_t rows,
                                 cudaStream_t stream)
{
  std::int32_t result = 0;
  check_cuda(
    cudaMemcpyAsync(&result, offsets + rows, sizeof(result), cudaMemcpyDeviceToHost, stream),
    "copy materialized output size");
  check_cuda(cudaStreamSynchronize(stream), "materialized output size synchronization");
  if (result < 0) throw std::runtime_error("materialized output exceeds cuDF's int32 limit");
  return result;
}

std::unique_ptr<cudf::column> make_regex_ir_replace_column(
  benchmark_workload& workload, loaded_materializing_kernel const& kernel, cudaStream_t stream)
{
  auto offsets     = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                           workload.rows() + 1,
                                           cudf::mask_state::UNALLOCATED,
                                           rmm::cuda_stream_view{stream});
  auto offset_data = offsets->mutable_view().data<std::int32_t>();
  kernel.launch_size(workload.chars(), workload.offsets(), workload.rows(), offset_data, stream);
  scan_transform_sizes(offset_data, workload.rows(), stream);
  auto output_bytes = copy_transform_size(offset_data, workload.rows(), stream);
  rmm::device_buffer chars(static_cast<std::size_t>(output_bytes), rmm::cuda_stream_view{stream});
  kernel.launch_emit(
    workload.chars(), workload.offsets(), workload.rows(), offset_data, chars.data(), stream);
  return cudf::make_strings_column(
    workload.rows(), std::move(offsets), std::move(chars), 0, rmm::device_buffer{});
}

std::unique_ptr<cudf::column> make_regex_ir_split_column(benchmark_workload& workload,
                                                         loaded_materializing_kernel const& kernel,
                                                         cudaStream_t stream)
{
  auto offsets     = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                           workload.rows() + 1,
                                           cudf::mask_state::UNALLOCATED,
                                           rmm::cuda_stream_view{stream});
  auto offset_data = offsets->mutable_view().data<std::int32_t>();
  kernel.launch_size(workload.chars(), workload.offsets(), workload.rows(), offset_data, stream);
  scan_transform_sizes(offset_data, workload.rows(), stream);
  auto field_count = copy_transform_size(offset_data, workload.rows(), stream);
  auto fields      = static_cast<std::size_t>(field_count);
  rmm::device_buffer spans(fields * 2U * sizeof(std::uint64_t), rmm::cuda_stream_view{stream});
  rmm::device_buffer pairs(fields * sizeof(cudf_string_pair), rmm::cuda_stream_view{stream});
  kernel.launch_split_emit(workload.chars(),
                           workload.offsets(),
                           workload.rows(),
                           offset_data,
                           static_cast<std::uint64_t*>(spans.data()),
                           pairs.data(),
                           stream);
  auto pair_data = static_cast<cudf_string_pair const*>(pairs.data());
  auto strings   = cudf::make_strings_column(
    cudf::device_span<cudf_string_pair const>{pair_data, fields}, rmm::cuda_stream_view{stream});
  return cudf::make_lists_column(
    workload.rows(), std::move(offsets), std::move(strings), 0, rmm::device_buffer{});
}

struct target_architecture {
  std::string compute;
  std::string sm;
};

target_architecture get_target_architecture(nvbench::state& state)
{
  if (!state.get_device().has_value()) throw std::runtime_error("GPU benchmark has no device");
  auto sm_number = state.get_device()->get_sm_version() / 10;
  return {fmt::format("compute_{}", sm_number), fmt::format("sm_{}", sm_number)};
}

std::unique_ptr<loaded_kernel> make_regex_ir_kernel(target_architecture const& architecture,
                                                    std::string_view pattern)
{
  check_driver(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 64U * 1024U), "cuCtxSetLimit");
  auto compiled = regex_ir::compile(pattern, regex_ir::operation::contains());
  if (!compiled) throw std::runtime_error("Regex IR compilation failed");
  regex_ir::nvvm_ir_codegen_options options;
  options.symbol_prefix    = "regex_ir_benchmark";
  options.execute_function = "regex_ir_benchmark_execute";
  auto nvvm_ir             = regex_ir::generate_nvvm_ir(*compiled.value, options);
  auto ptx                 = compile_to_ptx(nvvm_ir, architecture.compute);
  auto wrapper             = std::span<unsigned char const>{regex_ir_benchmark_contains_fatbin,
                                                            regex_ir_benchmark_contains_fatbinLength};
  auto cubin               = link_cubin(ptx, wrapper, architecture.sm);
  auto artifact_id         = next_artifact_id();
  dump_artifact(
    artifact_id, "contains.nvvm.ll", std::span<char const>{nvvm_ir.data(), nvvm_ir.size()});
  dump_artifact(artifact_id, "contains.ptx", std::span<char const>{ptx.data(), ptx.size()});
  dump_artifact(artifact_id, "contains.cubin", std::span<char const>{cubin.data(), cubin.size()});
  return std::make_unique<loaded_kernel>(cubin);
}

std::unique_ptr<loaded_extract_kernel> make_regex_ir_extract_kernel(
  target_architecture const& architecture, std::string_view pattern)
{
  check_driver(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 64U * 1024U), "cuCtxSetLimit");
  auto compiled = regex_ir::compile(pattern, regex_ir::operation::extract());
  if (!compiled) throw std::runtime_error("Regex IR compilation failed");
  if (compiled.value->capture_count > 4U) {
    throw std::runtime_error("GPU benchmark kernel supports at most four capture groups");
  }
  regex_ir::nvvm_ir_codegen_options options;
  options.symbol_prefix    = "regex_ir_benchmark_extract_ir";
  options.execute_function = "regex_ir_benchmark_extract";
  auto nvvm_ir             = regex_ir::generate_nvvm_ir(*compiled.value, options);
  auto ptx                 = compile_to_ptx(nvvm_ir, architecture.compute);
  auto wrapper             = std::span<unsigned char const>{regex_ir_benchmark_extract_fatbin,
                                                            regex_ir_benchmark_extract_fatbinLength};
  auto cubin               = link_cubin(ptx, wrapper, architecture.sm);
  auto artifact_id         = next_artifact_id();
  dump_artifact(
    artifact_id, "extract.nvvm.ll", std::span<char const>{nvvm_ir.data(), nvvm_ir.size()});
  dump_artifact(artifact_id, "extract.ptx", std::span<char const>{ptx.data(), ptx.size()});
  dump_artifact(artifact_id, "extract.cubin", std::span<char const>{cubin.data(), cubin.size()});
  return std::make_unique<loaded_extract_kernel>(cubin);
}

std::vector<char> compile_operation_cubin(target_architecture const& architecture,
                                          std::string_view pattern,
                                          regex_ir::operation selected,
                                          std::string_view symbol_prefix,
                                          std::string_view execute_function,
                                          std::span<unsigned char const> wrapper,
                                          std::string_view artifact_name)
{
  check_driver(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 64U * 1024U), "cuCtxSetLimit");
  auto compiled = regex_ir::compile(pattern, selected);
  if (!compiled) throw std::runtime_error("Regex IR compilation failed");
  regex_ir::nvvm_ir_codegen_options options;
  options.symbol_prefix    = symbol_prefix;
  options.execute_function = execute_function;
  auto nvvm_ir             = regex_ir::generate_nvvm_ir(*compiled.value, options);
  auto ptx                 = compile_to_ptx(nvvm_ir, architecture.compute);
  auto cubin               = link_cubin(ptx, wrapper, architecture.sm);
  auto artifact_id         = next_artifact_id();
  dump_artifact(artifact_id,
                fmt::format("{}.nvvm.ll", artifact_name),
                std::span<char const>{nvvm_ir.data(), nvvm_ir.size()});
  dump_artifact(artifact_id,
                fmt::format("{}.ptx", artifact_name),
                std::span<char const>{ptx.data(), ptx.size()});
  dump_artifact(artifact_id,
                fmt::format("{}.cubin", artifact_name),
                std::span<char const>{cubin.data(), cubin.size()});
  return cubin;
}

std::unique_ptr<loaded_count_kernel> make_regex_ir_count_kernel(
  target_architecture const& architecture, std::string_view pattern)
{
  auto wrapper = std::span<unsigned char const>{regex_ir_benchmark_count_fatbin,
                                                regex_ir_benchmark_count_fatbinLength};
  auto cubin   = compile_operation_cubin(architecture,
                                       pattern,
                                       regex_ir::operation::count(),
                                       "regex_ir_benchmark_count_ir",
                                       "regex_ir_benchmark_count",
                                       wrapper,
                                       "count");
  return std::make_unique<loaded_count_kernel>(cubin);
}

std::unique_ptr<loaded_materializing_kernel> make_regex_ir_replace_kernel(
  target_architecture const& architecture, std::string_view pattern, std::string replacement)
{
  auto wrapper = std::span<unsigned char const>{regex_ir_benchmark_replace_fatbin,
                                                regex_ir_benchmark_replace_fatbinLength};
  auto cubin   = compile_operation_cubin(architecture,
                                       pattern,
                                       regex_ir::operation::replace(std::move(replacement)),
                                       "regex_ir_benchmark_replace_ir",
                                       "regex_ir_benchmark_replace",
                                       wrapper,
                                       "replace");
  return std::make_unique<loaded_materializing_kernel>(
    cubin, "regex_ir_benchmark_replace_size", "regex_ir_benchmark_replace_emit");
}

std::unique_ptr<loaded_materializing_kernel> make_regex_ir_split_kernel(
  target_architecture const& architecture, std::string_view pattern)
{
  auto wrapper = std::span<unsigned char const>{regex_ir_benchmark_split_fatbin,
                                                regex_ir_benchmark_split_fatbinLength};
  auto cubin   = compile_operation_cubin(architecture,
                                       pattern,
                                       regex_ir::operation::split(),
                                       "regex_ir_benchmark_split_ir",
                                       "regex_ir_benchmark_split",
                                       wrapper,
                                       "split");
  return std::make_unique<loaded_materializing_kernel>(
    cubin, "regex_ir_benchmark_split_size", "regex_ir_benchmark_split_emit");
}

void add_throughput_counters(nvbench::state& state, benchmark_workload const& workload)
{
  state.add_element_count(static_cast<std::size_t>(workload.rows()), "Rows");
  state.add_global_memory_reads<std::uint8_t>(workload.bytes(), "InputBytes");
  state.add_global_memory_writes<std::uint8_t>(static_cast<std::size_t>(workload.rows()),
                                               "OutputBytes");
}

void validate_boolean_output(benchmark_workload& workload, cudf::column const& output)
{
  std::vector<std::uint8_t> actual(workload.expected().size());
  check_cuda(cudaMemcpyAsync(actual.data(),
                             output.view().data<bool>(),
                             actual.size(),
                             cudaMemcpyDeviceToHost,
                             workload.stream()),
             "copy boolean result");
  check_cuda(cudaStreamSynchronize(workload.stream()), "boolean validation synchronization");
  if (actual != workload.expected()) throw std::runtime_error("boolean result validation failed");
}

void validate_column(cudf::column_view actual,
                     cudf::column_view expected,
                     cudaStream_t stream,
                     std::string_view context)
{
  if (actual.type() != expected.type() || actual.size() != expected.size() ||
      actual.null_count() != expected.null_count() ||
      actual.num_children() != expected.num_children()) {
    throw std::runtime_error(fmt::format("{} column shape differs from cuDF", context));
  }
  if (cudf::is_fixed_width(actual.type()) && actual.size() != 0) {
    auto width       = cudf::size_of(actual.type());
    auto bytes       = static_cast<std::size_t>(actual.size()) * width;
    auto actual_data = static_cast<char const*>(actual.head<void>()) +
                       static_cast<std::size_t>(actual.offset()) * width;
    auto expected_data = static_cast<char const*>(expected.head<void>()) +
                         static_cast<std::size_t>(expected.offset()) * width;
    std::vector<char> actual_host(bytes);
    std::vector<char> expected_host(bytes);
    check_cuda(
      cudaMemcpyAsync(actual_host.data(), actual_data, bytes, cudaMemcpyDeviceToHost, stream),
      "copy Regex IR column validation data");
    check_cuda(
      cudaMemcpyAsync(expected_host.data(), expected_data, bytes, cudaMemcpyDeviceToHost, stream),
      "copy cuDF column validation data");
    check_cuda(cudaStreamSynchronize(stream), "column validation synchronization");
    if (actual_host != expected_host) {
      throw std::runtime_error(fmt::format("{} column data differs from cuDF", context));
    }
  }
  for (cudf::size_type child = 0; child < actual.num_children(); ++child) {
    validate_column(actual.child(child), expected.child(child), stream, context);
  }
}

void validate_table(cudf::table_view actual,
                    cudf::table_view expected,
                    cudaStream_t stream,
                    std::string_view context)
{
  if (actual.num_columns() != expected.num_columns() || actual.num_rows() != expected.num_rows()) {
    throw std::runtime_error(fmt::format("{} table shape differs from cuDF", context));
  }
  for (cudf::size_type column = 0; column < actual.num_columns(); ++column) {
    validate_column(actual.column(column), expected.column(column), stream, context);
  }
}

void regex_ir_warm(nvbench::state& state)
{
  auto& pattern = get_pattern(state);
  if (skip_unsupported_workload(state, pattern)) return;
  check_driver(cuInit(0), "cuInit");
  benchmark_workload workload(state, pattern);
  add_throughput_counters(state, workload);
  auto architecture = get_target_architecture(state);
  auto kernel       = make_regex_ir_kernel(architecture, pattern.expression);
  auto output       = make_regex_ir_boolean_column(workload, *kernel, workload.stream());
  validate_boolean_output(workload, *output);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    output = make_regex_ir_boolean_column(workload, *kernel, launch.get_stream().get_stream());
  });
}

void cudf_warm(nvbench::state& state)
{
  auto& pattern = get_pattern(state);
  if (skip_unsupported_workload(state, pattern)) return;
  benchmark_workload workload(state, pattern);
  add_throughput_counters(state, workload);
  auto program = cudf::strings::regex_program::create(std::string(pattern.expression));
  auto output  = cudf::strings::contains_re(
    workload.strings_view(), *program, rmm::cuda_stream_view{workload.stream()});
  validate_boolean_output(workload, *output);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    output = cudf::strings::contains_re(
      workload.strings_view(), *program, rmm::cuda_stream_view{launch.get_stream().get_stream()});
  });
}

void regex_ir_cold(nvbench::state& state)
{
  auto& pattern = get_pattern(state);
  if (skip_unsupported_workload(state, pattern)) return;
  check_driver(cuInit(0), "cuInit");
  benchmark_workload workload(state, pattern);
  add_throughput_counters(state, workload);
  auto architecture = get_target_architecture(state);

  auto validation_kernel = make_regex_ir_kernel(architecture, pattern.expression);
  auto validation_output =
    make_regex_ir_boolean_column(workload, *validation_kernel, workload.stream());
  validate_boolean_output(workload, *validation_output);
  validation_kernel.reset();

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               timer.start();
               auto kernel = make_regex_ir_kernel(architecture, pattern.expression);
               auto output =
                 make_regex_ir_boolean_column(workload, *kernel, launch.get_stream().get_stream());
               timer.stop();
             });
}

void cudf_cold(nvbench::state& state)
{
  auto& pattern = get_pattern(state);
  if (skip_unsupported_workload(state, pattern)) return;
  benchmark_workload workload(state, pattern);
  add_throughput_counters(state, workload);

  auto validation_program = cudf::strings::regex_program::create(std::string(pattern.expression));
  auto validation_output  = cudf::strings::contains_re(
    workload.strings_view(), *validation_program, rmm::cuda_stream_view{workload.stream()});
  validate_boolean_output(workload, *validation_output);

  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      timer.start();
      auto program = cudf::strings::regex_program::create(std::string(pattern.expression));
      auto output  = cudf::strings::contains_re(
        workload.strings_view(), *program, rmm::cuda_stream_view{launch.get_stream().get_stream()});
      timer.stop();
    });
}

enum class cudf_benchmark_operation : std::uint8_t {
  COUNT   = 0,
  EXTRACT = 1,
  REPLACE = 2,
  SPLIT   = 3,
};

void regex_ir_cudf_contains(nvbench::state& state)
{
  benchmark_pattern pattern = get_indexed_pattern(state, cudf_contains_patterns);
  if (skip_unsupported_workload(state, pattern)) return;
  std::int32_t hit_rate = get_axis(state, "HitRate");
  check_driver(cuInit(0), "cuInit");
  benchmark_workload workload(
    state,
    make_cudf_contains_input(get_axis(state, "Rows"), get_axis(state, "StringBytes"), hit_rate));
  add_throughput_counters(state, workload);
  target_architecture architecture = get_target_architecture(state);
  auto kernel                      = make_regex_ir_kernel(architecture, pattern.expression);
  auto output = make_regex_ir_boolean_column(workload, *kernel, workload.stream());
  validate_boolean_output(workload, *output);
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    output = make_regex_ir_boolean_column(workload, *kernel, launch.get_stream().get_stream());
  });
}

void cudf_cudf_contains(nvbench::state& state)
{
  benchmark_pattern pattern = get_indexed_pattern(state, cudf_contains_patterns);
  if (skip_unsupported_workload(state, pattern)) return;
  std::int32_t hit_rate = get_axis(state, "HitRate");
  benchmark_workload workload(
    state,
    make_cudf_contains_input(get_axis(state, "Rows"), get_axis(state, "StringBytes"), hit_rate));
  add_throughput_counters(state, workload);
  auto program = cudf::strings::regex_program::create(std::string{pattern.expression});
  auto output  = cudf::strings::contains_re(
    workload.strings_view(), *program, rmm::cuda_stream_view{workload.stream()});
  validate_boolean_output(workload, *output);
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    output = cudf::strings::contains_re(
      workload.strings_view(), *program, rmm::cuda_stream_view{launch.get_stream().get_stream()});
  });
}

std::pair<std::string, std::string> operation_pattern(nvbench::state& state,
                                                      cudf_benchmark_operation operation)
{
  if (operation != cudf_benchmark_operation::EXTRACT) {
    benchmark_pattern pattern = get_indexed_pattern(state, cudf_transform_patterns);
    std::string expression{pattern.expression};
    if (operation == cudf_benchmark_operation::REPLACE && state.get_string("Type") == "backref") {
      expression = "(" + expression + ")";
    }
    return {std::move(expression), std::string{pattern.matching_text}};
  }
  std::int32_t groups = get_axis(state, "Groups");
  std::string expression;
  std::string matching;
  for (std::int32_t group = 0; group < groups; ++group) {
    expression += R"REGEX((\d+) )REGEX";
    matching += std::to_string(100 + group) + " ";
  }
  return {std::move(expression), std::move(matching)};
}

void regex_ir_operation(nvbench::state& state, cudf_benchmark_operation operation)
{
  auto [expression, matching] = operation_pattern(state, operation);
  benchmark_pattern pattern{"operation", expression, matching};
  if (skip_unsupported_workload(state, pattern)) return;
  check_driver(cuInit(0), "cuInit");
  auto input =
    operation == cudf_benchmark_operation::EXTRACT
      ? make_extract_input(get_axis(state, "Rows"), get_axis(state, "StringBytes"))
      : make_random_string_input(get_axis(state, "Rows"), get_axis(state, "StringBytes"));
  benchmark_workload workload(state, std::move(input));
  state.add_element_count(static_cast<std::size_t>(workload.rows()), "Rows");
  state.add_global_memory_reads<std::uint8_t>(workload.bytes(), "InputBytes");
  target_architecture architecture = get_target_architecture(state);
  if (operation == cudf_benchmark_operation::COUNT) {
    auto kernel             = make_regex_ir_count_kernel(architecture, expression);
    auto output             = make_regex_ir_count_column(workload, *kernel, workload.stream());
    auto validation_program = cudf::strings::regex_program::create(expression);
    auto expected           = cudf::strings::count_re(workload.strings_view(), *validation_program);
    validate_column(output->view(), expected->view(), workload.stream(), "count");
    check_cuda(cudaStreamSynchronize(workload.stream()), "count warm-up synchronization");
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      output = make_regex_ir_count_column(workload, *kernel, launch.get_stream().get_stream());
    });
    return;
  }

  if (operation == cudf_benchmark_operation::EXTRACT) {
    auto kernel   = make_regex_ir_extract_kernel(architecture, expression);
    auto captures = static_cast<std::uint32_t>(get_axis(state, "Groups"));
    auto output   = make_regex_ir_extract_table(workload, *kernel, captures, workload.stream());
    auto validation_program = cudf::strings::regex_program::create(expression);
    auto expected           = cudf::strings::extract(workload.strings_view(), *validation_program);
    validate_table(output->view(), expected->view(), workload.stream(), "extract");
    check_cuda(cudaStreamSynchronize(workload.stream()), "extract warm-up synchronization");
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      output =
        make_regex_ir_extract_table(workload, *kernel, captures, launch.get_stream().get_stream());
    });
    return;
  }

  if (operation == cudf_benchmark_operation::REPLACE) {
    auto replacement =
      state.get_string("Type") == "backref" ? std::string{"#$1X"} : std::string{"77"};
    auto kernel             = make_regex_ir_replace_kernel(architecture, expression, replacement);
    auto output             = make_regex_ir_replace_column(workload, *kernel, workload.stream());
    auto validation_program = cudf::strings::regex_program::create(expression);
    std::unique_ptr<cudf::column> expected;
    if (state.get_string("Type") == "backref") {
      expected = cudf::strings::replace_with_backrefs(
        workload.strings_view(), *validation_program, std::string{"#\\1X"});
    } else {
      cudf::string_scalar replacement_scalar{"77"};
      expected =
        cudf::strings::replace_re(workload.strings_view(), *validation_program, replacement_scalar);
    }
    validate_column(output->view(), expected->view(), workload.stream(), "replace");
    check_cuda(cudaStreamSynchronize(workload.stream()), "replace warm-up synchronization");
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      output = make_regex_ir_replace_column(workload, *kernel, launch.get_stream().get_stream());
    });
    return;
  }

  auto kernel             = make_regex_ir_split_kernel(architecture, expression);
  auto output             = make_regex_ir_split_column(workload, *kernel, workload.stream());
  auto validation_program = cudf::strings::regex_program::create(expression);
  auto expected = cudf::strings::split_record_re(workload.strings_view(), *validation_program);
  validate_column(output->view(), expected->view(), workload.stream(), "split");
  check_cuda(cudaStreamSynchronize(workload.stream()), "split warm-up synchronization");
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    output = make_regex_ir_split_column(workload, *kernel, launch.get_stream().get_stream());
  });
}

void cudf_operation(nvbench::state& state, cudf_benchmark_operation operation)
{
  auto [expression, matching] = operation_pattern(state, operation);
  benchmark_pattern pattern{"operation", expression, matching};
  if (skip_unsupported_workload(state, pattern)) return;
  auto input =
    operation == cudf_benchmark_operation::EXTRACT
      ? make_extract_input(get_axis(state, "Rows"), get_axis(state, "StringBytes"))
      : make_random_string_input(get_axis(state, "Rows"), get_axis(state, "StringBytes"));
  benchmark_workload workload(state, std::move(input));
  state.add_element_count(static_cast<std::size_t>(workload.rows()), "Rows");
  state.add_global_memory_reads<std::uint8_t>(workload.bytes(), "InputBytes");
  auto program = cudf::strings::regex_program::create(expression);
  cudf::string_scalar replacement{"77"};
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    if (operation == cudf_benchmark_operation::COUNT) {
      auto output = cudf::strings::count_re(workload.strings_view(), *program);
    } else if (operation == cudf_benchmark_operation::EXTRACT) {
      auto output = cudf::strings::extract(workload.strings_view(), *program);
    } else if (operation == cudf_benchmark_operation::REPLACE) {
      if (state.get_string("Type") == "backref") {
        auto output = cudf::strings::replace_with_backrefs(
          workload.strings_view(), *program, std::string{"#\\1X"});
      } else {
        auto output = cudf::strings::replace_re(workload.strings_view(), *program, replacement);
      }
    } else {
      auto output = cudf::strings::split_record_re(workload.strings_view(), *program);
    }
  });
}

void regex_ir_count(nvbench::state& state)
{
  regex_ir_operation(state, cudf_benchmark_operation::COUNT);
}

void cudf_count(nvbench::state& state) { cudf_operation(state, cudf_benchmark_operation::COUNT); }

void regex_ir_extract(nvbench::state& state)
{
  regex_ir_operation(state, cudf_benchmark_operation::EXTRACT);
}

void cudf_extract(nvbench::state& state)
{
  cudf_operation(state, cudf_benchmark_operation::EXTRACT);
}

void regex_ir_replace(nvbench::state& state)
{
  regex_ir_operation(state, cudf_benchmark_operation::REPLACE);
}

void cudf_replace(nvbench::state& state)
{
  cudf_operation(state, cudf_benchmark_operation::REPLACE);
}

void regex_ir_split(nvbench::state& state)
{
  regex_ir_operation(state, cudf_benchmark_operation::SPLIT);
}

void cudf_split(nvbench::state& state) { cudf_operation(state, cudf_benchmark_operation::SPLIT); }

}  // namespace

NVBENCH_BENCH(regex_ir_warm)
  .set_name("regex_ir/warm")
  .add_string_axis("Pattern", {"log", "email", "uri", "ipv4"})
  .add_int64_axis("Rows", {1'000, 10'000, 100'000, 1'000'000, 10'000'000, 100'000'000})
  .add_int64_axis("StringBytes", {16, 32, 128, 512});

NVBENCH_BENCH(cudf_warm)
  .set_name("cudf/warm")
  .add_string_axis("Pattern", {"log", "email", "uri", "ipv4"})
  .add_int64_axis("Rows", {1'000, 10'000, 100'000, 1'000'000, 10'000'000, 100'000'000})
  .add_int64_axis("StringBytes", {16, 32, 128, 512});

NVBENCH_BENCH(regex_ir_cold)
  .set_name("regex_ir/cold")
  .set_disable_blocking_kernel(true)
  .add_string_axis("Pattern", {"log", "email", "uri", "ipv4"})
  .add_int64_axis("Rows", {1'000, 10'000, 100'000, 1'000'000, 10'000'000, 100'000'000})
  .add_int64_axis("StringBytes", {16, 32, 128, 512});

NVBENCH_BENCH(cudf_cold)
  .set_name("cudf/cold")
  .set_disable_blocking_kernel(true)
  .add_string_axis("Pattern", {"log", "email", "uri", "ipv4"})
  .add_int64_axis("Rows", {1'000, 10'000, 100'000, 1'000'000, 10'000'000, 100'000'000})
  .add_int64_axis("StringBytes", {16, 32, 128, 512});

NVBENCH_BENCH(regex_ir_cudf_contains)
  .set_name("regex_ir/contains")
  .add_int64_axis("StringBytes", {64, 128, 256})
  .add_int64_axis("Rows", api_row_counts())
  .add_int64_axis("HitRate", {50, 100})
  .add_int64_axis("Pattern", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

NVBENCH_BENCH(cudf_cudf_contains)
  .set_name("cudf/contains")
  .add_int64_axis("StringBytes", {64, 128, 256})
  .add_int64_axis("Rows", api_row_counts())
  .add_int64_axis("HitRate", {50, 100})
  .add_int64_axis("Pattern", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

NVBENCH_BENCH(regex_ir_count)
  .set_name("regex_ir/count")
  .add_int64_axis("StringBytes", {64, 128, 256})
  .add_int64_axis("Rows", api_row_counts())
  .add_int64_axis("Pattern", {0, 1, 2, 3, 4, 5, 6});

NVBENCH_BENCH(cudf_count)
  .set_name("cudf/count")
  .add_int64_axis("StringBytes", {64, 128, 256})
  .add_int64_axis("Rows", api_row_counts())
  .add_int64_axis("Pattern", {0, 1, 2, 3, 4, 5, 6});

NVBENCH_BENCH(regex_ir_extract)
  .set_name("regex_ir/extract")
  .add_int64_axis("StringBytes", {32, 64, 128, 256})
  .add_int64_axis("Rows", api_row_counts())
  .add_int64_axis("Groups", {1, 2, 4});

NVBENCH_BENCH(cudf_extract)
  .set_name("cudf/extract")
  .add_int64_axis("StringBytes", {32, 64, 128, 256})
  .add_int64_axis("Rows", api_row_counts())
  .add_int64_axis("Groups", {1, 2, 4});

NVBENCH_BENCH(regex_ir_replace)
  .set_name("regex_ir/replace")
  .add_int64_axis("StringBytes", {64, 128, 256})
  .add_int64_axis("Rows", api_row_counts())
  .add_int64_axis("Pattern", {0, 1, 2, 3, 4, 5, 6})
  .add_string_axis("Type", {"replace", "backref"});

NVBENCH_BENCH(cudf_replace)
  .set_name("cudf/replace")
  .add_int64_axis("StringBytes", {64, 128, 256})
  .add_int64_axis("Rows", api_row_counts())
  .add_int64_axis("Pattern", {0, 1, 2, 3, 4, 5, 6})
  .add_string_axis("Type", {"replace", "backref"});

NVBENCH_BENCH(regex_ir_split)
  .set_name("regex_ir/split")
  .add_int64_axis("StringBytes", {64, 128, 256})
  .add_int64_axis("Rows", api_row_counts())
  .add_int64_axis("Pattern", {0, 1, 2, 3, 4, 5, 6});

NVBENCH_BENCH(cudf_split)
  .set_name("cudf/split")
  .add_int64_axis("StringBytes", {64, 128, 256})
  .add_int64_axis("Rows", api_row_counts())
  .add_int64_axis("Pattern", {0, 1, 2, 3, 4, 5, 6});
