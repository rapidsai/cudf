/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "corpus_benchmark.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <fmt/format.h>
#include <nvJitLink.h>
#include <nvbench/nvbench.cuh>
#include <nvvm.h>
#include <regex_ir.hpp>
#include <regex_ir_benchmark_contains.fatbin.inc>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using regex_ir_benchmark::corpus_case;
using regex_ir_benchmark::corpus_source;

void trace(std::string_view message)
{
  if (std::getenv("REGEX_IR_BENCHMARK_TRACE") != nullptr) {
    std::cerr << "[corpus benchmark] " << message << '\n';
  }
}

// expressions and source-corpus roles follow openresty.org/misc/re/bench/
constexpr std::array openresty_cases{
  corpus_case{.name       = "01_literal_miss",
              .family     = "literal",
              .expression = "d",
              .corpus     = corpus_source::OpenRestyAlphabet},
  corpus_case{.name       = "02_short_alt_miss",
              .family     = "alternation",
              .expression = "d|de",
              .corpus     = corpus_source::OpenRestyRandomAlphabet},
  corpus_case{.name       = "03_suffix_alt_miss",
              .family     = "alternation",
              .expression = "dfa|efa|ufa|zfa",
              .corpus     = corpus_source::OpenRestyAlphabet},
  corpus_case{.name       = "04_suffix_alt_prose",
              .family     = "alternation",
              .expression = "dfa|efa|ufa|zfa",
              .corpus     = corpus_source::Mtent12},
  corpus_case{.name       = "05_wide_class_miss",
              .family     = "character_class",
              .expression = "[d-z]",
              .corpus     = corpus_source::OpenRestyAlphabet},
  corpus_case{.name       = "06_split_class_miss",
              .family     = "character_class",
              .expression = "[d-hx-z]",
              .corpus     = corpus_source::OpenRestyAlphabet},
  corpus_case{.name       = "07_split_class_prose",
              .family     = "character_class",
              .expression = "[d-hx-z]",
              .corpus     = corpus_source::Mtent12},
  corpus_case{
    .name   = "08_large_alt_prose",
    .family = "large_alternation",
    .expression =
      "ddd|fff|eee|ggg|hhh|iii|jjj|kkk|[l-n]mm|ooo|ppp|qqq|rrr|sss|ttt|uuu|vvv|www|[x-z]yy",
    .corpus = corpus_source::Mtent12},
  corpus_case{
    .name   = "09_large_alt_miss",
    .family = "large_alternation",
    .expression =
      "ddd|fff|eee|ggg|hhh|iii|jjj|kkk|[l-n]mm|ooo|ppp|qqq|rrr|sss|ttt|uuu|vvv|www|[x-z]yy",
    .corpus = corpus_source::OpenRestyAlphabet},
  corpus_case{.name       = "10_nested_alt",
              .family     = "nested_alternation",
              .expression = "(?:a|b)aa(?:aa|bb)cc(?:a|b)",
              .corpus     = corpus_source::OpenRestyAlphabet},
  corpus_case{.name       = "11_long_nested_alt_miss",
              .family     = "nested_alternation",
              .expression = "(?:a|b)aa(?:aa|bb)cc(?:a|b)abcabcabd",
              .corpus     = corpus_source::OpenRestyRandomAlphabet},
  corpus_case{.name       = "12_capture_chain_miss",
              .family     = "captures",
              .expression = "(a|b)aa(aa|bb)cc(a|b)abcabcabc",
              .corpus     = corpus_source::OpenRestyAlphabet},
  corpus_case{.name       = "13_capture_chain_random_miss",
              .family     = "captures",
              .expression = "(a|b)aa(aa|bb)cc(a|b)abcabcabc",
              .corpus     = corpus_source::OpenRestyRandomAlphabet},
  corpus_case{.name       = "14_lazy_class_repeat",
              .family     = "repetition",
              .expression = "d[abc]*?d",
              .corpus     = corpus_source::OpenRestyDelimiter},
  corpus_case{.name       = "15_lazy_dot_repeat",
              .family     = "repetition",
              .expression = "d.*?d",
              .corpus     = corpus_source::OpenRestyDelimiter},
  corpus_case{.name       = "16_greedy_dot_repeat",
              .family     = "repetition",
              .expression = "d.*d",
              .corpus     = corpus_source::OpenRestyDelimiter},
  corpus_case{.name       = "17_anchored_literal",
              .family     = "anchor",
              .expression = "^Twain",
              .corpus     = corpus_source::Mtent12,
              .multiline  = true},
  corpus_case{.name       = "18_literal_prose",
              .family     = "literal",
              .expression = "Twain",
              .corpus     = corpus_source::Mtent12},
  corpus_case{.name             = "19_folded_literal",
              .family           = "case_fold",
              .expression       = "Twain",
              .corpus           = corpus_source::Mtent12,
              .case_insensitive = true},
  corpus_case{.name       = "20_class_suffix",
              .family     = "character_class",
              .expression = "[a-z]shing",
              .corpus     = corpus_source::Mtent12},
  corpus_case{.name       = "21_name_alternation",
              .family     = "alternation",
              .expression = "Huck[a-zA-Z]+|Saw[a-zA-Z]+",
              .corpus     = corpus_source::Mtent12},
  corpus_case{.name       = "22_word_boundary",
              .family     = "assertion",
              .expression = R"REGEX(\b\w+nn\b)REGEX",
              .corpus     = corpus_source::Mtent12},
  corpus_case{.name       = "23_negated_bounded",
              .family     = "bounded_repeat",
              .expression = "[a-q][^u-z]{13}x",
              .corpus     = corpus_source::Mtent12},
  corpus_case{.name       = "24_name_literals",
              .family     = "alternation",
              .expression = "Tom|Sawyer|Huckleberry|Finn",
              .corpus     = corpus_source::Mtent12},
  corpus_case{.name             = "25_folded_names",
              .family           = "case_fold",
              .expression       = "Tom|Sawyer|Huckleberry|Finn",
              .corpus           = corpus_source::Mtent12,
              .case_insensitive = true},
  corpus_case{.name       = "26_short_prefix_names",
              .family     = "bounded_repeat",
              .expression = ".{0,2}(Tom|Sawyer|Huckleberry|Finn)",
              .corpus     = corpus_source::Mtent12},
  corpus_case{.name       = "27_required_prefix_names",
              .family     = "bounded_repeat",
              .expression = ".{2,4}(Tom|Sawyer|Huckleberry|Finn)",
              .corpus     = corpus_source::Mtent12},
  corpus_case{.name       = "28_word_suffix",
              .family     = "repetition",
              .expression = "[a-zA-Z]+ing",
              .corpus     = corpus_source::Mtent12},
  corpus_case{.name       = "29_bounded_word_suffix",
              .family     = "bounded_repeat",
              .expression = R"REGEX(\s[a-zA-Z]{0,12}ing\s)REGEX",
              .corpus     = corpus_source::Mtent12},
  corpus_case{.name       = "30_captured_name_suffix",
              .family     = "captures",
              .expression = R"REGEX(([A-Za-z]awyer|[A-Za-z]inn)\s)REGEX",
              .corpus     = corpus_source::Mtent12},
  corpus_case{.name       = "31_quoted_sentence",
              .family     = "bounded_repeat",
              .expression = R"REGEX(["'][^"']{0,30}[?!.]["'])REGEX",
              .corpus     = corpus_source::Mtent12}};

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
    check_nvvm(
      nvvmAddModuleToProgram(program, nvvm_ir.data(), nvvm_ir.size(), "openresty_regex.nvvm"),
      program,
      "nvvmAddModuleToProgram");
    auto architecture     = fmt::format("-arch={}", compute_arch);
    char const* verify[]  = {architecture.c_str()};
    char const* compile[] = {architecture.c_str(), "-opt=3"};
    check_nvvm(nvvmVerifyProgram(program, 1, verify), program, "nvvmVerifyProgram");
    check_nvvm(nvvmCompileProgram(program, 2, compile), program, "nvvmCompileProgram");
    std::size_t result_size = 0;
    check_nvvm(
      nvvmGetCompiledResultSize(program, &result_size), program, "nvvmGetCompiledResultSize");
    std::string ptx(result_size, '\0');
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
  auto architecture      = fmt::format("-arch={}", sm_arch);
  char const* options[]  = {architecture.c_str(), "-lto", "-O3"};
  nvJitLinkHandle linker = nullptr;
  if (nvJitLinkCreate(&linker, 3, options) != NVJITLINK_SUCCESS) {
    throw std::runtime_error("nvJitLinkCreate failed");
  }
  try {
    check_jitlink(nvJitLinkAddData(linker,
                                   NVJITLINK_INPUT_FATBIN,
                                   const_cast<unsigned char*>(kernel_fatbin.data()),
                                   kernel_fatbin.size(),
                                   "openresty_kernel.fatbin"),
                  linker,
                  "nvJitLinkAddData(kernel fatbin)");
    check_jitlink(
      nvJitLinkAddData(linker, NVJITLINK_INPUT_PTX, ptx.data(), ptx.size(), "openresty.ptx"),
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
      static_cast<void>(cuModuleUnload(module_));
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
      static_cast<unsigned>((static_cast<std::uint32_t>(rows) + block_size - 1U) / block_size);
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

std::uint64_t mix(std::uint64_t value)
{
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

std::string read_corpus_file(std::string_view filename, std::size_t expected_size)
{
  auto path = std::filesystem::path{REGEX_IR_BENCHMARK_CORPUS_DIRECTORY} / filename;
  std::ifstream input{path, std::ios::binary};
  if (!input) {
    throw std::runtime_error(fmt::format(
      "missing benchmark corpus {}; build the regex_ir_benchmark_corpora target", path.string()));
  }
  std::string result{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
  if (result.size() != expected_size) {
    throw std::runtime_error(fmt::format("benchmark corpus {} has {} bytes; expected {}",
                                         path.string(),
                                         result.size(),
                                         expected_size));
  }
  return result;
}

std::string make_openresty_alphabet()
{
  constexpr std::size_t repetitions = 5U * 1024U * 1024U;
  std::string result;
  result.reserve(repetitions * 5U + 8U);
  for (std::size_t index = 0; index < repetitions; ++index) {
    result.append("abccc");
  }
  result.append("aaabbccb");
  return result;
}

std::string make_openresty_random_alphabet(bool delimiter)
{
  constexpr std::size_t seed_bytes = 1024U * 1024U;
  constexpr std::array alphabet{'a', 'b', 'c'};
  std::string seed(seed_bytes, '\0');
  auto salt = delimiter ? 0x6e5f64656c696dULL : 0x72616e645f616263ULL;
  for (std::size_t index = 0; index < seed.size(); ++index) {
    auto range   = std::max<std::size_t>(index, 1U);
    auto sampled = static_cast<std::size_t>(mix(index ^ salt) % range);
    seed[index]  = alphabet[sampled % alphabet.size()];
  }

  std::string result;
  result.reserve(seed.size() * 10U + 9U);
  if (delimiter) result.push_back('d');
  for (std::size_t repetition = 0; repetition < 10U; ++repetition) {
    result.append(seed);
  }
  result.append("aaabbccb");
  return result;
}

std::string_view corpus_text(corpus_case const& benchmark)
{
  static std::string const alphabet        = make_openresty_alphabet();
  static std::string const random          = make_openresty_random_alphabet(false);
  static std::string const delim           = make_openresty_random_alphabet(true);
  static std::string const mtent12         = read_corpus_file("openresty-mtent12.txt", 20'045'118U);
  static std::string const leipzig         = read_corpus_file("leipzig-3200.txt", 16'013'977U);
  static std::string const boost_crc       = read_corpus_file("boost-1.41-crc.hpp", 34'483U);
  static std::string const boost_libraries = read_corpus_file("boost-1.41-libraries.htm", 51'799U);
  static std::string const mariomka = read_corpus_file("mariomka-input-text.txt", 6'839'410U);

  switch (benchmark.corpus) {
    case corpus_source::OpenRestyAlphabet: return alphabet;
    case corpus_source::OpenRestyRandomAlphabet: return random;
    case corpus_source::OpenRestyDelimiter: return delim;
    case corpus_source::Mtent12: return mtent12;
    case corpus_source::Mtent12Prefix50K: return std::string_view{mtent12}.substr(0, 50'000U);
    case corpus_source::Leipzig3200: return leipzig;
    case corpus_source::BoostCrc: return boost_crc;
    case corpus_source::BoostLibraries: return boost_libraries;
    case corpus_source::MariomkaInput: return mariomka;
    case corpus_source::Inline: return benchmark.inline_text;
  }
  return "";
}

struct input_data {
  std::vector<std::int32_t> offsets = std::vector<std::int32_t>{};
  std::vector<char> chars           = std::vector<char>{};
};

bool is_utf8_continuation(char value)
{
  return (static_cast<unsigned char>(value) & 0xc0U) == 0x80U;
}

std::size_t choose_boundary(std::string_view source,
                            std::size_t begin,
                            std::size_t minimum,
                            std::size_t maximum,
                            std::size_t desired)
{
  auto lower                          = begin + minimum;
  auto upper                          = begin + maximum;
  auto target                         = begin + std::clamp(desired, minimum, maximum);
  constexpr std::size_t search_radius = 32U;
  for (std::size_t distance = 0; distance <= search_radius; ++distance) {
    if (target + distance < upper && source[target + distance] == '\n') {
      return target + distance + 1U;
    }
    if (target >= lower + distance && target - distance < source.size() &&
        source[target - distance] == '\n') {
      return target - distance + 1U;
    }
  }

  auto boundary = target;
  while (boundary < upper && boundary < source.size() && is_utf8_continuation(source[boundary])) {
    ++boundary;
  }
  if (boundary <= upper) return boundary;
  boundary = target;
  while (boundary < source.size() && boundary > lower && is_utf8_continuation(source[boundary])) {
    --boundary;
  }
  return boundary;
}

std::vector<input_data> make_table_input(corpus_case const& benchmark,
                                         std::int32_t rows,
                                         std::int32_t maximum_bytes,
                                         std::int32_t column_count)
{
  auto source  = corpus_text(benchmark);
  auto maximum = static_cast<std::size_t>(maximum_bytes);
  auto slots   = static_cast<std::size_t>(rows) * static_cast<std::size_t>(column_count);
  std::vector<input_data> result(static_cast<std::size_t>(column_count));
  for (auto& column : result) {
    column.offsets.resize(static_cast<std::size_t>(rows) + 1U);
  }

  if (benchmark.corpus == corpus_source::Inline) {
    for (auto& column : result) {
      column.chars.reserve(static_cast<std::size_t>(rows) * source.size());
      for (std::int32_t row = 0; row < rows; ++row) {
        column.chars.insert(column.chars.end(), source.begin(), source.end());
        column.offsets[static_cast<std::size_t>(row) + 1U] =
          static_cast<std::int32_t>(column.chars.size());
      }
    }
    return result;
  }

  std::size_t source_position = 0;
  std::size_t slot            = 0;
  for (std::int32_t column_index = 0; column_index < column_count; ++column_index) {
    auto& column = result[static_cast<std::size_t>(column_index)];
    column.chars.reserve(source.size() / static_cast<std::size_t>(column_count) + maximum);
    for (std::int32_t row = 0; row < rows; ++row, ++slot) {
      auto remaining       = source.size() - source_position;
      auto remaining_slots = slots - slot;
      auto future_capacity = (remaining_slots - 1U) * maximum;
      auto minimum         = remaining > future_capacity ? remaining - future_capacity : 0U;
      auto maximum_length  = std::min(maximum, remaining);
      auto average         = (remaining + remaining_slots - 1U) / remaining_slots;
      auto variation       = 50U + static_cast<std::size_t>(mix(slot) % 101U);
      auto desired         = std::min(maximum_length, average * variation / 100U);
      auto end = choose_boundary(source, source_position, minimum, maximum_length, desired);
      column.chars.insert(column.chars.end(),
                          source.begin() + static_cast<std::ptrdiff_t>(source_position),
                          source.begin() + static_cast<std::ptrdiff_t>(end));
      source_position = end;
      column.offsets[static_cast<std::size_t>(row) + 1U] =
        static_cast<std::int32_t>(column.chars.size());
    }
  }
  if (source_position != source.size()) {
    throw std::runtime_error("columnar corpus construction did not consume the full source");
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
  return cudf::make_strings_column(static_cast<cudf::size_type>(input.offsets.size() - 1U),
                                   std::move(offsets),
                                   std::move(chars),
                                   0,
                                   rmm::device_buffer{});
}

class device_text_column {
 public:
  device_text_column(input_data input, std::int32_t rows, cudaStream_t stream)
    : rows_(rows), stream_(stream), strings_(make_strings(input, stream))
  {
    check_cuda(cudaStreamSynchronize(stream_), "column construction synchronization");
  }

  [[nodiscard]] std::int32_t rows() const { return rows_; }
  [[nodiscard]] cudf::strings_column_view view() const
  {
    return cudf::strings_column_view{strings_->view()};
  }
  [[nodiscard]] char const* chars() const
  {
    return view().chars_begin(rmm::cuda_stream_view{stream_});
  }
  [[nodiscard]] std::int32_t const* offsets() const
  {
    return view().offsets().data<std::int32_t>();
  }

 private:
  std::int32_t rows_   = 0;
  cudaStream_t stream_ = nullptr;
  std::unique_ptr<cudf::column> strings_;
};

std::int32_t get_axis(nvbench::state& state, std::string const& name)
{
  auto value = state.get_int64(name);
  if (value < 0 || value > std::numeric_limits<std::int32_t>::max()) {
    throw std::invalid_argument(fmt::format("{} is outside int32 range", name));
  }
  return static_cast<std::int32_t>(value);
}

corpus_case const& get_case(nvbench::state& state)
{
  auto index = state.get_int64("Case");
  if (index < 1 || static_cast<std::size_t>(index) > openresty_cases.size()) {
    throw std::invalid_argument("Case is outside the registered OpenResty range");
  }
  return openresty_cases[static_cast<std::size_t>(index - 1)];
}

class table_workload {
 public:
  table_workload(nvbench::state& state, corpus_case const& benchmark)
    : rows_(get_axis(state, "Rows")), stream_(state.get_cuda_stream().get_stream())
  {
    auto column_count = get_axis(state, "Columns");
    auto maximum      = get_axis(state, "MaxStringBytes");
    auto inputs       = make_table_input(benchmark, rows_, maximum, column_count);
    columns_.reserve(static_cast<std::size_t>(column_count));
    for (std::int32_t column = 0; column < column_count; ++column) {
      auto input = std::move(inputs[static_cast<std::size_t>(column)]);
      bytes_ += input.chars.size();
      columns_.push_back(std::make_unique<device_text_column>(std::move(input), rows_, stream_));
    }
  }

  [[nodiscard]] std::int32_t rows() const { return rows_; }
  [[nodiscard]] std::size_t bytes() const { return bytes_; }
  [[nodiscard]] cudaStream_t stream() const { return stream_; }
  [[nodiscard]] std::vector<std::unique_ptr<device_text_column>>& columns() { return columns_; }

 private:
  std::int32_t rows_   = 0;
  std::size_t bytes_   = 0;
  cudaStream_t stream_ = nullptr;
  std::vector<std::unique_ptr<device_text_column>> columns_ =
    std::vector<std::unique_ptr<device_text_column>>{};
};

struct target_architecture {
  std::string compute = "";
  std::string sm      = "";
};

target_architecture get_target_architecture(nvbench::state& state)
{
  if (!state.get_device().has_value()) throw std::runtime_error("GPU benchmark has no device");
  auto sm_number = state.get_device()->get_sm_version() / 10;
  return {fmt::format("compute_{}", sm_number), fmt::format("sm_{}", sm_number)};
}

std::unique_ptr<loaded_kernel> make_kernel(target_architecture const& architecture,
                                           corpus_case const& benchmark)
{
  check_driver(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 64U * 1024U), "cuCtxSetLimit");
  regex_ir::compile_options compile_options;
  compile_options.case_insensitive = benchmark.case_insensitive;
  compile_options.multiline        = benchmark.multiline;
  auto compiled =
    regex_ir::compile(benchmark.expression, regex_ir::operation::contains(), compile_options);
  if (!compiled) {
    auto detail = compiled.diagnostics.empty() ? std::string{"unknown error"}
                                               : compiled.diagnostics.front().message;
    throw std::runtime_error(fmt::format("Regex IR compilation failed: {}", detail));
  }
  regex_ir::nvvm_ir_codegen_options options;
  options.symbol_prefix    = "regex_ir_benchmark";
  options.execute_function = "regex_ir_benchmark_execute";
  auto nvvm_ir             = regex_ir::generate_nvvm_ir(*compiled.value, options);
  auto ptx                 = compile_to_ptx(nvvm_ir, architecture.compute);
  auto wrapper             = std::span<unsigned char const>{regex_ir_benchmark_contains_fatbin,
                                                            regex_ir_benchmark_contains_fatbinLength};
  auto cubin               = link_cubin(ptx, wrapper, architecture.sm);
  return std::make_unique<loaded_kernel>(cubin);
}

cudf::strings::regex_flags cudf_flags(corpus_case const& benchmark)
{
  auto flags = benchmark.case_insensitive ? cudf::strings::regex_flags::IGNORECASE
                                          : cudf::strings::regex_flags::DEFAULT;
  if (benchmark.multiline) {
    flags = static_cast<cudf::strings::regex_flags>(flags | cudf::strings::regex_flags::MULTILINE);
  }
  return flags;
}

std::unique_ptr<cudf::strings::regex_program> make_cudf_program(corpus_case const& benchmark)
{
  auto expression =
    std::string{benchmark.comparison_expression.empty() ? benchmark.expression
                                                        : benchmark.comparison_expression};
  auto replace_all = [&](std::string_view source, std::string_view replacement) {
    std::size_t position = 0;
    while ((position = expression.find(source, position)) != std::string::npos) {
      expression.replace(position, source.size(), replacement);
      position += replacement.size();
    }
  };
  replace_all("[:alpha:]", "A-Za-z");
  replace_all("[:alnum:]", "A-Za-z0-9");
  replace_all("[:digit:]", "0-9");
  replace_all("[:xdigit:]", "A-Fa-f0-9");
  replace_all("[:space:]", R"REGEX(\s)REGEX");
  replace_all("[:word:]", "A-Za-z0-9_");
  replace_all("[:punct:]", R"REGEX(\x21-\x2F\x3A-\x40\x5B-\x60\x7B-\x7E)REGEX");
  return cudf::strings::regex_program::create(
    expression, cudf_flags(benchmark), cudf::strings::capture_groups::NON_CAPTURE);
}

std::vector<std::unique_ptr<cudf::column>> launch_regex_ir(table_workload& workload,
                                                           loaded_kernel const& kernel,
                                                           cudaStream_t stream)
{
  std::vector<std::unique_ptr<cudf::column>> outputs;
  outputs.reserve(workload.columns().size());
  for (auto& column : workload.columns()) {
    auto output = cudf::make_numeric_column(cudf::data_type{cudf::type_id::BOOL8},
                                            column->rows(),
                                            cudf::mask_state::UNALLOCATED,
                                            rmm::cuda_stream_view{stream});
    kernel.launch(column->chars(),
                  column->offsets(),
                  column->rows(),
                  reinterpret_cast<std::uint8_t*>(output->mutable_view().data<bool>()),
                  stream);
    outputs.push_back(std::move(output));
  }
  return outputs;
}

std::vector<std::uint8_t> copy_result(void const* data, std::int32_t rows, cudaStream_t stream)
{
  std::vector<std::uint8_t> result(static_cast<std::size_t>(rows));
  check_cuda(cudaMemcpyAsync(result.data(), data, result.size(), cudaMemcpyDeviceToHost, stream),
             "copy validation result");
  check_cuda(cudaStreamSynchronize(stream), "validation synchronization");
  return result;
}

void validate(table_workload& workload,
              loaded_kernel const& kernel,
              cudf::strings::regex_program const& program)
{
  trace("launch Regex IR validation");
  auto regex_outputs = launch_regex_ir(workload, kernel, workload.stream());
  check_cuda(cudaStreamSynchronize(workload.stream()), "Regex IR validation synchronization");
  trace("launch cuDF validation");
  for (std::size_t index = 0; index < workload.columns().size(); ++index) {
    auto& column = workload.columns()[index];
    auto cudf_output =
      cudf::strings::contains_re(column->view(), program, rmm::cuda_stream_view{workload.stream()});
    auto actual =
      copy_result(regex_outputs[index]->view().data<bool>(), column->rows(), workload.stream());
    auto expected =
      copy_result(cudf_output->view().data<bool>(), column->rows(), workload.stream());
    if (actual != expected) {
      throw std::runtime_error("Regex IR and cuDF disagree on the external corpus");
    }
  }
  trace("validation complete");
}

bool skip_unsupported(nvbench::state& state, corpus_case const& benchmark)
{
  auto rows       = get_axis(state, "Rows");
  auto maximum    = get_axis(state, "MaxStringBytes");
  auto columns    = get_axis(state, "Columns");
  auto byte_limit = static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
  if (columns == 0) {
    state.skip("Columns must be positive");
    return true;
  }
  auto source_size = corpus_text(benchmark).size();
  if (benchmark.corpus == corpus_source::Inline &&
      source_size > static_cast<std::size_t>(maximum)) {
    state.skip("MaxStringBytes is too small for this case's complete scalar input");
    return true;
  }
  auto capacity = static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(maximum) *
                  static_cast<std::uint64_t>(columns);
  if (benchmark.corpus != corpus_source::Inline && source_size > capacity) {
    state.skip("Rows, Columns, and MaxStringBytes cannot hold the complete source corpus");
    return true;
  }
  if (static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(maximum) > byte_limit) {
    state.skip("one text column could exceed cuDF's signed 32-bit strings offset limit");
    return true;
  }
  return false;
}

void add_counters(nvbench::state& state, table_workload const& workload, std::int32_t columns)
{
  auto values = static_cast<std::size_t>(workload.rows()) * static_cast<std::size_t>(columns);
  state.add_element_count(values, "Strings");
  state.add_global_memory_reads<std::uint8_t>(workload.bytes(), "InputBytes");
  state.add_global_memory_writes<std::uint8_t>(values, "OutputBytes");
}

void add_compile_time(nvbench::state& state, double seconds)
{
  nvbench::summary& summary = state.add_summary("regex_ir/corpus/compile_time");
  summary.set_string("name", "Cold Compile");
  summary.set_string("hint", "duration");
  summary.set_string("description", "Uncached regex compilation and engine setup time");
  summary.set_float64("value", seconds);
}

void run_regex_ir_case(nvbench::state& state, corpus_case const& benchmark)
{
  if (skip_unsupported(state, benchmark)) return;
  check_driver(cuInit(0), "cuInit");
  trace("construct corpus");
  table_workload workload(state, benchmark);
  add_counters(state, workload, get_axis(state, "Columns"));
  auto architecture = get_target_architecture(state);
  trace("compile Regex IR kernel");
  auto compile_start = std::chrono::steady_clock::now();
  auto kernel        = make_kernel(architecture, benchmark);
  auto compile_stop  = std::chrono::steady_clock::now();
  add_compile_time(state, std::chrono::duration<double>(compile_stop - compile_start).count());
  trace("compile cuDF program");
  auto program = make_cudf_program(benchmark);
  validate(workload, *kernel, *program);
  std::vector<std::unique_ptr<cudf::column>> outputs;
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    outputs = launch_regex_ir(workload, *kernel, launch.get_stream().get_stream());
  });
}

void run_cudf_case(nvbench::state& state, corpus_case const& benchmark)
{
  if (skip_unsupported(state, benchmark)) return;
  check_driver(cuInit(0), "cuInit");
  trace("construct corpus");
  table_workload workload(state, benchmark);
  add_counters(state, workload, get_axis(state, "Columns"));
  auto architecture = get_target_architecture(state);
  trace("compile Regex IR kernel");
  auto kernel = make_kernel(architecture, benchmark);
  trace("compile cuDF program");
  auto compile_start = std::chrono::steady_clock::now();
  auto program       = make_cudf_program(benchmark);
  auto compile_stop  = std::chrono::steady_clock::now();
  add_compile_time(state, std::chrono::duration<double>(compile_stop - compile_start).count());
  validate(workload, *kernel, *program);
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    std::vector<std::unique_ptr<cudf::column>> outputs;
    outputs.reserve(workload.columns().size());
    for (auto& column : workload.columns()) {
      outputs.push_back(cudf::strings::contains_re(
        column->view(), *program, rmm::cuda_stream_view{launch.get_stream().get_stream()}));
    }
  });
}

}  // namespace

namespace regex_ir_benchmark {

void run_regex_ir(nvbench::state& state, corpus_case const& benchmark)
{
  run_regex_ir_case(state, benchmark);
}

void run_cudf(nvbench::state& state, corpus_case const& benchmark)
{
  run_cudf_case(state, benchmark);
}

}  // namespace regex_ir_benchmark

namespace {

void regex_ir_openresty(nvbench::state& state)
{
  regex_ir_benchmark::run_regex_ir(state, get_case(state));
}

void cudf_openresty(nvbench::state& state) { regex_ir_benchmark::run_cudf(state, get_case(state)); }

}  // namespace

NVBENCH_BENCH(regex_ir_openresty)
  .set_name("regex_ir/openresty")
  .add_int64_axis("Case", {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                           17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31})
  .add_int64_axis("Rows", {4096, 32768, 262144})
  .add_int64_axis("Columns", {1, 8})
  .add_int64_axis("MaxStringBytes", {64, 256, 1024});

NVBENCH_BENCH(cudf_openresty)
  .set_name("cudf/openresty")
  .add_int64_axis("Case", {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                           17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31})
  .add_int64_axis("Rows", {4096, 32768, 262144})
  .add_int64_axis("Columns", {1, 8})
  .add_int64_axis("MaxStringBytes", {64, 256, 1024});
