/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <re2/re2.h>
#include <regex_ir.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace {

enum class benchmark_operation : std::uint8_t {
  CONTAINS = 0,
  COUNT    = 1,
  EXTRACT  = 2,
  REPLACE  = 3,
  SPLIT    = 4,
};

struct pattern_case {
  std::string_view expression = "";
  std::string_view match      = "";
};

constexpr std::array contains_patterns{
  pattern_case{R"REGEX(^\d+ [a-z]+)REGEX", "123 abc"},
  pattern_case{R"REGEX([A-Z ]+\d+ +\d+[A-Z]+\d+$)REGEX", "ABC 123  45XYZ6"},
  pattern_case{"^123 abc", "123 abc"},
  pattern_case{"0987 5W43$", "0987 5W43"},
  pattern_case{"0987 5W43", "0987 5W43"},
  pattern_case{R"REGEX(5[A-Z]\d+)REGEX", "5W43"},
  pattern_case{"5W43|X9Z8", "5W43"},
  pattern_case{"7 5W4{1,3}", "7 5W44"},
  pattern_case{"7 (?:5W){1,2}", "7 5W5W"},
  pattern_case{"7 5.4.", "7 5x4y"},
  pattern_case{".+5W", "abc5W"},
};

constexpr std::array transform_patterns{
  pattern_case{R"REGEX(\d+)REGEX", "123"},
  pattern_case{" ", " "},
  pattern_case{"[a-z]+[A-Z]+", "abcXYZ"},
  pattern_case{"[a-f]+|[0-5]+", "abcdef"},
  pattern_case{"[a-z][0-9]{0,3}[A-Z]", "a12Z"},
  pattern_case{".+[0-9]", "abc7"},
  pattern_case{"[a-z]+Z", "abcZ"},
};

struct benchmark_case {
  benchmark_operation operation = benchmark_operation::CONTAINS;
  std::size_t pattern           = 0;
  std::size_t rows              = 0;
  std::size_t width             = 0;
  std::size_t groups            = 0;
  std::size_t hit_rate          = 50;
  bool backrefs : 1             = false;
};

struct options {
  bool full : 1              = false;
  bool list : 1              = false;
  std::size_t iterations     = 3;
  std::size_t rows_override  = 0;
  std::size_t width_override = 0;
  std::string operation      = "all";
};

std::string_view name(benchmark_operation operation)
{
  switch (operation) {
    case benchmark_operation::CONTAINS: return "contains";
    case benchmark_operation::COUNT: return "count";
    case benchmark_operation::EXTRACT: return "extract";
    case benchmark_operation::REPLACE: return "replace";
    case benchmark_operation::SPLIT: return "split";
  }
  return "unknown";
}

regex_ir::operation make_operation(benchmark_case const& test, std::string const& pattern)
{
  switch (test.operation) {
    case benchmark_operation::CONTAINS: return regex_ir::operation::contains();
    case benchmark_operation::COUNT: return regex_ir::operation::count();
    case benchmark_operation::EXTRACT: return regex_ir::operation::extract();
    case benchmark_operation::REPLACE:
      return regex_ir::operation::replace(test.backrefs ? "#$1X" : "77");
    case benchmark_operation::SPLIT: return regex_ir::operation::split();
  }
  throw std::logic_error("invalid operation for pattern " + pattern);
}

std::uint64_t hash_bytes(std::uint64_t seed, std::string_view value)
{
  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    seed               = (seed ^ byte) * 1099511628211ULL;
  }
  return seed;
}

std::vector<std::string> make_rows(benchmark_case const& test, pattern_case pattern)
{
  std::mt19937 generator(0xC0FFEEU + static_cast<unsigned>(test.pattern));
  constexpr std::string_view alphabet = "abcdefghijklmnpqrstuvwxyzABCDEF0123456789 ";
  std::uniform_int_distribution<std::size_t> pick(0, alphabet.size() - 1);
  std::vector<std::string> rows(test.rows);
  for (std::size_t row = 0; row < rows.size(); ++row) {
    std::string value(test.width, 'x');
    for (char& character : value)
      character = alphabet[pick(generator)];
    bool insert = row % 100U < test.hit_rate && pattern.match.size() <= value.size();
    if (insert) {
      std::size_t offset =
        test.operation == benchmark_operation::CONTAINS && pattern.expression.starts_with('^')
          ? 0
          : (value.size() - pattern.match.size()) / 2U;
      if (pattern.expression.ends_with('$')) offset = value.size() - pattern.match.size();
      std::copy(pattern.match.begin(),
                pattern.match.end(),
                value.begin() + static_cast<std::ptrdiff_t>(offset));
    }
    rows[row] = std::move(value);
  }
  return rows;
}

std::pair<std::string, std::vector<std::string>> make_extract_rows(benchmark_case const& test)
{
  std::string pattern;
  for (std::size_t group = 0; group < test.groups; ++group)
    pattern += R"REGEX((\d+) )REGEX";
  std::mt19937 generator(0xC0FFEEU);
  std::uniform_int_distribution<int> value(0, 999);
  std::vector<std::string> samples(100);
  for (std::string& sample : samples) {
    while (sample.size() < test.width)
      sample += std::to_string(value(generator)) + " ";
  }
  std::vector<std::string> rows(test.rows);
  for (std::size_t row = 0; row < rows.size(); ++row)
    rows[row] = samples[row % samples.size()];
  return {std::move(pattern), std::move(rows)};
}

std::uint64_t run_regex_ir(regex_ir::instruction_ir const& ir,
                           std::span<std::string const> rows,
                           benchmark_case const& test)
{
  std::uint64_t checksum = 0;
  for (std::string const& row : rows) {
    regex_ir::testing::execution_result result = regex_ir::testing::execute(ir, row);
    if (test.operation == benchmark_operation::CONTAINS) {
      checksum = checksum * 131U + static_cast<std::uint64_t>(result.matched);
    } else if (test.operation == benchmark_operation::COUNT) {
      checksum = checksum * 131U + result.count;
    } else if (test.operation == benchmark_operation::EXTRACT) {
      checksum = checksum * 131U + static_cast<std::uint64_t>(result.matched);
      if (result.matched) {
        for (std::size_t capture = 1; capture < result.captures.size(); ++capture) {
          if (result.captures[capture]) {
            regex_ir::testing::match_span span = *result.captures[capture];
            checksum =
              hash_bytes(checksum, std::string_view{row}.substr(span.begin, span.end - span.begin));
          }
        }
      }
    } else if (test.operation == benchmark_operation::REPLACE) {
      checksum = hash_bytes(checksum, result.replaced);
    } else {
      for (std::string const& piece : result.pieces)
        checksum = hash_bytes(checksum, piece);
    }
  }
  return checksum;
}

std::uint64_t re2_contains(RE2 const& regex, std::span<std::string const> rows)
{
  std::uint64_t checksum = 0;
  for (std::string const& row : rows) {
    checksum = checksum * 131U + static_cast<std::uint64_t>(RE2::PartialMatch(row, regex));
  }
  return checksum;
}

std::uint64_t re2_count(RE2 const& regex, std::span<std::string const> rows)
{
  std::uint64_t checksum = 0;
  for (std::string const& row : rows) {
    re2::StringPiece remaining{row};
    std::size_t count = 0;
    re2::StringPiece match;
    while (regex.Match(remaining, 0, remaining.size(), RE2::UNANCHORED, &match, 1)) {
      ++count;
      std::size_t consumed =
        static_cast<std::size_t>(match.data() - remaining.data()) + match.size();
      if (match.empty()) {
        if (consumed == remaining.size()) break;
        ++consumed;
      }
      remaining.remove_prefix(consumed);
    }
    checksum = checksum * 131U + count;
  }
  return checksum;
}

std::uint64_t re2_extract(RE2 const& regex, std::span<std::string const> rows, std::size_t groups)
{
  std::uint64_t checksum = 0;
  std::vector<re2::StringPiece> captures(groups + 1U);
  for (std::string const& row : rows) {
    bool matched = regex.Match(
      row, 0, row.size(), RE2::UNANCHORED, captures.data(), static_cast<int>(captures.size()));
    checksum = checksum * 131U + static_cast<std::uint64_t>(matched);
    if (matched) {
      for (std::size_t group = 1; group < captures.size(); ++group) {
        checksum = hash_bytes(checksum, {captures[group].data(), captures[group].size()});
      }
    }
  }
  return checksum;
}

std::uint64_t re2_replace(RE2 const& regex, std::span<std::string const> rows, bool backrefs)
{
  std::uint64_t checksum = 0;
  for (std::string const& row : rows) {
    std::string output = row;
    RE2::GlobalReplace(&output, regex, backrefs ? "#\\1X" : "77");
    checksum = hash_bytes(checksum, output);
  }
  return checksum;
}

std::uint64_t re2_split(RE2 const& regex, std::span<std::string const> rows)
{
  std::uint64_t checksum = 0;
  for (std::string const& row : rows) {
    std::size_t cursor = 0;
    re2::StringPiece match;
    while (regex.Match(row, cursor, row.size(), RE2::UNANCHORED, &match, 1)) {
      std::size_t begin = static_cast<std::size_t>(match.data() - row.data());
      checksum = hash_bytes(checksum, std::string_view{row}.substr(cursor, begin - cursor));
      cursor   = begin + match.size();
      if (match.empty()) {
        if (cursor == row.size()) break;
        ++cursor;
      }
    }
    checksum = hash_bytes(checksum, std::string_view{row}.substr(cursor));
  }
  return checksum;
}

std::uint64_t run_re2(RE2 const& regex,
                      std::span<std::string const> rows,
                      benchmark_case const& test)
{
  switch (test.operation) {
    case benchmark_operation::CONTAINS: return re2_contains(regex, rows);
    case benchmark_operation::COUNT: return re2_count(regex, rows);
    case benchmark_operation::EXTRACT: return re2_extract(regex, rows, test.groups);
    case benchmark_operation::REPLACE: return re2_replace(regex, rows, test.backrefs);
    case benchmark_operation::SPLIT: return re2_split(regex, rows);
  }
  return 0;
}

std::vector<benchmark_case> benchmark_cases(options const& selected)
{
  std::vector<benchmark_case> cases;
  auto include = [&](benchmark_operation operation) {
    return selected.operation == "all" || selected.operation == name(operation);
  };
  std::vector<std::size_t> standard_rows =
    selected.full
      ? std::vector<std::size_t>{262144, 2097152}
      : std::vector<std::size_t>{selected.rows_override == 0 ? 32768 : selected.rows_override};
  std::vector<std::size_t> standard_widths =
    selected.full
      ? std::vector<std::size_t>{64, 128, 256}
      : std::vector<std::size_t>{selected.width_override == 0 ? 64 : selected.width_override};

  if (include(benchmark_operation::CONTAINS)) {
    for (std::size_t rows : standard_rows) {
      for (std::size_t width : standard_widths) {
        for (std::size_t hit_rate : {50U, 100U}) {
          for (std::size_t pattern = 0; pattern < contains_patterns.size(); ++pattern) {
            cases.push_back({benchmark_operation::CONTAINS, pattern, rows, width, 0, hit_rate});
          }
        }
      }
    }
  }
  for (benchmark_operation operation :
       {benchmark_operation::COUNT, benchmark_operation::REPLACE, benchmark_operation::SPLIT}) {
    if (!include(operation)) continue;
    for (std::size_t rows : standard_rows) {
      for (std::size_t width : standard_widths) {
        for (std::size_t pattern = 0; pattern < transform_patterns.size(); ++pattern) {
          cases.push_back({operation, pattern, rows, width, 0, 50, false});
          if (operation == benchmark_operation::REPLACE) {
            cases.push_back({operation, pattern, rows, width, 0, 50, true});
          }
        }
      }
    }
  }
  if (include(benchmark_operation::EXTRACT)) {
    std::vector<std::size_t> extract_rows =
      selected.full ? std::vector<std::size_t>{32768, 262144, 2097152} : standard_rows;
    std::vector<std::size_t> extract_widths =
      selected.full ? std::vector<std::size_t>{32, 64, 128, 256} : standard_widths;
    for (std::size_t rows : extract_rows) {
      for (std::size_t width : extract_widths) {
        for (std::size_t groups : {1U, 2U, 4U}) {
          cases.push_back({benchmark_operation::EXTRACT, 0, rows, width, groups, 100});
        }
      }
    }
  }
  return cases;
}

options parse_options(int argc, char** argv)
{
  options result;
  for (int index = 1; index < argc; ++index) {
    std::string_view argument{argv[index]};
    auto value = [&](std::string_view option) {
      if (++index == argc) throw std::invalid_argument(std::string{option} + " requires a value");
      return std::string_view{argv[index]};
    };
    if (argument == "--full")
      result.full = true;
    else if (argument == "--list")
      result.list = true;
    else if (argument == "--operation")
      result.operation = value(argument);
    else if (argument == "--rows")
      result.rows_override = std::stoull(std::string{value(argument)});
    else if (argument == "--width")
      result.width_override = std::stoull(std::string{value(argument)});
    else if (argument == "--iterations")
      result.iterations = std::stoull(std::string{value(argument)});
    else if (argument == "--help") {
      std::cout
        << R"HELP(usage: regex-ir-cpu-benchmark [--full] [--list] [--operation NAME] [--rows N] [--width N] [--iterations N]
)HELP";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown option: " + std::string{argument});
    }
  }
  return result;
}

template <typename Function>
double measure(Function&& function, std::size_t iterations, std::uint64_t& checksum)
{
  double best = std::numeric_limits<double>::max();
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    auto start = std::chrono::steady_clock::now();
    checksum ^= function();
    auto end = std::chrono::steady_clock::now();
    best     = std::min(best, std::chrono::duration<double, std::milli>(end - start).count());
  }
  return best;
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    options selected                  = parse_options(argc, argv);
    std::vector<benchmark_case> cases = benchmark_cases(selected);
    if (selected.list) {
      for (benchmark_case const& test : cases) {
        std::cout << name(test.operation) << " pattern=" << test.pattern << " rows=" << test.rows
                  << " width=" << test.width << " groups=" << test.groups
                  << " backrefs=" << test.backrefs << '\n';
      }
      return 0;
    }

    std::cout
      << "operation,pattern,rows,width,variant,regex_ir_ms,re2_ms,regex_ir_mrows_s,re2_mrows_s\n";
    std::uint64_t checksum = 0;
    for (benchmark_case const& test : cases) {
      std::string pattern;
      std::vector<std::string> rows;
      if (test.operation == benchmark_operation::EXTRACT) {
        std::tie(pattern, rows) = make_extract_rows(test);
      } else {
        pattern_case selected_pattern = test.operation == benchmark_operation::CONTAINS
                                          ? contains_patterns[test.pattern]
                                          : transform_patterns[test.pattern];
        pattern                       = std::string{selected_pattern.expression};
        if (test.operation == benchmark_operation::REPLACE && test.backrefs) {
          pattern = "(" + pattern + ")";
        }
        rows = make_rows(test, selected_pattern);
      }

      regex_ir::operation operation = make_operation(test, pattern);
      auto compiled                 = regex_ir::compile(pattern, operation);
      if (!compiled) throw std::runtime_error("Regex IR failed to compile " + pattern);
      RE2::Options re2_options;
      re2_options.set_log_errors(false);
      RE2 regex(pattern, re2_options);
      if (!regex.ok()) throw std::runtime_error("RE2 failed to compile " + pattern);

      std::uint64_t regex_ir_check = run_regex_ir(*compiled.value, rows, test);
      std::uint64_t re2_check      = run_re2(regex, rows, test);
      if (regex_ir_check != re2_check) {
        throw std::runtime_error("engine result checksums differ for " + pattern);
      }

      double regex_ir_ms = measure(
        [&] { return run_regex_ir(*compiled.value, rows, test); }, selected.iterations, checksum);
      double re2_ms =
        measure([&] { return run_re2(regex, rows, test); }, selected.iterations, checksum);
      double regex_ir_rate = static_cast<double>(test.rows) / regex_ir_ms / 1000.0;
      double re2_rate      = static_cast<double>(test.rows) / re2_ms / 1000.0;
      std::string variant  = "plain";
      if (test.operation == benchmark_operation::CONTAINS) {
        variant = "hit" + std::to_string(test.hit_rate);
      } else if (test.operation == benchmark_operation::EXTRACT) {
        variant = "groups" + std::to_string(test.groups);
      } else if (test.backrefs) {
        variant = "backrefs";
      }
      std::cout << name(test.operation) << ',' << test.pattern << ',' << test.rows << ','
                << test.width << ',' << variant << ',' << std::fixed << std::setprecision(3)
                << regex_ir_ms << ',' << re2_ms << ',' << regex_ir_rate << ',' << re2_rate << '\n';
    }
    if (checksum == 0xFFFFFFFFFFFFFFFFULL) std::cerr << "checksum: " << checksum << '\n';
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
