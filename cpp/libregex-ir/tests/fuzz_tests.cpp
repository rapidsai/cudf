/*
 * deterministic parser, optimizer, and executor fuzz testing.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <regex_ir.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using regex_ir::testing::execution_result;

constexpr std::size_t arbitrary_iterations  = 100000;
constexpr std::size_t structured_iterations = 1200;

bool equal(regex_ir::operation_kind kind,
           execution_result const& left,
           execution_result const& right)
{
  using regex_ir::operation_kind;
  switch (kind) {
    case operation_kind::CONTAINS:
    case operation_kind::MATCHES: return left.matched == right.matched;
    case operation_kind::FIND:
      return left.matched == right.matched && left.matches == right.matches;
    case operation_kind::COUNT: return left.count == right.count;
    case operation_kind::EXTRACT:
      return left.matched == right.matched && left.captures == right.captures;
    case operation_kind::REPLACE: return left.replaced == right.replaced;
    case operation_kind::SPLIT: return left.pieces == right.pieces;
  }
  return false;
}

char const* operation_name(regex_ir::operation_kind kind)
{
  using regex_ir::operation_kind;
  switch (kind) {
    case operation_kind::CONTAINS: return "contains";
    case operation_kind::MATCHES: return "matches";
    case operation_kind::COUNT: return "count";
    case operation_kind::EXTRACT: return "extract";
    case operation_kind::FIND: return "find";
    case operation_kind::REPLACE: return "replace";
    case operation_kind::SPLIT: return "split";
  }
  return "unknown";
}

std::vector<regex_ir::operation> operations()
{
  return {
    regex_ir::operation::contains(),
    regex_ir::operation::matches(),
    regex_ir::operation::find(),
    regex_ir::operation::count(),
    regex_ir::operation::extract(),
    regex_ir::operation::replace("<$0>"),
    regex_ir::operation::split(),
  };
}

std::string random_input(std::mt19937& generator, std::size_t maximum_length)
{
  static std::vector<std::string> const alphabet = {
    "a", "b", "c", "0", "7", "_", " ", "\n", "λ", "界"};
  std::uniform_int_distribution<std::size_t> length_distribution(0, maximum_length);
  std::uniform_int_distribution<std::size_t> symbol_distribution(0, alphabet.size() - 1);
  std::string result;
  auto length = length_distribution(generator);
  for (std::size_t index = 0; index < length; ++index) {
    result += alphabet[symbol_distribution(generator)];
  }
  return result;
}

std::string structured_pattern(std::mt19937& generator, int depth)
{
  static std::vector<std::string> const atoms = {
    "a", "b", "0", ".", "[ab]", "[^b]", "[0-7]", "\\d", "\\w", "\\s", "λ"};
  static std::vector<std::string> const quantifiers = {
    "*", "+", "?", "*?", "+?", "??", "{0,2}", "{1,3}", "{2,}"};
  std::uniform_int_distribution<std::size_t> atom_distribution(0, atoms.size() - 1);
  if (depth <= 0) return atoms[atom_distribution(generator)];

  std::uniform_int_distribution<int> form_distribution(0, 5);
  auto form = form_distribution(generator);
  if (form == 0) return atoms[atom_distribution(generator)];
  if (form == 1) {
    return structured_pattern(generator, depth - 1) + structured_pattern(generator, depth - 1);
  }
  if (form == 2) {
    return "(?:" + structured_pattern(generator, depth - 1) + "|" +
           structured_pattern(generator, depth - 1) + ")";
  }
  if (form == 3) return "(" + structured_pattern(generator, depth - 1) + ")";
  if (form == 4) {
    std::uniform_int_distribution<std::size_t> quantifier_distribution(0, quantifiers.size() - 1);
    return "(?:" + structured_pattern(generator, depth - 1) + ")" +
           quantifiers[quantifier_distribution(generator)];
  }
  return "(?:" + structured_pattern(generator, depth - 1) + ")";
}

bool check_optimization_equivalence(std::string const& pattern,
                                    std::string const& input,
                                    regex_ir::operation const& operation)
{
  regex_ir::optimization_options disabled;
  disabled.remove_unreachable        = false;
  disabled.fold_epsilon_jumps        = false;
  disabled.fuse_literals             = false;
  disabled.strip_unobserved_captures = false;

  regex_ir::compile_options limits;
  limits.limits.max_states      = 4096;
  limits.limits.max_transitions = 16384;
  limits.limits.max_repeat      = 32;

  auto optimized = regex_ir::compile(pattern, operation, limits);
  auto baseline  = regex_ir::compile(pattern, operation, limits, disabled);
  if (!optimized || !baseline) {
    std::cerr << "structured pattern failed to compile: " << pattern
              << " operation=" << operation_name(operation.kind) << '\n';
    return false;
  }
  if (!regex_ir::verify(*optimized.value).empty() || !regex_ir::verify(*baseline.value).empty()) {
    std::cerr << "structured pattern produced invalid IR: " << pattern << '\n';
    return false;
  }

  try {
    auto optimized_result = regex_ir::testing::execute(*optimized.value, input);
    auto baseline_result  = regex_ir::testing::execute(*baseline.value, input);
    if (!equal(operation.kind, optimized_result, baseline_result)) {
      std::cerr << "optimizer differential mismatch: pattern=" << pattern << " input=" << input
                << " operation=" << operation_name(operation.kind) << '\n';
      return false;
    }
  } catch (std::exception const& error) {
    std::cerr << "execution threw for pattern=" << pattern << ": " << error.what() << '\n';
    return false;
  }
  return true;
}

void generate_inputs(std::string& current, std::size_t maximum, std::vector<std::string>& output)
{
  output.push_back(current);
  if (current.size() == maximum) return;
  for (char const value : std::string_view{"ab0"}) {
    current.push_back(value);
    generate_inputs(current, maximum, output);
    current.pop_back();
  }
}

std::vector<std::string> exhaustive_patterns()
{
  std::vector<std::string> const atoms = {"a", "b", ".", "[ab]", "[^a]", "\\d", "(a)"};
  std::vector<std::string> result      = atoms;
  std::vector<std::string> const unary = {"*", "+", "?", "*?", "+?", "??", "{0,2}", "{1,2}"};
  for (auto& atom : atoms) {
    for (auto& suffix : unary)
      result.push_back("(?:" + atom + ")" + suffix);
  }
  for (std::size_t left = 0; left < 4; ++left) {
    for (std::size_t right = 0; right < 4; ++right) {
      result.push_back(atoms[left] + atoms[right]);
      result.push_back("(?:" + atoms[left] + "|" + atoms[right] + ")");
    }
  }
  return result;
}

bool arbitrary_parser_fuzz(std::mt19937& generator)
{
  constexpr std::string_view alphabet = "abcXYZ09()[]{}*+?|.^$\\-,:! \n\t";
  std::uniform_int_distribution<std::size_t> length_distribution(0, 96);
  std::uniform_int_distribution<std::size_t> character_distribution(0, alphabet.size() - 1);
  auto all_operations = operations();

  for (std::size_t iteration = 0; iteration < arbitrary_iterations; ++iteration) {
    std::string pattern;
    auto length = length_distribution(generator);
    pattern.reserve(length);
    for (std::size_t index = 0; index < length; ++index) {
      pattern.push_back(alphabet[character_distribution(generator)]);
    }

    regex_ir::compile_options options;
    options.case_insensitive = (iteration & 1U) != 0;
    options.multiline        = (iteration & 2U) != 0;
    options.dot_all          = (iteration & 4U) != 0;
    options.characters =
      (iteration & 8U) != 0 ? regex_ir::character_mode::BYTES : regex_ir::character_mode::UTF8;
    options.limits.max_repeat      = 32;
    options.limits.max_states      = 4096;
    options.limits.max_transitions = 16384;

    auto& operation = all_operations[iteration % all_operations.size()];
    auto result     = regex_ir::compile(pattern, operation, options);
    if (!result && result.diagnostics.empty()) {
      std::cerr << "uncategorized compile failure at iteration " << iteration << '\n';
      return false;
    }
    if (result) {
      if (!regex_ir::verify(*result.value).empty()) {
        std::cerr << "invalid compiled IR at iteration " << iteration << '\n';
        return false;
      }
      if (regex_ir::to_string(*result.value) != regex_ir::to_string(*result.value)) {
        std::cerr << "non-deterministic IR printer at iteration " << iteration << '\n';
        return false;
      }
      if ((operation.kind == regex_ir::operation_kind::CONTAINS ||
           operation.kind == regex_ir::operation_kind::MATCHES) &&
          (iteration % 1000U) == 0) {
        auto generated = regex_ir::generate_nvvm_ir(*result.value);
        if (generated.empty() || generated != regex_ir::generate_nvvm_ir(*result.value)) {
          std::cerr << "invalid or non-deterministic NVVM at iteration " << iteration << '\n';
          return false;
        }
      }
      if ((iteration % 100U) == 0) {
        try {
          static_cast<void>(regex_ir::testing::execute(*result.value, random_input(generator, 8)));
        } catch (std::exception const& error) {
          std::cerr << "execution failure at iteration " << iteration << ": " << error.what()
                    << '\n';
          return false;
        }
      }
    }
  }
  return true;
}

bool exhaustive_optimizer_fuzz()
{
  std::vector<std::string> inputs;
  std::string current;
  generate_inputs(current, 4, inputs);
  std::vector<regex_ir::operation> const tested_operations = {regex_ir::operation::contains(),
                                                              regex_ir::operation::matches(),
                                                              regex_ir::operation::find(),
                                                              regex_ir::operation::count()};

  for (auto& pattern : exhaustive_patterns()) {
    for (auto& operation : tested_operations) {
      for (auto& input : inputs) {
        if (!check_optimization_equivalence(pattern, input, operation)) return false;
      }
    }
  }
  return true;
}

bool structured_random_fuzz(std::mt19937& generator)
{
  auto all_operations = operations();
  for (std::size_t iteration = 0; iteration < structured_iterations; ++iteration) {
    auto pattern = structured_pattern(generator, 3);
    if (pattern.size() > 160) pattern = "a";
    for (auto& operation : all_operations) {
      for (int input_index = 0; input_index < 2; ++input_index) {
        if (!check_optimization_equivalence(pattern, random_input(generator, 9), operation)) {
          std::cerr << "structured iteration " << iteration << '\n';
          return false;
        }
      }
    }
  }
  return true;
}

bool verifier_rejects_mutations()
{
  auto compiled = regex_ir::compile("a(b|c)+", regex_ir::operation::extract());
  if (!compiled) return false;

  auto invalid_entry  = *compiled.value;
  invalid_entry.entry = regex_ir::invalid_block;
  if (regex_ir::verify(invalid_entry).empty()) return false;

  auto invalid_id              = *compiled.value;
  invalid_id.blocks.front().id = regex_ir::invalid_block;
  if (regex_ir::verify(invalid_id).empty()) return false;

  auto invalid_edge = *compiled.value;
  for (auto& block : invalid_edge.blocks) {
    if (!block.successors.empty()) {
      block.successors.front().target = regex_ir::invalid_block;
      return !regex_ir::verify(invalid_edge).empty();
    }
  }
  return false;
}

}  // namespace

#ifdef REGEX_IR_FUZZ_SMOKE
int main()
{
  std::mt19937 generator(0xC0FFEEU);
  if (!arbitrary_parser_fuzz(generator)) return 1;
  if (!exhaustive_optimizer_fuzz()) return 1;
  if (!structured_random_fuzz(generator)) return 1;
  if (!verifier_rejects_mutations()) {
    std::cerr << "IR verifier accepted a deliberately invalid mutation\n";
    return 1;
  }
  std::cout << arbitrary_iterations << " arbitrary patterns, " << structured_iterations
            << " structured patterns, and exhaustive optimizer differentials passed\n";
}
#endif

namespace {

regex_ir::operation select_operation(std::uint8_t control)
{
  switch (control % 7U) {
    case 0: return regex_ir::operation::contains();
    case 1: return regex_ir::operation::matches();
    case 2: return regex_ir::operation::find();
    case 3: return regex_ir::operation::count();
    case 4: return regex_ir::operation::extract();
    case 5: return regex_ir::operation::replace("<$0>");
    default: return regex_ir::operation::split();
  }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(std::uint8_t const* data, std::size_t size)
{
  if (size == 0 || size > 4096) return 0;
  std::uint8_t control = data[0];
  std::string_view payload(reinterpret_cast<char const*>(data + 1), size - 1);
  std::size_t separator = payload.find('\n');
  if (separator == std::string_view::npos) separator = payload.size() / 2;
  std::string_view pattern = payload.substr(0, separator);
  std::string_view input =
    separator < payload.size() ? payload.substr(separator + 1) : std::string_view{};

  regex_ir::compile_options options;
  options.case_insensitive = (control & 0x08U) != 0;
  options.multiline        = (control & 0x10U) != 0;
  options.dot_all          = (control & 0x20U) != 0;
  options.characters =
    (control & 0x40U) != 0 ? regex_ir::character_mode::BYTES : regex_ir::character_mode::UTF8;
  options.limits.max_pattern_bytes = 4096;
  options.limits.max_states        = 8192;
  options.limits.max_transitions   = 32768;
  options.limits.max_repeat        = 64;

  auto compiled = regex_ir::compile(pattern, select_operation(control), options);
  if (!compiled) return 0;
  if (!regex_ir::verify(*compiled.value).empty()) __builtin_trap();
  static_cast<void>(regex_ir::to_string(*compiled.value));
  static_cast<void>(regex_ir::testing::execute(*compiled.value, input));
  return 0;
}
