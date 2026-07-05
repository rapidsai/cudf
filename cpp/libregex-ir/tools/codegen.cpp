/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <regex_ir.hpp>

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct options {
  regex_ir::operation operation     = regex_ir::operation::contains();
  regex_ir::compile_options compile = regex_ir::compile_options{};
  std::string symbol_prefix         = "regex_ir_generated";
  std::string execute_function      = "regex_ir_execute";
  bool prefix_filter : 1            = true;
  bool branch_hints  : 1            = true;
};

void usage(char const* executable)
{
  std::cout << "usage: " << executable << " [OPTIONS] PATTERN\n\n"
            << "Generate CUDA-oriented NVVM IR on standard output.\n\n"
            << "Options:\n"
            << "  -o, --operation NAME  contains, matches, find, count, extract, or split\n"
            << "                        (default: contains)\n"
            << "  --replace TEXT        select replace mode with replacement TEXT\n"
            << "  --symbol-prefix NAME  prefix for generated internal symbols\n"
            << "  --execute-function N  generated NVVM entry function name\n"
            << "  --no-prefix-filter    disable recursive-fallback prefix filtering\n"
            << "  --no-branch-hints     disable recursive-fallback branch hints\n"
            << "  -i, --ignore-case     enable Unicode case-insensitive matching\n"
            << "  -m, --multiline       enable multiline anchors\n"
            << "  -s, --dot-all         make dot match every line terminator\n"
            << "  --unicode-classes     use Unicode predefined classes\n"
            << "  --extended-newline    recognize CR, NEL, LS, and PS as newlines\n"
            << "  --bytes               use byte mode instead of UTF-8\n"
            << "  -h, --help            show this help\n";
}

bool take_value(int& index, int argc, char** argv, std::string& output)
{
  if (++index >= argc) return false;
  output = argv[index];
  return true;
}

int report(std::vector<regex_ir::diagnostic> const& diagnostics)
{
  for (auto& diagnostic : diagnostics) {
    std::cerr << "pattern byte " << diagnostic.span.offset << ": " << diagnostic.message << '\n';
  }
  return 1;
}

}  // namespace

int main(int argc, char** argv)
{
  options parsed;
  std::string pattern;
  bool has_pattern{};
  for (int index = 1; index < argc; ++index) {
    std::string_view argument = argv[index];
    if (argument == "-h" || argument == "--help") {
      usage(argv[0]);
      return 0;
    }
    if (argument == "-o" || argument == "--operation") {
      std::string value;
      if (!take_value(index, argc, argv, value) ||
          (value != "contains" && value != "matches" && value != "find" && value != "count" &&
           value != "extract" && value != "split")) {
        std::cerr << "--operation requires contains, matches, find, count, extract, or split\n";
        return 2;
      }
      if (value == "contains") parsed.operation = regex_ir::operation::contains();
      if (value == "matches") parsed.operation = regex_ir::operation::matches();
      if (value == "find") parsed.operation = regex_ir::operation::find();
      if (value == "count") parsed.operation = regex_ir::operation::count();
      if (value == "extract") parsed.operation = regex_ir::operation::extract();
      if (value == "split") parsed.operation = regex_ir::operation::split();
    } else if (argument == "--replace") {
      std::string value;
      if (!take_value(index, argc, argv, value)) {
        std::cerr << "--replace requires replacement text\n";
        return 2;
      }
      parsed.operation = regex_ir::operation::replace(std::move(value));
    } else if (argument == "--symbol-prefix") {
      if (!take_value(index, argc, argv, parsed.symbol_prefix)) {
        std::cerr << "--symbol-prefix requires a name\n";
        return 2;
      }
    } else if (argument == "--execute-function") {
      if (!take_value(index, argc, argv, parsed.execute_function)) {
        std::cerr << "--execute-function requires a name\n";
        return 2;
      }
    } else if (argument == "--no-prefix-filter") {
      parsed.prefix_filter = false;
    } else if (argument == "--no-branch-hints") {
      parsed.branch_hints = false;
    } else if (argument == "-i" || argument == "--ignore-case") {
      parsed.compile.case_insensitive = true;
    } else if (argument == "-m" || argument == "--multiline") {
      parsed.compile.multiline = true;
    } else if (argument == "-s" || argument == "--dot-all") {
      parsed.compile.dot_all = true;
    } else if (argument == "--unicode-classes") {
      parsed.compile.ascii_classes = false;
    } else if (argument == "--extended-newline") {
      parsed.compile.extended_newline = true;
    } else if (argument == "--bytes") {
      parsed.compile.characters = regex_ir::character_mode::BYTES;
    } else if (!argument.empty() && argument.front() == '-') {
      std::cerr << "unknown option: " << argument << '\n';
      return 2;
    } else if (has_pattern) {
      std::cerr << "provide exactly one PATTERN (quote patterns containing spaces)\n";
      return 2;
    } else {
      pattern     = argument;
      has_pattern = true;
    }
  }

  if (!has_pattern) {
    std::cerr << "missing PATTERN\n";
    usage(argv[0]);
    return 2;
  }

  auto compiled = regex_ir::compile(pattern, parsed.operation, parsed.compile);
  if (!compiled) return report(compiled.diagnostics);

  try {
    regex_ir::nvvm_ir_codegen_options codegen;
    codegen.symbol_prefix    = parsed.symbol_prefix;
    codegen.execute_function = parsed.execute_function;
    codegen.prefix_filter    = parsed.prefix_filter;
    codegen.branch_hints     = parsed.branch_hints;
    std::cout << regex_ir::generate_nvvm_ir(*compiled.value, codegen);
  } catch (std::invalid_argument const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
