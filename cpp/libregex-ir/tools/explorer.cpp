/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <regex_ir.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <format>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct view_options {
  bool automata           : 1 = true;
  bool ir                 : 1 = true;
  bool nvvm               : 1 = true;
  bool positive_selection : 1 = false;
};

enum class view_kind : std::uint8_t {
  AUTOMATA = 0,
  IR       = 1,
  NVVM     = 2,
};

struct explorer_options {
  regex_ir::compile_options compile = {};
  regex_ir::operation operation     = regex_ir::operation::contains();
  view_options views                = {};
  std::string symbol_prefix         = "regex_ir_generated";
  std::string execute_function      = "regex_ir_execute";
  bool prefix_filter : 1            = true;
  bool branch_hints  : 1            = true;
};

char const* state_name(regex_ir::automata_state_kind kind)
{
  using regex_ir::automata_state_kind;
  switch (kind) {
    case automata_state_kind::JUMP: return "jump";
    case automata_state_kind::BRANCH: return "branch";
    case automata_state_kind::CONSUME: return "consume";
    case automata_state_kind::ASSERTION: return "assertion";
    case automata_state_kind::CAPTURE: return "capture";
    case automata_state_kind::ACCEPT: return "accept";
  }
  return "unknown";
}

std::string codepoint(char32_t value)
{
  std::ostringstream out;
  out << "U+" << std::uppercase << std::hex << std::setfill('0') << std::setw(4)
      << static_cast<std::uint32_t>(value);
  return out.str();
}

std::string predicate_description(regex_ir::character_predicate const& predicate)
{
  if (predicate.recognized == regex_ir::predicate_class::ANY) {
    return predicate.matches_newline ? "any character" : "any except CR/LF";
  }
  std::ostringstream out;
  if (predicate.negated) out << "not ";
  out << '[';
  for (std::size_t index = 0; index < predicate.ranges.size(); ++index) {
    if (index != 0) out << ' ';
    auto range = predicate.ranges[index];
    out << codepoint(range.first);
    if (range.first != range.last) out << '-' << codepoint(range.last);
  }
  out << ']';
  return out.str();
}

std::string state_description(regex_ir::automata_state const& state)
{
  using regex_ir::automata_state_kind;
  std::ostringstream out;
  out << '%' << state.id << ' ' << state_name(state.kind);
  if (state.kind == automata_state_kind::CONSUME) {
    out << " " << predicate_description(state.predicate);
  } else if (state.kind == automata_state_kind::ASSERTION) {
    out << " kind=" << static_cast<int>(state.assertion);
  } else if (state.kind == automata_state_kind::CAPTURE) {
    out << ' ' << (state.capture == regex_ir::capture_action::BEGIN ? "begin" : "end") << '['
        << state.capture_index << ']';
  }
  return out.str();
}

std::string render_thompson_ascii(regex_ir::automata_ir const& ir)
{
  std::ostringstream out;
  out << "entry = %" << ir.entry << ", accept = %" << ir.accept << '\n';
  std::vector<regex_ir::state_id> order;
  std::vector<bool> seen(ir.states.size());
  if (ir.entry < ir.states.size()) {
    order.push_back(ir.entry);
    seen[ir.entry] = true;
  }
  for (std::size_t cursor = 0; cursor < order.size(); ++cursor) {
    auto edges = ir.states[order[cursor]].edges;
    std::stable_sort(edges.begin(), edges.end(), [](auto& left, auto& right) {
      return left.priority < right.priority;
    });
    for (auto edge : edges) {
      if (edge.target < seen.size() && !seen[edge.target]) {
        seen[edge.target] = true;
        order.push_back(edge.target);
      }
    }
  }
  for (std::size_t state = 0; state < seen.size(); ++state) {
    if (!seen[state]) order.push_back(static_cast<regex_ir::state_id>(state));
  }

  for (auto state_id : order) {
    auto& state      = ir.states[state_id];
    auto description = state_description(state);
    auto width       = std::max<std::size_t>(description.size() + 2, 20);
    out << (state.id == ir.entry ? "-> " : "   ") << '+' << std::string(width, '-') << "+\n";
    out << "   | " << description << std::string(width - description.size() - 1, ' ') << "|";
    if (state.id == ir.accept) out << "  <accept>";
    out << '\n';
    out << "   +" << std::string(width, '-') << "+\n";

    auto edges = state.edges;
    std::stable_sort(edges.begin(), edges.end(), [](auto& left, auto& right) {
      return left.priority < right.priority;
    });
    for (std::size_t index = 0; index < edges.size(); ++index) {
      auto edge = edges[index];
      out << "      " << (index + 1 == edges.size() ? "`" : "|") << "-- p" << edge.priority
          << " --> %" << edge.target << '\n';
    }
    if (!state.edges.empty()) out << '\n';
  }
  return out.str();
}

std::string diagnostics_text(std::string const& pattern,
                             std::vector<regex_ir::diagnostic> const& diagnostics)
{
  std::ostringstream out;
  for (auto& diagnostic : diagnostics) {
    out << "error at pattern byte " << diagnostic.span.offset << ": " << diagnostic.message << '\n'
        << "  " << pattern << '\n'
        << "  " << std::string(diagnostic.span.offset, ' ') << '^' << '\n';
  }
  return out.str();
}

struct exploration_output {
  bool succeeded             = false;
  std::string automata_ir    = {};
  std::string automata_graph = {};
  std::string instruction_ir = {};
  std::string nvvm_ir        = {};
  std::string diagnostics    = {};
};

exploration_output generate(std::string const& pattern, explorer_options const& options)
{
  exploration_output output;
  auto automata = regex_ir::compile_automata(pattern, options.compile);
  if (!automata) {
    output.diagnostics = diagnostics_text(pattern, automata.diagnostics);
    return output;
  }

  auto lowered = regex_ir::lower(*automata.value, options.operation);
  if (!lowered) {
    output.diagnostics = diagnostics_text(pattern, lowered.diagnostics);
    return output;
  }
  auto instructions = regex_ir::optimize(std::move(*lowered.value));
  if (!instructions) {
    output.diagnostics = diagnostics_text(pattern, instructions.diagnostics);
    return output;
  }

  output.succeeded      = true;
  output.automata_ir    = regex_ir::to_string(*automata.value);
  output.automata_graph = render_thompson_ascii(*automata.value);
  output.instruction_ir = regex_ir::to_string(*instructions.value);
  regex_ir::nvvm_ir_codegen_options nvvm_options;
  nvvm_options.symbol_prefix    = options.symbol_prefix;
  nvvm_options.execute_function = options.execute_function;
  nvvm_options.prefix_filter    = options.prefix_filter;
  nvvm_options.branch_hints     = options.branch_hints;
  try {
    output.nvvm_ir = regex_ir::generate_nvvm_ir(*instructions.value, nvvm_options);
  } catch (std::invalid_argument const& error) {
    output.nvvm_ir = std::format("; NVVM IR unavailable: {}\n", error.what());
  }
  return output;
}

std::string render_selected(exploration_output const& output, view_options const& views)
{
  if (!output.succeeded) return output.diagnostics;
  std::ostringstream rendered;
  if (views.automata) {
    rendered << "=== Automata IR printer ==================================================\n"
             << output.automata_ir
             << "\n=== Thompson automata ASCII =============================================\n"
             << output.automata_graph;
  }
  if (views.ir) {
    rendered << "=== Optimized Instruction IR printer ====================================\n"
             << output.instruction_ir;
  }
  if (views.nvvm) {
    rendered << "\n=== Generated NVVM IR ====================================================\n"
             << output.nvvm_ir;
  }
  if (!views.automata && !views.ir && !views.nvvm) { rendered << "No output views selected.\n"; }
  return rendered.str();
}

bool explore(std::string const& pattern, explorer_options const& options)
{
  auto output = generate(pattern, options);
  if (!output.succeeded) {
    std::cerr << output.diagnostics;
    return false;
  }
  std::cout << "\n=== Regex ================================================================\n"
            << pattern << "\n\n"
            << render_selected(output, options.views) << '\n';
  return true;
}

void usage(char const* executable)
{
  std::cout << "usage: " << executable << " [OPTIONS] [PATTERN]\n\n"
            << "With no PATTERN, starts an interactive line-oriented regex explorer.\n\n"
            << "Options:\n"
            << "  -o, --operation NAME  contains, matches, find, count, extract, or split\n"
            << "  --replace TEXT        select replace mode with replacement TEXT\n"
            << "  --automata            show only the Automata IR and Thompson ASCII graph\n"
            << "  --ir                  show only optimized Instruction IR\n"
            << "  --nvvm                show only generated NVVM IR\n"
            << "  --all                 show every representation\n"
            << "  --no-automata         hide the Automata IR and graph\n"
            << "  --no-ir               hide optimized Instruction IR\n"
            << "  --no-nvvm             hide generated NVVM IR\n"
            << "  --symbol-prefix NAME  prefix internal generated symbols\n"
            << "  --execute-function N  generated NVVM entry function name\n"
            << "  --no-prefix-filter    disable recursive-fallback prefix filtering\n"
            << "  --no-branch-hints     disable recursive-fallback branch hints\n"
            << "  -i, --ignore-case     enable Unicode case-insensitive matching\n"
            << "  -m, --multiline       enable multiline anchors\n"
            << "  -s, --dot-all         make dot match every line terminator\n"
            << "  --unicode-classes     use Unicode predefined classes\n"
            << "  --extended-newline    recognize CR, NEL, LS, and PS as newlines\n"
            << "  --bytes               use byte mode instead of UTF-8\n"
            << "  -h, --help             show this help\n\n"
            << "Interactive commands: :help, :quit\n";
}

bool set_operation(std::string_view name, regex_ir::operation& operation)
{
  if (name == "contains")
    operation = regex_ir::operation::contains();
  else if (name == "matches")
    operation = regex_ir::operation::matches();
  else if (name == "find")
    operation = regex_ir::operation::find();
  else if (name == "count")
    operation = regex_ir::operation::count();
  else if (name == "extract")
    operation = regex_ir::operation::extract();
  else if (name == "split")
    operation = regex_ir::operation::split();
  else
    return false;
  return true;
}

void select_view(view_options& views, view_kind selected)
{
  if (!views.positive_selection) {
    views.automata           = false;
    views.ir                 = false;
    views.nvvm               = false;
    views.positive_selection = true;
  }
  switch (selected) {
    case view_kind::AUTOMATA: views.automata = true; break;
    case view_kind::IR: views.ir = true; break;
    case view_kind::NVVM: views.nvvm = true; break;
  }
}

int line_console(explorer_options const& options)
{
  std::cout << "Regex IR regex explorer (:help for help, :quit to exit)\n";
  std::string pattern;
  bool succeeded = true;
  while (std::cout << "regex> " << std::flush, std::getline(std::cin, pattern)) {
    if (pattern == ":quit" || pattern == ":q") break;
    if (pattern == ":help") {
      usage("regex-ir-explorer");
      continue;
    }
    succeeded = explore(pattern, options) && succeeded;
  }
  return succeeded ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv)
{
  explorer_options options;
  std::vector<std::string> patterns;
  for (int index = 1; index < argc; ++index) {
    std::string_view argument = argv[index];
    if (argument == "-h" || argument == "--help") {
      usage(argv[0]);
      return 0;
    }
    if (argument == "-i" || argument == "--ignore-case") {
      options.compile.case_insensitive = true;
    } else if (argument == "-m" || argument == "--multiline") {
      options.compile.multiline = true;
    } else if (argument == "-s" || argument == "--dot-all") {
      options.compile.dot_all = true;
    } else if (argument == "--unicode-classes") {
      options.compile.ascii_classes = false;
    } else if (argument == "--extended-newline") {
      options.compile.extended_newline = true;
    } else if (argument == "--bytes") {
      options.compile.characters = regex_ir::character_mode::BYTES;
    } else if (argument == "--automata") {
      select_view(options.views, view_kind::AUTOMATA);
    } else if (argument == "--ir") {
      select_view(options.views, view_kind::IR);
    } else if (argument == "--nvvm") {
      select_view(options.views, view_kind::NVVM);
    } else if (argument == "--all") {
      options.views = {true, true, true, true};
    } else if (argument == "--no-automata") {
      options.views.automata = false;
    } else if (argument == "--no-ir") {
      options.views.ir = false;
    } else if (argument == "--no-nvvm") {
      options.views.nvvm = false;
    } else if (argument == "--symbol-prefix") {
      if (++index >= argc) {
        std::cerr << "--symbol-prefix requires a name\n";
        return 2;
      }
      options.symbol_prefix = argv[index];
    } else if (argument == "--execute-function") {
      if (++index >= argc) {
        std::cerr << "--execute-function requires a name\n";
        return 2;
      }
      options.execute_function = argv[index];
    } else if (argument == "--no-prefix-filter") {
      options.prefix_filter = false;
    } else if (argument == "--no-branch-hints") {
      options.branch_hints = false;
    } else if (argument == "-o" || argument == "--operation") {
      if (++index >= argc || !set_operation(argv[index], options.operation)) {
        std::cerr << "invalid or missing operation\n";
        return 2;
      }
    } else if (argument == "--replace") {
      if (++index >= argc) {
        std::cerr << "--replace requires replacement text\n";
        return 2;
      }
      options.operation = regex_ir::operation::replace(argv[index]);
    } else if (!argument.empty() && argument.front() == '-') {
      std::cerr << "unknown option: " << argument << '\n';
      return 2;
    } else {
      patterns.emplace_back(argument);
    }
  }

  if (!patterns.empty()) {
    bool succeeded = true;
    for (auto& pattern : patterns)
      succeeded = explore(pattern, options) && succeeded;
    return succeeded ? 0 : 1;
  }

  return line_console(options);
}
