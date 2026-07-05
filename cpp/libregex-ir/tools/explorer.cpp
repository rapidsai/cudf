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
#include <optional>
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
  regex_ir::compile_options compile = regex_ir::compile_options{};
  regex_ir::operation operation     = regex_ir::operation::contains();
  view_options views                = view_options{};
  std::string replacement           = "";
  std::string symbol_prefix         = "regex_ir_generated";
  std::string execute_function      = "regex_ir_execute";
  bool prefix_filter : 1            = true;
  bool branch_hints  : 1            = true;
};

struct exploration_request {
  std::string pattern              = "";
  std::optional<std::string> input = std::nullopt;
};

std::string_view operation_name(regex_ir::operation_kind kind)
{
  using regex_ir::operation_kind;
  switch (kind) {
    case operation_kind::CONTAINS: return "contains";
    case operation_kind::MATCHES: return "match";
    case operation_kind::COUNT: return "count";
    case operation_kind::EXTRACT: return "extract";
    case operation_kind::FIND: return "find";
    case operation_kind::REPLACE: return "replace";
    case operation_kind::SPLIT: return "split";
  }
  return "unknown";
}

std::string_view operation_abi(regex_ir::operation_kind kind)
{
  using regex_ir::operation_kind;
  switch (kind) {
    case operation_kind::CONTAINS:
    case operation_kind::MATCHES: return "i1(i8* data, i64 size)";
    case operation_kind::FIND: return "i1(i8* data, i64 size, i64* span)";
    case operation_kind::COUNT: return "i64(i8* data, i64 size)";
    case operation_kind::EXTRACT: return "i1(i8* data, i64 size, i64 search_start, i64* captures)";
    case operation_kind::REPLACE: return "i64(i8* data, i64 size, i8* output)";
    case operation_kind::SPLIT: return "i64(i8* data, i64 size, i64* spans)";
  }
  return "unknown";
}

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
  bool succeeded                                     = false;
  std::string automata_ir                            = "";
  std::string automata_graph                         = "";
  std::string instruction_ir                         = "";
  std::string nvvm_ir                                = "";
  std::string diagnostics                            = "";
  std::optional<regex_ir::instruction_ir> executable = std::nullopt;
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
  output.executable = std::move(*instructions.value);
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

std::string render_span(regex_ir::testing::match_span span, std::string_view input)
{
  std::ostringstream rendered;
  rendered << '[' << span.begin << ", " << span.end << ')';
  if (span.begin <= span.end && span.end <= input.size()) {
    rendered << ' ' << std::quoted(std::string(input.substr(span.begin, span.end - span.begin)));
  }
  return rendered.str();
}

std::string render_cpu_result(regex_ir::testing::execution_result const& result,
                              regex_ir::operation_kind operation,
                              std::string_view input)
{
  std::ostringstream rendered;
  rendered << "\n=== CPU interpreter ======================================================\n"
           << "input = " << std::quoted(std::string(input)) << '\n';

  switch (operation) {
    case regex_ir::operation_kind::CONTAINS:
      rendered << "contains = " << std::boolalpha << result.matched << '\n';
      break;
    case regex_ir::operation_kind::MATCHES:
      rendered << "match = " << std::boolalpha << result.matched << '\n';
      break;
    case regex_ir::operation_kind::FIND:
      if (result.matches.empty())
        rendered << "find = no match\n";
      else
        rendered << "find = " << render_span(result.matches.front(), input) << '\n';
      break;
    case regex_ir::operation_kind::COUNT: rendered << "count = " << result.count << '\n'; break;
    case regex_ir::operation_kind::EXTRACT:
      rendered << "extract = " << std::boolalpha << result.matched << '\n';
      for (std::size_t index = 0; index < result.captures.size(); ++index) {
        rendered << '$' << index << " = ";
        if (result.captures[index])
          rendered << render_span(*result.captures[index], input);
        else
          rendered << "unmatched";
        rendered << '\n';
      }
      break;
    case regex_ir::operation_kind::REPLACE:
      rendered << "replace = " << std::quoted(result.replaced) << '\n';
      break;
    case regex_ir::operation_kind::SPLIT:
      rendered << "split = " << result.pieces.size() << " field(s)\n";
      for (std::size_t index = 0; index < result.pieces.size(); ++index) {
        rendered << '[' << index << "] = " << std::quoted(result.pieces[index]) << '\n';
      }
      break;
  }
  return rendered.str();
}

bool explore(exploration_request const& request, explorer_options const& options)
{
  auto output = generate(request.pattern, options);
  if (!output.succeeded) {
    std::cerr << output.diagnostics;
    return false;
  }
  std::cout << "\n=== Regex ================================================================\n"
            << request.pattern << "\n\n"
            << "=== API =================================================================\n"
            << operation_name(options.operation.kind) << " -> "
            << operation_abi(options.operation.kind) << '\n';
  if (options.operation.kind == regex_ir::operation_kind::REPLACE) {
    std::cout << "replacement = " << std::quoted(options.replacement) << '\n';
  }
  std::cout << '\n' << render_selected(output, options.views) << '\n';
  if (request.input) {
    try {
      auto result = regex_ir::testing::execute(*output.executable, *request.input);
      std::cout << render_cpu_result(result, options.operation.kind, *request.input);
    } catch (std::exception const& error) {
      std::cerr << "CPU interpreter error: " << error.what() << '\n';
      return false;
    }
  }
  return true;
}

void usage(char const* executable)
{
  std::cout << std::format(
    R"HELP(usage: {} [OPTIONS] [PATTERN]

With no PATTERN, starts an interactive regex-to-IR console.
Function calls compile the pattern and execute the printed IR with the CPU interpreter:
  contains("[0-9]", 12834)
  find("[a-z]+", "123abc")
  replace("([0-9]+)", "id=42", "<$1>")

The pattern must be quoted. Quoted or unquoted input values are interpreted as text.

Options:
  -o, --operation NAME  contains, match, find, count, extract, replace, or split
  -r, --replacement TXT replacement template for the replace API (`$N` captures)
  --replace TEXT        shorthand for --operation replace --replacement TEXT
  --automata            show only the Automata IR and Thompson ASCII graph
  --ir                  show only optimized Instruction IR
  --nvvm                show only generated NVVM IR
  --all                 show every representation
  --no-automata         hide the Automata IR and graph
  --no-ir               hide optimized Instruction IR
  --no-nvvm             hide generated NVVM IR
  --symbol-prefix NAME  prefix internal generated symbols
  --execute-function N  generated NVVM entry function name
  --no-prefix-filter    disable recursive-fallback prefix filtering
  --no-branch-hints     disable recursive-fallback branch hints
  -i, --ignore-case     enable Unicode case-insensitive matching
  -m, --multiline       enable multiline anchors
  -s, --dot-all         make dot match every line terminator
  --unicode-classes     use Unicode predefined classes
  --extended-newline    recognize CR, NEL, LS, and PS as newlines
  --bytes               use byte mode instead of UTF-8
  -h, --help            show this help

Interactive commands:
  :operation NAME       switch API and regenerate the last pattern
  :replace TEXT         switch to replace and set its replacement template
  :replacement TEXT     update the replacement template
  :show VIEW            show automata, ir, nvvm, or all
  :hide VIEW            hide automata, ir, nvvm, or all
  :status               show the active API, ABI, replacement, and views
  :help                 show interactive help
  :quit                 leave the explorer
)HELP",
    executable);
}

bool set_operation(std::string_view name, explorer_options& options)
{
  if (name == "contains")
    options.operation = regex_ir::operation::contains();
  else if (name == "match" || name == "matches")
    options.operation = regex_ir::operation::matches();
  else if (name == "find")
    options.operation = regex_ir::operation::find();
  else if (name == "count")
    options.operation = regex_ir::operation::count();
  else if (name == "extract")
    options.operation = regex_ir::operation::extract();
  else if (name == "replace")
    options.operation = regex_ir::operation::replace(options.replacement);
  else if (name == "split")
    options.operation = regex_ir::operation::split();
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

std::string_view trim_left(std::string_view value)
{
  auto first = value.find_first_not_of(" \t");
  return first == std::string_view::npos ? std::string_view{} : value.substr(first);
}

std::string_view trim(std::string_view value)
{
  auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) return {};
  auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

enum class call_parse_status : std::uint8_t {
  NOT_A_CALL = 0,
  VALID      = 1,
  INVALID    = 2,
};

struct call_expression {
  call_parse_status status           = call_parse_status::NOT_A_CALL;
  std::string function               = "";
  std::vector<std::string> arguments = std::vector<std::string>{};
  std::string error                  = "";
};

bool is_operation_function(std::string_view name)
{
  return name == "contains" || name == "match" || name == "matches" || name == "find" ||
         name == "count" || name == "extract" || name == "replace" || name == "split";
}

int hexadecimal_digit(char value)
{
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10;
  return -1;
}

bool decode_call_argument(std::string_view raw,
                          bool require_quoted,
                          std::string& decoded,
                          std::string& error)
{
  raw = trim(raw);
  if (raw.empty()) {
    error = "arguments may not be empty";
    return false;
  }

  auto quote  = raw.front();
  auto quoted = quote == '\'' || quote == '"';
  if (!quoted) {
    if (require_quoted) {
      error = "the regex pattern must be a quoted string";
      return false;
    }
    decoded.assign(raw);
    return true;
  }
  if (raw.size() < 2 || raw.back() != quote) {
    error = "unterminated quoted argument";
    return false;
  }

  for (std::size_t index = 1; index + 1 < raw.size(); ++index) {
    auto value = raw[index];
    if (value != '\\') {
      decoded.push_back(value);
      continue;
    }
    if (index + 2 >= raw.size()) {
      error = "quoted argument ends with an escape character";
      return false;
    }
    auto escaped = raw[++index];
    switch (escaped) {
      case 'n': decoded.push_back('\n'); break;
      case 'r': decoded.push_back('\r'); break;
      case 't': decoded.push_back('\t'); break;
      case '0': decoded.push_back('\0'); break;
      case '\\': decoded.push_back('\\'); break;
      case '\'': decoded.push_back('\''); break;
      case '"': decoded.push_back('"'); break;
      case 'x': {
        if (index + 2 >= raw.size() - 1) {
          error = "\\x requires two hexadecimal digits";
          return false;
        }
        auto high = hexadecimal_digit(raw[index + 1]);
        auto low  = hexadecimal_digit(raw[index + 2]);
        if (high < 0 || low < 0) {
          error = "\\x requires two hexadecimal digits";
          return false;
        }
        decoded.push_back(static_cast<char>((high << 4) | low));
        index += 2;
        break;
      }
      default:
        // preserve regex escapes such as \d without requiring C++-style double escaping
        decoded.push_back('\\');
        decoded.push_back(escaped);
        break;
    }
  }
  return true;
}

call_expression parse_call_expression(std::string_view line)
{
  call_expression parsed;
  auto text = trim(line);
  auto open = text.find('(');
  if (open == std::string_view::npos) return parsed;

  auto function = trim(text.substr(0, open));
  if (!is_operation_function(function)) return parsed;
  parsed.status   = call_parse_status::INVALID;
  parsed.function = std::string(function);
  if (text.size() <= open + 1 || text.back() != ')') {
    parsed.error = "function call must end with ')'";
    return parsed;
  }

  auto body = text.substr(open + 1, text.size() - open - 2);
  std::vector<std::string_view> raw_arguments;
  std::size_t argument_begin = 0;
  char quote                 = 0;
  bool escaped               = false;
  for (std::size_t index = 0; index < body.size(); ++index) {
    auto value = body[index];
    if (quote != 0) {
      if (escaped) {
        escaped = false;
      } else if (value == '\\') {
        escaped = true;
      } else if (value == quote) {
        quote = 0;
      }
    } else if (value == '\'' || value == '"') {
      quote = value;
    } else if (value == ',') {
      raw_arguments.push_back(body.substr(argument_begin, index - argument_begin));
      argument_begin = index + 1;
    }
  }
  if (quote != 0) {
    parsed.error = "unterminated quoted argument";
    return parsed;
  }
  raw_arguments.push_back(body.substr(argument_begin));

  auto expected_arguments = function == "replace" ? 3U : 2U;
  if (raw_arguments.size() != expected_arguments) {
    parsed.error = std::format("{} expects {} arguments", function, expected_arguments);
    return parsed;
  }

  parsed.arguments.reserve(raw_arguments.size());
  for (std::size_t index = 0; index < raw_arguments.size(); ++index) {
    std::string decoded;
    if (!decode_call_argument(raw_arguments[index], index == 0, decoded, parsed.error))
      return parsed;
    parsed.arguments.push_back(std::move(decoded));
  }
  parsed.status = call_parse_status::VALID;
  return parsed;
}

std::optional<exploration_request> parse_request(std::string const& text,
                                                 explorer_options& options,
                                                 std::string& error)
{
  auto call = parse_call_expression(text);
  if (call.status == call_parse_status::NOT_A_CALL) {
    return exploration_request{text, std::nullopt};
  }
  if (call.status == call_parse_status::INVALID) {
    error = std::move(call.error);
    return std::nullopt;
  }

  if (call.function == "replace") options.replacement = call.arguments[2];
  if (!set_operation(call.function, options)) {
    error = "unknown regex operation";
    return std::nullopt;
  }
  return exploration_request{std::move(call.arguments[0]), std::move(call.arguments[1])};
}

void print_status(explorer_options const& options)
{
  std::cout << "api: " << operation_name(options.operation.kind)
            << "\nabi: " << operation_abi(options.operation.kind) << "\nviews:";
  if (options.views.automata) std::cout << " automata";
  if (options.views.ir) std::cout << " ir";
  if (options.views.nvvm) std::cout << " nvvm";
  if (!options.views.automata && !options.views.ir && !options.views.nvvm) std::cout << " none";
  if (options.operation.kind == regex_ir::operation_kind::REPLACE) {
    std::cout << "\nreplacement: " << std::quoted(options.replacement);
  }
  std::cout << '\n';
}

bool set_view(view_options& views, std::string_view name, bool enabled)
{
  if (name == "all") {
    views.automata = enabled;
    views.ir       = enabled;
    views.nvvm     = enabled;
  } else if (name == "automata" || name == "graph") {
    views.automata = enabled;
  } else if (name == "ir") {
    views.ir = enabled;
  } else if (name == "nvvm") {
    views.nvvm = enabled;
  } else {
    return false;
  }
  views.positive_selection = true;
  return true;
}

bool handle_command(std::string_view line,
                    explorer_options& options,
                    std::optional<exploration_request> const& last_request,
                    bool& quit)
{
  auto body       = trim_left(line.substr(1));
  auto separator  = body.find_first_of(" \t");
  auto command    = body.substr(0, separator);
  auto argument   = separator == std::string_view::npos ? std::string_view{}
                                                        : trim_left(body.substr(separator + 1));
  bool regenerate = false;

  if (command == "quit" || command == "q") {
    quit = true;
    return true;
  }
  if (command == "help" || command == "h") {
    usage("regex-ir-explorer");
    return true;
  }
  if (command == "status") {
    print_status(options);
    return true;
  }
  if (command == "operation" || command == "op") {
    if (argument.empty() || !set_operation(argument, options)) {
      std::cerr << "unknown API; use contains, match, find, count, extract, replace, or split\n";
      return true;
    }
    regenerate = true;
  } else if (command == "replace") {
    options.replacement = std::string(argument);
    options.operation   = regex_ir::operation::replace(options.replacement);
    regenerate          = true;
  } else if (command == "replacement") {
    options.replacement = std::string(argument);
    if (options.operation.kind == regex_ir::operation_kind::REPLACE) {
      options.operation = regex_ir::operation::replace(options.replacement);
      regenerate        = true;
    }
  } else if (command == "show" || command == "hide") {
    if (!set_view(options.views, argument, command == "show")) {
      std::cerr << "unknown view; use automata, ir, nvvm, or all\n";
      return true;
    }
    regenerate = true;
  } else if (set_operation(command, options)) {
    regenerate = true;
  } else {
    std::cerr << "unknown command: :" << command << " (:help lists commands)\n";
    return true;
  }

  print_status(options);
  return !regenerate || !last_request || explore(*last_request, options);
}

int line_console(explorer_options options)
{
  std::cout << "Regex IR explorer — enter a pattern, an API call, or :help for commands\n";
  std::string line;
  std::optional<exploration_request> last_request;
  bool succeeded = true;
  bool quit      = false;
  while (std::cout << "regex-ir[" << operation_name(options.operation.kind) << "]> " << std::flush,
         std::getline(std::cin, line)) {
    if (!line.empty() && line.front() == ':') {
      succeeded = handle_command(line, options, last_request, quit) && succeeded;
      if (quit) break;
      continue;
    }
    std::string error;
    auto request = parse_request(line, options, error);
    if (!request) {
      std::cerr << "invalid call: " << error << '\n';
      succeeded = false;
      continue;
    }
    last_request = std::move(*request);
    succeeded    = explore(*last_request, options) && succeeded;
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
      if (++index >= argc || !set_operation(argv[index], options)) {
        std::cerr << "invalid or missing operation\n";
        return 2;
      }
    } else if (argument == "--replace" || argument == "-r" || argument == "--replacement") {
      if (++index >= argc) {
        std::cerr << argument << " requires replacement text\n";
        return 2;
      }
      options.replacement = argv[index];
      options.operation   = regex_ir::operation::replace(options.replacement);
    } else if (!argument.empty() && argument.front() == '-') {
      std::cerr << "unknown option: " << argument << '\n';
      return 2;
    } else {
      patterns.emplace_back(argument);
    }
  }

  if (!patterns.empty()) {
    bool succeeded = true;
    for (auto& pattern : patterns) {
      std::string error;
      auto request = parse_request(pattern, options, error);
      if (!request) {
        std::cerr << "invalid call: " << error << '\n';
        succeeded = false;
        continue;
      }
      succeeded = explore(*request, options) && succeeded;
    }
    return succeeded ? 0 : 1;
  }

  return line_console(options);
}
