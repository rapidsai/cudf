/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fmt/format.h>
#include <regex_ir.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <iterator>
#include <locale>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

// parser and automata IR

namespace regex_ir {
namespace {

#include "regex_unicode_data.inc"

struct parse_failure : std::exception {};

enum class node_kind : std::uint8_t {
  EMPTY       = 0,
  PREDICATE   = 1,
  CONCATENATE = 2,
  ALTERNATE   = 3,
  REPEAT      = 4,
  GROUP       = 5,
  ASSERTION   = 6,
};

struct node {
  node_kind kind                              = node_kind::EMPTY;
  source_span source                          = {};
  character_predicate predicate               = {};
  assertion_kind assertion                    = assertion_kind::BEGIN_INPUT;
  std::vector<std::unique_ptr<node>> children = {};
  std::uint32_t minimum                       = 0;
  std::uint32_t maximum                       = 0;
  bool greedy                                 = true;
  std::uint32_t capture_index                 = 0;
  bool capturing                              = false;
};

bool can_consume_character(node const& value)
{
  switch (value.kind) {
    case node_kind::PREDICATE: return true;
    case node_kind::EMPTY:
    case node_kind::ASSERTION: return false;
    case node_kind::GROUP: return can_consume_character(*value.children.front());
    case node_kind::CONCATENATE:
    case node_kind::ALTERNATE:
      return std::any_of(value.children.begin(), value.children.end(), [](auto& child) {
        return can_consume_character(*child);
      });
    case node_kind::REPEAT: return can_consume_character(*value.children.front());
  }
  return false;
}

void normalize_ranges(character_predicate& predicate)
{
  if (predicate.ranges.empty()) { return; }
  std::sort(predicate.ranges.begin(), predicate.ranges.end(), [](auto& lhs, auto& rhs) {
    return lhs.first < rhs.first || (lhs.first == rhs.first && lhs.last < rhs.last);
  });
  std::vector<codepoint_range> merged;
  for (auto range : predicate.ranges) {
    if (merged.empty() || static_cast<std::uint64_t>(range.first) >
                            static_cast<std::uint64_t>(merged.back().last) + 1U) {
      merged.push_back(range);
    } else if (range.last > merged.back().last) {
      merged.back().last = range.last;
    }
  }
  predicate.ranges = std::move(merged);
}

template <std::size_t Size>
void append_unicode_ranges(character_predicate& predicate, unicode_data_range const (&ranges)[Size])
{
  predicate.ranges.reserve(predicate.ranges.size() + Size);
  for (unicode_data_range range : ranges) {
    predicate.ranges.push_back(
      {static_cast<char32_t>(range.first), static_cast<char32_t>(range.last)});
  }
}

std::vector<codepoint_range> complement_ranges(std::vector<codepoint_range> ranges)
{
  character_predicate normalized;
  normalized.ranges = std::move(ranges);
  normalize_ranges(normalized);
  std::vector<codepoint_range> result;
  char32_t begin = U'\0';
  for (codepoint_range range : normalized.ranges) {
    if (begin < range.first) result.push_back({begin, static_cast<char32_t>(range.first - 1)});
    if (range.last == static_cast<char32_t>(0x10FFFF)) return result;
    begin = static_cast<char32_t>(range.last + 1);
  }
  result.push_back({begin, static_cast<char32_t>(0x10FFFF)});
  return result;
}

void remove_codepoint(std::vector<codepoint_range>& ranges, char32_t value)
{
  std::vector<codepoint_range> result;
  result.reserve(ranges.size() + 1);
  for (codepoint_range range : ranges) {
    if (value < range.first || value > range.last) {
      result.push_back(range);
      continue;
    }
    if (range.first < value) result.push_back({range.first, static_cast<char32_t>(value - 1)});
    if (value < range.last) result.push_back({static_cast<char32_t>(value + 1), range.last});
  }
  ranges = std::move(result);
}

char32_t swap_case(char32_t value)
{
  static std::locale locale{"C.UTF-8"};
  auto wide = static_cast<wchar_t>(value);
  return static_cast<char32_t>(std::isupper(wide, locale) ? std::tolower(wide, locale)
                                                          : std::toupper(wide, locale));
}

void add_case_pair(character_predicate& predicate, char32_t first, char32_t last)
{
  predicate.ranges.push_back({first, last});
  auto swapped_first = swap_case(first);
  auto swapped_last  = swap_case(last);
  if (swapped_first <= swapped_last) predicate.ranges.push_back({swapped_first, swapped_last});
}

class parser {
 public:
  parser(std::string_view pattern, compile_options const& options)
    : pattern_(pattern), options_(options)
  {
  }

  std::unique_ptr<node> parse()
  {
    if (pattern_.size() > options_.limits.max_pattern_bytes) {
      fail(
        diagnostic_code::RESOURCE_LIMIT, {0, pattern_.size()}, "pattern exceeds max_pattern_bytes");
    }
    auto expression = parse_alternation();
    if (position_ != pattern_.size()) {
      fail(diagnostic_code::UNEXPECTED_TOKEN, {position_, 1}, "unexpected token");
    }
    return expression;
  }

  std::vector<diagnostic> diagnostics = {};
  std::uint32_t capture_count         = 0;

 private:
  [[noreturn]] void fail(diagnostic_code code, source_span span, std::string message)
  {
    diagnostics.push_back({code, span, std::move(message)});
    throw parse_failure{};
  }

  bool at_end() const noexcept { return position_ >= pattern_.size(); }
  char peek() const noexcept { return at_end() ? '\0' : pattern_[position_]; }
  char take()
  {
    if (at_end()) {
      fail(diagnostic_code::UNEXPECTED_END, {position_, 0}, "unexpected end of pattern");
    }
    return pattern_[position_++];
  }

  bool consume(char value)
  {
    if (peek() != value) { return false; }
    ++position_;
    return true;
  }

  std::unique_ptr<node> make(node_kind kind, std::size_t start)
  {
    auto result    = std::make_unique<node>();
    result->kind   = kind;
    result->source = {start, position_ - start};
    return result;
  }

  std::unique_ptr<node> parse_alternation()
  {
    auto lhs = parse_concatenation();
    while (consume('|')) {
      auto rhs       = parse_concatenation();
      auto alternate = make(node_kind::ALTERNATE, lhs->source.offset);
      alternate->children.push_back(std::move(lhs));
      alternate->children.push_back(std::move(rhs));
      alternate->source.length = position_ - alternate->source.offset;
      lhs                      = std::move(alternate);
    }
    return lhs;
  }

  std::unique_ptr<node> parse_concatenation()
  {
    auto start = position_;
    std::vector<std::unique_ptr<node>> children;
    while (!at_end() && peek() != ')' && peek() != '|') {
      children.push_back(parse_quantified());
    }
    if (children.empty()) { return make(node_kind::EMPTY, start); }
    if (children.size() == 1) { return std::move(children.front()); }
    auto result      = make(node_kind::CONCATENATE, start);
    result->children = std::move(children);
    return result;
  }

  std::uint32_t parse_decimal()
  {
    std::uint64_t value{};
    std::size_t count{};
    while (std::isdigit(static_cast<unsigned char>(peek())) != 0) {
      value = value * 10U + static_cast<unsigned>(take() - '0');
      ++count;
      if (value > options_.limits.max_repeat) {
        fail(diagnostic_code::RESOURCE_LIMIT,
             {position_ - count, count},
             "repeat bound exceeds max_repeat");
      }
    }
    if (count == 0) {
      fail(diagnostic_code::INVALID_QUANTIFIER, {position_, 0}, "expected repeat bound");
    }
    return static_cast<std::uint32_t>(value);
  }

  std::unique_ptr<node> parse_quantified()
  {
    auto atom = parse_atom();
    if (at_end()) { return atom; }

    auto start = atom->source.offset;
    std::uint32_t minimum{};
    std::uint32_t maximum{};
    bool quantified = true;
    if (consume('*')) {
      minimum = 0;
      maximum = unbounded_repeat;
    } else if (consume('+')) {
      minimum = 1;
      maximum = unbounded_repeat;
    } else if (consume('?')) {
      minimum = 0;
      maximum = 1;
    } else if (peek() == '{' && position_ + 1 < pattern_.size() &&
               std::isdigit(static_cast<unsigned char>(pattern_[position_ + 1])) != 0) {
      ++position_;
      minimum = parse_decimal();
      maximum = minimum;
      if (consume(',')) { maximum = peek() == '}' ? unbounded_repeat : parse_decimal(); }
      if (!consume('}')) {
        fail(diagnostic_code::INVALID_QUANTIFIER, {position_, 0}, "unterminated quantifier");
      }
      if (maximum != unbounded_repeat && maximum < minimum) {
        fail(diagnostic_code::INVALID_QUANTIFIER,
             {start, position_ - start},
             "repeat maximum is smaller than minimum");
      }
    } else {
      quantified = false;
    }

    if (!quantified) { return atom; }
    if (!can_consume_character(*atom)) {
      fail(diagnostic_code::INVALID_QUANTIFIER,
           {start, position_ - start},
           "zero-width assertions cannot be repeated");
    }
    auto result = make(node_kind::REPEAT, start);
    result->children.push_back(std::move(atom));
    result->minimum       = minimum;
    result->maximum       = maximum;
    result->greedy        = !consume('?');
    result->source.length = position_ - start;
    if (peek() == '*' || peek() == '+' || peek() == '?' || peek() == '{') {
      fail(diagnostic_code::INVALID_QUANTIFIER, {position_, 1}, "multiple repeat operators");
    }
    return result;
  }

  char32_t decode_literal(std::size_t& length)
  {
    if (options_.characters == character_mode::BYTES) {
      length = 1;
      return static_cast<unsigned char>(pattern_[position_]);
    }
    auto first = static_cast<unsigned char>(pattern_[position_]);
    if (first < 0x80U) {
      length = 1;
      return first;
    }
    std::size_t count = first < 0xE0U ? 2 : (first < 0xF0U ? 3 : 4);
    if (position_ + count > pattern_.size() || first < 0xC2U || first > 0xF4U) {
      fail(diagnostic_code::INVALID_ESCAPE, {position_, 1}, "invalid UTF-8 in pattern");
    }
    char32_t value = first & (count == 2 ? 0x1FU : (count == 3 ? 0x0FU : 0x07U));
    for (std::size_t index = 1; index < count; ++index) {
      auto next = static_cast<unsigned char>(pattern_[position_ + index]);
      if ((next & 0xC0U) != 0x80U) {
        fail(diagnostic_code::INVALID_ESCAPE, {position_, count}, "invalid UTF-8 in pattern");
      }
      value = static_cast<char32_t>((value << 6U) | (next & 0x3FU));
    }
    if ((count == 3 && value < 0x800U) || (count == 4 && value < 0x10000U) || value > 0x10FFFFU ||
        (value >= 0xD800U && value <= 0xDFFFU)) {
      fail(diagnostic_code::INVALID_ESCAPE, {position_, count}, "invalid UTF-8 scalar value");
    }
    length = count;
    return value;
  }

  char32_t parse_hex(std::size_t digits, std::size_t escape_start)
  {
    char32_t value{};
    for (std::size_t index = 0; index < digits; ++index) {
      if (at_end()) {
        fail(diagnostic_code::INVALID_ESCAPE,
             {escape_start, position_ - escape_start},
             "truncated hexadecimal escape");
      }
      char const digit = take();
      value <<= 4U;
      if (digit >= '0' && digit <= '9') {
        value |= static_cast<char32_t>(digit - '0');
      } else if (digit >= 'a' && digit <= 'f') {
        value |= static_cast<char32_t>(digit - 'a' + 10);
      } else if (digit >= 'A' && digit <= 'F') {
        value |= static_cast<char32_t>(digit - 'A' + 10);
      } else {
        fail(diagnostic_code::INVALID_ESCAPE, {position_ - 1, 1}, "invalid hexadecimal digit");
      }
    }
    return value;
  }

  char32_t parse_octal(char first)
  {
    char32_t value     = static_cast<char32_t>(first - '0');
    std::size_t digits = 1;
    while (digits < 3 && peek() >= '0' && peek() <= '7') {
      value = static_cast<char32_t>((value << 3U) | static_cast<char32_t>(take() - '0'));
      ++digits;
    }
    return value;
  }

  character_predicate predefined(char value)
  {
    character_predicate result;
    auto append_ascii = [&](char kind) {
      if (kind == 'd') result.ranges = {{U'0', U'9'}};
      if (kind == 'w') { result.ranges = {{U'0', U'9'}, {U'A', U'Z'}, {U'_', U'_'}, {U'a', U'z'}}; }
      if (kind == 's') result.ranges = {{U'\t', U' '}};
    };
    auto append_unicode = [&](char kind) {
      if (kind == 'd') append_unicode_ranges(result, cudf_unicode_digit_ranges);
      if (kind == 'w') {
        append_unicode_ranges(result, cudf_unicode_word_ranges);
        result.ranges.push_back({U'_', U'_'});
      }
      if (kind == 's') append_unicode_ranges(result, cudf_unicode_space_ranges);
    };
    auto base = static_cast<char>(std::tolower(static_cast<unsigned char>(value)));
    if (options_.ascii_classes) {
      append_ascii(base);
    } else {
      append_unicode(base);
    }
    normalize_ranges(result);
    bool negative = value == 'D' || value == 'W' || value == 'S';
    if (negative) {
      result.ranges = complement_ranges(std::move(result.ranges));
      if (!options_.ascii_classes && (value == 'D' || value == 'W')) {
        remove_codepoint(result.ranges, U'\n');
      }
    }
    switch (value) {
      case 'd':
      case 'D':
        result.recognized = value == 'd' ? predicate_class::DIGIT : predicate_class::NOT_DIGIT;
        break;
      case 'w':
      case 'W':
        result.recognized = value == 'w' ? predicate_class::WORD : predicate_class::NOT_WORD;
        break;
      case 's':
      case 'S':
        result.recognized = value == 's' ? predicate_class::SPACE : predicate_class::NOT_SPACE;
        break;
      default: break;
    }
    return result;
  }

  char32_t escaped_literal(char value, std::size_t start)
  {
    switch (value) {
      case 'a': return U'\a';
      case 'b': return U'\b';
      case 'n': return U'\n';
      case 'r': return U'\r';
      case 't': return U'\t';
      case 'f': return U'\f';
      case 'v': return U'\v';
      case 'x': return parse_hex(2, start);
      case 'u': return parse_hex(4, start);
      default: return static_cast<unsigned char>(value);
    }
  }

  std::unique_ptr<node> parse_escape(bool in_class)
  {
    auto start = position_ - 1;
    if (at_end()) { fail(diagnostic_code::INVALID_ESCAPE, {start, 1}, "trailing backslash"); }
    char const value = take();
    if (!in_class &&
        (value == 'b' || value == 'B' || value == 'A' || value == 'Z' || value == 'z')) {
      auto result = make(node_kind::ASSERTION, start);
      if (value == 'b') result->assertion = assertion_kind::WORD_BOUNDARY;
      if (value == 'B') result->assertion = assertion_kind::NOT_WORD_BOUNDARY;
      if (value == 'A') result->assertion = assertion_kind::BEGIN_INPUT;
      if (value == 'Z' || value == 'z') result->assertion = assertion_kind::END_INPUT;
      return result;
    }
    auto result = make(node_kind::PREDICATE, start);
    if (value == 'd' || value == 'D' || value == 'w' || value == 'W' || value == 's' ||
        value == 'S') {
      result->predicate = predefined(value);
    } else {
      if (std::isalpha(static_cast<unsigned char>(value)) != 0 && value != 'a' && value != 'b' &&
          value != 'f' && value != 'n' && value != 'r' && value != 't' && value != 'u' &&
          value != 'v' && value != 'x') {
        fail(diagnostic_code::INVALID_ESCAPE, {start, 2}, "unknown alphabetic escape");
      }
      bool three_digit_octal = value >= '0' && value <= '7' && position_ + 1 < pattern_.size() &&
                               pattern_[position_] >= '0' && pattern_[position_] <= '7' &&
                               pattern_[position_ + 1] >= '0' && pattern_[position_ + 1] <= '7';
      if (value >= '1' && value <= '9' && !three_digit_octal) {
        fail(diagnostic_code::UNSUPPORTED_FEATURE, {start, 2}, "backreferences are not supported");
      }
      auto literal = three_digit_octal ? parse_octal(value) : escaped_literal(value, start);
      if (options_.case_insensitive) {
        add_case_pair(result->predicate, literal, literal);
      } else {
        result->predicate.ranges.push_back({literal, literal});
      }
      normalize_ranges(result->predicate);
    }
    result->source.length = position_ - start;
    return result;
  }

  std::unique_ptr<node> parse_class()
  {
    auto start                = position_ - 1;
    auto result               = make(node_kind::PREDICATE, start);
    result->predicate.negated = consume('^');
    bool first                = true;
    bool closed               = false;
    while (!at_end()) {
      if (peek() == ']' && !first) {
        ++position_;
        closed = true;
        break;
      }
      first = false;
      char32_t lower{};
      if (consume('\\')) {
        auto escaped = parse_escape(true);
        if (escaped->predicate.ranges.size() != 1 ||
            escaped->predicate.ranges.front().first != escaped->predicate.ranges.front().last) {
          result->predicate.ranges.insert(result->predicate.ranges.end(),
                                          escaped->predicate.ranges.begin(),
                                          escaped->predicate.ranges.end());
          continue;
        }
        lower = escaped->predicate.ranges.front().first;
      } else {
        std::size_t length{};
        lower = decode_literal(length);
        position_ += length;
      }

      char32_t upper = lower;
      if (peek() == '-' && position_ + 1 < pattern_.size() && pattern_[position_ + 1] != ']') {
        ++position_;
        if (consume('\\')) {
          auto escaped = parse_escape(true);
          if (!escaped->predicate.is_singleton()) {
            fail(diagnostic_code::INVALID_CHARACTER_CLASS,
                 {position_, 1},
                 "range endpoint must be a literal");
          }
          upper = escaped->predicate.singleton();
        } else {
          std::size_t length{};
          upper = decode_literal(length);
          position_ += length;
        }
        if (upper < lower) {
          fail(diagnostic_code::INVALID_CHARACTER_CLASS,
               {start, position_ - start},
               "descending character range");
        }
      }
      if (options_.case_insensitive) {
        add_case_pair(result->predicate, lower, upper);
      } else {
        result->predicate.ranges.push_back({lower, upper});
      }
    }
    if (!closed) {
      fail(diagnostic_code::INVALID_CHARACTER_CLASS,
           {start, position_ - start},
           "unterminated character class");
    }
    if (result->predicate.ranges.empty()) {
      fail(diagnostic_code::INVALID_CHARACTER_CLASS,
           {start, position_ - start},
           "empty character class");
    }
    normalize_ranges(result->predicate);
    result->source.length = position_ - start;
    return result;
  }

  std::unique_ptr<node> parse_atom()
  {
    auto start       = position_;
    char const value = take();
    if (value == '(') {
      bool capturing = true;
      if (consume('?')) {
        if (consume(':')) {
          capturing = false;
        } else {
          fail(diagnostic_code::UNSUPPORTED_FEATURE,
               {start, position_ - start + 1},
               "lookaround and inline group extensions are not supported");
        }
      }
      if (++depth_ > options_.limits.max_nesting) {
        fail(diagnostic_code::RESOURCE_LIMIT, {start, 1}, "group nesting exceeds max_nesting");
      }
      std::uint32_t capture{};
      if (capturing) {
        if (capture_count >= options_.limits.max_captures) {
          fail(diagnostic_code::RESOURCE_LIMIT, {start, 1}, "capture count exceeds max_captures");
        }
        capture = ++capture_count;
      }
      auto child = parse_alternation();
      if (!consume(')')) {
        fail(
          diagnostic_code::UNMATCHED_PARENTHESIS, {start, position_ - start}, "unterminated group");
      }
      --depth_;
      auto group = make(node_kind::GROUP, start);
      group->children.push_back(std::move(child));
      group->capturing     = capturing;
      group->capture_index = capture;
      return group;
    }
    if (value == '[') { return parse_class(); }
    if (value == '\\') { return parse_escape(false); }
    if (value == '.') {
      auto result                        = make(node_kind::PREDICATE, start);
      result->predicate.recognized       = predicate_class::ANY;
      result->predicate.matches_newline  = options_.dot_all;
      result->predicate.extended_newline = options_.extended_newline;
      return result;
    }
    if (value == '^' || value == '$') {
      auto result       = make(node_kind::ASSERTION, start);
      result->assertion = value == '^' ? assertion_kind::BEGIN_LINE : assertion_kind::END_LINE;
      return result;
    }
    if (value == ')' || value == '|' || value == '*' || value == '+' || value == '?' ||
        (value == '{' && std::isdigit(static_cast<unsigned char>(peek())) != 0)) {
      fail(diagnostic_code::UNEXPECTED_TOKEN, {start, 1}, "unexpected metacharacter");
    }

    --position_;
    std::size_t length{};
    auto literal = decode_literal(length);
    position_ += length;
    auto result = make(node_kind::PREDICATE, start);
    if (options_.case_insensitive) {
      add_case_pair(result->predicate, literal, literal);
    } else {
      result->predicate.ranges.push_back({literal, literal});
    }
    normalize_ranges(result->predicate);
    result->source.length = length;
    return result;
  }

  std::string_view pattern_;
  compile_options const& options_;
  std::size_t position_ = 0;
  std::size_t depth_    = 0;
};

struct patch_reference {
  state_id state   = invalid_state;
  std::size_t edge = 0;
};
struct fragment {
  state_id start                    = invalid_state;
  std::vector<patch_reference> outs = {};
};

class thompson_builder {
 public:
  thompson_builder(std::string_view pattern, compile_options const& options)
  {
    ir.pattern = std::string(pattern);
    ir.options = options;
  }

  automata_ir ir                      = {};
  std::vector<diagnostic> diagnostics = {};

  fragment build(node const& expression)
  {
    switch (expression.kind) {
      case node_kind::EMPTY: return make_empty(expression.source);
      case node_kind::PREDICATE: return make_predicate(expression);
      case node_kind::ASSERTION: return make_assertion(expression);
      case node_kind::GROUP: return make_group(expression);
      case node_kind::CONCATENATE: return make_concatenate(expression);
      case node_kind::ALTERNATE: return make_alternate(expression);
      case node_kind::REPEAT: return make_repeat(expression);
    }
    return {};
  }

  void finish(fragment expression, std::uint32_t captures)
  {
    auto accept = add_state(automata_state_kind::ACCEPT, {ir.pattern.size(), 0});
    patch(expression.outs, accept);
    auto entry = add_state(automata_state_kind::JUMP, {0, 0});
    add_edge(entry, expression.start, 0);
    ir.entry         = entry;
    ir.accept        = accept;
    ir.capture_count = captures;
  }

 private:
  state_id add_state(automata_state_kind kind, source_span source)
  {
    if (ir.states.size() >= ir.options.limits.max_states) {
      diagnostics.push_back(
        {diagnostic_code::RESOURCE_LIMIT, source, "state count exceeds max_states"});
      throw parse_failure{};
    }
    auto id = static_cast<state_id>(ir.states.size());
    automata_state state;
    state.id     = id;
    state.kind   = kind;
    state.source = source;
    ir.states.push_back(std::move(state));
    return id;
  }

  std::size_t add_edge(state_id from, state_id to, std::uint32_t priority)
  {
    if (transition_count_ >= ir.options.limits.max_transitions) {
      diagnostics.push_back({diagnostic_code::RESOURCE_LIMIT,
                             ir.states[from].source,
                             "transition count exceeds limit"});
      throw parse_failure{};
    }
    ++transition_count_;
    ir.states[from].edges.push_back({to, priority});
    return ir.states[from].edges.size() - 1;
  }

  patch_reference add_open_edge(state_id from, std::uint32_t priority)
  {
    return {from, add_edge(from, invalid_state, priority)};
  }

  void patch(std::vector<patch_reference> const& references, state_id target)
  {
    // open edges let Thompson fragments be joined without rebuilding either fragment
    for (auto reference : references) {
      ir.states[reference.state].edges[reference.edge].target = target;
    }
  }

  fragment make_empty(source_span span)
  {
    auto state = add_state(automata_state_kind::JUMP, span);
    return {state, {add_open_edge(state, 0)}};
  }

  fragment make_predicate(node const& expression)
  {
    auto state                 = add_state(automata_state_kind::CONSUME, expression.source);
    ir.states[state].predicate = expression.predicate;
    return {state, {add_open_edge(state, 0)}};
  }

  fragment make_assertion(node const& expression)
  {
    auto state                 = add_state(automata_state_kind::ASSERTION, expression.source);
    ir.states[state].assertion = expression.assertion;
    return {state, {add_open_edge(state, 0)}};
  }

  fragment make_group(node const& expression)
  {
    if (!expression.capturing) return build(*expression.children.front());

    auto begin                     = add_state(automata_state_kind::CAPTURE, expression.source);
    ir.states[begin].capture       = capture_action::BEGIN;
    ir.states[begin].capture_index = expression.capture_index;

    auto inner = build(*expression.children.front());
    add_edge(begin, inner.start, 0);

    auto end                     = add_state(automata_state_kind::CAPTURE, expression.source);
    ir.states[end].capture       = capture_action::END;
    ir.states[end].capture_index = expression.capture_index;
    patch(inner.outs, end);
    return {begin, {add_open_edge(end, 0)}};
  }

  fragment concatenate(fragment left, fragment right)
  {
    patch(left.outs, right.start);
    return {left.start, std::move(right.outs)};
  }

  fragment make_concatenate(node const& expression)
  {
    if (expression.children.empty()) return make_empty(expression.source);
    auto result = build(*expression.children.front());
    for (std::size_t index = 1; index < expression.children.size(); ++index) {
      result = concatenate(std::move(result), build(*expression.children[index]));
    }
    return result;
  }

  fragment alternate(fragment left, fragment right, source_span span)
  {
    auto branch = add_state(automata_state_kind::BRANCH, span);
    add_edge(branch, left.start, 0);
    add_edge(branch, right.start, 1);
    left.outs.insert(left.outs.end(),
                     std::make_move_iterator(right.outs.begin()),
                     std::make_move_iterator(right.outs.end()));
    return {branch, std::move(left.outs)};
  }

  fragment make_alternate(node const& expression)
  {
    if (expression.children.empty()) return make_empty(expression.source);
    auto result = build(*expression.children.front());
    for (std::size_t index = 1; index < expression.children.size(); ++index) {
      result = alternate(std::move(result), build(*expression.children[index]), expression.source);
    }
    return result;
  }

  fragment optional(node const& expression, source_span span, bool greedy)
  {
    auto inner  = build(expression);
    auto branch = add_state(automata_state_kind::BRANCH, span);
    // lower priorities are attempted first, so swapping them implements lazy quantifiers
    auto take_priority = greedy ? 0U : 1U;
    auto exit_priority = greedy ? 1U : 0U;
    add_edge(branch, inner.start, take_priority);
    inner.outs.push_back(add_open_edge(branch, exit_priority));
    return {branch, std::move(inner.outs)};
  }

  fragment star(node const& expression, source_span span, bool greedy)
  {
    auto inner           = build(expression);
    auto branch          = add_state(automata_state_kind::BRANCH, span);
    auto repeat_priority = greedy ? 0U : 1U;
    auto exit_priority   = greedy ? 1U : 0U;
    add_edge(branch, inner.start, repeat_priority);
    patch(inner.outs, branch);
    return {branch, {add_open_edge(branch, exit_priority)}};
  }

  fragment make_repeat(node const& expression)
  {
    auto& repeated = *expression.children.front();
    std::optional<fragment> result;
    auto append = [&](fragment next) {
      if (result) {
        *result = concatenate(std::move(*result), std::move(next));
      } else {
        result = std::move(next);
      }
    };

    for (std::uint32_t count = 0; count < expression.minimum; ++count) {
      append(build(repeated));
    }

    if (expression.maximum == unbounded_repeat) {
      append(star(repeated, expression.source, expression.greedy));
    } else {
      for (std::uint32_t count = expression.minimum; count < expression.maximum; ++count) {
        append(optional(repeated, expression.source, expression.greedy));
      }
    }

    return result ? std::move(*result) : make_empty(expression.source);
  }

  std::size_t transition_count_ = 0;
};

char const* state_name(automata_state_kind kind)
{
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

char const* assertion_name(assertion_kind kind)
{
  switch (kind) {
    case assertion_kind::BEGIN_INPUT: return "begin_input";
    case assertion_kind::END_INPUT: return "end_input";
    case assertion_kind::WORD_BOUNDARY: return "word_boundary";
    case assertion_kind::NOT_WORD_BOUNDARY: return "not_word_boundary";
    case assertion_kind::BEGIN_LINE: return "begin_line";
    case assertion_kind::END_LINE: return "end_line";
  }
  return "unknown";
}

}  // namespace

bool character_predicate::matches(char32_t value) const noexcept
{
  if (recognized == predicate_class::ANY) {
    if (matches_newline) return true;
    if (!extended_newline) return value != U'\n';
    return value != U'\n' && value != U'\r' && value != static_cast<char32_t>(0x85) &&
           value != static_cast<char32_t>(0x2028) && value != static_cast<char32_t>(0x2029);
  }
  bool contained = false;
  for (auto range : ranges) {
    if (value >= range.first && value <= range.last) {
      contained = true;
      break;
    }
  }
  return negated ? !contained : contained;
}

bool character_predicate::is_singleton() const noexcept
{
  return !negated && recognized == predicate_class::NONE && ranges.size() == 1 &&
         ranges.front().first == ranges.front().last;
}

char32_t character_predicate::singleton() const noexcept
{
  return is_singleton() ? ranges.front().first : U'\0';
}

std::vector<diagnostic> verify(automata_ir const& ir)
{
  std::vector<diagnostic> result;
  auto invalid = [&](source_span span, std::string message) {
    result.push_back({diagnostic_code::INVALID_AUTOMATA_IR, span, std::move(message)});
  };
  if (ir.entry >= ir.states.size()) invalid({}, "entry state is invalid");
  if (ir.accept >= ir.states.size()) invalid({}, "accept state is invalid");
  for (std::size_t index = 0; index < ir.states.size(); ++index) {
    auto& state = ir.states[index];
    if (state.id != index) invalid(state.source, "state ID does not match storage index");
    for (auto edge : state.edges) {
      if (edge.target >= ir.states.size()) invalid(state.source, "edge target is invalid");
    }
    if (state.kind == automata_state_kind::ACCEPT && !state.edges.empty()) {
      invalid(state.source, "accept state has outgoing edges");
    }
    if (state.kind == automata_state_kind::CONSUME && state.edges.size() != 1) {
      invalid(state.source, "consume state must have one edge");
    }
    if ((state.kind == automata_state_kind::JUMP || state.kind == automata_state_kind::ASSERTION ||
         state.kind == automata_state_kind::CAPTURE) &&
        state.edges.size() != 1) {
      invalid(state.source, "linear epsilon state must have one edge");
    }
    if (state.kind == automata_state_kind::BRANCH && state.edges.size() < 2) {
      invalid(state.source, "branch state must have at least two edges");
    }
    if (state.kind == automata_state_kind::CAPTURE &&
        (state.capture_index == 0 || state.capture_index > ir.capture_count)) {
      invalid(state.source, "capture index is out of range");
    }
  }
  return result;
}

std::string to_string(automata_ir const& ir)
{
  std::ostringstream out;
  out << "automata pattern=" << std::quoted(ir.pattern) << " entry=" << ir.entry
      << " accept=" << ir.accept << " captures=" << ir.capture_count << '\n';
  for (auto& state : ir.states) {
    out << '%' << state.id << ' ' << state_name(state.kind);
    if (state.kind == automata_state_kind::CONSUME) {
      if (state.predicate.is_singleton()) {
        out << " U+" << std::hex << std::uppercase
            << static_cast<std::uint32_t>(state.predicate.singleton()) << std::dec;
      } else {
        out << " ranges=" << state.predicate.ranges.size();
        if (state.predicate.negated) out << " negated";
      }
    } else if (state.kind == automata_state_kind::CAPTURE) {
      out << ' ' << (state.capture == capture_action::BEGIN ? "begin" : "end") << '['
          << state.capture_index << ']';
    } else if (state.kind == automata_state_kind::ASSERTION) {
      out << ' ' << assertion_name(state.assertion);
    }
    for (auto edge : state.edges) {
      out << " -> %" << edge.target << "(p" << edge.priority << ')';
    }
    out << '\n';
  }
  return out.str();
}

automata_result compile_automata(std::string_view pattern, compile_options const& options)
{
  parser parse_pattern(pattern, options);
  std::unique_ptr<node> expression;
  try {
    expression = parse_pattern.parse();
  } catch (parse_failure const&) {
    return {std::nullopt, std::move(parse_pattern.diagnostics)};
  }

  thompson_builder builder(pattern, options);
  try {
    auto fragment = builder.build(*expression);
    builder.finish(std::move(fragment), parse_pattern.capture_count);
  } catch (parse_failure const&) {
    auto diagnostics = std::move(builder.diagnostics);
    diagnostics.insert(
      diagnostics.end(), parse_pattern.diagnostics.begin(), parse_pattern.diagnostics.end());
    return {std::nullopt, std::move(diagnostics)};
  }

  auto diagnostics = verify(builder.ir);
  if (!diagnostics.empty()) { return {std::nullopt, std::move(diagnostics)}; }
  return {std::move(builder.ir), {}};
}

}  // namespace regex_ir

/*
 * test-only CPU interpreter for Regex IR Instruction IR.
 * SPDX-License-Identifier: Apache-2.0
 */

namespace regex_ir::testing {
namespace {

struct decoded {
  char32_t value     = U'\0';
  std::size_t length = 0;
};

std::optional<decoded> decode(std::string_view input, std::size_t position, character_mode mode)
{
  if (position >= input.size()) return std::nullopt;
  auto first = static_cast<unsigned char>(input[position]);
  if (mode == character_mode::BYTES || first < 0x80U) return decoded{first, 1};

  std::size_t count = first < 0xE0U ? 2 : (first < 0xF0U ? 3 : 4);
  if (first < 0xC2U || first > 0xF4U || position + count > input.size()) {
    return decoded{first, 1};
  }
  char32_t value = first & (count == 2 ? 0x1FU : (count == 3 ? 0x0FU : 0x07U));
  for (std::size_t index = 1; index < count; ++index) {
    auto next = static_cast<unsigned char>(input[position + index]);
    if ((next & 0xC0U) != 0x80U) return decoded{first, 1};
    value = static_cast<char32_t>((value << 6U) | (next & 0x3FU));
  }
  if ((count == 3 && value < 0x800U) || (count == 4 && value < 0x10000U) || value > 0x10FFFFU ||
      (value >= 0xD800U && value <= 0xDFFFU)) {
    return decoded{first, 1};
  }
  return decoded{value, count};
}

std::size_t advance(std::string_view input,
                    std::size_t position,
                    std::uint32_t count,
                    character_mode mode)
{
  for (std::uint32_t index = 0; index < count && position < input.size(); ++index) {
    position += decode(input, position, mode)->length;
  }
  return position;
}

bool can_peek_count(std::string_view input,
                    std::size_t position,
                    std::uint32_t count,
                    character_mode mode)
{
  for (std::uint32_t index = 0; index < count; ++index) {
    auto current = decode(input, position, mode);
    if (!current) return false;
    position += current->length;
  }
  return true;
}

bool is_word(char32_t value, compile_options const& options)
{
  if (value == U'_') return true;
  if (options.ascii_classes || options.characters == character_mode::BYTES) {
    return (value >= U'a' && value <= U'z') || (value >= U'A' && value <= U'Z') ||
           (value >= U'0' && value <= U'9');
  }
  return std::any_of(std::begin(cudf_unicode_word_ranges),
                     std::end(cudf_unicode_word_ranges),
                     [value](unicode_data_range range) {
                       auto codepoint = static_cast<std::uint32_t>(value);
                       return codepoint >= range.first && codepoint <= range.last;
                     });
}

bool is_newline(char32_t value, compile_options const& options)
{
  if (!options.extended_newline) return value == U'\n';
  return value == U'\n' || value == U'\r' || value == static_cast<char32_t>(0x85) ||
         value == static_cast<char32_t>(0x2028) || value == static_cast<char32_t>(0x2029);
}

bool assertion_matches(assertion_kind kind,
                       std::string_view input,
                       std::size_t position,
                       compile_options const& options)
{
  auto current            = decode(input, position, options.characters);
  char32_t previous_value = U'\0';
  bool previous_word{};
  if (position > 0) {
    std::size_t previous = position - 1;
    while (previous > 0 && (static_cast<unsigned char>(input[previous]) & 0xC0U) == 0x80U) {
      --previous;
    }
    auto value     = decode(input, previous, options.characters);
    previous_value = value ? value->value : U'\0';
    previous_word  = value && is_word(value->value, options);
  }
  auto current_value   = current ? current->value : U'\0';
  bool current_newline = current && is_newline(current_value, options);
  bool current_word    = current && is_word(current_value, options);

  switch (kind) {
    case assertion_kind::BEGIN_INPUT: return position == 0;
    case assertion_kind::END_INPUT: return position == input.size();
    case assertion_kind::WORD_BOUNDARY: return previous_word != current_word;
    case assertion_kind::NOT_WORD_BOUNDARY: return previous_word == current_word;
    case assertion_kind::BEGIN_LINE:
      return position == 0 ||
             (options.multiline && is_newline(previous_value, options) &&
              !(options.extended_newline && previous_value == U'\r' && current_value == U'\n'));
    case assertion_kind::END_LINE:
      if (position == input.size()) return true;
      if (!current_newline) return false;
      if (options.extended_newline && current_value == U'\n' && previous_value == U'\r') {
        return false;
      }
      if (options.multiline) return true;
      if (current_value == U'\r' && options.extended_newline) {
        if (position + current->length == input.size()) return true;
        auto next = decode(input, position + current->length, options.characters);
        return next && next->value == U'\n' &&
               position + current->length + next->length == input.size();
      }
      return position + current->length == input.size();
  }
  return false;
}

struct capture_value {
  std::optional<std::size_t> begin = {};
  std::optional<std::size_t> end   = {};
};

struct found_match {
  std::size_t begin                   = 0;
  std::size_t end                     = 0;
  std::vector<capture_value> captures = {};
};

bool run_block(instruction_ir const& ir,
               std::string_view input,
               block_id block_id_value,
               std::size_t position,
               std::vector<capture_value> captures,
               bool require_end,
               std::unordered_set<std::uint64_t>& active,
               found_match& result,
               std::size_t& steps)
{
  if (++steps > 1000000U || block_id_value >= ir.blocks.size()) return false;
  auto key =
    (static_cast<std::uint64_t>(block_id_value) << 32U) ^ static_cast<std::uint64_t>(position);
  if (!active.insert(key).second) return false;

  auto local_position = position;
  auto local_captures = std::move(captures);
  bool accepted{};
  for (auto& item : ir.blocks[block_id_value].instructions) {
    if (auto* peek_op = std::get_if<can_peek>(&item)) {
      if (!can_peek_count(input, local_position, peek_op->characters, ir.options.characters)) {
        active.erase(key);
        return false;
      }
    } else if (auto* character_op = std::get_if<match_character>(&item)) {
      auto current = decode(input, local_position, ir.options.characters);
      if (!current || !character_op->predicate.matches(current->value)) {
        active.erase(key);
        return false;
      }
    } else if (auto* literal_op = std::get_if<match_literal>(&item)) {
      auto cursor  = local_position;
      bool matches = true;
      for (auto expected : literal_op->value) {
        auto current = decode(input, cursor, ir.options.characters);
        if (!current || current->value != expected) {
          matches = false;
          break;
        }
        cursor += current->length;
      }
      if (!matches) {
        active.erase(key);
        return false;
      }
    } else if (auto* advance_op = std::get_if<advance_cursor>(&item)) {
      local_position =
        advance(input, local_position, advance_op->characters, ir.options.characters);
    } else if (auto* assertion_op = std::get_if<test_assertion>(&item)) {
      if (!assertion_matches(assertion_op->kind, input, local_position, ir.options)) {
        active.erase(key);
        return false;
      }
    } else if (auto* capture_op = std::get_if<write_capture>(&item)) {
      if (capture_op->capture_index >= local_captures.size()) {
        active.erase(key);
        return false;
      }
      if (capture_op->action == capture_action::BEGIN) {
        local_captures[capture_op->capture_index].begin = local_position;
        local_captures[capture_op->capture_index].end.reset();
      } else {
        local_captures[capture_op->capture_index].end = local_position;
      }
    } else if (std::holds_alternative<emit_accept>(item)) {
      accepted = true;
    }
  }

  if (accepted) {
    active.erase(key);
    if (require_end && local_position != input.size()) return false;
    result.end      = local_position;
    result.captures = std::move(local_captures);
    return true;
  }

  auto successors = ir.blocks[block_id_value].successors;
  std::stable_sort(successors.begin(), successors.end(), [](auto& lhs, auto& rhs) {
    return lhs.priority < rhs.priority;
  });
  for (auto edge : successors) {
    if (run_block(ir,
                  input,
                  edge.target,
                  local_position,
                  local_captures,
                  require_end,
                  active,
                  result,
                  steps)) {
      active.erase(key);
      return true;
    }
  }
  active.erase(key);
  return false;
}

std::optional<found_match> find_first(instruction_ir const& ir,
                                      std::string_view input,
                                      std::size_t search_start,
                                      bool only_start,
                                      bool require_end)
{
  auto start = search_start;
  while (start <= input.size()) {
    found_match match;
    match.begin = start;
    std::vector<capture_value> captures(ir.capture_count + 1U);
    captures[0].begin = start;
    std::unordered_set<std::uint64_t> active;
    std::size_t steps{};
    if (run_block(ir, input, ir.entry, start, captures, require_end, active, match, steps)) {
      match.captures[0].end = match.end;
      return match;
    }
    if (only_start || start == input.size()) break;
    start = advance(input, start, 1, ir.options.characters);
  }
  return std::nullopt;
}

std::size_t next_search(std::string_view input, found_match const& match, character_mode mode)
{
  if (match.end != match.begin) return match.end;
  return match.end == input.size() ? input.size() + 1U : advance(input, match.end, 1, mode);
}

std::vector<std::optional<match_span>> public_captures(found_match const& match)
{
  std::vector<std::optional<match_span>> result;
  result.reserve(match.captures.size());
  for (auto& capture : match.captures) {
    if (capture.begin && capture.end) {
      result.push_back(match_span{*capture.begin, *capture.end});
    } else {
      result.push_back(std::nullopt);
    }
  }
  return result;
}

void append_replacement(std::string& output,
                        instruction_ir const& ir,
                        std::string_view input,
                        found_match const& match)
{
  for (auto& token : ir.replacement) {
    if (token.type == replacement_token::kind::LITERAL) {
      output += token.literal;
    } else if (token.capture_index < match.captures.size()) {
      auto& capture = match.captures[token.capture_index];
      if (capture.begin && capture.end) {
        output.append(input.substr(*capture.begin, *capture.end - *capture.begin));
      }
    }
  }
}

}  // namespace

execution_result execute(instruction_ir const& ir, std::string_view input)
{
  auto diagnostics = verify(ir);
  if (!diagnostics.empty()) throw std::invalid_argument("cannot execute invalid Instruction IR");

  execution_result output;
  auto kind = ir.selected_operation.kind;
  if (kind == operation_kind::MATCHES) {
    auto match     = find_first(ir, input, 0, true, true);
    output.matched = match.has_value();
    if (match) {
      output.count = 1;
      output.matches.push_back({match->begin, match->end});
      output.captures = public_captures(*match);
    }
    return output;
  }

  if (kind == operation_kind::CONTAINS || kind == operation_kind::FIND ||
      kind == operation_kind::EXTRACT) {
    auto match     = find_first(ir, input, 0, false, false);
    output.matched = match.has_value();
    if (match) {
      output.count = 1;
      output.matches.push_back({match->begin, match->end});
      if (kind == operation_kind::EXTRACT) output.captures = public_captures(*match);
    }
    return output;
  }

  std::size_t search{};
  std::size_t copied{};
  while (search <= input.size()) {
    auto match = find_first(ir, input, search, false, false);
    if (!match) break;
    output.matched = true;
    ++output.count;
    output.matches.push_back({match->begin, match->end});

    if (kind == operation_kind::REPLACE) {
      output.replaced.append(input.substr(copied, match->begin - copied));
      append_replacement(output.replaced, ir, input, *match);
      copied = match->end;
    } else if (kind == operation_kind::SPLIT) {
      output.pieces.emplace_back(input.substr(copied, match->begin - copied));
      copied = match->end;
    }

    auto next = next_search(input, *match, ir.options.characters);
    if (next > input.size()) break;
    search = next;
  }

  if (kind == operation_kind::REPLACE) {
    output.replaced.append(input.substr(copied));
  } else if (kind == operation_kind::SPLIT) {
    output.pieces.emplace_back(input.substr(copied));
  }
  return output;
}

execution_result enumerate(instruction_ir const& ir, std::string_view input)
{
  auto diagnostics = verify(ir);
  if (!diagnostics.empty()) throw std::invalid_argument("cannot execute invalid Instruction IR");

  execution_result output;
  std::size_t search = 0;
  while (search <= input.size()) {
    auto match = find_first(ir, input, search, false, false);
    if (!match) break;
    output.matched = true;
    ++output.count;
    output.matches.push_back({match->begin, match->end});
    output.capture_matches.push_back(public_captures(*match));
    auto next = next_search(input, *match, ir.options.characters);
    if (next > input.size()) break;
    search = next;
  }
  if (!output.capture_matches.empty()) output.captures = output.capture_matches.front();
  return output;
}

}  // namespace regex_ir::testing

// instruction IR lowering

namespace regex_ir {
namespace {

std::vector<diagnostic> parse_replacement(std::string const& replacement,
                                          std::uint32_t capture_count,
                                          std::vector<replacement_token>& output)
{
  std::vector<diagnostic> diagnostics;
  std::string literal;
  auto flush_literal = [&] {
    if (!literal.empty()) {
      output.push_back({replacement_token::kind::LITERAL, std::move(literal), 0});
      literal.clear();
    }
  };

  for (std::size_t position = 0; position < replacement.size();) {
    if (replacement[position] != '$') {
      literal.push_back(replacement[position++]);
      continue;
    }
    auto start = position++;
    if (position < replacement.size() && replacement[position] == '$') {
      literal.push_back('$');
      ++position;
      continue;
    }
    if (position == replacement.size() ||
        std::isdigit(static_cast<unsigned char>(replacement[position])) == 0) {
      diagnostics.push_back({diagnostic_code::INVALID_REPLACEMENT,
                             {start, 1},
                             "dollar must be followed by a capture number or dollar"});
      return diagnostics;
    }
    std::uint64_t capture{};
    while (position < replacement.size() &&
           std::isdigit(static_cast<unsigned char>(replacement[position])) != 0) {
      capture = capture * 10U + static_cast<unsigned>(replacement[position++] - '0');
      if (capture > capture_count) {
        diagnostics.push_back({diagnostic_code::INVALID_REPLACEMENT,
                               {start, position - start},
                               "replacement capture is out of range"});
        return diagnostics;
      }
    }
    flush_literal();
    output.push_back({replacement_token::kind::CAPTURE, {}, static_cast<std::uint32_t>(capture)});
  }
  flush_literal();
  return diagnostics;
}

char const* operation_name(operation_kind kind)
{
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

std::string instruction_name(instruction const& value)
{
  return std::visit(
    [](auto& item) {
      using type = std::decay_t<decltype(item)>;
      std::ostringstream out;
      if constexpr (std::is_same_v<type, can_peek>) {
        out << "can_peek " << item.characters;
      } else if constexpr (std::is_same_v<type, read_character>) {
        out << "read_character";
      } else if constexpr (std::is_same_v<type, match_character>) {
        if (item.predicate.is_singleton()) {
          out << "match U+" << std::hex << std::uppercase
              << static_cast<std::uint32_t>(item.predicate.singleton());
        } else {
          out << "match_class ranges=" << item.predicate.ranges.size();
          if (item.predicate.negated) out << " negated";
        }
      } else if constexpr (std::is_same_v<type, match_literal>) {
        out << "match_literal length=" << item.value.size();
      } else if constexpr (std::is_same_v<type, advance_cursor>) {
        out << "advance " << item.characters;
      } else if constexpr (std::is_same_v<type, test_assertion>) {
        out << "assert " << assertion_name(item.kind);
      } else if constexpr (std::is_same_v<type, write_capture>) {
        out << "capture " << (item.action == capture_action::BEGIN ? "begin " : "end ")
            << item.capture_index;
      } else if constexpr (std::is_same_v<type, emit_accept>) {
        out << "accept";
      }
      return out.str();
    },
    value);
}

}  // namespace

ir_metrics measure(instruction_ir const& ir)
{
  ir_metrics result;
  result.blocks = ir.blocks.size();
  for (auto& block : ir.blocks) {
    if (block.successors.size() > 1) ++result.branches;
    for (auto& item : block.instructions) {
      std::visit(
        [&](auto& instruction_value) {
          using type = std::decay_t<decltype(instruction_value)>;
          if constexpr (std::is_same_v<type, read_character>) {
            ++result.stream_reads;
          } else if constexpr (std::is_same_v<type, match_character>) {
            ++result.predicates;
          } else if constexpr (std::is_same_v<type, match_literal>) {
            ++result.predicates;
            ++result.stream_reads;
            result.literal_codepoints += instruction_value.value.size();
          } else if constexpr (std::is_same_v<type, write_capture>) {
            ++result.capture_writes;
          }
        },
        item);
    }
  }
  return result;
}

std::vector<diagnostic> verify(instruction_ir const& ir)
{
  std::vector<diagnostic> result;
  auto invalid = [&](source_span span, std::string message) {
    result.push_back({diagnostic_code::INVALID_INSTRUCTION_IR, span, std::move(message)});
  };
  if (ir.entry >= ir.blocks.size()) invalid({}, "entry block is invalid");
  if (ir.accept >= ir.blocks.size()) invalid({}, "accept block is invalid");
  for (std::size_t index = 0; index < ir.blocks.size(); ++index) {
    auto& block = ir.blocks[index];
    if (block.id != index) invalid(block.source, "block ID does not match storage index");
    for (auto edge : block.successors) {
      if (edge.target >= ir.blocks.size()) invalid(block.source, "successor target is invalid");
    }
    bool accepting{};
    std::size_t character_tests{};
    std::size_t advances{};
    for (auto& item : block.instructions) {
      if (std::holds_alternative<emit_accept>(item)) accepting = true;
      if (std::holds_alternative<match_character>(item) ||
          std::holds_alternative<match_literal>(item)) {
        ++character_tests;
      }
      if (std::holds_alternative<advance_cursor>(item)) ++advances;
      if (auto* capture = std::get_if<write_capture>(&item);
          capture != nullptr &&
          (capture->capture_index == 0 || capture->capture_index > ir.capture_count)) {
        invalid(block.source, "capture write index is out of range");
      }
    }
    if (accepting && !block.successors.empty()) {
      invalid(block.source, "accept block has successors");
    }
    if (character_tests > 1) invalid(block.source, "block has multiple character tests");
    if (advances > 1) invalid(block.source, "block advances more than once");
  }
  return result;
}

std::string to_string(instruction_ir const& ir)
{
  std::ostringstream out;
  out << "instruction_ir operation=" << operation_name(ir.selected_operation.kind)
      << " pattern=" << std::quoted(ir.pattern) << " entry=^" << ir.entry << " accept=^"
      << ir.accept << " captures=" << ir.capture_count << " scan=" << ir.control.scan_input
      << " require_end=" << ir.control.require_end << " first_only=" << ir.control.first_only
      << '\n';
  for (auto& block : ir.blocks) {
    out << '^' << block.id << ':';
    if (block.instructions.empty()) out << " nop";
    for (auto& item : block.instructions)
      out << ' ' << instruction_name(item) << ';';
    auto edges = block.successors;
    std::stable_sort(
      edges.begin(), edges.end(), [](auto& lhs, auto& rhs) { return lhs.priority < rhs.priority; });
    for (auto edge : edges) {
      out << " -> ^" << edge.target << "(p" << edge.priority << ')';
    }
    out << '\n';
  }
  auto metrics = measure(ir);
  out << "metrics blocks=" << metrics.blocks << " branches=" << metrics.branches
      << " predicates=" << metrics.predicates << " reads=" << metrics.stream_reads
      << " captures=" << metrics.capture_writes
      << " literal_codepoints=" << metrics.literal_codepoints << '\n';
  return out.str();
}

instruction_result lower(automata_ir const& automata, operation const& selected)
{
  auto diagnostics = verify(automata);
  if (!diagnostics.empty()) { return {std::nullopt, std::move(diagnostics)}; }

  instruction_ir result;
  result.pattern            = automata.pattern;
  result.options            = automata.options;
  result.selected_operation = selected;
  switch (selected.kind) {
    case operation_kind::MATCHES:
      result.control = {false, true, true, true, result_shape::BOOLEAN};
      break;
    case operation_kind::CONTAINS:
      result.control = {true, false, true, true, result_shape::BOOLEAN};
      break;
    case operation_kind::FIND:
      result.control = {true, false, true, true, result_shape::MATCH_SPAN};
      break;
    case operation_kind::COUNT:
      result.control = {true, false, false, true, result_shape::MATCH_COUNT};
      break;
    case operation_kind::EXTRACT:
      result.control = {true, false, true, true, result_shape::CAPTURES};
      break;
    case operation_kind::REPLACE:
      result.control = {true, false, false, true, result_shape::REPLACEMENT};
      break;
    case operation_kind::SPLIT:
      result.control = {true, false, false, true, result_shape::SPLIT_FIELDS};
      break;
  }
  result.entry         = automata.entry;
  result.accept        = automata.accept;
  result.capture_count = automata.capture_count;
  result.blocks.reserve(automata.states.size());

  for (auto& state : automata.states) {
    instruction_block block;
    block.id     = state.id;
    block.source = state.source;
    block.successors.reserve(state.edges.size());
    for (auto edge : state.edges)
      block.successors.push_back({edge.target, edge.priority});

    switch (state.kind) {
      case automata_state_kind::JUMP:
      case automata_state_kind::BRANCH: break;
      case automata_state_kind::CONSUME:
        block.instructions.push_back(can_peek{1});
        block.instructions.push_back(read_character{});
        block.instructions.push_back(match_character{state.predicate});
        block.instructions.push_back(advance_cursor{1});
        break;
      case automata_state_kind::ASSERTION:
        block.instructions.push_back(test_assertion{state.assertion});
        break;
      case automata_state_kind::CAPTURE:
        block.instructions.push_back(write_capture{state.capture, state.capture_index});
        break;
      case automata_state_kind::ACCEPT: block.instructions.push_back(emit_accept{}); break;
    }
    result.blocks.push_back(std::move(block));
  }

  if (selected.kind == operation_kind::REPLACE) {
    auto replacement_diagnostics =
      parse_replacement(selected.replacement, result.capture_count, result.replacement);
    diagnostics.insert(diagnostics.end(),
                       std::make_move_iterator(replacement_diagnostics.begin()),
                       std::make_move_iterator(replacement_diagnostics.end()));
  }

  if (!diagnostics.empty()) return {std::nullopt, std::move(diagnostics)};
  result.metrics = measure(result);
  diagnostics    = verify(result);
  if (!diagnostics.empty()) return {std::nullopt, std::move(diagnostics)};
  return {std::move(result), {}};
}

instruction_result compile(std::string_view pattern,
                           operation const& selected,
                           compile_options const& options,
                           optimization_options const& optimization)
{
  auto automata = compile_automata(pattern, options);
  if (!automata) return {std::nullopt, std::move(automata.diagnostics)};
  auto instructions = lower(*automata.value, selected);
  if (!instructions) return instructions;
  return optimize(std::move(*instructions.value), optimization);
}

}  // namespace regex_ir

// instruction IR optimization

namespace regex_ir {
namespace {

bool observes_captures(operation_kind kind)
{
  return kind == operation_kind::EXTRACT || kind == operation_kind::REPLACE;
}

std::optional<char32_t> singleton(instruction_block const& block)
{
  for (auto& item : block.instructions) {
    if (auto* match = std::get_if<match_character>(&item);
        match != nullptr && match->predicate.is_singleton()) {
      return match->predicate.singleton();
    }
  }
  return std::nullopt;
}

void strip_captures(instruction_ir& ir)
{
  if (observes_captures(ir.selected_operation.kind)) return;
  for (auto& block : ir.blocks) {
    block.instructions.erase(std::remove_if(block.instructions.begin(),
                                            block.instructions.end(),
                                            [](instruction const& item) {
                                              return std::holds_alternative<write_capture>(item);
                                            }),
                             block.instructions.end());
  }
}

block_id resolve_empty(instruction_ir const& ir, block_id start)
{
  // stop at cycles because nullable repetition can produce an all-empty component
  std::unordered_set<block_id> visited;
  auto current = start;
  while (current < ir.blocks.size() && visited.insert(current).second) {
    auto& block = ir.blocks[current];
    if (!block.instructions.empty() || block.successors.size() != 1) break;
    current = block.successors.front().target;
  }
  return current;
}

void fold_empty_jumps(instruction_ir& ir)
{
  ir.entry = resolve_empty(ir, ir.entry);
  for (auto& block : ir.blocks) {
    for (auto& edge : block.successors)
      edge.target = resolve_empty(ir, edge.target);
  }
}

void fuse_literals(instruction_ir& ir, std::size_t limit)
{
  if (limit < 2) return;
  std::vector<std::size_t> incoming(ir.blocks.size());
  for (auto& block : ir.blocks) {
    for (auto edge : block.successors) {
      if (edge.target < incoming.size()) ++incoming[edge.target];
    }
  }

  for (auto& block : ir.blocks) {
    auto first = singleton(block);
    if (!first || block.successors.size() != 1) continue;

    std::u32string value{*first};
    auto next = block.successors.front().target;
    std::unordered_set<block_id> visited{block.id};
    // a single incoming edge makes it safe to consume the candidate into this block
    while (value.size() < limit && next < ir.blocks.size() && incoming[next] == 1 &&
           visited.insert(next).second) {
      auto& candidate = ir.blocks[next];
      auto character  = singleton(candidate);
      if (!character || candidate.successors.size() != 1) break;
      value.push_back(*character);
      next = candidate.successors.front().target;
    }
    if (value.size() < 2) continue;

    block.instructions.clear();
    block.instructions.push_back(can_peek{static_cast<std::uint32_t>(value.size())});
    block.instructions.push_back(match_literal{std::move(value)});
    block.instructions.push_back(advance_cursor{
      static_cast<std::uint32_t>(std::get<match_literal>(block.instructions[1]).value.size())});
    block.successors = {{next, 0}};
  }
}

void remove_unreachable(instruction_ir& ir)
{
  if (ir.entry >= ir.blocks.size()) return;
  std::vector<bool> reachable(ir.blocks.size());
  std::vector<block_id> work{ir.entry};
  reachable[ir.entry] = true;
  while (!work.empty()) {
    auto current = work.back();
    work.pop_back();
    for (auto edge : ir.blocks[current].successors) {
      if (edge.target < reachable.size() && !reachable[edge.target]) {
        reachable[edge.target] = true;
        work.push_back(edge.target);
      }
    }
  }

  std::vector<block_id> remap(ir.blocks.size(), invalid_block);
  std::vector<instruction_block> blocks;
  blocks.reserve(ir.blocks.size());
  for (std::size_t old = 0; old < ir.blocks.size(); ++old) {
    if (!reachable[old]) continue;
    remap[old] = static_cast<block_id>(blocks.size());
    auto block = std::move(ir.blocks[old]);
    block.id   = static_cast<block_id>(blocks.size());
    blocks.push_back(std::move(block));
  }
  // rewrite dense IDs only after every old-to-new mapping has been established
  for (auto& block : blocks) {
    for (auto& edge : block.successors)
      edge.target = remap[edge.target];
  }
  ir.entry  = remap[ir.entry];
  ir.accept = remap[ir.accept];
  ir.blocks = std::move(blocks);
}

}  // namespace

instruction_result optimize(instruction_ir ir, optimization_options const& options)
{
  auto diagnostics = verify(ir);
  if (!diagnostics.empty()) return {std::nullopt, std::move(diagnostics)};

  if (options.strip_unobserved_captures) strip_captures(ir);
  if (options.fold_epsilon_jumps) fold_empty_jumps(ir);
  if (options.fuse_literals) fuse_literals(ir, options.literal_fusion_limit);
  if (options.fold_epsilon_jumps) fold_empty_jumps(ir);
  if (options.remove_unreachable) remove_unreachable(ir);

  ir.metrics  = measure(ir);
  diagnostics = verify(ir);
  if (!diagnostics.empty()) return {std::nullopt, std::move(diagnostics)};
  return {std::move(ir), {}};
}

}  // namespace regex_ir

// device IR generation

namespace regex_ir {
namespace {

class source_buffer {
 public:
  template <typename... Args>
  void append(fmt::format_string<Args...> format, Args&&... args)
  {
    value_ += fmt::format(format, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void line(fmt::format_string<Args...> format, Args&&... args)
  {
    append(format, std::forward<Args>(args)...);
    value_ += '\n';
  }

  void blank() { value_ += '\n'; }
  [[nodiscard]] std::string take() { return std::move(value_); }

 private:
  std::string value_ = {};
};

void require_identifier(std::string_view value, std::string_view field)
{
  auto first_is_valid = [](unsigned char character) {
    return std::isalpha(character) != 0 || character == '_';
  };
  auto rest_is_valid = [&](unsigned char character) {
    return first_is_valid(character) || std::isdigit(character) != 0;
  };
  if (value.empty() || !first_is_valid(static_cast<unsigned char>(value.front())) ||
      !std::all_of(value.begin() + 1, value.end(), [&](char character) {
        return rest_is_valid(static_cast<unsigned char>(character));
      })) {
    throw std::invalid_argument(fmt::format("{} must be a valid source identifier", field));
  }
  if (value.starts_with("llvm.") || value.starts_with("nvvm.")) {
    throw std::invalid_argument(fmt::format("{} uses a reserved identifier", field));
  }
}

void require_codegen_ir(instruction_ir const& ir)
{
  auto diagnostics = verify(ir);
  if (!diagnostics.empty()) throw std::invalid_argument("cannot generate code from invalid IR");
}

std::string nvvm_symbol(std::string_view prefix, std::string_view suffix)
{
  return fmt::format("{}_{}", prefix, suffix);
}

struct deterministic_nfa_node {
  character_predicate predicate    = {};
  std::vector<std::size_t> targets = {};
  bool consumes : 1                = false;
  bool accepts  : 1                = false;
};

struct deterministic_interval {
  std::uint32_t first    = 0;
  std::uint32_t last     = 0;
  std::uint16_t class_id = 0;
};

struct deterministic_machine {
  std::array<std::uint16_t, 256> byte_classes           = {};
  std::vector<deterministic_interval> unicode_intervals = {};
  std::vector<std::uint16_t> transitions                = {};
  std::uint16_t initial_state                           = 0;
  std::uint16_t class_count                             = 0;
  std::uint16_t state_count                             = 0;
  bool scan_input : 1                                   = false;
};

void set_machine_bit(std::vector<std::uint64_t>& bits, std::size_t index)
{
  bits[index / 64] |= std::uint64_t{1} << (index % 64);
}

bool machine_bit(std::vector<std::uint64_t> const& bits, std::size_t index)
{
  return (bits[index / 64] & (std::uint64_t{1} << (index % 64))) != 0;
}

character_predicate singleton_predicate(char32_t value)
{
  character_predicate result;
  result.ranges.push_back({value, value});
  return result;
}

std::optional<deterministic_machine> make_deterministic_machine(instruction_ir const& ir)
{
  if (ir.selected_operation.kind != operation_kind::CONTAINS &&
      ir.selected_operation.kind != operation_kind::MATCHES) {
    return std::nullopt;
  }

  std::vector<std::size_t> block_starts(ir.blocks.size());
  std::vector<std::size_t> block_lengths(ir.blocks.size(), 1);
  std::size_t node_count = 0;
  for (auto& block : ir.blocks) {
    match_character const* match  = nullptr;
    match_literal const* literal  = nullptr;
    can_peek const* peek          = nullptr;
    advance_cursor const* advance = nullptr;
    bool accepts                  = false;
    for (auto& item : block.instructions) {
      if (std::holds_alternative<test_assertion>(item)) return std::nullopt;
      if (auto* candidate = std::get_if<match_character>(&item)) match = candidate;
      if (auto* candidate = std::get_if<match_literal>(&item)) literal = candidate;
      if (auto* candidate = std::get_if<can_peek>(&item)) peek = candidate;
      if (auto* candidate = std::get_if<advance_cursor>(&item)) advance = candidate;
      if (std::holds_alternative<emit_accept>(item)) accepts = true;
    }
    if (match != nullptr && literal != nullptr) return std::nullopt;
    if ((match != nullptr || literal != nullptr) && accepts) return std::nullopt;
    if (match != nullptr && (peek == nullptr || peek->characters != 1 || advance == nullptr ||
                             advance->characters != 1)) {
      return std::nullopt;
    }
    if (literal != nullptr) {
      if (literal->value.empty() || peek == nullptr || peek->characters != literal->value.size() ||
          advance == nullptr || advance->characters != literal->value.size()) {
        return std::nullopt;
      }
      block_lengths[block.id] = literal->value.size();
    }
    block_starts[block.id] = node_count;
    node_count += block_lengths[block.id];
  }
  if (node_count == 0) return std::nullopt;

  std::vector<deterministic_nfa_node> nodes(node_count);
  for (auto& block : ir.blocks) {
    auto start                   = block_starts[block.id];
    match_character const* match = nullptr;
    match_literal const* literal = nullptr;
    for (auto& item : block.instructions) {
      if (auto* candidate = std::get_if<match_character>(&item)) match = candidate;
      if (auto* candidate = std::get_if<match_literal>(&item)) literal = candidate;
      if (std::holds_alternative<emit_accept>(item)) nodes[start].accepts = true;
    }

    auto append_successors = [&](deterministic_nfa_node& node) {
      for (auto edge : block.successors)
        node.targets.push_back(block_starts[edge.target]);
    };

    if (match != nullptr) {
      nodes[start].predicate = match->predicate;
      nodes[start].consumes  = true;
      append_successors(nodes[start]);
    } else if (literal != nullptr) {
      for (std::size_t index = 0; index < literal->value.size(); ++index) {
        auto& node     = nodes[start + index];
        node.predicate = singleton_predicate(literal->value[index]);
        node.consumes  = true;
        if (index + 1 < literal->value.size()) {
          node.targets.push_back(start + index + 1);
        } else {
          append_successors(node);
        }
      }
    } else {
      append_successors(nodes[start]);
    }
  }

  auto bit_count  = nodes.size() + 1;
  auto word_count = (bit_count + 63) / 64;
  auto accept_bit = nodes.size();
  auto empty_bits = [&] { return std::vector<std::uint64_t>(word_count); };
  auto closure    = [&](std::vector<std::uint64_t> const& seeds) {
    auto result  = empty_bits();
    auto visited = empty_bits();
    std::vector<std::size_t> work;
    for (std::size_t index = 0; index < nodes.size(); ++index) {
      if (machine_bit(seeds, index)) work.push_back(index);
    }
    while (!work.empty()) {
      auto index = work.back();
      work.pop_back();
      if (machine_bit(visited, index)) continue;
      set_machine_bit(visited, index);
      auto& node = nodes[index];
      if (node.accepts) set_machine_bit(result, accept_bit);
      if (node.consumes) {
        set_machine_bit(result, index);
        continue;
      }
      for (auto target : node.targets)
        work.push_back(target);
    }
    return result;
  };

  auto entry_seeds = empty_bits();
  set_machine_bit(entry_seeds, block_starts[ir.entry]);
  auto start_state = closure(entry_seeds);

  constexpr std::uint32_t unicode_limit = 0x110000;
  std::vector<std::uint32_t> boundaries{0, 256, unicode_limit};
  for (auto& node : nodes) {
    if (!node.consumes) continue;
    if (node.predicate.recognized == predicate_class::ANY && !node.predicate.matches_newline) {
      boundaries.insert(boundaries.end(), {10, 11, 13, 14});
    }
    for (auto range : node.predicate.ranges) {
      auto first = static_cast<std::uint32_t>(range.first);
      auto last  = static_cast<std::uint32_t>(range.last);
      if (first < unicode_limit) boundaries.push_back(first);
      if (last < unicode_limit - 1) boundaries.push_back(last + 1);
    }
  }
  std::sort(boundaries.begin(), boundaries.end());
  boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());

  std::map<std::vector<std::uint64_t>, std::uint16_t> class_ids;
  std::vector<std::uint32_t> representatives;
  std::vector<deterministic_interval> intervals;
  for (std::size_t index = 0; index + 1 < boundaries.size(); ++index) {
    auto first = boundaries[index];
    auto last  = boundaries[index + 1] - 1;
    if (first > last || first >= unicode_limit) continue;
    auto signature = empty_bits();
    for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
      if (nodes[node_index].consumes &&
          nodes[node_index].predicate.matches(static_cast<char32_t>(first))) {
        set_machine_bit(signature, node_index);
      }
    }
    auto existing          = class_ids.find(signature);
    std::uint16_t class_id = 0;
    if (existing == class_ids.end()) {
      if (class_ids.size() >= 32767) return std::nullopt;
      class_id = static_cast<std::uint16_t>(class_ids.size());
      class_ids.emplace(std::move(signature), class_id);
      representatives.push_back(first);
    } else {
      class_id = existing->second;
    }
    intervals.push_back({first, last, class_id});
  }
  if (class_ids.empty()) return std::nullopt;

  deterministic_machine machine;
  machine.class_count        = static_cast<std::uint16_t>(class_ids.size());
  machine.scan_input         = ir.control.scan_input;
  std::size_t interval_index = 0;
  for (std::size_t value = 0; value < machine.byte_classes.size(); ++value) {
    while (interval_index + 1 < intervals.size() && value > intervals[interval_index].last)
      ++interval_index;
    machine.byte_classes[value] = intervals[interval_index].class_id;
  }
  for (auto interval : intervals) {
    if (interval.last < 256) continue;
    interval.first = std::max(interval.first, 256U);
    if (!machine.unicode_intervals.empty() &&
        machine.unicode_intervals.back().class_id == interval.class_id &&
        machine.unicode_intervals.back().last + 1 == interval.first) {
      machine.unicode_intervals.back().last = interval.last;
    } else {
      machine.unicode_intervals.push_back(interval);
    }
  }

  constexpr std::size_t max_dfa_states  = 32767;
  constexpr std::size_t max_table_items = 4 * 1024 * 1024;
  std::map<std::vector<std::uint64_t>, std::uint16_t> state_ids;
  std::vector<std::vector<std::uint64_t>> states;
  state_ids.emplace(start_state, 0);
  states.push_back(start_state);
  for (std::size_t state_index = 0; state_index < states.size(); ++state_index) {
    if (states.size() * representatives.size() > max_table_items) return std::nullopt;
    for (auto representative : representatives) {
      auto seeds = empty_bits();
      for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
        if (!machine_bit(states[state_index], node_index)) continue;
        auto& node = nodes[node_index];
        if (!node.predicate.matches(static_cast<char32_t>(representative))) continue;
        for (auto target : node.targets)
          set_machine_bit(seeds, target);
      }
      auto next = closure(seeds);
      if (machine.scan_input) {
        for (std::size_t word = 0; word < next.size(); ++word)
          next[word] |= start_state[word];
      }
      auto existing        = state_ids.find(next);
      std::uint16_t target = 0;
      if (existing == state_ids.end()) {
        if (states.size() >= max_dfa_states) return std::nullopt;
        target = static_cast<std::uint16_t>(states.size());
        state_ids.emplace(next, target);
        states.push_back(std::move(next));
      } else {
        target = existing->second;
      }
      if (machine_bit(states[target], accept_bit)) target |= 0x8000U;
      machine.transitions.push_back(target);
    }
  }
  machine.state_count   = static_cast<std::uint16_t>(states.size());
  machine.initial_state = machine_bit(start_state, accept_bit) ? 0x8000U : 0U;
  return machine;
}

class nvvm_ir_renderer {
 public:
  nvvm_ir_renderer(instruction_ir const& ir, nvvm_ir_codegen_options const& options)
    : ir_(ir), options_(options)
  {
  }

  std::string render()
  {
    require_codegen_ir(ir_);
    require_identifier(options_.symbol_prefix, "symbol_prefix");
    require_identifier(options_.execute_function, "execute_function");
    if (ir_.control.result != result_shape::BOOLEAN &&
        ir_.control.result != result_shape::CAPTURES) {
      throw std::invalid_argument(
        "NVVM IR generation supports boolean operations and capture enumeration");
    }
    deterministic_ =
      ir_.control.result == result_shape::BOOLEAN ? make_deterministic_machine(ir_) : std::nullopt;
    output_.line(
      R"NVVM(; NVVM IR generated by Regex IR
; pattern: {}
target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64")NVVM",
      escaped_comment(ir_.pattern));
    output_.line("; executor: {}",
                 deterministic_.has_value() ? "deterministic table" : "recursive Thompson");
    if (deterministic_.has_value()) {
      output_.line("; dfa states: {}, alphabet classes: {}",
                   deterministic_->state_count,
                   deterministic_->class_count);
    }
    output_.blank();

    emit_load_byte();
    emit_decode_width();
    emit_decode_codepoint();
    if (deterministic_.has_value()) {
      emit_deterministic_globals(*deterministic_);
      emit_deterministic_classifier(*deterministic_);
      emit_deterministic_execute(*deterministic_);
    } else {
      emit_optimizer_intrinsics();
      emit_advance();
      emit_can_peek();
      emit_is_word();
      emit_previous_position();
      emit_assertion();
      emit_predicate_helpers();
      emit_literal_helpers();
      emit_blocks();
      if (ir_.control.result == result_shape::CAPTURES) {
        emit_capture_execute();
      } else {
        emit_execute();
      }
    }
    output_.line(
      R"NVVM(!nvvmir.version = !{{!0}}
!0 = !{{i32 2, i32 0}})NVVM");
    return output_.take();
  }

 private:
  [[nodiscard]] std::string name(std::string_view suffix) const
  {
    return nvvm_symbol(options_.symbol_prefix, suffix);
  }

  static std::string escaped_comment(std::string_view value)
  {
    std::string result;
    result.reserve(value.size());
    for (auto character : value)
      result += character == '\n' ? ' ' : character;
    return result;
  }

  [[nodiscard]] std::optional<std::uint8_t> required_ascii_prefix() const
  {
    if (!options_.prefix_filter || !ir_.control.scan_input || ir_.entry >= ir_.blocks.size()) {
      return std::nullopt;
    }
    for (auto& instruction : ir_.blocks[ir_.entry].instructions) {
      if (auto* literal = std::get_if<match_literal>(&instruction)) {
        if (!literal->value.empty() && literal->value.front() <= 0x7f) {
          return static_cast<std::uint8_t>(literal->value.front());
        }
        return std::nullopt;
      }
      if (auto* character = std::get_if<match_character>(&instruction)) {
        if (character->predicate.is_singleton() && character->predicate.singleton() <= 0x7f) {
          return static_cast<std::uint8_t>(character->predicate.singleton());
        }
        return std::nullopt;
      }
      if (std::holds_alternative<advance_cursor>(instruction) ||
          std::holds_alternative<emit_accept>(instruction)) {
        return std::nullopt;
      }
    }
    return std::nullopt;
  }

  static std::int32_t llvm_i16(std::uint16_t value) { return static_cast<std::int16_t>(value); }

  static std::string format_i16_array(std::vector<std::uint16_t> const& values)
  {
    std::string result;
    result.reserve(values.size() * 8);
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (index != 0) result += ", ";
      fmt::format_to(std::back_inserter(result), "i16 {}", llvm_i16(values[index]));
    }
    return result;
  }

  /**
   * @brief emits constant character-class and DFA-transition tables
   *
   * @param machine deterministic machine to render
   */
  void emit_deterministic_globals(deterministic_machine const& machine)
  {
    std::vector<std::uint16_t> byte_classes(machine.byte_classes.begin(),
                                            machine.byte_classes.end());
    output_.line("@{} = internal addrspace(4) constant [256 x i16] [{}], align 2",
                 name("dfa_byte_classes"),
                 format_i16_array(byte_classes));
    output_.line("@{} = internal addrspace(4) constant [{} x i16] [{}], align 2",
                 name("dfa_transitions"),
                 machine.transitions.size(),
                 format_i16_array(machine.transitions));
    output_.blank();
  }

  /**
   * @brief emits the Unicode code-point to deterministic alphabet-class mapper
   *
   * @param machine deterministic machine whose classes are rendered
   */
  void emit_deterministic_classifier(deterministic_machine const& machine)
  {
    auto function = name("dfa_classify");
    auto table    = name("dfa_byte_classes");
    output_.line("{}",
                 fmt::format(
                   R"NVVM(define internal i32 @{function}(i32 %cp) alwaysinline nounwind readonly {{
entry:
  %is_byte = icmp ult i32 %cp, 256
  br i1 %is_byte, label %byte, label %unicode
byte:
  %byte_index = zext i32 %cp to i64
  %byte_class_ptr = getelementptr [256 x i16], [256 x i16] addrspace(4)* @{table}, i64 0, i64 %byte_index
  %byte_class_i16 = load i16, i16 addrspace(4)* %byte_class_ptr, align 2
  %byte_class = zext i16 %byte_class_i16 to i32
  ret i32 %byte_class
unicode:)NVVM",
                   fmt::arg("function", function),
                   fmt::arg("table", table)));

    if (machine.unicode_intervals.empty()) {
      output_.line("  ret i32 0");
    } else if (machine.unicode_intervals.size() == 1) {
      output_.line("  ret i32 {}", machine.unicode_intervals.front().class_id);
    } else {
      auto result = fmt::format("{}", machine.unicode_intervals.back().class_id);
      for (std::size_t reverse = machine.unicode_intervals.size() - 1; reverse > 0; --reverse) {
        auto index     = reverse - 1;
        auto& interval = machine.unicode_intervals[index];
        output_.line("{}",
                     fmt::format(R"NVVM(  %unicode_le_{index} = icmp ule i32 %cp, {last}
  %unicode_class_{index} = select i1 %unicode_le_{index}, i32 {class_id}, i32 {fallback})NVVM",
                                 fmt::arg("index", index),
                                 fmt::arg("last", interval.last),
                                 fmt::arg("class_id", interval.class_id),
                                 fmt::arg("fallback", result)));
        result = fmt::format("%unicode_class_{}", index);
      }
      output_.line("  ret i32 {}", result);
    }
    output_.line("}}");
    output_.blank();
  }

  /**
   * @brief emits the single-pass deterministic contains or matches executor
   *
   * @param machine deterministic machine to execute
   */
  void emit_deterministic_execute(deterministic_machine const& machine)
  {
    if ((machine.initial_state & 0x8000U) != 0 && machine.scan_input) {
      output_.line(
        R"NVVM(define zeroext i1 @{}(i8* %data, i64 %size) alwaysinline nounwind readnone {{
entry:
  ret i1 true
}})NVVM",
        options_.execute_function);
      output_.blank();
      return;
    }

    auto load_byte   = name("load_byte");
    auto decode      = name("decode_codepoint");
    auto width       = name("decode_width");
    auto classify    = name("dfa_classify");
    auto transitions = name("dfa_transitions");
    output_.line("{}",
                 fmt::format(
                   R"NVVM(define zeroext i1 @{execute}(i8* %data, i64 %size) nounwind readonly {{
entry:
  br label %loop
loop:
  %position = phi i64 [ 0, %entry ], [ %next_position, %continue ]
  %state = phi i32 [ {initial_state}, %entry ], [ %next_state, %continue ]
  %at_end = icmp eq i64 %position, %size
  br i1 %at_end, label %done, label %load
load:
  %input_ptr = getelementptr i8, i8* %data, i64 %position
  %first = call i32 @{load_byte}(i8* %input_ptr)
  %is_ascii = icmp ult i32 %first, 128
  br i1 %is_ascii, label %ascii, label %unicode
ascii:
  %ascii_class = call i32 @{classify}(i32 %first)
  br label %transition
unicode:
  %codepoint = call i32 @{decode}(i8* %data, i64 %size, i64 %position)
  %unicode_width = call i64 @{width}(i8* %data, i64 %size, i64 %position)
  %unicode_class = call i32 @{classify}(i32 %codepoint)
  br label %transition
transition:
  %character_class = phi i32 [ %ascii_class, %ascii ], [ %unicode_class, %unicode ]
  %character_width = phi i64 [ 1, %ascii ], [ %unicode_width, %unicode ]
  %state_index = and i32 %state, 32767
  %state_offset = mul nuw i32 %state_index, {class_count}
  %transition_index = add nuw i32 %state_offset, %character_class
  %transition_index_i64 = zext i32 %transition_index to i64
  %transition_ptr = getelementptr [{transition_count} x i16], [{transition_count} x i16] addrspace(4)* @{transitions}, i64 0, i64 %transition_index_i64
  %next_state_i16 = load i16, i16 addrspace(4)* %transition_ptr, align 2
  %next_state = zext i16 %next_state_i16 to i32
  %next_position = add i64 %position, %character_width)NVVM",
                   fmt::arg("execute", options_.execute_function),
                   fmt::arg("initial_state", machine.initial_state),
                   fmt::arg("load_byte", load_byte),
                   fmt::arg("classify", classify),
                   fmt::arg("decode", decode),
                   fmt::arg("width", width),
                   fmt::arg("class_count", machine.class_count),
                   fmt::arg("transition_count", machine.transitions.size()),
                   fmt::arg("transitions", transitions)));
    if (machine.scan_input) {
      output_.line(
        R"NVVM(  %accept_bits = and i32 %next_state, 32768
  %accepted = icmp ne i32 %accept_bits, 0
  br i1 %accepted, label %yes, label %continue
continue:
  br label %loop
done:
  ret i1 false
yes:
  ret i1 true
}})NVVM");
    } else {
      output_.line(
        R"NVVM(  br label %continue
continue:
  br label %loop
done:
  %accept_bits = and i32 %state, 32768
  %accepted = icmp ne i32 %accept_bits, 0
  ret i1 %accepted
}})NVVM");
    }
    output_.blank();
  }

  /**
   * @brief emits NVVM-supported optimizer intrinsic declarations used by generated branch hints
   */
  void emit_optimizer_intrinsics()
  {
    if (required_ascii_prefix().has_value() && options_.branch_hints) {
      output_.line("declare i1 @llvm.expect.i1(i1, i1)");
      output_.blank();
    }
  }

  /**
   * @brief emits the compiler-cached byte load used by every input decoder
   */
  void emit_load_byte()
  {
    auto function = name("load_byte");
    output_.line(
      R"NVVM(define internal i32 @{}(i8* %ptr) alwaysinline nounwind readonly {{
entry:
  %value8 = load i8, i8* %ptr, align 1
  %value = zext i8 %value8 to i32
  ret i32 %value
}})NVVM",
      function);
    output_.blank();
  }

  /**
   * @brief emits the helper that returns the byte width of the character at a cursor position
   */
  void emit_decode_width()
  {
    auto function  = name("decode_width");
    auto load_byte = name("load_byte");
    output_.line(
      R"NVVM(define internal i64 @{}(i8* %data, i64 %size, i64 %pos) alwaysinline nounwind readonly {{
entry:
  %in_bounds = icmp ult i64 %pos, %size
  br i1 %in_bounds, label %load, label %missing
missing:
  ret i64 0
load:)NVVM",
      function);
    if (ir_.options.characters == character_mode::BYTES) {
      output_.line(
        R"NVVM(  ret i64 1
}})NVVM");
      output_.blank();
      return;
    }
    output_.line(
      R"NVVM(  %ptr = getelementptr i8, i8* %data, i64 %pos
  %first = call i32 @{}(i8* %ptr)
  %ascii = icmp ult i32 %first, 128
  br i1 %ascii, label %one, label %classify
one:
  ret i64 1
classify:
  %ge2 = icmp uge i32 %first, 194
  %le2 = icmp ule i32 %first, 223
  %is2 = and i1 %ge2, %le2
  %ge3 = icmp uge i32 %first, 224
  %le3 = icmp ule i32 %first, 239
  %is3 = and i1 %ge3, %le3
  %ge4 = icmp uge i32 %first, 240
  %le4 = icmp ule i32 %first, 244
  %is4 = and i1 %ge4, %le4
  %maybe3 = select i1 %is3, i64 3, i64 1
  %maybe4 = select i1 %is4, i64 4, i64 %maybe3
  %required = select i1 %is2, i64 2, i64 %maybe4
  %is_multibyte = icmp ugt i64 %required, 1
  br i1 %is_multibyte, label %bounds, label %invalid
invalid:
  ret i64 1
bounds:
  %end = add i64 %pos, %required
  %enough = icmp ule i64 %end, %size
  br i1 %enough, label %continuation_loop, label %invalid_short
invalid_short:
  ret i64 1
continuation_loop:
  %index = phi i64 [ 1, %bounds ], [ %next_index, %continuation_ok ]
  %continuation_pos = add i64 %pos, %index
  %continuation_ptr = getelementptr i8, i8* %data, i64 %continuation_pos
  %continuation = call i32 @{}(i8* %continuation_ptr)
  %tag = and i32 %continuation, 192
  %valid = icmp eq i32 %tag, 128
  br i1 %valid, label %continuation_ok, label %invalid_continuation
invalid_continuation:
  ret i64 1
continuation_ok:
  %next_index = add i64 %index, 1
  %done = icmp eq i64 %next_index, %required
  br i1 %done, label %valid_multibyte, label %continuation_loop
valid_multibyte:
  ret i64 %required
}})NVVM",
      load_byte,
      load_byte);
    output_.blank();
  }

  /**
   * @brief emits the helper that decodes the character at a cursor position into a code point
   */
  void emit_decode_codepoint()
  {
    auto function  = name("decode_codepoint");
    auto width     = name("decode_width");
    auto load_byte = name("load_byte");
    output_.line(
      R"NVVM(define internal i32 @{}(i8* %data, i64 %size, i64 %pos) alwaysinline nounwind readonly {{
entry:
  %available = icmp ult i64 %pos, %size
  br i1 %available, label %load, label %missing
missing:
  ret i32 0
load:
  %ptr = getelementptr i8, i8* %data, i64 %pos
  %first = call i32 @{}(i8* %ptr))NVVM",
      function,
      load_byte);
    if (ir_.options.characters == character_mode::BYTES) {
      output_.line(
        R"NVVM(  ret i32 %first
}})NVVM");
      output_.blank();
      return;
    }
    output_.line(
      R"NVVM(  %width = call i64 @{}(i8* %data, i64 %size, i64 %pos)
  %single = icmp eq i64 %width, 1
  br i1 %single, label %single_byte, label %initialize
single_byte:
  ret i32 %first
initialize:
  %is2 = icmp eq i64 %width, 2
  %is3 = icmp eq i64 %width, 3
  %mask3 = select i1 %is3, i32 15, i32 7
  %mask = select i1 %is2, i32 31, i32 %mask3
  %initial = and i32 %first, %mask
  br label %loop
loop:
  %index = phi i64 [ 1, %initialize ], [ %next_index, %body ]
  %value = phi i32 [ %initial, %initialize ], [ %combined, %body ]
  %done = icmp uge i64 %index, %width
  br i1 %done, label %exit, label %body
body:
  %byte_pos = add i64 %pos, %index
  %byte_ptr = getelementptr i8, i8* %data, i64 %byte_pos
  %byte = call i32 @{}(i8* %byte_ptr)
  %payload = and i32 %byte, 63
  %shifted = shl i32 %value, 6
  %combined = or i32 %shifted, %payload
  %next_index = add i64 %index, 1
  br label %loop
exit:
  ret i32 %value
}})NVVM",
      width,
      load_byte);
    output_.blank();
  }

  /**
   * @brief emits the helper that advances a byte cursor by a requested number of characters
   */
  void emit_advance()
  {
    auto function = name("advance");
    auto width    = name("decode_width");
    output_.line(
      "{}",
      fmt::format(
        R"NVVM(define internal i64 @{function}(i8* %data, i64 %size, i64 %pos, i64 %count) alwaysinline nounwind readonly {{
entry:
  br label %loop
loop:
  %cursor = phi i64 [ %pos, %entry ], [ %next_cursor, %step ]
  %index = phi i64 [ 0, %entry ], [ %next_index, %step ]
  %done = icmp uge i64 %index, %count
  br i1 %done, label %exit, label %check
check:
  %width = call i64 @{width}(i8* %data, i64 %size, i64 %cursor)
  %missing = icmp eq i64 %width, 0
  br i1 %missing, label %exit, label %step
step:
  %next_cursor = add i64 %cursor, %width
  %next_index = add i64 %index, 1
  br label %loop
exit:
  ret i64 %cursor
}})NVVM",
        fmt::arg("function", function),
        fmt::arg("width", width)));
    output_.blank();
  }

  /**
   * @brief emits the helper that checks whether a requested number of characters is available
   */
  void emit_can_peek()
  {
    auto function = name("can_peek");
    auto width    = name("decode_width");
    output_.line(
      "{}",
      fmt::format(
        R"NVVM(define internal i1 @{function}(i8* %data, i64 %size, i64 %pos, i64 %count) alwaysinline nounwind readonly {{
entry:
  br label %loop
loop:
  %cursor = phi i64 [ %pos, %entry ], [ %next_cursor, %step ]
  %index = phi i64 [ 0, %entry ], [ %next_index, %step ]
  %done = icmp uge i64 %index, %count
  br i1 %done, label %yes, label %check
check:
  %width = call i64 @{width}(i8* %data, i64 %size, i64 %cursor)
  %missing = icmp eq i64 %width, 0
  br i1 %missing, label %no, label %step
step:
  %next_cursor = add i64 %cursor, %width
  %next_index = add i64 %index, 1
  br label %loop
yes:
  ret i1 true
no:
  ret i1 false
}})NVVM",
        fmt::arg("function", function),
        fmt::arg("width", width)));
    output_.blank();
  }

  /**
   * @brief reports whether the program evaluates a Unicode word boundary
   *
   * @return true when a boundary assertion needs cuDF's Unicode word table
   */
  [[nodiscard]] bool uses_unicode_word_boundaries() const
  {
    if (ir_.options.ascii_classes || ir_.options.characters == character_mode::BYTES) return false;
    for (instruction_block const& block : ir_.blocks) {
      for (instruction const& item : block.instructions) {
        auto* assertion = std::get_if<test_assertion>(&item);
        if (assertion != nullptr && (assertion->kind == assertion_kind::WORD_BOUNDARY ||
                                     assertion->kind == assertion_kind::NOT_WORD_BOUNDARY)) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * @brief emits the configured word-character classifier used by boundary assertions
   */
  void emit_is_word()
  {
    auto function = name("is_word");
    if (uses_unicode_word_boundaries()) {
      output_.line(
        R"NVVM(define internal i1 @{}(i32 %cp) alwaysinline nounwind readnone {{
entry:)NVVM",
        function);
      std::string combined;
      for (std::size_t index = 0; index < std::size(cudf_unicode_word_ranges); ++index) {
        unicode_data_range range = cudf_unicode_word_ranges[index];
        output_.line("  %word_ge_{} = icmp uge i32 %cp, {}", index, range.first);
        output_.line("  %word_le_{} = icmp ule i32 %cp, {}", index, range.last);
        output_.line("  %word_in_{} = and i1 %word_ge_{}, %word_le_{}", index, index, index);
        if (index == 0) {
          combined = "%word_in_0";
        } else {
          output_.line("  %word_combined_{} = or i1 {}, %word_in_{}", index, combined, index);
          combined = fmt::format("%word_combined_{}", index);
        }
      }
      output_.line("  %word_underscore = icmp eq i32 %cp, 95");
      output_.line("  %word_result = or i1 {}, %word_underscore", combined);
      output_.line(
        R"NVVM(  ret i1 %word_result
}})NVVM");
      output_.blank();
      return;
    }
    output_.line(
      R"NVVM(define internal i1 @{}(i32 %cp) alwaysinline nounwind readnone {{
entry:
  %ge_lower = icmp uge i32 %cp, 97
  %le_lower = icmp ule i32 %cp, 122
  %lower = and i1 %ge_lower, %le_lower
  %ge_upper = icmp uge i32 %cp, 65
  %le_upper = icmp ule i32 %cp, 90
  %upper = and i1 %ge_upper, %le_upper
  %ge_digit = icmp uge i32 %cp, 48
  %le_digit = icmp ule i32 %cp, 57
  %digit = and i1 %ge_digit, %le_digit
  %alpha = or i1 %lower, %upper
  %alnum = or i1 %alpha, %digit
  %underscore = icmp eq i32 %cp, 95
  %result = or i1 %alnum, %underscore
  ret i1 %result
}})NVVM",
      function);
    output_.blank();
  }

  /**
   * @brief emits the helper that locates the character immediately before a byte position
   */
  void emit_previous_position()
  {
    auto function = name("previous_position");
    auto advance  = name("advance");
    output_.line(
      "{}",
      fmt::format(
        R"NVVM(define internal i64 @{function}(i8* %data, i64 %size, i64 %target) alwaysinline nounwind readonly {{
entry:
  %at_begin = icmp eq i64 %target, 0
  br i1 %at_begin, label %zero, label %loop
zero:
  ret i64 0
loop:
  %cursor = phi i64 [ 0, %entry ], [ %next, %continue ]
  %next = call i64 @{advance}(i8* %data, i64 %size, i64 %cursor, i64 1)
  %reached = icmp uge i64 %next, %target
  br i1 %reached, label %found, label %continue
continue:
  br label %loop
found:
  ret i64 %cursor
}})NVVM",
        fmt::arg("function", function),
        fmt::arg("advance", advance)));
    output_.blank();
  }

  /**
   * @brief emits the dispatcher for begin, end, word-boundary, and non-boundary assertions
   */
  void emit_assertion()
  {
    auto function = name("assertion");
    auto previous = name("previous_position");
    auto decode   = name("decode_codepoint");
    auto is_word  = name("is_word");
    auto advance  = name("advance");
    output_.line(
      "{}",
      fmt::format(
        R"NVVM(define internal i1 @{function}(i8* %data, i64 %size, i64 %pos, i32 %kind) inlinehint nounwind readonly {{
entry:
  switch i32 %kind, label %not_boundary [
    i32 0, label %begin_input
    i32 1, label %end_input
    i32 2, label %boundary
    i32 4, label %begin_line
    i32 5, label %end_line
  ]
begin_input:
  %is_begin_input = icmp eq i64 %pos, 0
  ret i1 %is_begin_input
end_input:
  %is_end_input = icmp eq i64 %pos, %size
  ret i1 %is_end_input
begin_line:
  %is_begin_line_input = icmp eq i64 %pos, 0
  br i1 %is_begin_line_input, label %true, label %begin_line_nonzero
begin_line_nonzero:
  br i1 {multiline}, label %begin_line_previous, label %false
begin_line_previous:
  %begin_prev_pos = call i64 @{previous}(i8* %data, i64 %size, i64 %pos)
  %begin_prev = call i32 @{decode}(i8* %data, i64 %size, i64 %begin_prev_pos)
  %begin_prev_lf = icmp eq i32 %begin_prev, 10
  %begin_prev_cr = icmp eq i32 %begin_prev, 13
  %begin_prev_nel = icmp eq i32 %begin_prev, 133
  %begin_prev_ls = icmp eq i32 %begin_prev, 8232
  %begin_prev_ps = icmp eq i32 %begin_prev, 8233
  %begin_prev_crlf = or i1 %begin_prev_lf, %begin_prev_cr
  %begin_prev_extended_0 = or i1 %begin_prev_crlf, %begin_prev_nel
  %begin_prev_extended_1 = or i1 %begin_prev_ls, %begin_prev_ps
  %begin_prev_extended = or i1 %begin_prev_extended_0, %begin_prev_extended_1
  %begin_prev_newline = select i1 {extended}, i1 %begin_prev_extended, i1 %begin_prev_lf
  %begin_current = call i32 @{decode}(i8* %data, i64 %size, i64 %pos)
  %begin_current_lf = icmp eq i32 %begin_current, 10
  %begin_mid_crlf_0 = and i1 %begin_prev_cr, %begin_current_lf
  %begin_mid_crlf = and i1 {extended}, %begin_mid_crlf_0
  %begin_not_mid_crlf = xor i1 %begin_mid_crlf, true
  %begin_line_result = and i1 %begin_prev_newline, %begin_not_mid_crlf
  ret i1 %begin_line_result
end_line:
  %is_end_line_input = icmp eq i64 %pos, %size
  br i1 %is_end_line_input, label %true, label %end_line_current
end_line_current:
  %end_cp = call i32 @{decode}(i8* %data, i64 %size, i64 %pos)
  %end_lf = icmp eq i32 %end_cp, 10
  %end_cr = icmp eq i32 %end_cp, 13
  %end_nel = icmp eq i32 %end_cp, 133
  %end_ls = icmp eq i32 %end_cp, 8232
  %end_ps = icmp eq i32 %end_cp, 8233
  %end_crlf = or i1 %end_lf, %end_cr
  %end_extended_0 = or i1 %end_crlf, %end_nel
  %end_extended_1 = or i1 %end_ls, %end_ps
  %end_extended = or i1 %end_extended_0, %end_extended_1
  %end_newline = select i1 {extended}, i1 %end_extended, i1 %end_lf
  br i1 %end_newline, label %end_line_newline, label %false
end_line_newline:
  %end_prev_pos = call i64 @{previous}(i8* %data, i64 %size, i64 %pos)
  %end_prev = call i32 @{decode}(i8* %data, i64 %size, i64 %end_prev_pos)
  %end_prev_cr = icmp eq i32 %end_prev, 13
  %end_mid_crlf_0 = and i1 %end_prev_cr, %end_lf
  %end_mid_crlf = and i1 {extended}, %end_mid_crlf_0
  br i1 %end_mid_crlf, label %false, label %end_line_not_mid
end_line_not_mid:
  br i1 {multiline}, label %true, label %end_line_final
end_line_final:
  %end_next = call i64 @{advance}(i8* %data, i64 %size, i64 %pos, i64 1)
  %end_is_final = icmp eq i64 %end_next, %size
  br i1 %end_is_final, label %true, label %end_line_possible_crlf
end_line_possible_crlf:
  %end_can_be_crlf = and i1 {extended}, %end_cr
  br i1 %end_can_be_crlf, label %end_line_after_cr, label %false
end_line_after_cr:
  %end_next_cp = call i32 @{decode}(i8* %data, i64 %size, i64 %end_next)
  %end_next_lf = icmp eq i32 %end_next_cp, 10
  %end_after_lf = call i64 @{advance}(i8* %data, i64 %size, i64 %end_next, i64 1)
  %end_lf_is_final = icmp eq i64 %end_after_lf, %size
  %end_final_crlf = and i1 %end_next_lf, %end_lf_is_final
  ret i1 %end_final_crlf
boundary:
  %current_in_bounds = icmp ult i64 %pos, %size
  br i1 %current_in_bounds, label %current_decode, label %current_missing
current_decode:
  %current_cp = call i32 @{decode}(i8* %data, i64 %size, i64 %pos)
  %current_word_value = call i1 @{is_word}(i32 %current_cp)
  br label %current_join
current_missing:
  br label %current_join
current_join:
  %current_word = phi i1 [ %current_word_value, %current_decode ], [ false, %current_missing ]
  %has_previous = icmp ugt i64 %pos, 0
  br i1 %has_previous, label %previous_decode, label %previous_missing
previous_decode:
  %previous_pos = call i64 @{previous}(i8* %data, i64 %size, i64 %pos)
  %previous_cp = call i32 @{decode}(i8* %data, i64 %size, i64 %previous_pos)
  %previous_word_value = call i1 @{is_word}(i32 %previous_cp)
  br label %previous_join
previous_missing:
  br label %previous_join
previous_join:
  %previous_word = phi i1 [ %previous_word_value, %previous_decode ], [ false, %previous_missing ]
  %is_boundary = xor i1 %previous_word, %current_word
  ret i1 %is_boundary
not_boundary:
  %boundary_value = call i1 @{function}(i8* %data, i64 %size, i64 %pos, i32 2)
  %not_boundary_value = xor i1 %boundary_value, true
  ret i1 %not_boundary_value
true:
  ret i1 true
false:
  ret i1 false
}})NVVM",
        fmt::arg("decode", decode),
        fmt::arg("is_word", is_word),
        fmt::arg("previous", previous),
        fmt::arg("function", function),
        fmt::arg("advance", advance),
        fmt::arg("multiline", ir_.options.multiline ? "true" : "false"),
        fmt::arg("extended", ir_.options.extended_newline ? "true" : "false")));
    output_.blank();
  }

  /**
   * @brief emits a range-test helper for every character-predicate instruction
   */
  void emit_predicate_helpers()
  {
    for (auto& block : ir_.blocks) {
      for (auto& instruction : block.instructions) {
        auto* match = std::get_if<match_character>(&instruction);
        if (match == nullptr) continue;
        auto function = name(fmt::format("predicate_{}", block.id));
        auto decode   = name("decode_codepoint");
        output_.line(
          "{}",
          fmt::format(
            R"NVVM(define internal i1 @{function}(i8* %data, i64 %size, i64 %pos) alwaysinline nounwind readonly {{
entry:
  %cp = call i32 @{decode}(i8* %data, i64 %size, i64 %pos))NVVM",
            fmt::arg("function", function),
            fmt::arg("decode", decode)));
        if (match->predicate.recognized == predicate_class::ANY) {
          if (match->predicate.matches_newline) {
            output_.line("  ret i1 true");
          } else if (match->predicate.extended_newline) {
            output_.line(
              R"NVVM(  %not_lf = icmp ne i32 %cp, 10
  %not_cr = icmp ne i32 %cp, 13
  %not_nel = icmp ne i32 %cp, 133
  %not_ls = icmp ne i32 %cp, 8232
  %not_ps = icmp ne i32 %cp, 8233
  %not_crlf = and i1 %not_lf, %not_cr
  %not_extended_0 = and i1 %not_nel, %not_ls
  %not_extended_1 = and i1 %not_extended_0, %not_ps
  %result = and i1 %not_crlf, %not_extended_1
  ret i1 %result)NVVM");
          } else {
            output_.line(
              R"NVVM(  %not_lf = icmp ne i32 %cp, 10
  ret i1 %not_lf)NVVM");
          }
        } else {
          std::vector<std::string> comparisons;
          for (std::size_t index = 0; index < match->predicate.ranges.size(); ++index) {
            auto range = match->predicate.ranges[index];
            if (range.first == range.last) {
              auto comparison = fmt::format("equal_{}", index);
              output_.line(
                "  %{} = icmp eq i32 %cp, {}", comparison, static_cast<std::uint32_t>(range.first));
              comparisons.push_back(comparison);
            } else {
              auto ge     = fmt::format("ge_{}", index);
              auto le     = fmt::format("le_{}", index);
              auto inside = fmt::format("inside_{}", index);
              output_.line("{}",
                           fmt::format(R"NVVM(  %{ge} = icmp uge i32 %cp, {first}
  %{le} = icmp ule i32 %cp, {last}
  %{inside} = and i1 %{ge}, %{le})NVVM",
                                       fmt::arg("ge", ge),
                                       fmt::arg("le", le),
                                       fmt::arg("inside", inside),
                                       fmt::arg("first", static_cast<std::uint32_t>(range.first)),
                                       fmt::arg("last", static_cast<std::uint32_t>(range.last))));
              comparisons.push_back(inside);
            }
          }
          if (comparisons.empty()) {
            output_.line("  ret i1 {}", match->predicate.negated ? "true" : "false");
          } else {
            auto combined = comparisons.front();
            for (std::size_t index = 1; index < comparisons.size(); ++index) {
              auto next = fmt::format("combined_{}", index);
              output_.line("  %{} = or i1 %{}, %{}", next, combined, comparisons[index]);
              combined = next;
            }
            if (match->predicate.negated) {
              output_.line(
                R"NVVM(  %negated = xor i1 %{}, true
  ret i1 %negated)NVVM",
                combined);
            } else {
              output_.line("  ret i1 %{}", combined);
            }
          }
        }
        output_.line("}}");
        output_.blank();
      }
    }
  }

  /**
   * @brief emits a code-point comparison helper for every literal instruction
   */
  void emit_literal_helpers()
  {
    auto can_peek  = name("can_peek");
    auto decode    = name("decode_codepoint");
    auto width     = name("decode_width");
    auto load_byte = name("load_byte");
    for (auto& block : ir_.blocks) {
      for (auto& instruction : block.instructions) {
        auto* literal = std::get_if<match_literal>(&instruction);
        if (literal == nullptr) continue;
        auto function = name(fmt::format("literal_{}", block.id));
        auto ascii    = !literal->value.empty() &&
                     std::all_of(literal->value.begin(), literal->value.end(), [](char32_t value) {
                       return value <= 0x7f;
                     });
        if (ascii) {
          output_.line(
            "{}",
            fmt::format(
              R"NVVM(define internal i1 @{function}(i8* %data, i64 %size, i64 %pos) alwaysinline nounwind readonly {{
entry:
  %in_bounds = icmp ule i64 %pos, %size
  %remaining = sub i64 %size, %pos
  %enough = icmp uge i64 %remaining, {size}
  %available = and i1 %in_bounds, %enough
  br i1 %available, label %check_0, label %fail)NVVM",
              fmt::arg("function", function),
              fmt::arg("size", literal->value.size())));
          for (std::size_t index = 0; index < literal->value.size(); ++index) {
            auto success =
              index + 1 == literal->value.size() ? "success" : fmt::format("check_{}", index + 1);
            output_.line(
              "{}",
              fmt::format(R"NVVM(check_{index}:
  %byte_pos_{index} = add i64 %pos, {index}
  %byte_ptr_{index} = getelementptr i8, i8* %data, i64 %byte_pos_{index}
  %byte_{index} = call i32 @{load_byte}(i8* %byte_ptr_{index})
  %matches_{index} = icmp eq i32 %byte_{index}, {value}
  br i1 %matches_{index}, label %{success}, label %fail)NVVM",
                          fmt::arg("index", index),
                          fmt::arg("load_byte", load_byte),
                          fmt::arg("value", static_cast<std::uint32_t>(literal->value[index])),
                          fmt::arg("success", success)));
          }
          output_.line(
            R"NVVM(success:
  ret i1 true
fail:
  ret i1 false
}})NVVM");
          output_.blank();
          continue;
        }

        output_.line(
          "{}",
          fmt::format(
            R"NVVM(define internal i1 @{function}(i8* %data, i64 %size, i64 %pos) alwaysinline nounwind readonly {{
entry:
  %available = call i1 @{can_peek}(i8* %data, i64 %size, i64 %pos, i64 {size})
  br i1 %available, label %check_0, label %fail)NVVM",
            fmt::arg("function", function),
            fmt::arg("can_peek", can_peek),
            fmt::arg("size", literal->value.size())));
        for (std::size_t index = 0; index < literal->value.size(); ++index) {
          auto position = index == 0 ? "%pos" : fmt::format("%pos_{}", index);
          output_.line(
            "{}",
            fmt::format(R"NVVM(check_{index}:
  %cp_{index} = call i32 @{decode}(i8* %data, i64 %size, i64 {position})
  %matches_{index} = icmp eq i32 %cp_{index}, {codepoint})NVVM",
                        fmt::arg("index", index),
                        fmt::arg("decode", decode),
                        fmt::arg("position", position),
                        fmt::arg("codepoint", static_cast<std::uint32_t>(literal->value[index]))));
          if (index + 1 == literal->value.size()) {
            output_.line("  br i1 %matches_{}, label %success, label %fail", index);
          } else {
            output_.line(
              "{}",
              fmt::format(
                R"NVVM(  %width_{index} = call i64 @{width}(i8* %data, i64 %size, i64 {position})
  %pos_{next} = add i64 {position}, %width_{index}
  br i1 %matches_{index}, label %check_{next}, label %fail)NVVM",
                fmt::arg("index", index),
                fmt::arg("next", index + 1),
                fmt::arg("width", width),
                fmt::arg("position", position)));
          }
        }
        output_.line(
          R"NVVM(success:
  ret i1 true
fail:
  ret i1 false
}})NVVM");
        output_.blank();
      }
    }
  }

  /**
   * @brief emits the recursive dispatcher that executes the instruction block graph
   */
  void emit_blocks()
  {
    // one recursive dispatcher represents cyclic block graphs without forward declarations
    auto function = name("run_block");
    output_.line(
      R"NVVM(define internal i1 @{}(i32 %block, i8* %data, i64 %size, i64* %position, i64* %captures, i64 %steps) nounwind {{
entry:
  %exhausted = icmp eq i64 %steps, 0
  %next_steps = sub i64 %steps, 1
  br i1 %exhausted, label %return_fail, label %dispatch
dispatch:
  switch i32 %block, label %return_fail [)NVVM",
      function);
    for (auto& block : ir_.blocks)
      output_.line("    i32 {}, label %b{}_op_0", block.id, block.id);
    output_.line("  ]");
    for (auto& block : ir_.blocks)
      emit_block(block);
    output_.line(
      R"NVVM(return_success:
  ret i1 true
return_fail:
  ret i1 false
}})NVVM");
    output_.blank();
  }

  /**
   * @brief emits one block's instructions and prioritized successor attempts
   *
   * @param block instruction block to lower into NVVM IR
   */
  void emit_block(instruction_block const& block)
  {
    auto prefix                 = fmt::format("b{}", block.id);
    std::size_t operation_index = 0;
    bool returned               = false;
    for (auto& item : block.instructions) {
      output_.line("{}_op_{}:", prefix, operation_index);
      auto next = fmt::format("{}_op_{}", prefix, operation_index + 1);
      if (auto* peek = std::get_if<can_peek>(&item)) {
        auto* next_literal =
          operation_index + 1 < block.instructions.size()
            ? std::get_if<match_literal>(&block.instructions[operation_index + 1])
            : nullptr;
        auto fused_ascii_literal =
          next_literal != nullptr && next_literal->value.size() == peek->characters &&
          std::all_of(next_literal->value.begin(), next_literal->value.end(), [](char32_t value) {
            return value <= 0x7f;
          });
        if (fused_ascii_literal) {
          // the ASCII literal helper performs one byte-count bounds check for the fused sequence
          output_.line("  br label %{}", next);
        } else {
          output_.line(
            "{}",
            fmt::format(R"NVVM(  %{prefix}_pos_{index} = load i64, i64* %position, align 8
  %{prefix}_peek_{index} = call i1 @{can_peek}(i8* %data, i64 %size, i64 %{prefix}_pos_{index}, i64 {characters})
  br i1 %{prefix}_peek_{index}, label %{next}, label %return_fail)NVVM",
                        fmt::arg("prefix", prefix),
                        fmt::arg("index", operation_index),
                        fmt::arg("can_peek", name("can_peek")),
                        fmt::arg("characters", peek->characters),
                        fmt::arg("next", next)));
        }
      } else if (std::holds_alternative<read_character>(item)) {
        output_.line("  br label %{}", next);
      } else if (auto* capture = std::get_if<write_capture>(&item)) {
        auto slot = static_cast<std::size_t>(capture->capture_index) * 2U +
                    (capture->action == capture_action::END ? 1U : 0U);
        output_.line("{}",
                     fmt::format(
                       R"NVVM(  %{prefix}_capture_pos_{index} = load i64, i64* %position, align 8
  %{prefix}_capture_ptr_{index} = getelementptr i64, i64* %captures, i64 {slot}
  store i64 %{prefix}_capture_pos_{index}, i64* %{prefix}_capture_ptr_{index}, align 8)NVVM",
                       fmt::arg("prefix", prefix),
                       fmt::arg("index", operation_index),
                       fmt::arg("slot", slot)));
        if (capture->action == capture_action::BEGIN) {
          output_.line(
            "{}",
            fmt::format(
              R"NVVM(  %{prefix}_capture_end_ptr_{index} = getelementptr i64, i64* %captures, i64 {end_slot}
  store i64 -1, i64* %{prefix}_capture_end_ptr_{index}, align 8)NVVM",
              fmt::arg("prefix", prefix),
              fmt::arg("index", operation_index),
              fmt::arg("end_slot", slot + 1U)));
        }
        output_.line("  br label %{}", next);
      } else if (std::holds_alternative<match_character>(item)) {
        output_.line(
          "{}",
          fmt::format(R"NVVM(  %{prefix}_match_pos_{index} = load i64, i64* %position, align 8
  %{prefix}_match_{index} = call i1 @{predicate}(i8* %data, i64 %size, i64 %{prefix}_match_pos_{index})
  br i1 %{prefix}_match_{index}, label %{next}, label %return_fail)NVVM",
                      fmt::arg("prefix", prefix),
                      fmt::arg("index", operation_index),
                      fmt::arg("predicate", name(fmt::format("predicate_{}", block.id))),
                      fmt::arg("next", next)));
      } else if (std::holds_alternative<match_literal>(item)) {
        output_.line(
          "{}",
          fmt::format(R"NVVM(  %{prefix}_literal_pos_{index} = load i64, i64* %position, align 8
  %{prefix}_literal_{index} = call i1 @{literal}(i8* %data, i64 %size, i64 %{prefix}_literal_pos_{index})
  br i1 %{prefix}_literal_{index}, label %{next}, label %return_fail)NVVM",
                      fmt::arg("prefix", prefix),
                      fmt::arg("index", operation_index),
                      fmt::arg("literal", name(fmt::format("literal_{}", block.id))),
                      fmt::arg("next", next)));
      } else if (auto* advance = std::get_if<advance_cursor>(&item)) {
        auto* previous_literal =
          operation_index > 0 ? std::get_if<match_literal>(&block.instructions[operation_index - 1])
                              : nullptr;
        auto matched_ascii_literal = previous_literal != nullptr &&
                                     previous_literal->value.size() == advance->characters &&
                                     std::all_of(previous_literal->value.begin(),
                                                 previous_literal->value.end(),
                                                 [](char32_t value) { return value <= 0x7f; });
        if (matched_ascii_literal) {
          output_.line(
            "{}",
            fmt::format(R"NVVM(  %{prefix}_advance_pos_{index} = load i64, i64* %position, align 8
  %{prefix}_advanced_{index} = add i64 %{prefix}_advance_pos_{index}, {characters}
  store i64 %{prefix}_advanced_{index}, i64* %position, align 8
  br label %{next})NVVM",
                        fmt::arg("prefix", prefix),
                        fmt::arg("index", operation_index),
                        fmt::arg("characters", advance->characters),
                        fmt::arg("next", next)));
        } else {
          output_.line("{}",
                       fmt::format(
                         R"NVVM(  %{prefix}_advance_pos_{index} = load i64, i64* %position, align 8
  %{prefix}_advanced_{index} = call i64 @{advance}(i8* %data, i64 %size, i64 %{prefix}_advance_pos_{index}, i64 {characters})
  store i64 %{prefix}_advanced_{index}, i64* %position, align 8
  br label %{next})NVVM",
                         fmt::arg("prefix", prefix),
                         fmt::arg("index", operation_index),
                         fmt::arg("advance", name("advance")),
                         fmt::arg("characters", advance->characters),
                         fmt::arg("next", next)));
        }
      } else if (auto* assertion = std::get_if<test_assertion>(&item)) {
        output_.line(
          "{}",
          fmt::format(R"NVVM(  %{prefix}_assert_pos_{index} = load i64, i64* %position, align 8
  %{prefix}_assert_{index} = call i1 @{assertion}(i8* %data, i64 %size, i64 %{prefix}_assert_pos_{index}, i32 {kind})
  br i1 %{prefix}_assert_{index}, label %{next}, label %return_fail)NVVM",
                      fmt::arg("prefix", prefix),
                      fmt::arg("index", operation_index),
                      fmt::arg("assertion", name("assertion")),
                      fmt::arg("kind", static_cast<std::uint32_t>(assertion->kind)),
                      fmt::arg("next", next)));
      } else if (std::holds_alternative<emit_accept>(item)) {
        if (ir_.control.require_end) {
          output_.line(
            R"NVVM(  %{}_accept_pos = load i64, i64* %position, align 8
  %{}_accept = icmp eq i64 %{}_accept_pos, %size
  ret i1 %{}_accept)NVVM",
            prefix,
            prefix,
            prefix,
            prefix);
        } else {
          output_.line("  ret i1 true");
        }
        returned = true;
      }
      ++operation_index;
    }

    if (returned) return;
    output_.line("{}_op_{}:", prefix, operation_index);
    auto edges = block.successors;
    std::stable_sort(edges.begin(), edges.end(), [](auto& left, auto& right) {
      return left.priority < right.priority;
    });
    if (edges.empty()) {
      output_.line("  br label %return_fail");
      return;
    }
    output_.line("  %{}_saved = load i64, i64* %position, align 8", prefix);
    auto capture_slots = ir_.control.result == result_shape::CAPTURES
                           ? static_cast<std::size_t>(ir_.capture_count + 1U) * 2U
                           : 0U;
    for (std::size_t slot = 0; slot < capture_slots; ++slot) {
      output_.line(R"NVVM(  %{}_saved_capture_{} = getelementptr i64, i64* %captures, i64 {})NVVM",
                   prefix,
                   slot,
                   slot);
      output_.line("  %{}_saved_capture_value_{} = load i64, i64* %{}_saved_capture_{}, align 8",
                   prefix,
                   slot,
                   prefix,
                   slot);
    }
    output_.line("  br label %{}_attempt_0", prefix);
    for (std::size_t index = 0; index < edges.size(); ++index) {
      auto failure = index + 1 < edges.size() ? fmt::format("{}_attempt_{}", prefix, index + 1)
                                              : std::string{"return_fail"};
      output_.line("{}",
                   fmt::format(
                     R"NVVM({prefix}_attempt_{index}:
  %{prefix}_child_{index} = call i1 @{run_block}(i32 {target}, i8* %data, i64 %size, i64* %position, i64* %captures, i64 %next_steps)
  br i1 %{prefix}_child_{index}, label %return_success, label %{prefix}_restore_{index}
{prefix}_restore_{index}:
  store i64 %{prefix}_saved, i64* %position, align 8)NVVM",
                     fmt::arg("prefix", prefix),
                     fmt::arg("index", index),
                     fmt::arg("run_block", name("run_block")),
                     fmt::arg("target", edges[index].target),
                     fmt::arg("failure", failure)));
      for (std::size_t slot = 0; slot < capture_slots; ++slot) {
        output_.line("  store i64 %{}_saved_capture_value_{}, i64* %{}_saved_capture_{}, align 8",
                     prefix,
                     slot,
                     prefix,
                     slot);
      }
      output_.line("  br label %{}", failure);
    }
  }

  /**
   * @brief emits the externally callable anchored or scanning regex execution function
   */
  void emit_execute()
  {
    auto run_block  = name("run_block");
    auto advance    = name("advance");
    auto multiplier = static_cast<std::uint64_t>(ir_.blocks.size()) * 8U + 32U;
    output_.line(
      "{}",
      fmt::format(R"NVVM(define zeroext i1 @{execute}(i8* %data, i64 %size) nounwind readonly {{
entry:
  %position = alloca i64, align 8
  %size_plus_one = add i64 %size, 1
  %step_limit = mul i64 %size_plus_one, {multiplier})NVVM",
                  fmt::arg("execute", options_.execute_function),
                  fmt::arg("multiplier", multiplier)));
    if (!ir_.control.scan_input) {
      output_.line("{}",
                   fmt::format(R"NVVM(  store i64 0, i64* %position, align 8
  %matched = call i1 @{run_block}(i32 {entry}, i8* %data, i64 %size, i64* %position, i64* null, i64 %step_limit)
  ret i1 %matched
}})NVVM",
                               fmt::arg("run_block", run_block),
                               fmt::arg("entry", ir_.entry)));
      output_.blank();
      return;
    }

    auto prefix = required_ascii_prefix();
    if (prefix.has_value()) {
      std::string hint;
      if (options_.branch_hints) {
        hint = R"NVVM(  %candidate_likely = call i1 @llvm.expect.i1(i1 %candidate, i1 false)
)NVVM";
      }
      auto condition = options_.branch_hints ? "%candidate_likely" : "%candidate";
      output_.line("{}",
                   fmt::format(
                     R"NVVM(  br label %search
search:
  %start = phi i64 [ 0, %entry ], [ %next_start, %next ]
  %at_end = icmp eq i64 %start, %size
  br i1 %at_end, label %attempt, label %filter
filter:
  %start_ptr = getelementptr i8, i8* %data, i64 %start
  %start_byte = call i32 @{load_byte}(i8* %start_ptr)
  %candidate = icmp eq i32 %start_byte, {prefix}
{hint}  br i1 {condition}, label %attempt, label %skip
skip:
  %is_ascii = icmp ult i32 %start_byte, 128
  br i1 %is_ascii, label %continue_ascii, label %continue_utf8
attempt:
  store i64 %start, i64* %position, align 8
  %matched = call i1 @{run_block}(i32 {entry}, i8* %data, i64 %size, i64* %position, i64* null, i64 %step_limit)
  br i1 %matched, label %yes, label %check_end
check_end:
  br i1 %at_end, label %no, label %continue_ascii
continue_ascii:
  %ascii_next = add i64 %start, 1
  br label %next
continue_utf8:
  %utf8_next = call i64 @{advance}(i8* %data, i64 %size, i64 %start, i64 1)
  br label %next
next:
  %next_start = phi i64 [ %ascii_next, %continue_ascii ], [ %utf8_next, %continue_utf8 ]
  br label %search
yes:
  ret i1 true
no:
  ret i1 false
}})NVVM",
                     fmt::arg("load_byte", name("load_byte")),
                     fmt::arg("prefix", static_cast<std::uint32_t>(*prefix)),
                     fmt::arg("hint", hint),
                     fmt::arg("condition", condition),
                     fmt::arg("run_block", run_block),
                     fmt::arg("entry", ir_.entry),
                     fmt::arg("advance", advance)));
      output_.blank();
      return;
    }

    output_.line("{}",
                 fmt::format(R"NVVM(  br label %search
search:
  %start = phi i64 [ 0, %entry ], [ %next_start, %continue ]
  store i64 %start, i64* %position, align 8
  %matched = call i1 @{run_block}(i32 {entry}, i8* %data, i64 %size, i64* %position, i64* null, i64 %step_limit)
  br i1 %matched, label %yes, label %check_end
check_end:
  %at_end = icmp eq i64 %start, %size
  br i1 %at_end, label %no, label %continue
continue:
  %next_start = call i64 @{advance}(i8* %data, i64 %size, i64 %start, i64 1)
  br label %search
yes:
  ret i1 true
no:
  ret i1 false
}})NVVM",
                             fmt::arg("run_block", run_block),
                             fmt::arg("entry", ir_.entry),
                             fmt::arg("advance", advance)));
    output_.blank();
  }

  /**
   * @brief emits the capture-enumeration ABI used by exact CUDA result materialization
   */
  void emit_capture_execute()
  {
    auto run_block  = name("run_block");
    auto advance    = name("advance");
    auto multiplier = static_cast<std::uint64_t>(ir_.blocks.size()) * 8U + 32U;
    auto slots      = static_cast<std::size_t>(ir_.capture_count + 1U) * 2U;
    output_.line(
      "{}",
      fmt::format(
        R"NVVM(define zeroext i1 @{execute}(i8* %data, i64 %size, i64 %search_start, i64* %captures) nounwind {{
entry:
  %position = alloca i64, align 8
  %size_plus_one = add i64 %size, 1
  %step_limit = mul i64 %size_plus_one, {multiplier}
  br label %search
search:
  %start = phi i64 [ %search_start, %entry ], [ %next_start, %continue ]
  %in_range = icmp ule i64 %start, %size
  br i1 %in_range, label %initialize, label %no
initialize:)NVVM",
        fmt::arg("execute", options_.execute_function),
        fmt::arg("multiplier", multiplier)));
    for (std::size_t slot = 0; slot < slots; ++slot) {
      output_.line("  %capture_ptr_{} = getelementptr i64, i64* %captures, i64 {}", slot, slot);
      output_.line("  store i64 -1, i64* %capture_ptr_{}, align 8", slot);
    }
    output_.line("{}",
                 fmt::format(
                   R"NVVM(  store i64 %start, i64* %capture_ptr_0, align 8
  store i64 %start, i64* %position, align 8
  %matched = call i1 @{run_block}(i32 {entry}, i8* %data, i64 %size, i64* %position, i64* %captures, i64 %step_limit)
  br i1 %matched, label %yes, label %check_end
check_end:
  %at_end = icmp eq i64 %start, %size
  br i1 %at_end, label %no, label %continue
continue:
  %next_start = call i64 @{advance}(i8* %data, i64 %size, i64 %start, i64 1)
  br label %search
yes:
  %match_end = load i64, i64* %position, align 8
  store i64 %match_end, i64* %capture_ptr_1, align 8
  ret i1 true
no:
  ret i1 false
}})NVVM",
                   fmt::arg("run_block", run_block),
                   fmt::arg("entry", ir_.entry),
                   fmt::arg("advance", advance)));
    output_.blank();
  }

  instruction_ir const& ir_;
  nvvm_ir_codegen_options const& options_;
  std::optional<deterministic_machine> deterministic_ = std::nullopt;
  source_buffer output_                               = {};
};

}  // namespace

std::string generate_nvvm_ir(instruction_ir const& ir, nvvm_ir_codegen_options const& options)
{
  return nvvm_ir_renderer(ir, options).render();
}

}  // namespace regex_ir
