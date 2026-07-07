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
  source_span source                          = source_span{};
  character_predicate predicate               = character_predicate{};
  assertion_kind assertion                    = assertion_kind::BEGIN_INPUT;
  std::vector<std::unique_ptr<node>> children = std::vector<std::unique_ptr<node>>{};
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

bool is_unconditional_empty(node const& value)
{
  switch (value.kind) {
    case node_kind::EMPTY: return true;
    case node_kind::GROUP: return is_unconditional_empty(*value.children.front());
    case node_kind::CONCATENATE:
    case node_kind::ALTERNATE:
      return std::all_of(value.children.begin(), value.children.end(), [](auto& child) {
        return is_unconditional_empty(*child);
      });
    case node_kind::PREDICATE:
    case node_kind::REPEAT:
    case node_kind::ASSERTION: return false;
  }
  return false;
}

bool contains_capture(node const& value)
{
  if (value.kind == node_kind::GROUP && value.capturing) return true;
  return std::any_of(value.children.begin(), value.children.end(), [](auto& child) {
    return contains_capture(*child);
  });
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

bool append_posix_class(character_predicate& predicate, std::string_view name)
{
  if (name == "alpha") {
    predicate.ranges.insert(predicate.ranges.end(), {{U'A', U'Z'}, {U'a', U'z'}});
  } else if (name == "alnum") {
    predicate.ranges.insert(predicate.ranges.end(), {{U'0', U'9'}, {U'A', U'Z'}, {U'a', U'z'}});
  } else if (name == "digit") {
    predicate.ranges.push_back({U'0', U'9'});
  } else if (name == "xdigit") {
    predicate.ranges.insert(predicate.ranges.end(), {{U'0', U'9'}, {U'A', U'F'}, {U'a', U'f'}});
  } else if (name == "space") {
    predicate.ranges.insert(predicate.ranges.end(), {{U'\t', U'\r'}, {U' ', U' '}});
  } else if (name == "word") {
    predicate.ranges.insert(predicate.ranges.end(),
                            {{U'0', U'9'}, {U'A', U'Z'}, {U'_', U'_'}, {U'a', U'z'}});
  } else if (name == "punct") {
    predicate.ranges.insert(predicate.ranges.end(),
                            {{U'!', U'/'}, {U':', U'@'}, {U'[', U'`'}, {U'{', U'~'}});
  } else {
    return false;
  }
  return true;
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

  std::vector<diagnostic> diagnostics = std::vector<diagnostic>{};
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
    auto greedy = !consume('?');
    if (peek() == '*' || peek() == '+' || peek() == '?' || peek() == '{') {
      fail(diagnostic_code::INVALID_QUANTIFIER, {position_, 1}, "multiple repeat operators");
    }
    if (is_unconditional_empty(*atom) && !contains_capture(*atom)) {
      // repeating an unconditional empty expression is still empty
      atom->source.length = position_ - start;
      return atom;
    }
    if (!can_consume_character(*atom)) {
      fail(diagnostic_code::INVALID_QUANTIFIER,
           {start, position_ - start},
           "zero-width assertions cannot be repeated");
    }
    auto result = make(node_kind::REPEAT, start);
    result->children.push_back(std::move(atom));
    result->minimum       = minimum;
    result->maximum       = maximum;
    result->greedy        = greedy;
    result->source.length = position_ - start;
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
    if (value == 'p' || value == 'P') {
      if (!consume('{')) {
        fail(diagnostic_code::INVALID_ESCAPE, {start, position_ - start}, "missing property name");
      }
      auto property_begin = position_;
      while (!at_end() && peek() != '}')
        static_cast<void>(take());
      if (!consume('}')) {
        fail(diagnostic_code::INVALID_ESCAPE,
             {start, position_ - start},
             "unterminated Unicode property");
      }
      auto property = pattern_.substr(property_begin, position_ - property_begin - 1U);
      if (property != "Sm") {
        fail(diagnostic_code::UNSUPPORTED_FEATURE,
             {start, position_ - start},
             "unsupported Unicode property");
      }
      append_unicode_ranges(result->predicate, unicode_math_symbol_ranges);
      if (value == 'P') {
        result->predicate.ranges = complement_ranges(std::move(result->predicate.ranges));
      }
      normalize_ranges(result->predicate);
    } else if (value == 'd' || value == 'D' || value == 'w' || value == 'W' || value == 's' ||
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
      if (peek() == '[' && position_ + 1U < pattern_.size() && pattern_[position_ + 1U] == ':') {
        auto class_start = position_;
        position_ += 2U;
        auto name_start = position_;
        while (position_ + 1U < pattern_.size() &&
               !(pattern_[position_] == ':' && pattern_[position_ + 1U] == ']')) {
          ++position_;
        }
        if (position_ + 1U >= pattern_.size()) {
          fail(diagnostic_code::INVALID_CHARACTER_CLASS,
               {class_start, position_ - class_start},
               "unterminated POSIX character class");
        }
        auto name = pattern_.substr(name_start, position_ - name_start);
        position_ += 2U;
        if (!append_posix_class(result->predicate, name)) {
          fail(diagnostic_code::UNSUPPORTED_FEATURE,
               {class_start, position_ - class_start},
               "unsupported POSIX character class");
        }
        continue;
      }
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
  std::vector<patch_reference> outs = std::vector<patch_reference>{};
};

class thompson_builder {
 public:
  thompson_builder(std::string_view pattern, compile_options const& options)
  {
    ir.pattern = std::string(pattern);
    ir.options = options;
  }

  automata_ir ir                      = automata_ir{};
  std::vector<diagnostic> diagnostics = std::vector<diagnostic>{};

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
  std::optional<std::size_t> begin = std::nullopt;
  std::optional<std::size_t> end   = std::nullopt;
};

struct found_match {
  std::size_t begin                   = 0;
  std::size_t end                     = 0;
  std::vector<capture_value> captures = std::vector<capture_value>{};
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
  std::vector<bool> observed(ir.capture_count + 1U, false);
  if (ir.selected_operation.kind == operation_kind::EXTRACT) {
    std::fill(observed.begin(), observed.end(), true);
  } else if (ir.selected_operation.kind == operation_kind::REPLACE) {
    for (auto& token : ir.replacement) {
      if (token.type == replacement_token::kind::CAPTURE) { observed[token.capture_index] = true; }
    }
  }
  for (auto& block : ir.blocks) {
    block.instructions.erase(std::remove_if(block.instructions.begin(),
                                            block.instructions.end(),
                                            [&](instruction const& item) {
                                              auto* capture = std::get_if<write_capture>(&item);
                                              return capture != nullptr &&
                                                     !observed[capture->capture_index];
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
  void emit(fmt::format_string<Args...> format, Args&&... args)
  {
    append(format, std::forward<Args>(args)...);
    value_ += '\n';
  }

  void blank() { value_ += '\n'; }
  [[nodiscard]] std::string take() { return std::move(value_); }

 private:
  std::string value_ = "";
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
  character_predicate predicate           = character_predicate{};
  std::vector<std::size_t> targets        = std::vector<std::size_t>{};
  std::optional<write_capture> capture    = std::nullopt;
  std::optional<assertion_kind> assertion = std::nullopt;
  bool consumes : 1                       = false;
  bool accepts  : 1                       = false;
};

struct deterministic_nfa_graph {
  std::vector<deterministic_nfa_node> nodes = std::vector<deterministic_nfa_node>{};
  std::size_t entry                         = 0;
};

struct deterministic_capture_action {
  std::uint32_t slot = 0;
  bool reset_end : 1 = false;

  bool operator==(deterministic_capture_action const&) const = default;
};

struct deterministic_interval {
  std::uint32_t first    = 0;
  std::uint32_t last     = 0;
  std::uint16_t class_id = 0;
};

struct deterministic_machine {
  std::array<std::uint16_t, 256> byte_classes           = std::array<std::uint16_t, 256>{};
  std::array<std::uint64_t, 4> start_byte_bitmap        = std::array<std::uint64_t, 4>{};
  std::vector<deterministic_interval> unicode_intervals = std::vector<deterministic_interval>{};
  std::vector<std::uint16_t> transitions                = std::vector<std::uint16_t>{};
  std::vector<std::vector<deterministic_capture_action>> transition_capture_actions =
    std::vector<std::vector<deterministic_capture_action>>{};
  std::vector<std::vector<deterministic_capture_action>> accept_capture_actions =
    std::vector<std::vector<deterministic_capture_action>>{};
  std::vector<std::uint8_t> boundary_accepts     = std::vector<std::uint8_t>{};
  std::uint16_t initial_state                    = 0;
  std::uint16_t dead_state                       = 0;
  std::uint16_t class_count                      = 0;
  std::uint16_t state_count                      = 0;
  std::uint16_t state_mask                       = 16383;
  std::uint16_t restart_state                    = std::numeric_limits<std::uint16_t>::max();
  std::uint8_t transition_address_space          = 4;
  std::uint8_t boundary_class_count              = 0;
  std::uint8_t assertion_mask                    = 0;
  std::uint8_t start_byte_range_count            = 0;
  std::optional<assertion_kind> accept_assertion = std::nullopt;
  bool scan_input        : 1                     = false;
  bool accept_at_end     : 1                     = false;
  bool capture_one_pass  : 1                     = false;
  bool assertion_aware   : 1                     = false;
  bool start_byte_filter : 1                     = false;
};

void build_start_byte_filter(deterministic_machine& machine)
{
  if ((machine.initial_state & 0x8000U) != 0 || machine.dead_state > machine.state_mask ||
      machine.class_count == 0U) {
    return;
  }
  auto initial = static_cast<std::size_t>(machine.initial_state & machine.state_mask);
  auto offset  = initial * machine.class_count;
  if (offset + machine.class_count > machine.transitions.size()) return;

  std::size_t candidates        = 0;
  auto previous_ascii_candidate = false;
  for (std::size_t byte = 0; byte < machine.byte_classes.size(); ++byte) {
    auto transition = machine.transitions[offset + machine.byte_classes[byte]];
    auto target     = static_cast<std::uint16_t>(transition & machine.state_mask);
    auto candidate  = target != machine.dead_state;
    if (byte < 128U && candidate && !previous_ascii_candidate) { ++machine.start_byte_range_count; }
    if (byte < 128U) previous_ascii_candidate = candidate;
    if (!candidate) continue;
    machine.start_byte_bitmap[byte / 64U] |= std::uint64_t{1} << (byte % 64U);
    ++candidates;
  }
  // sparse, simple ranges repay the extra candidate-dispatch control flow.
  machine.start_byte_filter =
    candidates != 0U && candidates <= 16U && machine.start_byte_range_count <= 2U;
}

void build_restart_acceleration(deterministic_machine& machine)
{
  if ((machine.initial_state & 0x8000U) != 0 || machine.class_count == 0U ||
      machine.dead_state > machine.state_mask) {
    return;
  }
  auto initial    = static_cast<std::uint16_t>(machine.initial_state & machine.state_mask);
  auto transition = [&](std::uint16_t state, std::uint16_t character_class) {
    return machine
      .transitions[static_cast<std::size_t>(state) * machine.class_count + character_class];
  };

  auto prefix_state = std::numeric_limits<std::uint16_t>::max();
  std::vector<bool> prefix_classes(machine.class_count);
  // every skipped suffix must be in the same prefix state at the failure byte.
  for (std::uint16_t character_class = 0; character_class < machine.class_count;
       ++character_class) {
    auto encoded = transition(initial, character_class);
    auto target  = static_cast<std::uint16_t>(encoded & machine.state_mask);
    if (target == machine.dead_state) continue;
    if ((encoded & 0xC000U) != 0U || target == initial) return;
    if (prefix_state == std::numeric_limits<std::uint16_t>::max()) {
      prefix_state = target;
    } else if (target != prefix_state) {
      return;
    }
    prefix_classes[character_class] = true;
  }
  if (prefix_state == std::numeric_limits<std::uint16_t>::max()) return;

  for (std::uint16_t character_class = 0; character_class < machine.class_count;
       ++character_class) {
    if (!prefix_classes[character_class]) continue;
    auto encoded = transition(prefix_state, character_class);
    if ((encoded & 0x4000U) != 0U || (encoded & machine.state_mask) != prefix_state) return;
  }
  for (std::uint16_t state = 0; state < machine.state_count; ++state) {
    if (state == initial || state == prefix_state) continue;
    for (std::uint16_t character_class = 0; character_class < machine.class_count;
         ++character_class) {
      if ((transition(state, character_class) & machine.state_mask) == prefix_state) return;
    }
  }
  machine.restart_state = prefix_state;
}

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

std::optional<deterministic_nfa_graph> make_deterministic_graph(instruction_ir const& ir)
{
  if (ir.entry >= ir.blocks.size()) return std::nullopt;
  std::vector<std::size_t> block_starts(ir.blocks.size());
  std::vector<std::size_t> block_lengths(ir.blocks.size(), 1);
  std::size_t node_count = 0;
  for (auto& block : ir.blocks) {
    match_character const* match    = nullptr;
    match_literal const* literal    = nullptr;
    can_peek const* peek            = nullptr;
    advance_cursor const* advance   = nullptr;
    test_assertion const* assertion = nullptr;
    bool accepts                    = false;
    for (auto& item : block.instructions) {
      if (auto* candidate = std::get_if<match_character>(&item)) match = candidate;
      if (auto* candidate = std::get_if<match_literal>(&item)) literal = candidate;
      if (auto* candidate = std::get_if<can_peek>(&item)) peek = candidate;
      if (auto* candidate = std::get_if<advance_cursor>(&item)) advance = candidate;
      if (auto* candidate = std::get_if<test_assertion>(&item)) {
        if (assertion != nullptr) return std::nullopt;
        assertion = candidate;
      }
      if (std::holds_alternative<emit_accept>(item)) accepts = true;
    }
    if (match != nullptr && literal != nullptr) return std::nullopt;
    if ((match != nullptr || literal != nullptr) && (accepts || assertion != nullptr)) {
      return std::nullopt;
    }
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

  deterministic_nfa_graph graph;
  graph.nodes.resize(node_count);
  graph.entry = block_starts[ir.entry];
  for (auto& block : ir.blocks) {
    auto start                      = block_starts[block.id];
    match_character const* match    = nullptr;
    match_literal const* literal    = nullptr;
    write_capture const* capture    = nullptr;
    test_assertion const* assertion = nullptr;
    for (auto& item : block.instructions) {
      if (auto* candidate = std::get_if<match_character>(&item)) match = candidate;
      if (auto* candidate = std::get_if<match_literal>(&item)) literal = candidate;
      if (auto* candidate = std::get_if<write_capture>(&item)) capture = candidate;
      if (auto* candidate = std::get_if<test_assertion>(&item)) assertion = candidate;
      if (std::holds_alternative<emit_accept>(item)) graph.nodes[start].accepts = true;
    }
    if (capture != nullptr) graph.nodes[start].capture = *capture;
    if (assertion != nullptr) graph.nodes[start].assertion = assertion->kind;

    auto append_successors = [&](deterministic_nfa_node& node) {
      auto successors = block.successors;
      std::stable_sort(successors.begin(), successors.end(), [](auto& left, auto& right) {
        return left.priority < right.priority;
      });
      for (auto edge : successors)
        node.targets.push_back(block_starts[edge.target]);
    };

    if (match != nullptr) {
      graph.nodes[start].predicate = match->predicate;
      graph.nodes[start].consumes  = true;
      append_successors(graph.nodes[start]);
    } else if (literal != nullptr) {
      for (std::size_t index = 0; index < literal->value.size(); ++index) {
        auto& node     = graph.nodes[start + index];
        node.predicate = singleton_predicate(literal->value[index]);
        node.consumes  = true;
        if (index + 1 < literal->value.size()) {
          node.targets.push_back(start + index + 1);
        } else {
          append_successors(node);
        }
      }
    } else {
      append_successors(graph.nodes[start]);
    }
  }
  return graph;
}

std::optional<std::vector<std::uint32_t>> make_deterministic_alphabet(
  std::vector<deterministic_nfa_node> const& nodes, deterministic_machine& machine)
{
  constexpr std::uint32_t unicode_limit = 0x110000;
  auto word_count                       = (nodes.size() + 63U) / 64U;
  std::vector<std::uint32_t> boundaries{0, 256, unicode_limit};
  for (auto& node : nodes) {
    if (!node.consumes) continue;
    if (node.predicate.recognized == predicate_class::ANY && !node.predicate.matches_newline) {
      boundaries.insert(boundaries.end(), {10, 11});
      if (node.predicate.extended_newline) {
        boundaries.insert(boundaries.end(), {13, 14, 133, 134, 8232, 8234});
      }
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
    std::vector<std::uint64_t> signature(word_count);
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

  machine.class_count        = static_cast<std::uint16_t>(class_ids.size());
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
  return representatives;
}

std::uint8_t assertion_bit(assertion_kind assertion)
{
  return static_cast<std::uint8_t>(1U << static_cast<std::uint8_t>(assertion));
}

std::optional<deterministic_machine> make_assertion_deterministic_machine(
  instruction_ir const& ir, deterministic_nfa_graph const& graph, bool scan_input)
{
  deterministic_machine machine;
  machine.scan_input      = scan_input;
  machine.accept_at_end   = ir.control.require_end;
  machine.state_mask      = 32767U;
  machine.assertion_aware = true;
  for (auto& node : graph.nodes) {
    if (node.capture.has_value()) return std::nullopt;
    if (node.assertion.has_value()) machine.assertion_mask |= assertion_bit(*node.assertion);
  }
  if (machine.assertion_mask == 0) return std::nullopt;

  std::vector<std::uint8_t> context_masks;
  for (std::uint8_t mask = 0; mask < 64U; ++mask) {
    auto relevant = static_cast<std::uint8_t>(mask & machine.assertion_mask);
    if (std::find(context_masks.begin(), context_masks.end(), relevant) == context_masks.end()) {
      context_masks.push_back(relevant);
    }
  }
  machine.boundary_class_count = static_cast<std::uint8_t>(context_masks.size());

  auto representatives = make_deterministic_alphabet(graph.nodes, machine);
  if (!representatives) return std::nullopt;

  struct assertion_closure {
    std::vector<std::size_t> consumers = std::vector<std::size_t>{};
    bool accepts : 1                   = false;
  };
  auto close = [&](std::vector<std::size_t> seeds, std::uint8_t assertion_mask) {
    assertion_closure result;
    std::vector<bool> visited(graph.nodes.size());
    std::vector<std::size_t> work;
    for (auto seed = seeds.rbegin(); seed != seeds.rend(); ++seed)
      work.push_back(*seed);
    while (!work.empty()) {
      auto index = work.back();
      work.pop_back();
      if (visited[index]) continue;
      visited[index] = true;
      auto& node     = graph.nodes[index];
      if (node.assertion.has_value() && (assertion_mask & assertion_bit(*node.assertion)) == 0) {
        continue;
      }
      if (node.accepts) result.accepts = true;
      if (node.consumes) {
        result.consumers.push_back(index);
        continue;
      }
      for (auto target = node.targets.rbegin(); target != node.targets.rend(); ++target)
        work.push_back(*target);
    }
    std::sort(result.consumers.begin(), result.consumers.end());
    result.consumers.erase(std::unique(result.consumers.begin(), result.consumers.end()),
                           result.consumers.end());
    return result;
  };
  auto canonicalize = [](std::vector<std::size_t>& state) {
    std::sort(state.begin(), state.end());
    state.erase(std::unique(state.begin(), state.end()), state.end());
  };

  constexpr std::size_t max_table_items = 4U * 1024U * 1024U;
  std::map<std::vector<std::size_t>, std::uint16_t> state_ids;
  std::vector<std::vector<std::size_t>> states{{graph.entry}};
  state_ids.emplace(states.front(), 0);
  machine.dead_state = std::numeric_limits<std::uint16_t>::max();
  for (std::size_t state_index = 0; state_index < states.size(); ++state_index) {
    if (states.size() * context_masks.size() * representatives->size() > max_table_items) {
      return std::nullopt;
    }
    for (auto context : context_masks) {
      auto seeds = states[state_index];
      if (scan_input) {
        seeds.push_back(graph.entry);
        canonicalize(seeds);
      }
      auto closure = close(std::move(seeds), context);
      machine.boundary_accepts.push_back(closure.accepts ? 1U : 0U);
      for (auto representative : *representatives) {
        std::vector<std::size_t> next;
        for (auto node_index : closure.consumers) {
          auto& node = graph.nodes[node_index];
          if (!node.predicate.matches(static_cast<char32_t>(representative))) continue;
          next.insert(next.end(), node.targets.begin(), node.targets.end());
        }
        canonicalize(next);
        auto existing        = state_ids.find(next);
        std::uint16_t target = 0;
        if (existing == state_ids.end()) {
          if (states.size() >= machine.state_mask) return std::nullopt;
          target = static_cast<std::uint16_t>(states.size());
          state_ids.emplace(next, target);
          states.push_back(std::move(next));
        } else {
          target = existing->second;
        }
        if (states[target].empty()) machine.dead_state = target;
        machine.transitions.push_back(
          static_cast<std::uint16_t>(target | (closure.accepts ? 0x8000U : 0U)));
      }
    }
  }
  machine.initial_state = 0;
  machine.state_count   = static_cast<std::uint16_t>(states.size());
  if (machine.transitions.size() * sizeof(std::uint16_t) > 32U * 1024U) {
    machine.transition_address_space = 1;
  }
  return machine;
}

std::optional<deterministic_machine> make_deterministic_machine(instruction_ir const& ir,
                                                                bool scan_input,
                                                                bool preserve_priority)
{
  auto block_assertion = [](instruction_block const& block) -> std::optional<assertion_kind> {
    if (block.instructions.size() != 1U ||
        !std::holds_alternative<test_assertion>(block.instructions.front())) {
      return std::nullopt;
    }
    return std::get<test_assertion>(block.instructions.front()).kind;
  };
  std::optional<block_id> begin_anchor;
  std::optional<block_id> end_anchor;
  std::optional<assertion_kind> end_assertion;
  bool has_assertions = false;
  if (ir.entry < ir.blocks.size()) {
    auto assertion = block_assertion(ir.blocks[ir.entry]);
    if (assertion == assertion_kind::BEGIN_INPUT ||
        (assertion == assertion_kind::BEGIN_LINE && !ir.options.multiline)) {
      begin_anchor = ir.entry;
    }
  }
  if (ir.accept < ir.blocks.size()) {
    std::vector<block_id> predecessors;
    for (auto& block : ir.blocks) {
      for (auto edge : block.successors) {
        if (edge.target == ir.accept) predecessors.push_back(block.id);
      }
    }
    if (predecessors.size() == 1U) {
      auto assertion = block_assertion(ir.blocks[predecessors.front()]);
      if (assertion == assertion_kind::END_INPUT || assertion == assertion_kind::END_LINE) {
        end_anchor    = predecessors.front();
        end_assertion = assertion;
      }
    }
  }
  for (auto& block : ir.blocks) {
    for (auto& item : block.instructions) {
      if (std::holds_alternative<test_assertion>(item)) {
        has_assertions = true;
        if (preserve_priority && block.id != begin_anchor && block.id != end_anchor) {
          return std::nullopt;
        }
      }
    }
  }
  if (has_assertions && !preserve_priority) {
    auto graph = make_deterministic_graph(ir);
    if (!graph) return std::nullopt;
    return make_assertion_deterministic_machine(ir, *graph, scan_input);
  }
  if ((begin_anchor.has_value() || end_anchor.has_value()) &&
      ir.control.result != result_shape::BOOLEAN) {
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
    write_capture const* capture = nullptr;
    for (auto& item : block.instructions) {
      if (auto* candidate = std::get_if<match_character>(&item)) match = candidate;
      if (auto* candidate = std::get_if<match_literal>(&item)) literal = candidate;
      if (auto* candidate = std::get_if<write_capture>(&item)) capture = candidate;
      if (std::holds_alternative<emit_accept>(item)) nodes[start].accepts = true;
    }
    if (capture != nullptr) nodes[start].capture = *capture;

    auto append_successors = [&](deterministic_nfa_node& node) {
      auto successors = block.successors;
      std::stable_sort(successors.begin(), successors.end(), [](auto& left, auto& right) {
        return left.priority < right.priority;
      });
      for (auto edge : successors)
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
  struct closure_state {
    std::vector<std::uint64_t> bits  = std::vector<std::uint64_t>{};
    std::vector<std::size_t> ordered = std::vector<std::size_t>{};
    std::vector<std::vector<deterministic_capture_action>> captures =
      std::vector<std::vector<deterministic_capture_action>>{};
  };
  struct closure_work_item {
    std::size_t node                                  = 0;
    std::vector<deterministic_capture_action> actions = std::vector<deterministic_capture_action>{};
  };
  auto capture_paths_deterministic = true;
  auto empty_bits                  = [&] { return std::vector<std::uint64_t>(word_count); };
  auto closure                     = [&](std::vector<std::size_t> const& seeds) {
    closure_state result{empty_bits(), {}, {}};
    auto visited = empty_bits();
    std::vector<std::optional<std::vector<deterministic_capture_action>>> visited_actions(
      nodes.size());
    std::vector<closure_work_item> work;
    for (auto seed = seeds.rbegin(); seed != seeds.rend(); ++seed)
      work.push_back({*seed, {}});
    while (!work.empty()) {
      auto current = std::move(work.back());
      work.pop_back();
      auto index = current.node;
      if (machine_bit(visited, index)) {
        if (visited_actions[index] != current.actions) capture_paths_deterministic = false;
        continue;
      }
      set_machine_bit(visited, index);
      visited_actions[index] = current.actions;
      auto& node             = nodes[index];
      if (node.capture.has_value()) {
        auto slot = node.capture->capture_index * 2U +
                    (node.capture->action == capture_action::END ? 1U : 0U);
        current.actions.push_back({slot, node.capture->action == capture_action::BEGIN});
      }
      if (node.accepts) {
        set_machine_bit(result.bits, accept_bit);
        result.ordered.push_back(accept_bit);
        result.captures.push_back(current.actions);
        if (preserve_priority) break;
      }
      if (node.consumes) {
        set_machine_bit(result.bits, index);
        result.ordered.push_back(index);
        result.captures.push_back(current.actions);
        continue;
      }
      for (auto target = node.targets.rbegin(); target != node.targets.rend(); ++target) {
        work.push_back({*target, current.actions});
      }
    }
    if (!preserve_priority) {
      std::sort(result.ordered.begin(), result.ordered.end());
      result.ordered.erase(std::unique(result.ordered.begin(), result.ordered.end()),
                           result.ordered.end());
      result.captures.assign(result.ordered.size(), {});
    }
    return result;
  };

  auto start_state = closure({block_starts[ir.entry]});

  constexpr std::uint32_t unicode_limit = 0x110000;
  std::vector<std::uint32_t> boundaries{0, 256, unicode_limit};
  for (auto& node : nodes) {
    if (!node.consumes) continue;
    if (node.predicate.recognized == predicate_class::ANY && !node.predicate.matches_newline) {
      boundaries.insert(boundaries.end(), {10, 11});
      if (node.predicate.extended_newline) {
        boundaries.insert(boundaries.end(), {13, 14, 133, 134, 8232, 8234});
      }
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
  machine.scan_input         = scan_input && !begin_anchor.has_value();
  machine.accept_at_end      = ir.control.require_end;
  machine.accept_assertion   = end_assertion;
  machine.state_mask         = preserve_priority ? 16383U : 32767U;
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

  auto max_dfa_states                   = static_cast<std::size_t>(machine.state_mask);
  constexpr std::size_t max_table_items = 4 * 1024 * 1024;
  std::map<std::vector<std::size_t>, std::uint16_t> state_ids;
  std::vector<std::vector<std::uint64_t>> states;
  std::vector<std::vector<std::size_t>> state_orders;
  std::vector<std::vector<std::vector<deterministic_capture_action>>> state_capture_actions;
  state_ids.emplace(start_state.ordered, 0);
  states.push_back(start_state.bits);
  state_orders.push_back(start_state.ordered);
  state_capture_actions.push_back(start_state.captures);
  machine.dead_state   = std::numeric_limits<std::uint16_t>::max();
  auto strict_one_pass = !machine_bit(start_state.bits, accept_bit);
  auto terminal_accept = true;
  for (std::size_t state_index = 0; state_index < states.size(); ++state_index) {
    if (states.size() * representatives.size() > max_table_items) return std::nullopt;
    for (auto representative : representatives) {
      std::vector<std::size_t> seeds;
      std::vector<deterministic_capture_action> transition_captures;
      std::vector<deterministic_capture_action> deferred_accept_captures;
      auto matching_consumers  = 0U;
      auto has_deferred_accept = false;
      for (std::size_t order_index = 0; order_index < state_orders[state_index].size();
           ++order_index) {
        auto node_index = state_orders[state_index][order_index];
        if (node_index == accept_bit) {
          if (preserve_priority) {
            has_deferred_accept      = true;
            deferred_accept_captures = state_capture_actions[state_index][order_index];
            break;
          }
          continue;
        }
        auto& node = nodes[node_index];
        if (!node.predicate.matches(static_cast<char32_t>(representative))) continue;
        if (matching_consumers++ == 0U) {
          transition_captures = state_capture_actions[state_index][order_index];
        }
        seeds.insert(seeds.end(), node.targets.begin(), node.targets.end());
      }
      if (matching_consumers > 1U) strict_one_pass = false;
      auto next              = closure(seeds);
      auto discovered_accept = machine_bit(next.bits, accept_bit);
      auto stop_before       = preserve_priority && has_deferred_accept && matching_consumers == 0U;
      if (preserve_priority && has_deferred_accept && matching_consumers != 0U &&
          !discovered_accept) {
        // preserve a lower-priority accept while descendants of earlier threads continue.
        set_machine_bit(next.bits, accept_bit);
        next.ordered.push_back(accept_bit);
        next.captures.push_back(std::move(deferred_accept_captures));
      }
      if (machine.scan_input) {
        for (std::size_t word = 0; word < next.bits.size(); ++word)
          next.bits[word] |= start_state.bits[word];
        for (std::size_t start_index = 0; start_index < start_state.ordered.size(); ++start_index) {
          auto node_index = start_state.ordered[start_index];
          if (std::find(next.ordered.begin(), next.ordered.end(), node_index) ==
              next.ordered.end()) {
            next.ordered.push_back(node_index);
            next.captures.push_back(start_state.captures[start_index]);
          }
        }
        if (!preserve_priority) {
          std::sort(next.ordered.begin(), next.ordered.end());
          next.ordered.erase(std::unique(next.ordered.begin(), next.ordered.end()),
                             next.ordered.end());
          next.captures.assign(next.ordered.size(), {});
        }
      }
      auto existing        = state_ids.find(next.ordered);
      std::uint16_t target = 0;
      if (existing == state_ids.end()) {
        if (states.size() >= max_dfa_states) return std::nullopt;
        target = static_cast<std::uint16_t>(states.size());
        state_ids.emplace(next.ordered, target);
        states.push_back(std::move(next.bits));
        state_orders.push_back(std::move(next.ordered));
        state_capture_actions.push_back(std::move(next.captures));
      } else {
        target = existing->second;
        if (state_capture_actions[target] != next.captures) { capture_paths_deterministic = false; }
      }
      if (std::none_of(
            states[target].begin(), states[target].end(), [](auto word) { return word != 0; })) {
        machine.dead_state = target;
      }
      auto accepts  = machine_bit(states[target], accept_bit);
      auto consumes = std::any_of(state_orders[target].begin(),
                                  state_orders[target].end(),
                                  [&](auto node_index) { return node_index != accept_bit; });
      if (accepts && consumes) terminal_accept = false;
      std::vector<deterministic_capture_action> accept_captures;
      if (accepts) {
        auto accept =
          std::find(state_orders[target].begin(), state_orders[target].end(), accept_bit);
        auto index      = static_cast<std::size_t>(accept - state_orders[target].begin());
        accept_captures = state_capture_actions[target][index];
      }
      auto update_accept =
        preserve_priority ? discovered_accept : machine_bit(states[target], accept_bit);
      if (update_accept) target |= 0x8000U;
      if (stop_before) target |= 0x4000U;
      machine.transitions.push_back(target);
      machine.transition_capture_actions.push_back(std::move(transition_captures));
      machine.accept_capture_actions.push_back(std::move(accept_captures));
    }
  }
  machine.state_count      = static_cast<std::uint16_t>(states.size());
  machine.initial_state    = machine_bit(start_state.bits, accept_bit) ? 0x8000U : 0U;
  machine.capture_one_pass = strict_one_pass && terminal_accept && capture_paths_deterministic;
  if (machine.transitions.size() * sizeof(std::uint16_t) > 32U * 1024U) {
    machine.transition_address_space = 1;
  }
  build_start_byte_filter(machine);
  build_restart_acceleration(machine);
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
    capture_slots_      = live_capture_slots();
    auto boolean_result = ir_.control.result == result_shape::BOOLEAN;
    ascii_literal_      = exact_ascii_literal();
    // short early-hit scans favor the compact DFA; long literals repay wide candidate scans.
    if (boolean_result && ir_.control.scan_input && ascii_literal_.has_value() &&
        ascii_literal_->size() > 1U && ascii_literal_->size() < 16U) {
      ascii_literal_.reset();
    }
    if (!ascii_literal_.has_value()) {
      if (boolean_result && begins_at_input_start() &&
          ir_.blocks[ir_.entry].successors.size() == 1) {
        // the selected successor is already constrained to byte zero by the removed assertion.
        auto anchored_ir               = ir_;
        anchored_ir.entry              = ir_.blocks[ir_.entry].successors.front().target;
        anchored_ir.control.scan_input = false;
        anchored_ir.blocks[ir_.entry].instructions.clear();
        anchored_ir.blocks[ir_.entry].successors.clear();
        deterministic_ = make_deterministic_machine(anchored_ir, false, false);
      } else {
        deterministic_ = make_deterministic_machine(
          ir_, boolean_result && ir_.control.scan_input, !boolean_result);
      }
    }
    auto tagged_result = ir_.control.result == result_shape::CAPTURES &&
                         deterministic_.has_value() && deterministic_->capture_one_pass;
    if (!boolean_result &&
        (!deterministic_.has_value() || (uses_capture_buffer() && !tagged_result))) {
      deterministic_.reset();
    }
    auto executor = std::string_view{"recursive Thompson"};
    if (ascii_literal_.has_value()) {
      executor =
        ascii_literal_->size() == 1U ? "single-byte literal scan" : "packed ASCII literal scan";
    } else if (deterministic_.has_value()) {
      executor = deterministic_->assertion_aware ? "assertion-aware deterministic table"
                 : boolean_result                ? "deterministic table"
                 : tagged_result                 ? "tagged prioritized deterministic table"
                                                 : "prioritized deterministic table";
    }
    output_.emit("{}",
                 fmt::format(R"NVVM(; NVVM IR generated by Regex IR
; pattern: {pattern}
target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
; executor: {executor})NVVM",
                             fmt::arg("pattern", escaped_comment(ir_.pattern)),
                             fmt::arg("executor", executor)));
    if (deterministic_.has_value()) {
      output_.emit("; dfa states: {}, alphabet classes: {}",
                   deterministic_->state_count,
                   deterministic_->class_count);
    }
    output_.blank();

    emit_load_byte();
    emit_decode_width();
    emit_decode_codepoint();
    if (ascii_literal_.has_value() && boolean_result) {
      emit_ascii_literal_execute(*ascii_literal_);
    } else if (deterministic_.has_value() && boolean_result) {
      if (deterministic_->assertion_aware) {
        auto word_assertions =
          static_cast<std::uint8_t>(assertion_bit(assertion_kind::WORD_BOUNDARY) |
                                    assertion_bit(assertion_kind::NOT_WORD_BOUNDARY));
        if ((deterministic_->assertion_mask & word_assertions) != 0) emit_is_word();
      } else if (deterministic_->accept_assertion.has_value()) {
        emit_advance();
        emit_is_word();
        emit_previous_position();
        emit_assertion();
      }
      emit_deterministic_globals(*deterministic_);
      emit_deterministic_classifier(*deterministic_);
      if (deterministic_->assertion_aware) {
        emit_deterministic_boundary_classifier(*deterministic_);
        emit_assertion_deterministic_execute(*deterministic_);
      } else {
        emit_deterministic_execute(*deterministic_);
      }
    } else {
      emit_optimizer_intrinsics();
      emit_advance();
      emit_replacement_globals();
      if (ascii_literal_.has_value()) {
        if (ascii_literal_->size() == 1U) {
          emit_single_byte_find_from(static_cast<std::uint8_t>(ascii_literal_->front()));
        } else {
          emit_ascii_literal_at(*ascii_literal_);
          emit_ascii_literal_find_from(*ascii_literal_);
        }
      } else if (deterministic_.has_value()) {
        emit_deterministic_globals(*deterministic_);
        emit_deterministic_classifier(*deterministic_);
        if (tagged_result) {
          emit_tagged_deterministic_find_from(*deterministic_);
        } else {
          emit_deterministic_find_from(*deterministic_);
        }
      } else {
        emit_can_peek();
        emit_is_word();
        emit_previous_position();
        emit_assertion();
        emit_predicate_helpers();
        emit_literal_helpers();
        emit_blocks();
      }
      if (boolean_result) {
        emit_execute();
      } else {
        if (!deterministic_.has_value() && !ascii_literal_.has_value()) emit_find_from();
        switch (ir_.control.result) {
          case result_shape::MATCH_SPAN: emit_find_execute(); break;
          case result_shape::MATCH_COUNT: emit_count_execute(); break;
          case result_shape::CAPTURES: emit_capture_execute(); break;
          case result_shape::REPLACEMENT: emit_replace_execute(); break;
          case result_shape::SPLIT_FIELDS: emit_split_execute(); break;
          case result_shape::BOOLEAN: break;
        }
      }
    }
    output_.emit(
      R"NVVM(!nvvmir.version = !{{!0}}
!0 = !{{i32 2, i32 0}})NVVM");
    return output_.take();
  }

 private:
  [[nodiscard]] std::string name(std::string_view suffix) const
  {
    return nvvm_symbol(options_.symbol_prefix, suffix);
  }

  [[nodiscard]] std::optional<std::string> exact_ascii_literal() const
  {
    if (ir_.entry >= ir_.blocks.size() || ir_.accept >= ir_.blocks.size()) { return std::nullopt; }

    std::vector<bool> visited(ir_.blocks.size(), false);
    std::string literal;
    auto current = ir_.entry;
    while (current < ir_.blocks.size() && !visited[current]) {
      visited[current] = true;
      auto& block      = ir_.blocks[current];
      if (current == ir_.accept) {
        auto accepting = block.instructions.size() == 1U &&
                         std::holds_alternative<emit_accept>(block.instructions.front()) &&
                         block.successors.empty();
        return accepting && !literal.empty() ? std::optional<std::string>{std::move(literal)}
                                             : std::nullopt;
      }
      if (block.successors.size() != 1U) return std::nullopt;

      std::u32string consumed;
      std::optional<std::uint32_t> peek_count;
      std::optional<std::uint32_t> advance_count;
      auto reads_character = false;
      for (auto& item : block.instructions) {
        if (auto* peek = std::get_if<can_peek>(&item)) {
          if (peek_count.has_value()) return std::nullopt;
          peek_count = peek->characters;
        } else if (std::holds_alternative<read_character>(item)) {
          if (reads_character) return std::nullopt;
          reads_character = true;
        } else if (auto* character_match = std::get_if<match_character>(&item)) {
          if (!consumed.empty() || !character_match->predicate.is_singleton()) {
            return std::nullopt;
          }
          consumed.push_back(character_match->predicate.singleton());
        } else if (auto* literal_match = std::get_if<match_literal>(&item)) {
          if (!consumed.empty()) return std::nullopt;
          consumed = literal_match->value;
        } else if (auto* advance = std::get_if<advance_cursor>(&item)) {
          if (advance_count.has_value()) return std::nullopt;
          advance_count = advance->characters;
        } else {
          return std::nullopt;
        }
      }

      if (!block.instructions.empty()) {
        auto count = static_cast<std::uint32_t>(consumed.size());
        if (count == 0U || !peek_count.has_value() || *peek_count != count ||
            !advance_count.has_value() || *advance_count != count ||
            (reads_character && count != 1U)) {
          return std::nullopt;
        }
        for (auto codepoint : consumed) {
          if (codepoint > 0x7f) return std::nullopt;
          literal.push_back(static_cast<char>(codepoint));
        }
      }
      current = block.successors.front().target;
    }
    return std::nullopt;
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

  [[nodiscard]] bool begins_at_input_start() const
  {
    if (ir_.entry >= ir_.blocks.size()) return false;
    auto& entry = ir_.blocks[ir_.entry];
    if (entry.instructions.size() != 1U ||
        !std::holds_alternative<test_assertion>(entry.instructions.front())) {
      return false;
    }
    auto assertion = std::get<test_assertion>(entry.instructions.front()).kind;
    return assertion == assertion_kind::BEGIN_INPUT ||
           (assertion == assertion_kind::BEGIN_LINE && !ir_.options.multiline);
  }

  [[nodiscard]] std::vector<std::size_t> live_capture_slots() const
  {
    std::vector<bool> live(static_cast<std::size_t>(ir_.capture_count + 1U) * 2U, false);
    if (ir_.control.result == result_shape::CAPTURES) {
      std::fill(live.begin(), live.end(), true);
    } else if (ir_.control.result == result_shape::REPLACEMENT) {
      for (auto& token : ir_.replacement) {
        if (token.type == replacement_token::kind::CAPTURE && token.capture_index != 0 &&
            !is_whole_match_capture(token.capture_index)) {
          auto slot       = static_cast<std::size_t>(token.capture_index) * 2U;
          live[slot]      = true;
          live[slot + 1U] = true;
        }
      }
    }
    std::vector<std::size_t> result;
    for (std::size_t slot = 0; slot < live.size(); ++slot) {
      if (live[slot]) result.push_back(slot);
    }
    return result;
  }

  [[nodiscard]] bool uses_capture_buffer() const { return !capture_slots_.empty(); }

  [[nodiscard]] bool is_whole_match_capture(std::uint32_t capture_index) const
  {
    auto is_capture = [&](instruction_block const& block, capture_action action) {
      return block.instructions.size() == 1U &&
             std::holds_alternative<write_capture>(block.instructions.front()) &&
             std::get<write_capture>(block.instructions.front()).capture_index == capture_index &&
             std::get<write_capture>(block.instructions.front()).action == action;
    };

    std::vector<bool> visited(ir_.blocks.size(), false);
    auto block           = ir_.entry;
    auto begins_at_start = false;
    while (block < ir_.blocks.size() && !visited[block]) {
      visited[block]  = true;
      auto& candidate = ir_.blocks[block];
      if (is_capture(candidate, capture_action::BEGIN)) begins_at_start = true;
      if (!candidate.instructions.empty() && !is_capture(candidate, capture_action::BEGIN)) {
        break;
      }
      if (candidate.successors.size() != 1U) break;
      block = candidate.successors.front().target;
    }
    if (!begins_at_start) return false;

    std::vector<std::vector<block_id>> predecessors(ir_.blocks.size());
    for (auto& candidate : ir_.blocks) {
      for (auto edge : candidate.successors)
        predecessors[edge.target].push_back(candidate.id);
    }
    std::fill(visited.begin(), visited.end(), false);
    block               = ir_.accept;
    auto ends_at_accept = false;
    while (block < ir_.blocks.size() && !visited[block]) {
      visited[block]  = true;
      auto& candidate = ir_.blocks[block];
      if (is_capture(candidate, capture_action::END)) ends_at_accept = true;
      auto is_accept = candidate.instructions.size() == 1U &&
                       std::holds_alternative<emit_accept>(candidate.instructions.front());
      if (!candidate.instructions.empty() && !is_capture(candidate, capture_action::END) &&
          !is_accept) {
        break;
      }
      if (predecessors[block].size() != 1U) break;
      block = predecessors[block].front();
    }
    return ends_at_accept;
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

  template <typename Range>
  static std::string format_i8_array(Range const& values)
  {
    std::string result;
    result.reserve(values.size() * 7U);
    std::size_t index = 0;
    for (auto value : values) {
      if (index++ != 0) result += ", ";
      fmt::format_to(std::back_inserter(result), "i8 {}", value);
    }
    return result;
  }

  static std::string format_byte_array(std::string_view value)
  {
    std::string result;
    result.reserve(value.size() * 3U);
    for (auto byte : value) {
      fmt::format_to(std::back_inserter(result), R"(\{:02X})", static_cast<unsigned char>(byte));
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
    auto common = fmt::format(
      R"NVVM(@{classes} = internal addrspace(4) constant [256 x i16] [{class_values}], align 2
@{transitions} = internal addrspace({transition_address_space}) constant [{transition_count} x i16] [{transition_values}], align 2)NVVM",
      fmt::arg("classes", name("dfa_byte_classes")),
      fmt::arg("class_values", format_i16_array(byte_classes)),
      fmt::arg("transitions", name("dfa_transitions")),
      fmt::arg("transition_address_space", machine.transition_address_space),
      fmt::arg("transition_count", machine.transitions.size()),
      fmt::arg("transition_values", format_i16_array(machine.transitions)));
    if (machine.assertion_aware) {
      output_.emit("{}",
                   fmt::format(
                     R"NVVM({common}
@{boundary_accepts} = internal addrspace(4) constant [{accept_count} x i8] [{accept_values}], align 1)NVVM",
                     fmt::arg("common", common),
                     fmt::arg("boundary_accepts", name("dfa_boundary_accepts")),
                     fmt::arg("accept_count", machine.boundary_accepts.size()),
                     fmt::arg("accept_values", format_i8_array(machine.boundary_accepts))));
    } else {
      output_.emit("{}", common);
    }
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
    std::vector<std::size_t> frequencies(machine.class_count);
    for (auto character_class : machine.byte_classes)
      ++frequencies[character_class];
    auto default_class = static_cast<std::uint16_t>(
      std::max_element(frequencies.begin(), frequencies.end()) - frequencies.begin());
    std::vector<deterministic_interval> byte_intervals;
    for (std::uint32_t byte = 0; byte < machine.byte_classes.size();) {
      auto character_class = machine.byte_classes[byte];
      if (character_class == default_class) {
        ++byte;
        continue;
      }
      auto first = byte;
      while (byte + 1U < machine.byte_classes.size() &&
             machine.byte_classes[byte + 1U] == character_class) {
        ++byte;
      }
      byte_intervals.push_back({first, byte, character_class});
      ++byte;
    }
    // more fragmented mappings favor one cached table load over comparison chains.
    auto inline_byte_classes = !byte_intervals.empty() && byte_intervals.size() <= 2U;
    std::string byte_classifier;
    if (inline_byte_classes) {
      auto result = fmt::format("{}", default_class);
      for (std::size_t index = 0; index < byte_intervals.size(); ++index) {
        auto interval = byte_intervals[index];
        if (interval.first == interval.last) {
          fmt::format_to(std::back_inserter(byte_classifier),
                         "  %byte_in_{} = icmp eq i32 %cp, {}\n",
                         index,
                         interval.first);
        } else {
          fmt::format_to(std::back_inserter(byte_classifier),
                         R"NVVM(  %byte_offset_{} = sub i32 %cp, {}
  %byte_in_{} = icmp ule i32 %byte_offset_{}, {}
)NVVM",
                         index,
                         interval.first,
                         index,
                         index,
                         interval.last - interval.first);
        }
        fmt::format_to(std::back_inserter(byte_classifier),
                       "  %byte_class_{} = select i1 %byte_in_{}, i32 {}, i32 {}\n",
                       index,
                       index,
                       interval.class_id,
                       result);
        result = fmt::format("%byte_class_{}", index);
      }
      fmt::format_to(std::back_inserter(byte_classifier), "  ret i32 {}", result);
    } else {
      byte_classifier = fmt::format(
        R"NVVM(  %byte_index = zext i32 %cp to i64
  %byte_class_ptr = getelementptr [256 x i16], [256 x i16] addrspace(4)* @{table}, i64 0, i64 %byte_index
  %byte_class_i16 = load i16, i16 addrspace(4)* %byte_class_ptr, align 2
  %byte_class = zext i16 %byte_class_i16 to i32
  ret i32 %byte_class)NVVM",
        fmt::arg("table", table));
    }
    output_.emit("{}",
                 fmt::format(
                   R"NVVM(define internal i32 @{function}(i32 %cp) alwaysinline nounwind readonly {{
entry:
  %is_byte = icmp ult i32 %cp, 256
  br i1 %is_byte, label %byte, label %unicode
byte:
{byte_classifier}
unicode:)NVVM",
                   fmt::arg("function", function),
                   fmt::arg("byte_classifier", byte_classifier)));

    if (machine.unicode_intervals.empty()) {
      output_.emit("  ret i32 0");
    } else if (machine.unicode_intervals.size() == 1) {
      output_.emit("  ret i32 {}", machine.unicode_intervals.front().class_id);
    } else {
      auto result = fmt::format("{}", machine.unicode_intervals.back().class_id);
      for (std::size_t reverse = machine.unicode_intervals.size() - 1; reverse > 0; --reverse) {
        auto index     = reverse - 1;
        auto& interval = machine.unicode_intervals[index];
        output_.emit("{}",
                     fmt::format(R"NVVM(  %unicode_le_{index} = icmp ule i32 %cp, {last}
  %unicode_class_{index} = select i1 %unicode_le_{index}, i32 {class_id}, i32 {fallback})NVVM",
                                 fmt::arg("index", index),
                                 fmt::arg("last", interval.last),
                                 fmt::arg("class_id", interval.class_id),
                                 fmt::arg("fallback", result)));
        result = fmt::format("%unicode_class_{}", index);
      }
      output_.emit("  ret i32 {}", result);
    }
    output_.emit("}}");
    output_.blank();
  }

  /**
   * @brief emits the position-context classifier used by assertion-aware DFA transitions
   *
   * @param machine deterministic machine whose assertion truth values are classified
   */
  void emit_deterministic_boundary_classifier(deterministic_machine const& machine)
  {
    auto function = name("dfa_boundary_classify");
    auto is_word  = name("is_word");
    auto decode   = name("decode_codepoint");
    output_.emit(
      R"NVVM(define internal i32 @{}(i8* %data, i64 %size, i64 %position, i32 %previous_cp, i32 %current_cp, i64 %current_width) alwaysinline nounwind readonly {{
entry:)NVVM",
      function);

    auto mask_value        = std::string{"0"};
    std::size_t mask_index = 0;
    auto append_bit        = [&](std::string_view condition) {
      auto bit = std::uint32_t{1} << mask_index;
      output_.emit("  %boundary_bit_{} = select i1 {}, i32 {}, i32 0", mask_index, condition, bit);
      if (mask_value == "0") {
        mask_value = fmt::format("%boundary_bit_{}", mask_index);
      } else {
        output_.emit(
          "  %boundary_mask_{} = or i32 {}, %boundary_bit_{}", mask_index, mask_value, mask_index);
        mask_value = fmt::format("%boundary_mask_{}", mask_index);
      }
      ++mask_index;
    };

    auto begin_input_bit = assertion_bit(assertion_kind::BEGIN_INPUT);
    auto end_input_bit   = assertion_bit(assertion_kind::END_INPUT);
    auto word_bits       = static_cast<std::uint8_t>(assertion_bit(assertion_kind::WORD_BOUNDARY) |
                                               assertion_bit(assertion_kind::NOT_WORD_BOUNDARY));
    auto line_bits       = static_cast<std::uint8_t>(assertion_bit(assertion_kind::BEGIN_LINE) |
                                               assertion_bit(assertion_kind::END_LINE));
    auto needs_begin =
      (machine.assertion_mask &
       static_cast<std::uint8_t>(begin_input_bit | assertion_bit(assertion_kind::BEGIN_LINE))) != 0;
    auto needs_end =
      (machine.assertion_mask &
       static_cast<std::uint8_t>(end_input_bit | assertion_bit(assertion_kind::END_LINE))) != 0;
    if (needs_begin) output_.emit("  %at_begin = icmp eq i64 %position, 0");
    if (needs_end) output_.emit("  %at_end = icmp eq i64 %position, %size");
    if ((machine.assertion_mask & begin_input_bit) != 0) { append_bit("%at_begin"); }
    if ((machine.assertion_mask & end_input_bit) != 0) { append_bit("%at_end"); }

    if ((machine.assertion_mask & word_bits) != 0) {
      output_.emit("{}",
                   fmt::format(
                     R"NVVM(  %has_previous = icmp ne i64 %position, 0
  %has_current = icmp ult i64 %position, %size
  %previous_word_raw = call i1 @{is_word}(i32 %previous_cp)
  %current_word_raw = call i1 @{is_word}(i32 %current_cp)
  %previous_word = and i1 %has_previous, %previous_word_raw
  %current_word = and i1 %has_current, %current_word_raw
  %word_boundary = xor i1 %previous_word, %current_word
  %not_word_boundary = xor i1 %word_boundary, true)NVVM",
                     fmt::arg("is_word", is_word)));
      if ((machine.assertion_mask & assertion_bit(assertion_kind::WORD_BOUNDARY)) != 0) {
        append_bit("%word_boundary");
      }
      if ((machine.assertion_mask & assertion_bit(assertion_kind::NOT_WORD_BOUNDARY)) != 0) {
        append_bit("%not_word_boundary");
      }
    }

    if ((machine.assertion_mask & line_bits) != 0) {
      if (ir_.options.extended_newline) {
        output_.emit(
          R"NVVM(  %current_lf = icmp eq i32 %current_cp, 10
  %current_cr = icmp eq i32 %current_cp, 13
  %current_nel = icmp eq i32 %current_cp, 133
  %current_ls = icmp eq i32 %current_cp, 8232
  %current_ps = icmp eq i32 %current_cp, 8233
  %current_crlf = or i1 %current_lf, %current_cr
  %current_extended_0 = or i1 %current_crlf, %current_nel
  %current_extended_1 = or i1 %current_ls, %current_ps
  %current_newline = or i1 %current_extended_0, %current_extended_1
  %previous_lf = icmp eq i32 %previous_cp, 10
  %previous_cr = icmp eq i32 %previous_cp, 13
  %previous_nel = icmp eq i32 %previous_cp, 133
  %previous_ls = icmp eq i32 %previous_cp, 8232
  %previous_ps = icmp eq i32 %previous_cp, 8233
  %previous_crlf = or i1 %previous_lf, %previous_cr
  %previous_extended_0 = or i1 %previous_crlf, %previous_nel
  %previous_extended_1 = or i1 %previous_ls, %previous_ps
  %previous_newline = or i1 %previous_extended_0, %previous_extended_1
  %mid_crlf = and i1 %previous_cr, %current_lf
  %not_mid_crlf = xor i1 %mid_crlf, true)NVVM");
      } else {
        output_.emit(
          R"NVVM(  %current_lf = icmp eq i32 %current_cp, 10
  %current_newline = icmp eq i32 %current_cp, 10
  %previous_newline = icmp eq i32 %previous_cp, 10
  %not_mid_crlf = icmp eq i32 0, 0)NVVM");
      }

      if ((machine.assertion_mask & assertion_bit(assertion_kind::BEGIN_LINE)) != 0) {
        if (ir_.options.multiline) {
          output_.emit(R"NVVM(  %begin_after_newline = and i1 %previous_newline, %not_mid_crlf
  %begin_line = or i1 %at_begin, %begin_after_newline)NVVM");
        } else {
          output_.emit("  %begin_line = icmp eq i64 %position, 0");
        }
        append_bit("%begin_line");
      }

      if ((machine.assertion_mask & assertion_bit(assertion_kind::END_LINE)) != 0) {
        if (ir_.options.multiline) {
          output_.emit(R"NVVM(  %end_at_newline = and i1 %current_newline, %not_mid_crlf
  %end_line = or i1 %at_end, %end_at_newline)NVVM");
        } else {
          output_.emit("  %next_position = add i64 %position, %current_width");
          output_.emit("  %current_is_final = icmp eq i64 %next_position, %size");
          auto final_sequence = std::string{"%current_is_final"};
          if (ir_.options.extended_newline) {
            output_.emit(
              "{}",
              fmt::format(
                R"NVVM(  %next_cp = call i32 @{decode}(i8* %data, i64 %size, i64 %next_position)
  %next_width = call i64 @{width}(i8* %data, i64 %size, i64 %next_position)
  %after_next = add i64 %next_position, %next_width
  %next_lf = icmp eq i32 %next_cp, 10
  %after_next_is_end = icmp eq i64 %after_next, %size
  %crlf_tail_0 = and i1 %current_cr, %next_lf
  %crlf_tail = and i1 %crlf_tail_0, %after_next_is_end
  %final_sequence = or i1 %current_is_final, %crlf_tail)NVVM",
                fmt::arg("decode", decode),
                fmt::arg("width", name("decode_width"))));
            final_sequence = "%final_sequence";
          }
          output_.emit("  %end_final_newline_0 = and i1 %current_newline, %not_mid_crlf");
          output_.emit("  %end_final_newline = and i1 %end_final_newline_0, {}", final_sequence);
          output_.emit("  %end_line = or i1 %at_end, %end_final_newline");
        }
        append_bit("%end_line");
      }
    }

    output_.emit("  ret i32 {}\n}}", mask_value);
    output_.blank();
  }

  /**
   * @brief emits the single-pass deterministic contains or matches executor
   *
   * @param machine deterministic machine to execute
   */
  void emit_deterministic_execute(deterministic_machine const& machine)
  {
    if ((machine.initial_state & 0x8000U) != 0 && !machine.accept_at_end &&
        !machine.accept_assertion.has_value()) {
      output_.emit(
        R"NVVM(define zeroext i1 @{}(i8* %data, i64 %size) alwaysinline nounwind readnone {{
entry:
  ret i1 true
}})NVVM",
        options_.execute_function);
      output_.blank();
      return;
    }

    auto load_byte            = name("load_byte");
    auto decode               = name("decode_codepoint");
    auto width                = name("decode_width");
    auto classify             = name("dfa_classify");
    auto transitions          = name("dfa_transitions");
    auto dead_guard           = std::string{};
    auto reject               = std::string{};
    auto prefix_loop_position = std::string{};
    auto prefix_loop_state    = std::string{};
    auto ascii_block          = fmt::format(
      R"NVVM(ascii:
  %ascii_class = call i32 @{classify}(i32 %first)
  br label %transition)NVVM",
      fmt::arg("classify", classify));
    auto ascii_predecessor = std::string{"%ascii"};
    auto prefix            = machine.scan_input ? required_ascii_prefix() : std::nullopt;
    if (prefix.has_value()) {
      prefix_loop_position = ", [ %prefix_next_position, %prefix_skip ]";
      prefix_loop_state    = ", [ " + std::to_string(machine.initial_state) + ", %prefix_skip ]";
      ascii_predecessor    = "%ascii_classify";
      ascii_block          = fmt::format(
        R"NVVM(ascii:
  %prefix_state_index = and i32 %state, {state_mask}
  %prefix_at_initial = icmp eq i32 %prefix_state_index, {initial_state_index}
  %prefix_equal = icmp eq i32 %first, {prefix}
  %prefix_mismatch = xor i1 %prefix_equal, true
  %prefix_skip_candidate = and i1 %prefix_at_initial, %prefix_mismatch
  br i1 %prefix_skip_candidate, label %prefix_skip, label %ascii_classify
ascii_classify:
  %ascii_class = call i32 @{classify}(i32 %first)
  br label %transition
prefix_skip:
  %prefix_next_position = add nuw i64 %position, 1
  br label %loop)NVVM",
        fmt::arg("state_mask", machine.state_mask),
        fmt::arg("initial_state_index", machine.initial_state & machine.state_mask),
        fmt::arg("prefix", static_cast<std::uint32_t>(*prefix)),
        fmt::arg("classify", classify));
    }
    if (!machine.scan_input && machine.dead_state <= machine.state_mask) {
      dead_guard = fmt::format(
        R"NVVM(  %next_state_index = and i32 %next_state, {state_mask}
  %dead = icmp eq i32 %next_state_index, {dead_state}
  br i1 %dead, label %reject, label %transition_live
transition_live:)NVVM",
        fmt::arg("state_mask", machine.state_mask),
        fmt::arg("dead_state", machine.dead_state));
      reject = R"NVVM(reject:
  ret i1 false
)NVVM";
    }
    output_.emit("{}",
                 fmt::format(
                   R"NVVM(define zeroext i1 @{execute}(i8* %data, i64 %size) nounwind readonly {{
entry:
  br label %loop
loop:
  %position = phi i64 [ 0, %entry ], [ %next_position, %continue ]{prefix_loop_position}
  %state = phi i32 [ {initial_state}, %entry ], [ %next_state, %continue ]{prefix_loop_state}
  %at_end = icmp eq i64 %position, %size
  br i1 %at_end, label %done, label %load
load:
  %input_ptr = getelementptr i8, i8* %data, i64 %position
  %first = call i32 @{load_byte}(i8* %input_ptr)
  %is_ascii = icmp ult i32 %first, 128
  br i1 %is_ascii, label %ascii, label %unicode
{ascii_block}
unicode:
  %codepoint = call i32 @{decode}(i8* %data, i64 %size, i64 %position)
  %unicode_width = call i64 @{width}(i8* %data, i64 %size, i64 %position)
  %unicode_class = call i32 @{classify}(i32 %codepoint)
  br label %transition
transition:
  %character_class = phi i32 [ %ascii_class, {ascii_predecessor} ], [ %unicode_class, %unicode ]
  %character_width = phi i64 [ 1, {ascii_predecessor} ], [ %unicode_width, %unicode ]
  %state_index = and i32 %state, {state_mask}
  %state_offset = mul nuw i32 %state_index, {class_count}
  %transition_index = add nuw i32 %state_offset, %character_class
  %transition_index_i64 = zext i32 %transition_index to i64
  %transition_ptr = getelementptr [{transition_count} x i16], [{transition_count} x i16] addrspace({transition_address_space})* @{transitions}, i64 0, i64 %transition_index_i64
  %next_state_i16 = load i16, i16 addrspace({transition_address_space})* %transition_ptr, align 2
  %next_state = zext i16 %next_state_i16 to i32
  %next_position = add i64 %position, %character_width
{dead_guard})NVVM",
                   fmt::arg("execute", options_.execute_function),
                   fmt::arg("initial_state", machine.initial_state),
                   fmt::arg("load_byte", load_byte),
                   fmt::arg("classify", classify),
                   fmt::arg("decode", decode),
                   fmt::arg("width", width),
                   fmt::arg("class_count", machine.class_count),
                   fmt::arg("state_mask", machine.state_mask),
                   fmt::arg("transition_count", machine.transitions.size()),
                   fmt::arg("transition_address_space", machine.transition_address_space),
                   fmt::arg("transitions", transitions),
                   fmt::arg("dead_guard", dead_guard),
                   fmt::arg("prefix_loop_position", prefix_loop_position),
                   fmt::arg("prefix_loop_state", prefix_loop_state),
                   fmt::arg("ascii_block", ascii_block),
                   fmt::arg("ascii_predecessor", ascii_predecessor)));
    if (!machine.accept_at_end && machine.accept_assertion.has_value()) {
      output_.emit("{}",
                   fmt::format(
                     R"NVVM(  %accept_bits = and i32 %next_state, 32768
  %accepted = icmp ne i32 %accept_bits, 0
  br i1 %accepted, label %check_accept, label %continue
check_accept:
  %accept_assertion = call i1 @{assertion}(i8* %data, i64 %size, i64 %next_position, i32 {kind})
  br i1 %accept_assertion, label %yes, label %continue
continue:
  br label %loop
done:
  %done_accept_bits = and i32 %state, 32768
  %done_accepted = icmp ne i32 %done_accept_bits, 0
  %done_assertion = call i1 @{assertion}(i8* %data, i64 %size, i64 %position, i32 {kind})
  %done_result = and i1 %done_accepted, %done_assertion
  ret i1 %done_result
yes:
  ret i1 true
{reject}
}})NVVM",
                     fmt::arg("assertion", name("assertion")),
                     fmt::arg("kind", static_cast<std::uint32_t>(*machine.accept_assertion)),
                     fmt::arg("reject", reject)));
    } else if (!machine.accept_at_end) {
      output_.emit("{}",
                   fmt::format(R"NVVM(  %accept_bits = and i32 %next_state, 32768
  %accepted = icmp ne i32 %accept_bits, 0
  br i1 %accepted, label %yes, label %continue
continue:
  br label %loop
done:
  ret i1 false
yes:
  ret i1 true
{reject}
}})NVVM",
                               fmt::arg("reject", reject)));
    } else {
      auto assertion = std::string{};
      auto result    = std::string{"%accepted"};
      if (machine.accept_assertion.has_value()) {
        assertion = fmt::format(
          R"NVVM(  %done_assertion = call i1 @{assertion}(i8* %data, i64 %size, i64 %position, i32 {kind})
  %done_result = and i1 %accepted, %done_assertion
)NVVM",
          fmt::arg("assertion", name("assertion")),
          fmt::arg("kind", static_cast<std::uint32_t>(*machine.accept_assertion)));
        result = "%done_result";
      }
      output_.emit("{}",
                   fmt::format(
                     R"NVVM(  br label %continue
continue:
  br label %loop
done:
  %accept_bits = and i32 %state, 32768
  %accepted = icmp ne i32 %accept_bits, 0
{assertion}  ret i1 {result}
{reject}
}})NVVM",
                     fmt::arg("assertion", assertion),
                     fmt::arg("result", result),
                     fmt::arg("reject", reject)));
    }
    output_.blank();
  }

  /**
   * @brief emits a single-pass DFA whose epsilon closure is selected by boundary context
   *
   * @param machine assertion-aware deterministic machine to execute
   */
  void emit_assertion_deterministic_execute(deterministic_machine const& machine)
  {
    auto load_byte        = name("load_byte");
    auto decode           = name("decode_codepoint");
    auto width            = name("decode_width");
    auto classify         = name("dfa_classify");
    auto boundary         = name("dfa_boundary_classify");
    auto transitions      = name("dfa_transitions");
    auto boundary_accepts = name("dfa_boundary_accepts");
    output_.emit("{}",
                 fmt::format(
                   R"NVVM(define zeroext i1 @{execute}(i8* %data, i64 %size) nounwind readonly {{
entry:
  br label %loop
loop:
  %position = phi i64 [ 0, %entry ], [ %next_position, %continue ]
  %state = phi i32 [ {initial_state}, %entry ], [ %next_state, %continue ]
  %previous_cp = phi i32 [ 0, %entry ], [ %current_cp, %continue ]
  %at_end = icmp eq i64 %position, %size
  br i1 %at_end, label %finish, label %load
load:
  %input_ptr = getelementptr i8, i8* %data, i64 %position
  %first = call i32 @{load_byte}(i8* %input_ptr)
  %is_ascii = icmp ult i32 %first, 128
  br i1 %is_ascii, label %ascii, label %unicode
ascii:
  %ascii_class = call i32 @{classify}(i32 %first)
  br label %boundary
unicode:
  %codepoint = call i32 @{decode}(i8* %data, i64 %size, i64 %position)
  %unicode_width = call i64 @{width}(i8* %data, i64 %size, i64 %position)
  %unicode_class = call i32 @{classify}(i32 %codepoint)
  br label %boundary
boundary:
  %current_cp = phi i32 [ %first, %ascii ], [ %codepoint, %unicode ]
  %character_class = phi i32 [ %ascii_class, %ascii ], [ %unicode_class, %unicode ]
  %character_width = phi i64 [ 1, %ascii ], [ %unicode_width, %unicode ]
  %boundary_class = call i32 @{boundary}(i8* %data, i64 %size, i64 %position, i32 %previous_cp, i32 %current_cp, i64 %character_width)
  br label %transition)NVVM",
                   fmt::arg("execute", options_.execute_function),
                   fmt::arg("initial_state", machine.initial_state),
                   fmt::arg("load_byte", load_byte),
                   fmt::arg("classify", classify),
                   fmt::arg("decode", decode),
                   fmt::arg("width", width),
                   fmt::arg("boundary", boundary)));
    output_.emit("{}",
                 fmt::format(
                   R"NVVM(transition:
  %transition_state_offset = mul nuw i32 %state, {boundary_class_count}
  %transition_context_index = add nuw i32 %transition_state_offset, %boundary_class
  %transition_context_offset = mul nuw i32 %transition_context_index, {class_count}
  %transition_index = add nuw i32 %transition_context_offset, %character_class
  %transition_index_i64 = zext i32 %transition_index to i64
  %transition_ptr = getelementptr [{transition_count} x i16], [{transition_count} x i16] addrspace({transition_address_space})* @{transitions}, i64 0, i64 %transition_index_i64
  %next_state_i16 = load i16, i16 addrspace({transition_address_space})* %transition_ptr, align 2
  %next_state_raw = zext i16 %next_state_i16 to i32
  %accept_bits = and i32 %next_state_raw, 32768
  %accepted = icmp ne i32 %accept_bits, 0
  %next_state = and i32 %next_state_raw, 32767
  %next_position = add i64 %position, %character_width)NVVM",
                   fmt::arg("boundary_class_count", machine.boundary_class_count),
                   fmt::arg("class_count", machine.class_count),
                   fmt::arg("transition_count", machine.transitions.size()),
                   fmt::arg("transition_address_space", machine.transition_address_space),
                   fmt::arg("transitions", transitions)));
    if (machine.accept_at_end) {
      output_.emit("  br label %continue");
    } else {
      output_.emit("  br i1 %accepted, label %yes, label %continue");
    }
    output_.emit("{}",
                 fmt::format(
                   R"NVVM(continue:
  br label %loop
finish:
  %finish_boundary_class = call i32 @{boundary}(i8* %data, i64 %size, i64 %position, i32 %previous_cp, i32 0, i64 0)
  %finish_state_offset = mul nuw i32 %state, {boundary_class_count}
  %finish_accept_index = add nuw i32 %finish_state_offset, %finish_boundary_class
  %finish_accept_index_i64 = zext i32 %finish_accept_index to i64
  %finish_accept_ptr = getelementptr [{accept_count} x i8], [{accept_count} x i8] addrspace(4)* @{boundary_accepts}, i64 0, i64 %finish_accept_index_i64
  %finish_accept_i8 = load i8, i8 addrspace(4)* %finish_accept_ptr, align 1
  %finish_accepted = icmp ne i8 %finish_accept_i8, 0
  ret i1 %finish_accepted
yes:
  ret i1 true
}})NVVM",
                   fmt::arg("boundary_class_count", machine.boundary_class_count),
                   fmt::arg("boundary", boundary),
                   fmt::arg("accept_count", machine.boundary_accepts.size()),
                   fmt::arg("boundary_accepts", boundary_accepts)));
    output_.blank();
  }

  /**
   * @brief emits NVVM intrinsics used by branch hints and replacement range copies
   */
  void emit_optimizer_intrinsics()
  {
    auto expect      = required_ascii_prefix().has_value() && options_.branch_hints;
    auto memory_copy = ir_.control.result == result_shape::REPLACEMENT;
    if (expect && memory_copy) {
      output_.emit(R"NVVM(declare i1 @llvm.expect.i1(i1, i1)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1))NVVM");
    } else if (expect) {
      output_.emit("declare i1 @llvm.expect.i1(i1, i1)");
    } else if (memory_copy) {
      output_.emit("declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)");
    }
    if (expect || memory_copy) output_.blank();
  }

  /**
   * @brief emits the compiler-cached byte load used by every input decoder
   */
  void emit_load_byte()
  {
    auto function = name("load_byte");
    output_.emit(
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
    output_.emit(
      R"NVVM(define internal i64 @{}(i8* %data, i64 %size, i64 %pos) alwaysinline nounwind readonly {{
entry:
  %in_bounds = icmp ult i64 %pos, %size
  br i1 %in_bounds, label %load, label %missing
missing:
  ret i64 0
load:)NVVM",
      function);
    if (ir_.options.characters == character_mode::BYTES) {
      output_.emit(
        R"NVVM(  ret i64 1
}})NVVM");
      output_.blank();
      return;
    }
    output_.emit(
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
    output_.emit(
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
      output_.emit(
        R"NVVM(  ret i32 %first
}})NVVM");
      output_.blank();
      return;
    }
    output_.emit(
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
    output_.emit(
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
    output_.emit(
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
      output_.emit(
        R"NVVM(define internal i1 @{}(i32 %cp) alwaysinline nounwind readnone {{
entry:)NVVM",
        function);
      std::string combined;
      for (std::size_t index = 0; index < std::size(cudf_unicode_word_ranges); ++index) {
        unicode_data_range range = cudf_unicode_word_ranges[index];
        output_.emit("{}",
                     fmt::format(R"NVVM(  %word_ge_{index} = icmp uge i32 %cp, {first}
  %word_le_{index} = icmp ule i32 %cp, {last}
  %word_in_{index} = and i1 %word_ge_{index}, %word_le_{index})NVVM",
                                 fmt::arg("index", index),
                                 fmt::arg("first", range.first),
                                 fmt::arg("last", range.last)));
        if (index == 0) {
          combined = "%word_in_0";
        } else {
          output_.emit("  %word_combined_{} = or i1 {}, %word_in_{}", index, combined, index);
          combined = fmt::format("%word_combined_{}", index);
        }
      }
      output_.emit("{}",
                   fmt::format(R"NVVM(  %word_underscore = icmp eq i32 %cp, 95
  %word_result = or i1 {combined}, %word_underscore
  ret i1 %word_result
}})NVVM",
                               fmt::arg("combined", combined)));
      output_.blank();
      return;
    }
    output_.emit(
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
    if (ir_.options.characters == character_mode::BYTES) {
      output_.emit(
        R"NVVM(define internal i64 @{}(i8* %data, i64 %size, i64 %target) alwaysinline nounwind readonly {{
entry:
  %at_begin = icmp eq i64 %target, 0
  %previous = sub i64 %target, 1
  %result = select i1 %at_begin, i64 0, i64 %previous
  ret i64 %result
}})NVVM",
        function);
      output_.blank();
      return;
    }

    auto load_byte = name("load_byte");
    output_.emit(
      "{}",
      fmt::format(
        R"NVVM(define internal i64 @{function}(i8* %data, i64 %size, i64 %target) alwaysinline nounwind readonly {{
entry:
  %at_begin = icmp eq i64 %target, 0
  br i1 %at_begin, label %zero, label %start
zero:
  ret i64 0
start:
  %initial = sub i64 %target, 1
  br label %loop
loop:
  %cursor = phi i64 [ %initial, %start ], [ %previous, %continue ]
  %pointer = getelementptr i8, i8* %data, i64 %cursor
  %byte = call i32 @{load_byte}(i8* %pointer)
  %prefix = and i32 %byte, 192
  %continuation = icmp eq i32 %prefix, 128
  %at_zero = icmp eq i64 %cursor, 0
  %leading = xor i1 %continuation, true
  %done = or i1 %leading, %at_zero
  br i1 %done, label %found, label %continue
continue:
  %previous = sub i64 %cursor, 1
  br label %loop
found:
  ret i64 %cursor
}})NVVM",
        fmt::arg("function", function),
        fmt::arg("load_byte", load_byte)));
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
    output_.emit(
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
        output_.emit(
          "{}",
          fmt::format(
            R"NVVM(define internal i1 @{function}(i8* %data, i64 %size, i64 %pos) alwaysinline nounwind readonly {{
entry:
  %cp = call i32 @{decode}(i8* %data, i64 %size, i64 %pos))NVVM",
            fmt::arg("function", function),
            fmt::arg("decode", decode)));
        if (match->predicate.recognized == predicate_class::ANY) {
          if (match->predicate.matches_newline) {
            output_.emit("  ret i1 true");
          } else if (match->predicate.extended_newline) {
            output_.emit(
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
            output_.emit(
              R"NVVM(  %not_lf = icmp ne i32 %cp, 10
  ret i1 %not_lf)NVVM");
          }
        } else {
          std::vector<std::string> comparisons;
          for (std::size_t index = 0; index < match->predicate.ranges.size(); ++index) {
            auto range = match->predicate.ranges[index];
            if (range.first == range.last) {
              auto comparison = fmt::format("equal_{}", index);
              output_.emit(
                "  %{} = icmp eq i32 %cp, {}", comparison, static_cast<std::uint32_t>(range.first));
              comparisons.push_back(comparison);
            } else {
              auto ge     = fmt::format("ge_{}", index);
              auto le     = fmt::format("le_{}", index);
              auto inside = fmt::format("inside_{}", index);
              output_.emit("{}",
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
            output_.emit("  ret i1 {}", match->predicate.negated ? "true" : "false");
          } else {
            auto combined = comparisons.front();
            for (std::size_t index = 1; index < comparisons.size(); ++index) {
              auto next = fmt::format("combined_{}", index);
              output_.emit("  %{} = or i1 %{}, %{}", next, combined, comparisons[index]);
              combined = next;
            }
            if (match->predicate.negated) {
              output_.emit(
                R"NVVM(  %negated = xor i1 %{}, true
  ret i1 %negated)NVVM",
                combined);
            } else {
              output_.emit("  ret i1 %{}", combined);
            }
          }
        }
        output_.emit("}}");
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
          output_.emit(
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
            output_.emit(
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
          output_.emit(
            R"NVVM(success:
  ret i1 true
fail:
  ret i1 false
}})NVVM");
          output_.blank();
          continue;
        }

        output_.emit(
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
          output_.emit(
            "{}",
            fmt::format(R"NVVM(check_{index}:
  %cp_{index} = call i32 @{decode}(i8* %data, i64 %size, i64 {position})
  %matches_{index} = icmp eq i32 %cp_{index}, {codepoint})NVVM",
                        fmt::arg("index", index),
                        fmt::arg("decode", decode),
                        fmt::arg("position", position),
                        fmt::arg("codepoint", static_cast<std::uint32_t>(literal->value[index]))));
          if (index + 1 == literal->value.size()) {
            output_.emit("  br i1 %matches_{}, label %success, label %fail", index);
          } else {
            output_.emit(
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
        output_.emit(
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
   * @brief emits constant byte arrays used by a specialized replacement executor
   */
  void emit_replacement_globals()
  {
    if (ir_.control.result != result_shape::REPLACEMENT) return;
    for (std::size_t index = 0; index < ir_.replacement.size(); ++index) {
      auto& token = ir_.replacement[index];
      if (token.type != replacement_token::kind::LITERAL || token.literal.empty()) continue;
      output_.emit(R"NVVM(@{} = internal addrspace(4) constant [{} x i8] c"{}", align 1)NVVM",
                   name(fmt::format("replacement_{}", index)),
                   token.literal.size(),
                   format_byte_array(token.literal));
    }
    output_.blank();
  }

  /**
   * @brief emits the recursive dispatcher that executes the instruction block graph
   */
  void emit_blocks()
  {
    // one recursive dispatcher represents cyclic block graphs without forward declarations
    auto function = name("run_block");
    std::string cases;
    for (auto& block : ir_.blocks) {
      fmt::format_to(
        std::back_inserter(cases), "    i32 {}, label %b{}_op_0\n", block.id, block.id);
    }
    output_.emit(
      "{}",
      fmt::format(
        R"NVVM(define internal i1 @{function}(i32 %block, i8* %data, i64 %size, i64* %position, i64* %captures, i64 %steps) nounwind {{
entry:
  %exhausted = icmp eq i64 %steps, 0
  %next_steps = sub i64 %steps, 1
  br i1 %exhausted, label %return_fail, label %dispatch
dispatch:
  switch i32 %block, label %return_fail [
{cases}  ])NVVM",
        fmt::arg("function", function),
        fmt::arg("cases", cases)));
    for (auto& block : ir_.blocks)
      emit_block(block);
    output_.emit(
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
      output_.emit("{}_op_{}:", prefix, operation_index);
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
          output_.emit("  br label %{}", next);
        } else {
          output_.emit(
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
        output_.emit("  br label %{}", next);
      } else if (auto* capture = std::get_if<write_capture>(&item)) {
        auto slot = static_cast<std::size_t>(capture->capture_index) * 2U +
                    (capture->action == capture_action::END ? 1U : 0U);
        if (std::find(capture_slots_.begin(), capture_slots_.end(), slot) == capture_slots_.end()) {
          output_.emit("  br label %{}", next);
          ++operation_index;
          continue;
        }
        auto reset = std::string{};
        if (capture->action == capture_action::BEGIN) {
          reset = fmt::format(
            R"NVVM(  %{prefix}_capture_end_ptr_{index} = getelementptr i64, i64* %captures, i64 {end_slot}
  store i64 -1, i64* %{prefix}_capture_end_ptr_{index}, align 8
)NVVM",
            fmt::arg("prefix", prefix),
            fmt::arg("index", operation_index),
            fmt::arg("end_slot", slot + 1U));
        }
        output_.emit("{}",
                     fmt::format(
                       R"NVVM(  %{prefix}_capture_pos_{index} = load i64, i64* %position, align 8
  %{prefix}_capture_ptr_{index} = getelementptr i64, i64* %captures, i64 {slot}
  store i64 %{prefix}_capture_pos_{index}, i64* %{prefix}_capture_ptr_{index}, align 8
{reset}  br label %{next})NVVM",
                       fmt::arg("prefix", prefix),
                       fmt::arg("index", operation_index),
                       fmt::arg("slot", slot),
                       fmt::arg("reset", reset),
                       fmt::arg("next", next)));
      } else if (std::holds_alternative<match_character>(item)) {
        output_.emit(
          "{}",
          fmt::format(R"NVVM(  %{prefix}_match_pos_{index} = load i64, i64* %position, align 8
  %{prefix}_match_{index} = call i1 @{predicate}(i8* %data, i64 %size, i64 %{prefix}_match_pos_{index})
  br i1 %{prefix}_match_{index}, label %{next}, label %return_fail)NVVM",
                      fmt::arg("prefix", prefix),
                      fmt::arg("index", operation_index),
                      fmt::arg("predicate", name(fmt::format("predicate_{}", block.id))),
                      fmt::arg("next", next)));
      } else if (std::holds_alternative<match_literal>(item)) {
        output_.emit(
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
          output_.emit(
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
          output_.emit("{}",
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
        output_.emit(
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
          output_.emit(
            R"NVVM(  %{}_accept_pos = load i64, i64* %position, align 8
  %{}_accept = icmp eq i64 %{}_accept_pos, %size
  ret i1 %{}_accept)NVVM",
            prefix,
            prefix,
            prefix,
            prefix);
        } else {
          output_.emit("  ret i1 true");
        }
        returned = true;
      }
      ++operation_index;
    }

    if (returned) return;
    output_.emit("{}_op_{}:", prefix, operation_index);
    auto edges = block.successors;
    std::stable_sort(edges.begin(), edges.end(), [](auto& left, auto& right) {
      return left.priority < right.priority;
    });
    if (edges.empty()) {
      output_.emit("  br label %return_fail");
      return;
    }
    output_.emit("  %{}_saved = load i64, i64* %position, align 8", prefix);
    for (auto slot : capture_slots_) {
      output_.emit(
        "{}",
        fmt::format(
          R"NVVM(  %{prefix}_saved_capture_{slot} = getelementptr i64, i64* %captures, i64 {slot}
  %{prefix}_saved_capture_value_{slot} = load i64, i64* %{prefix}_saved_capture_{slot}, align 8)NVVM",
          fmt::arg("prefix", prefix),
          fmt::arg("slot", slot)));
    }
    output_.emit("  br label %{}_attempt_0", prefix);
    for (std::size_t index = 0; index < edges.size(); ++index) {
      auto failure = index + 1 < edges.size() ? fmt::format("{}_attempt_{}", prefix, index + 1)
                                              : std::string{"return_fail"};
      output_.emit("{}",
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
      for (auto slot : capture_slots_) {
        output_.emit("  store i64 %{}_saved_capture_value_{}, i64* %{}_saved_capture_{}, align 8",
                     prefix,
                     slot,
                     prefix,
                     slot);
      }
      output_.emit("  br label %{}", failure);
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
    output_.emit(
      "{}",
      fmt::format(R"NVVM(define zeroext i1 @{execute}(i8* %data, i64 %size) nounwind readonly {{
entry:
  %position = alloca i64, align 8
  %size_plus_one = add i64 %size, 1
  %step_limit = mul i64 %size_plus_one, {multiplier})NVVM",
                  fmt::arg("execute", options_.execute_function),
                  fmt::arg("multiplier", multiplier)));
    if (!ir_.control.scan_input || begins_at_input_start()) {
      output_.emit("{}",
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
      output_.emit("{}",
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

    output_.emit("{}",
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
   * @brief emits capture stores selected by a deterministic transition index
   *
   * @param prefix unique label and temporary prefix
   * @param programs capture-update program for each deterministic transition
   * @param position SSA value recorded by every capture update
   * @param continuation block reached after applying a program
   */
  void emit_capture_action_dispatch(
    std::string_view prefix,
    std::vector<std::vector<deterministic_capture_action>> const& programs,
    std::string_view position,
    std::string_view continuation)
  {
    std::vector<std::size_t> active;
    for (std::size_t index = 0; index < programs.size(); ++index) {
      if (!programs[index].empty()) active.push_back(index);
    }
    if (active.empty()) {
      output_.emit("  br label %{}", continuation);
      return;
    }

    std::string source;
    fmt::format_to(
      std::back_inserter(source), "  switch i32 %transition_index, label %{}_done [\n", prefix);
    for (auto index : active) {
      fmt::format_to(
        std::back_inserter(source), "    i32 {}, label %{}_{}\n", index, prefix, index);
    }
    fmt::format_to(std::back_inserter(source), "  ]\n");
    for (auto index : active) {
      fmt::format_to(std::back_inserter(source), "{}_{}:\n", prefix, index);
      for (std::size_t action_index = 0; action_index < programs[index].size(); ++action_index) {
        auto action = programs[index][action_index];
        fmt::format_to(
          std::back_inserter(source),
          R"NVVM(  %{prefix}_{transition}_{action}_ptr = getelementptr i64, i64* %captures, i64 {slot}
  store i64 {position}, i64* %{prefix}_{transition}_{action}_ptr, align 8
)NVVM",
          fmt::arg("prefix", prefix),
          fmt::arg("transition", index),
          fmt::arg("action", action_index),
          fmt::arg("slot", action.slot),
          fmt::arg("position", position));
        if (action.reset_end) {
          fmt::format_to(
            std::back_inserter(source),
            R"NVVM(  %{prefix}_{transition}_{action}_end_ptr = getelementptr i64, i64* %captures, i64 {slot}
  store i64 -1, i64* %{prefix}_{transition}_{action}_end_ptr, align 8
)NVVM",
            fmt::arg("prefix", prefix),
            fmt::arg("transition", index),
            fmt::arg("action", action_index),
            fmt::arg("slot", action.slot + 1U));
        }
      }
      fmt::format_to(std::back_inserter(source), "  br label %{}_done\n", prefix);
    }
    fmt::format_to(std::back_inserter(source), "{}_done:\n  br label %{}", prefix, continuation);
    output_.emit("{}", source);
  }

  std::string render_start_byte_filter(deterministic_machine const& machine,
                                       std::string_view candidate_target)
  {
    if (!machine.start_byte_filter) return {};

    std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges;
    for (std::uint32_t byte = 0; byte < 128U;) {
      auto is_candidate =
        (machine.start_byte_bitmap[byte / 64U] & (std::uint64_t{1} << (byte % 64U))) != 0U;
      if (!is_candidate) {
        ++byte;
        continue;
      }
      auto first = byte;
      while (byte + 1U < 128U && (machine.start_byte_bitmap[(byte + 1U) / 64U] &
                                  (std::uint64_t{1} << ((byte + 1U) % 64U))) != 0U) {
        ++byte;
      }
      ranges.emplace_back(first, byte);
      ++byte;
    }

    std::string checks;
    std::string predicate = "false";
    for (std::size_t index = 0; index < ranges.size(); ++index) {
      auto [first, last] = ranges[index];
      auto range_value   = fmt::format("%start_filter_range_{}", index);
      if (first == last) {
        fmt::format_to(std::back_inserter(checks),
                       "  {} = icmp eq i32 %start_filter_byte, {}\n",
                       range_value,
                       first);
      } else {
        fmt::format_to(std::back_inserter(checks),
                       R"NVVM(  %start_filter_offset_{} = sub i32 %start_filter_byte, {}
  {} = icmp ule i32 %start_filter_offset_{}, {}
)NVVM",
                       index,
                       first,
                       range_value,
                       index,
                       last - first);
      }
      if (index == 0U) {
        predicate = range_value;
      } else {
        auto any_value = fmt::format("%start_filter_any_{}", index);
        fmt::format_to(
          std::back_inserter(checks), "  {} = or i1 {}, {}\n", any_value, predicate, range_value);
        predicate = std::move(any_value);
      }
    }

    return fmt::format(
      R"NVVM(start_filter:
  %filter_at_end = icmp eq i64 %start, %size
  br i1 %filter_at_end, label %{candidate_target}, label %start_filter_load
start_filter_load:
  %start_filter_ptr = getelementptr i8, i8* %data, i64 %start
  %start_filter_byte = call i32 @{load_byte}(i8* %start_filter_ptr)
  %start_filter_ascii = icmp ult i32 %start_filter_byte, 128
  br i1 %start_filter_ascii, label %start_filter_ascii_byte, label %{candidate_target}
start_filter_ascii_byte:
{checks}  br i1 {predicate}, label %{candidate_target}, label %advance_start
)NVVM",
      fmt::arg("candidate_target", candidate_target),
      fmt::arg("load_byte", name("load_byte")),
      fmt::arg("checks", checks),
      fmt::arg("predicate", predicate));
  }

  /**
   * @brief emits a direct tagged-DFA matcher for capture-safe deterministic automata
   *
   * @param machine deterministic machine and transition capture programs to execute
   */
  void emit_tagged_deterministic_find_from(deterministic_machine const& machine)
  {
    auto search_target = machine.start_byte_filter ? "start_filter" : "initialize";
    auto start_filter  = render_start_byte_filter(machine, "initialize");
    auto restart       = machine.restart_state <= machine.state_mask
                           ? fmt::format(
                         R"NVVM(  %restart_state_match = icmp eq i32 %state, {restart_state}
  %restart_consumed = icmp ugt i64 %position, %start
  %restart_before_end = icmp ult i64 %position, %size
  %restart_prefix_candidate = and i1 %restart_state_match, %restart_consumed
  %restart_prefix = and i1 %restart_prefix_candidate, %restart_before_end
  %restart_base = select i1 %restart_prefix, i64 %position, i64 %start
)NVVM",
                         fmt::arg("restart_state", machine.restart_state))
                           : std::string{};
    auto advance_phi =
      machine.restart_state <= machine.state_mask && machine.start_byte_filter
        ? std::
            string{R"NVVM(  %restart_advance_base = phi i64 [ %restart_base, %candidate_fail ], [ %start, %start_filter_ascii_byte ]
)NVVM"}
        : std::string{};
    auto advance_base = machine.restart_state <= machine.state_mask
                          ? (machine.start_byte_filter ? "%restart_advance_base" : "%restart_base")
                          : "%start";
    output_.emit(
      "{}",
      fmt::format(
        R"NVVM(define internal i1 @{function}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* %captures) nounwind {{
entry:
  br label %search
search:
  %start = phi i64 [ %search_start, %entry ], [ %next_start, %advance_start ]
  %in_range = icmp ule i64 %start, %size
  br i1 %in_range, label %{search_target}, label %no
{start_filter}
initialize:)NVVM",
        fmt::arg("function", name("find_from")),
        fmt::arg("search_target", search_target),
        fmt::arg("start_filter", start_filter)));
    for (auto slot : capture_slots_) {
      output_.emit(
        "{}",
        fmt::format(R"NVVM(  %capture_{slot}_ptr = getelementptr i64, i64* %captures, i64 {slot}
  store i64 -1, i64* %capture_{slot}_ptr, align 8)NVVM",
                    fmt::arg("slot", slot)));
    }
    output_.emit("{}",
                 fmt::format(
                   R"NVVM(  store i64 %start, i64* %match_begin, align 8
  br label %loop
loop:
  %position = phi i64 [ %start, %initialize ], [ %next_position, %continue ]
  %state = phi i32 [ {initial_state}, %initialize ], [ %next_state, %continue ]
  %at_end = icmp eq i64 %position, %size
  br i1 %at_end, label %candidate_fail, label %load
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
  %state_index = and i32 %state, 16383
  %state_offset = mul nuw i32 %state_index, {class_count}
  %transition_index = add nuw i32 %state_offset, %character_class
  %transition_index_i64 = zext i32 %transition_index to i64
  %transition_ptr = getelementptr [{transition_count} x i16], [{transition_count} x i16] addrspace({transition_address_space})* @{transitions}, i64 0, i64 %transition_index_i64
  %encoded_i16 = load i16, i16 addrspace({transition_address_space})* %transition_ptr, align 2
  %encoded = zext i16 %encoded_i16 to i32
  %next_state = and i32 %encoded, 16383
  %dead = icmp eq i32 %next_state, {dead_state}
  br i1 %dead, label %candidate_fail, label %capture_transition
capture_transition:)NVVM",
                   fmt::arg("initial_state", machine.initial_state),
                   fmt::arg("load_byte", name("load_byte")),
                   fmt::arg("classify", name("dfa_classify")),
                   fmt::arg("decode", name("decode_codepoint")),
                   fmt::arg("width", name("decode_width")),
                   fmt::arg("class_count", machine.class_count),
                   fmt::arg("transition_count", machine.transitions.size()),
                   fmt::arg("transition_address_space", machine.transition_address_space),
                   fmt::arg("transitions", name("dfa_transitions")),
                   fmt::arg("dead_state", machine.dead_state)));
    emit_capture_action_dispatch(
      "transition_capture", machine.transition_capture_actions, "%position", "consume");
    output_.emit(
      R"NVVM(consume:
  %next_position = add i64 %position, %character_width
  %accept_bits = and i32 %encoded, 32768
  %accepted = icmp ne i32 %accept_bits, 0
  br i1 %accepted, label %capture_accept, label %continue
continue:
  br label %loop
capture_accept:)NVVM");
    emit_capture_action_dispatch(
      "accept_capture", machine.accept_capture_actions, "%next_position", "yes");
    output_.emit("{}",
                 fmt::format(
                   R"NVVM(candidate_fail:
{restart}  %at_input_end = icmp eq i64 %start, %size
  br i1 %at_input_end, label %no, label %advance_start
advance_start:
{advance_phi}  %next_start = call i64 @{advance}(i8* %data, i64 %size, i64 {advance_base}, i64 1)
  br label %search
yes:
  store i64 %next_position, i64* %match_end, align 8
  ret i1 true
no:
  ret i1 false
}})NVVM",
                   fmt::arg("restart", restart),
                   fmt::arg("advance_phi", advance_phi),
                   fmt::arg("advance", name("advance")),
                   fmt::arg("advance_base", advance_base)));
    output_.blank();
  }

  /**
   * @brief emits a packed fixed-width comparison at one byte position
   *
   * @param literal non-empty ASCII literal to compare
   */
  void emit_ascii_literal_at(std::string_view literal)
  {
    std::string comparisons;
    std::string matched;
    std::size_t offset = 0;
    std::size_t index  = 0;
    while (offset < literal.size()) {
      auto remaining      = literal.size() - offset;
      auto width          = remaining >= 8U ? 8U : remaining >= 4U ? 4U : remaining >= 2U ? 2U : 1U;
      std::uint64_t value = 0;
      for (std::size_t byte = 0; byte < width; ++byte) {
        value |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(literal[offset + byte]))
                 << (byte * 8U);
      }
      auto bits = width * 8U;
      fmt::format_to(
        std::back_inserter(comparisons),
        "  %literal_byte_ptr_{} = getelementptr i8, i8* %data, i64 %literal_offset_{}\n",
        index,
        index);
      if (width == 1U) {
        fmt::format_to(std::back_inserter(comparisons),
                       "  %literal_chunk_{} = load i8, i8* %literal_byte_ptr_{}, align 1\n",
                       index,
                       index);
      } else {
        fmt::format_to(std::back_inserter(comparisons),
                       R"NVVM(  %literal_chunk_ptr_{} = bitcast i8* %literal_byte_ptr_{} to i{}*
  %literal_chunk_{} = load i{}, i{}* %literal_chunk_ptr_{}, align 1
)NVVM",
                       index,
                       index,
                       bits,
                       index,
                       bits,
                       bits,
                       index);
      }
      fmt::format_to(std::back_inserter(comparisons),
                     "  %literal_equal_{} = icmp eq i{} %literal_chunk_{}, {}\n",
                     index,
                     bits,
                     index,
                     value);
      if (matched.empty()) {
        matched = fmt::format("%literal_equal_{}", index);
      } else {
        fmt::format_to(std::back_inserter(comparisons),
                       "  %literal_equal_through_{} = and i1 {}, %literal_equal_{}\n",
                       index,
                       matched,
                       index);
        matched = fmt::format("%literal_equal_through_{}", index);
      }
      offset += width;
      ++index;
    }

    std::string offsets;
    offset = 0;
    for (std::size_t chunk = 0; chunk < index; ++chunk) {
      fmt::format_to(std::back_inserter(offsets),
                     "  %literal_offset_{} = add i64 %position, {}\n",
                     chunk,
                     offset);
      auto remaining = literal.size() - offset;
      offset += remaining >= 8U ? 8U : remaining >= 4U ? 4U : remaining >= 2U ? 2U : 1U;
    }

    output_.emit(
      "{}",
      fmt::format(
        R"NVVM(define internal i1 @{function}(i8* %data, i64 %size, i64 %position) alwaysinline nounwind readonly {{
entry:
  %position_valid = icmp ule i64 %position, %size
  %remaining = sub i64 %size, %position
  %enough_bytes = icmp uge i64 %remaining, {literal_size}
  %in_bounds = and i1 %position_valid, %enough_bytes
  br i1 %in_bounds, label %compare, label %no
compare:
{offsets}{comparisons}  ret i1 {matched}
no:
  ret i1 false
}})NVVM",
        fmt::arg("function", name("ascii_literal_at")),
        fmt::arg("literal_size", literal.size()),
        fmt::arg("offsets", offsets),
        fmt::arg("comparisons", comparisons),
        fmt::arg("matched", matched)));
    output_.blank();
  }

  /**
   * @brief emits a direct boolean executor for an exact ASCII literal
   *
   * @param literal non-empty ASCII literal to search or match
   */
  void emit_ascii_literal_execute(std::string_view literal)
  {
    emit_ascii_literal_at(literal);
    if (!ir_.control.scan_input) {
      auto required_size =
        ir_.control.require_end
          ? fmt::format("  %required_size = icmp eq i64 %size, {}\n", literal.size())
          : std::string{"  %required_size = icmp uge i64 %size, 0\n"};
      output_.emit("{}",
                   fmt::format(
                     R"NVVM(define zeroext i1 @{execute}(i8* %data, i64 %size) nounwind readonly {{
entry:
{required_size}  %literal_match = call i1 @{literal_at}(i8* %data, i64 %size, i64 0)
  %matched = and i1 %required_size, %literal_match
  ret i1 %matched
}})NVVM",
                     fmt::arg("execute", options_.execute_function),
                     fmt::arg("required_size", required_size),
                     fmt::arg("literal_at", name("ascii_literal_at"))));
      output_.blank();
      return;
    }

    if (literal.size() >= 8U) {
      // one has-zero mask tests eight possible first bytes before packed verification.
      std::uint64_t repeated_first = 0;
      for (std::size_t byte = 0; byte < 8U; ++byte) {
        repeated_first |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(literal.front()))
                          << (byte * 8U);
      }
      output_.emit("{}",
                   fmt::format(
                     R"NVVM(declare i64 @llvm.cttz.i64(i64, i1)

define zeroext i1 @{execute}(i8* %data, i64 %size) nounwind readonly {{
entry:
  br label %search
search:
  %position = phi i64 [ 0, %entry ], [ %next_chunk, %chunk_continue ]
  %candidate_end = add i64 %position, {literal_size}
  %in_range = icmp ule i64 %candidate_end, %size
  br i1 %in_range, label %load_chunk, label %no
load_chunk:
  %chunk_byte_ptr = getelementptr i8, i8* %data, i64 %position
  %chunk_ptr = bitcast i8* %chunk_byte_ptr to i64*
  %chunk = load i64, i64* %chunk_ptr, align 1
  %candidate_x = xor i64 %chunk, {repeated_first}
  %candidate_minus_ones = sub i64 %candidate_x, 72340172838076673
  %candidate_not = xor i64 %candidate_x, -1
  %candidate_zero_bytes = and i64 %candidate_minus_ones, %candidate_not
  %candidate_mask = and i64 %candidate_zero_bytes, -9187201950435737472
  %has_candidate = icmp ne i64 %candidate_mask, 0
  br i1 %has_candidate, label %candidate, label %chunk_continue
candidate:
  %remaining_candidates = phi i64 [ %candidate_mask, %load_chunk ], [ %next_candidates, %verify_continue ]
  %candidate_bit = call i64 @llvm.cttz.i64(i64 %remaining_candidates, i1 false)
  %candidate_byte = lshr i64 %candidate_bit, 3
  %candidate_position = add i64 %position, %candidate_byte
  %matched = call i1 @{literal_at}(i8* %data, i64 %size, i64 %candidate_position)
  br i1 %matched, label %yes, label %verify_continue
verify_continue:
  %candidate_mask_minus_one = sub i64 %remaining_candidates, 1
  %next_candidates = and i64 %remaining_candidates, %candidate_mask_minus_one
  %has_more_candidates = icmp ne i64 %next_candidates, 0
  br i1 %has_more_candidates, label %candidate, label %chunk_continue
chunk_continue:
  %next_chunk = add nuw i64 %position, 8
  br label %search
yes:
  ret i1 true
no:
  ret i1 false
}})NVVM",
                     fmt::arg("execute", options_.execute_function),
                     fmt::arg("literal_size", literal.size()),
                     fmt::arg("repeated_first", repeated_first),
                     fmt::arg("literal_at", name("ascii_literal_at"))));
      output_.blank();
      return;
    }

    auto verify = literal.size() == 1U
                    ? std::string{"  br i1 %candidate, label %yes, label %continue\n"}
                    : fmt::format(
                        R"NVVM(  br i1 %candidate, label %verify, label %continue
verify:
  %matched = call i1 @{literal_at}(i8* %data, i64 %size, i64 %position)
  br i1 %matched, label %yes, label %continue
)NVVM",
                        fmt::arg("literal_at", name("ascii_literal_at")));
    output_.emit(
      "{}",
      fmt::format(
        R"NVVM(define zeroext i1 @{execute}(i8* %data, i64 %size) nounwind readonly {{
entry:
  br label %search
search:
  %position = phi i64 [ 0, %entry ], [ %next_position, %continue ]
  %candidate_end = add i64 %position, {literal_size}
  %in_range = icmp ule i64 %candidate_end, %size
  br i1 %in_range, label %load, label %no
load:
  %input_ptr = getelementptr i8, i8* %data, i64 %position
  %first = call i32 @{load_byte}(i8* %input_ptr)
  %candidate = icmp eq i32 %first, {first}
{verify}continue:
  %next_position = add nuw i64 %position, 1
  br label %search
yes:
  ret i1 true
no:
  ret i1 false
}})NVVM",
        fmt::arg("execute", options_.execute_function),
        fmt::arg("literal_size", literal.size()),
        fmt::arg("load_byte", name("load_byte")),
        fmt::arg("first", static_cast<std::uint32_t>(static_cast<std::uint8_t>(literal.front()))),
        fmt::arg("verify", verify)));
    output_.blank();
  }

  /**
   * @brief emits a packed ASCII-literal finder for span-producing operations
   *
   * @param literal non-empty multi-byte ASCII literal to search
   */
  void emit_ascii_literal_find_from(std::string_view literal)
  {
    output_.emit(
      "{}",
      fmt::format(
        R"NVVM(define internal i1 @{function}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* %captures) alwaysinline nounwind {{
entry:
  br label %search
search:
  %position = phi i64 [ %search_start, %entry ], [ %next_position, %continue ]
  %candidate_end = add i64 %position, {literal_size}
  %in_range = icmp ule i64 %candidate_end, %size
  br i1 %in_range, label %load, label %no
load:
  %input_ptr = getelementptr i8, i8* %data, i64 %position
  %first = call i32 @{load_byte}(i8* %input_ptr)
  %candidate = icmp eq i32 %first, {first}
  br i1 %candidate, label %verify, label %continue
verify:
  %matched = call i1 @{literal_at}(i8* %data, i64 %size, i64 %position)
  br i1 %matched, label %yes, label %continue
continue:
  %next_position = add nuw i64 %position, 1
  br label %search
yes:
  store i64 %position, i64* %match_begin, align 8
  store i64 %candidate_end, i64* %match_end, align 8
  ret i1 true
no:
  ret i1 false
}})NVVM",
        fmt::arg("function", name("find_from")),
        fmt::arg("literal_size", literal.size()),
        fmt::arg("load_byte", name("load_byte")),
        fmt::arg("first", static_cast<std::uint32_t>(static_cast<std::uint8_t>(literal.front()))),
        fmt::arg("literal_at", name("ascii_literal_at"))));
    output_.blank();
  }

  /**
   * @brief emits a direct byte-search primitive for an exact ASCII character
   *
   * @param byte ASCII byte that forms the complete regex
   */
  void emit_single_byte_find_from(std::uint8_t byte)
  {
    output_.emit(
      "{}",
      fmt::format(
        R"NVVM(define internal i1 @{function}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* %captures) alwaysinline nounwind {{
entry:
  br label %search
search:
  %position = phi i64 [ %search_start, %entry ], [ %next_position, %continue ]
  %in_range = icmp ult i64 %position, %size
  br i1 %in_range, label %load, label %no
load:
  %input_ptr = getelementptr i8, i8* %data, i64 %position
  %value = call i32 @{load_byte}(i8* %input_ptr)
  %matched = icmp eq i32 %value, {byte}
  br i1 %matched, label %yes, label %continue
continue:
  %next_position = add nuw i64 %position, 1
  br label %search
yes:
  %end = add nuw i64 %position, 1
  store i64 %position, i64* %match_begin, align 8
  store i64 %end, i64* %match_end, align 8
  ret i1 true
no:
  ret i1 false
}})NVVM",
        fmt::arg("function", name("find_from")),
        fmt::arg("load_byte", name("load_byte")),
        fmt::arg("byte", static_cast<std::uint32_t>(byte))));
    output_.blank();
  }

  /**
   * @brief emits a non-recursive leftmost-match primitive for an ordered deterministic automaton
   *
   * @param machine deterministic machine whose transitions are known to preserve priority
   */
  void emit_deterministic_find_from(deterministic_machine const& machine)
  {
    auto initial_accept = (machine.initial_state & 0x8000U) != 0 ? "%start" : "-1";
    auto search_target  = machine.start_byte_filter ? "start_filter" : "candidate";
    auto start_filter   = render_start_byte_filter(machine, "candidate");
    auto restart        = machine.restart_state <= machine.state_mask
                            ? fmt::format(
                         R"NVVM(  %restart_state_match = icmp eq i32 %state, {restart_state}
  %restart_consumed = icmp ugt i64 %position, %start
  %restart_before_end = icmp ult i64 %position, %size
  %restart_prefix_candidate = and i1 %restart_state_match, %restart_consumed
  %restart_prefix = and i1 %restart_prefix_candidate, %restart_before_end
  %restart_base = select i1 %restart_prefix, i64 %position, i64 %start
)NVVM",
                         fmt::arg("restart_state", machine.restart_state))
                            : std::string{};
    auto advance_phi =
      machine.restart_state <= machine.state_mask && machine.start_byte_filter
        ? std::
            string{R"NVVM(  %restart_advance_base = phi i64 [ %restart_base, %candidate_fail ], [ %start, %start_filter_ascii_byte ]
)NVVM"}
        : std::string{};
    auto advance_base = machine.restart_state <= machine.state_mask
                          ? (machine.start_byte_filter ? "%restart_advance_base" : "%restart_base")
                          : "%start";
    output_.emit(
      "{}",
      fmt::format(
        R"NVVM(define internal i1 @{function}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* %captures) nounwind {{
entry:
  br label %search
search:
  %start = phi i64 [ %search_start, %entry ], [ %next_start, %advance_start ]
  %in_range = icmp ule i64 %start, %size
  br i1 %in_range, label %{search_target}, label %no
{start_filter}
candidate:
  br label %loop
loop:
  %position = phi i64 [ %start, %candidate ], [ %next_position, %continue ]
  %state = phi i32 [ {initial_state}, %candidate ], [ %next_state, %continue ]
  %last_accept = phi i64 [ {initial_accept}, %candidate ], [ %next_accept, %continue ]
  %at_end = icmp eq i64 %position, %size
  br i1 %at_end, label %candidate_done, label %load
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
  %state_index = and i32 %state, 16383
  %state_offset = mul nuw i32 %state_index, {class_count}
  %transition_index = add nuw i32 %state_offset, %character_class
  %transition_index_i64 = zext i32 %transition_index to i64
  %transition_ptr = getelementptr [{transition_count} x i16], [{transition_count} x i16] addrspace({transition_address_space})* @{transitions}, i64 0, i64 %transition_index_i64
  %encoded_i16 = load i16, i16 addrspace({transition_address_space})* %transition_ptr, align 2
  %encoded = zext i16 %encoded_i16 to i32
  %stop_bits = and i32 %encoded, 16384
  %stop_before = icmp ne i32 %stop_bits, 0
  br i1 %stop_before, label %candidate_done, label %check_transition
check_transition:
  %next_state = and i32 %encoded, 16383
  %dead = icmp eq i32 %next_state, {dead_state}
  br i1 %dead, label %candidate_done, label %consume
consume:
  %next_position = add i64 %position, %character_width
  %accept_bits = and i32 %encoded, 32768
  %accepted = icmp ne i32 %accept_bits, 0
  %next_accept = select i1 %accepted, i64 %next_position, i64 %last_accept
  br label %continue
continue:
  br label %loop
candidate_done:
  %accepted_end = phi i64 [ %last_accept, %loop ], [ %last_accept, %transition ], [ %last_accept, %check_transition ]
  %matched = icmp ne i64 %accepted_end, -1
  br i1 %matched, label %yes, label %candidate_fail
candidate_fail:
{restart}  %at_input_end = icmp eq i64 %start, %size
  br i1 %at_input_end, label %no, label %advance_start
advance_start:
{advance_phi}  %next_start = call i64 @{advance}(i8* %data, i64 %size, i64 {advance_base}, i64 1)
  br label %search
yes:
  store i64 %start, i64* %match_begin, align 8
  store i64 %accepted_end, i64* %match_end, align 8
  ret i1 true
no:
  ret i1 false
}})NVVM",
        fmt::arg("function", name("find_from")),
        fmt::arg("search_target", search_target),
        fmt::arg("start_filter", start_filter),
        fmt::arg("initial_state", machine.initial_state),
        fmt::arg("initial_accept", initial_accept),
        fmt::arg("load_byte", name("load_byte")),
        fmt::arg("classify", name("dfa_classify")),
        fmt::arg("decode", name("decode_codepoint")),
        fmt::arg("width", name("decode_width")),
        fmt::arg("class_count", machine.class_count),
        fmt::arg("transition_count", machine.transitions.size()),
        fmt::arg("transition_address_space", machine.transition_address_space),
        fmt::arg("transitions", name("dfa_transitions")),
        fmt::arg("dead_state", machine.dead_state),
        fmt::arg("restart", restart),
        fmt::arg("advance_phi", advance_phi),
        fmt::arg("advance", name("advance")),
        fmt::arg("advance_base", advance_base)));
    output_.blank();
  }

  /**
   * @brief emits the shared leftmost-match primitive used by materializing operations
   */
  void emit_find_from()
  {
    auto function   = name("find_from");
    auto run_block  = name("run_block");
    auto advance    = name("advance");
    auto multiplier = static_cast<std::uint64_t>(ir_.blocks.size()) * 8U + 32U;
    output_.emit(
      "{}",
      fmt::format(
        R"NVVM(define internal i1 @{function}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* %captures) nounwind {{
entry:
  %position = alloca i64, align 8
  %size_plus_one = add i64 %size, 1
  %step_limit = mul i64 %size_plus_one, {multiplier}
  br label %search
search:
  %start = phi i64 [ %search_start, %entry ], [ %next_start, %continue ]
  %in_range = icmp ule i64 %start, %size)NVVM",
        fmt::arg("function", function),
        fmt::arg("multiplier", multiplier)));

    auto prefix = required_ascii_prefix();
    if (prefix.has_value()) {
      auto hint = std::string{};
      if (options_.branch_hints) {
        hint = R"NVVM(  %prefix_likely = call i1 @llvm.expect.i1(i1 %prefix_candidate, i1 false)
)NVVM";
      }
      auto condition = options_.branch_hints ? "%prefix_likely" : "%prefix_candidate";
      output_.emit("{}",
                   fmt::format(R"NVVM(  br i1 %in_range, label %prefix_end, label %no
prefix_end:
  %at_end = icmp eq i64 %start, %size
  br i1 %at_end, label %initialize, label %prefix_filter
prefix_filter:
  %prefix_ptr = getelementptr i8, i8* %data, i64 %start
  %prefix_byte = call i32 @{load_byte}(i8* %prefix_ptr)
  %prefix_candidate = icmp eq i32 %prefix_byte, {prefix}
{hint}  br i1 {condition}, label %initialize, label %continue)NVVM",
                               fmt::arg("load_byte", name("load_byte")),
                               fmt::arg("prefix", static_cast<std::uint32_t>(*prefix)),
                               fmt::arg("hint", hint),
                               fmt::arg("condition", condition)));
    } else {
      output_.emit("  br i1 %in_range, label %initialize, label %no");
    }

    output_.emit("initialize:");
    for (auto slot : capture_slots_) {
      output_.emit(
        "{}",
        fmt::format(
          R"NVVM(  %find_capture_ptr_{slot} = getelementptr i64, i64* %captures, i64 {slot}
  store i64 -1, i64* %find_capture_ptr_{slot}, align 8)NVVM",
          fmt::arg("slot", slot)));
    }
    if (ir_.control.result == result_shape::CAPTURES) {
      output_.emit("  store i64 %start, i64* %find_capture_ptr_0, align 8");
    }
    auto captures = uses_capture_buffer() ? "%captures" : "null";
    output_.emit("{}",
                 fmt::format(R"NVVM(  store i64 %start, i64* %position, align 8
  %matched = call i1 @{run_block}(i32 {entry}, i8* %data, i64 %size, i64* %position, i64* {captures}, i64 %step_limit)
  br i1 %matched, label %yes, label %check_end
check_end:)NVVM",
                             fmt::arg("run_block", run_block),
                             fmt::arg("entry", ir_.entry),
                             fmt::arg("captures", captures)));
    if (prefix.has_value()) {
      output_.emit("  br i1 %at_end, label %no, label %continue");
    } else {
      output_.emit(R"NVVM(  %at_end = icmp eq i64 %start, %size
  br i1 %at_end, label %no, label %continue)NVVM");
    }
    output_.emit("{}",
                 fmt::format(R"NVVM(continue:
  %next_start = call i64 @{advance}(i8* %data, i64 %size, i64 %start, i64 1)
  br label %search
yes:
  %accepted_end = load i64, i64* %position, align 8
  store i64 %start, i64* %match_begin, align 8
  store i64 %accepted_end, i64* %match_end, align 8)NVVM",
                             fmt::arg("advance", advance)));
    if (ir_.control.result == result_shape::CAPTURES) {
      output_.emit("  store i64 %accepted_end, i64* %find_capture_ptr_1, align 8");
    }
    output_.emit(R"NVVM(  ret i1 true
no:
  ret i1 false
}})NVVM");
    output_.blank();
  }

  /**
   * @brief emits the first-match span ABI for a find operation
   */
  void emit_find_execute()
  {
    output_.emit(
      "{}",
      fmt::format(R"NVVM(define zeroext i1 @{execute}(i8* %data, i64 %size, i64* %span) nounwind {{
entry:
  %match_begin = getelementptr i64, i64* %span, i64 0
  %match_end = getelementptr i64, i64* %span, i64 1
  %matched = call i1 @{find_from}(i8* %data, i64 %size, i64 0, i64* %match_begin, i64* %match_end, i64* null)
  ret i1 %matched
}})NVVM",
                  fmt::arg("execute", options_.execute_function),
                  fmt::arg("find_from", name("find_from"))));
    output_.blank();
  }

  /**
   * @brief emits the first-match capture ABI used by extract and enumeration wrappers
   */
  void emit_capture_execute()
  {
    output_.emit(
      "{}",
      fmt::format(
        R"NVVM(define zeroext i1 @{execute}(i8* %data, i64 %size, i64 %search_start, i64* %captures) nounwind {{
entry:
  %match_begin = getelementptr i64, i64* %captures, i64 0
  %match_end = getelementptr i64, i64* %captures, i64 1
  %matched = call i1 @{find_from}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* %captures)
  ret i1 %matched
}})NVVM",
        fmt::arg("execute", options_.execute_function),
        fmt::arg("find_from", name("find_from"))));
    output_.blank();
  }

  /**
   * @brief emits a complete non-overlapping match-count executor
   */
  void emit_count_execute()
  {
    output_.emit("{}",
                 fmt::format(R"NVVM(define i64 @{execute}(i8* %data, i64 %size) nounwind readonly {{
entry:
  %match_begin = alloca i64, align 8
  %match_end = alloca i64, align 8
  br label %loop
loop:
  %search_start = phi i64 [ 0, %entry ], [ %end_value, %nonempty ], [ %empty_next, %advance_empty ]
  %count = phi i64 [ 0, %entry ], [ %next_count, %nonempty ], [ %next_count, %advance_empty ]
  %matched = call i1 @{find_from}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* null)
  br i1 %matched, label %found, label %done
found:
  %begin_value = load i64, i64* %match_begin, align 8
  %end_value = load i64, i64* %match_end, align 8
  %next_count = add i64 %count, 1
  %nonempty_match = icmp ne i64 %begin_value, %end_value
  br i1 %nonempty_match, label %nonempty, label %empty
nonempty:
  br label %loop
empty:
  %empty_at_end = icmp eq i64 %end_value, %size
  br i1 %empty_at_end, label %done_after_match, label %advance_empty
advance_empty:
  %empty_next = call i64 @{advance}(i8* %data, i64 %size, i64 %end_value, i64 1)
  br label %loop
done_after_match:
  ret i64 %next_count
done:
  ret i64 %count
}})NVVM",
                             fmt::arg("execute", options_.execute_function),
                             fmt::arg("find_from", name("find_from")),
                             fmt::arg("advance", name("advance"))));
    output_.blank();
  }

  /**
   * @brief emits the range-copy primitive used by the replacement executor
   */
  void emit_append_range()
  {
    output_.emit(
      "{}",
      fmt::format(
        R"NVVM(define internal i64 @{function}(i8* %source, i64 %begin, i64 %end, i8* %output, i64 %cursor) alwaysinline nounwind {{
entry:
  %length = sub i64 %end, %begin
  %next_cursor = add i64 %cursor, %length
  %output_missing = icmp eq i8* %output, null
  %empty = icmp eq i64 %length, 0
  %skip = or i1 %output_missing, %empty
  br i1 %skip, label %done, label %copy
copy:
  %source_ptr = getelementptr i8, i8* %source, i64 %begin
  %output_ptr = getelementptr i8, i8* %output, i64 %cursor
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %output_ptr, i8* align 1 %source_ptr, i64 %length, i1 false)
  br label %done
done:
  ret i64 %next_cursor
}})NVVM",
        fmt::arg("function", name("append_range"))));
    output_.blank();
  }

  /**
   * @brief emits a sizing-and-materialization executor specialized for a replacement template
   */
  void emit_replace_execute()
  {
    emit_append_range();
    auto capture_argument    = std::string{"null"};
    auto capture_declaration = std::string{};
    if (uses_capture_buffer()) {
      auto slots          = static_cast<std::size_t>(ir_.capture_count + 1U) * 2U;
      capture_declaration = fmt::format(R"NVVM(  %capture_array = alloca [{slots} x i64], align 8
  %captures = getelementptr [{slots} x i64], [{slots} x i64]* %capture_array, i64 0, i64 0)NVVM",
                                        fmt::arg("slots", slots));
      capture_argument    = "%captures";
    }
    output_.emit("{}",
                 fmt::format(
                   R"NVVM(define i64 @{execute}(i8* %data, i64 %size, i8* %output) nounwind {{
entry:
  %match_begin = alloca i64, align 8
  %match_end = alloca i64, align 8
{capture_declaration}
  br label %loop
loop:
  %search_start = phi i64 [ 0, %entry ], [ %match_end_value, %nonempty ], [ %advanced_start, %advance_empty ]
  %copied = phi i64 [ 0, %entry ], [ %match_end_value, %nonempty ], [ %match_end_value, %advance_empty ]
  %cursor = phi i64 [ 0, %entry ], [ %replacement_cursor, %nonempty ], [ %replacement_cursor, %advance_empty ]
  %matched = call i1 @{find_from}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* {captures})
  br i1 %matched, label %found, label %no_match
found:
  %match_begin_value = load i64, i64* %match_begin, align 8
  %match_end_value = load i64, i64* %match_end, align 8
  %unmatched_cursor = call i64 @{append}(i8* %data, i64 %copied, i64 %match_begin_value, i8* %output, i64 %cursor))NVVM",
                   fmt::arg("execute", options_.execute_function),
                   fmt::arg("capture_declaration", capture_declaration),
                   fmt::arg("find_from", name("find_from")),
                   fmt::arg("captures", capture_argument),
                   fmt::arg("append", name("append_range"))));
    std::string cursor = "%unmatched_cursor";
    for (std::size_t index = 0; index < ir_.replacement.size(); ++index) {
      auto& token = ir_.replacement[index];
      if (token.type == replacement_token::kind::LITERAL) {
        if (token.literal.empty()) continue;
        output_.emit(
          "{}",
          fmt::format(
            R"NVVM(  %replacement_constant_{index} = getelementptr [{size} x i8], [{size} x i8] addrspace(4)* @{constant}, i64 0, i64 0
  %replacement_literal_{index} = addrspacecast i8 addrspace(4)* %replacement_constant_{index} to i8*
  %replacement_cursor_{index} = call i64 @{append}(i8* %replacement_literal_{index}, i64 0, i64 {size}, i8* %output, i64 {cursor}))NVVM",
            fmt::arg("index", index),
            fmt::arg("size", token.literal.size()),
            fmt::arg("constant", name(fmt::format("replacement_{}", index))),
            fmt::arg("append", name("append_range")),
            fmt::arg("cursor", cursor)));
      } else if (token.capture_index == 0 || is_whole_match_capture(token.capture_index)) {
        output_.emit(
          "{}",
          fmt::format(
            R"NVVM(  %replacement_cursor_{index} = call i64 @{append}(i8* %data, i64 %match_begin_value, i64 %match_end_value, i8* %output, i64 {cursor}))NVVM",
            fmt::arg("index", index),
            fmt::arg("append", name("append_range")),
            fmt::arg("cursor", cursor)));
      } else {
        auto slot = static_cast<std::size_t>(token.capture_index) * 2U;
        output_.emit(
          "{}",
          fmt::format(
            R"NVVM(  %replacement_capture_begin_ptr_{index} = getelementptr i64, i64* %captures, i64 {slot}
  %replacement_capture_end_ptr_{index} = getelementptr i64, i64* %captures, i64 {end_slot}
  %replacement_capture_begin_{index} = load i64, i64* %replacement_capture_begin_ptr_{index}, align 8
  %replacement_capture_end_{index} = load i64, i64* %replacement_capture_end_ptr_{index}, align 8
  %replacement_capture_has_begin_{index} = icmp ne i64 %replacement_capture_begin_{index}, -1
  %replacement_capture_has_end_{index} = icmp ne i64 %replacement_capture_end_{index}, -1
  %replacement_capture_valid_{index} = and i1 %replacement_capture_has_begin_{index}, %replacement_capture_has_end_{index}
  %replacement_capture_safe_begin_{index} = select i1 %replacement_capture_valid_{index}, i64 %replacement_capture_begin_{index}, i64 0
  %replacement_capture_safe_end_{index} = select i1 %replacement_capture_valid_{index}, i64 %replacement_capture_end_{index}, i64 0
  %replacement_cursor_{index} = call i64 @{append}(i8* %data, i64 %replacement_capture_safe_begin_{index}, i64 %replacement_capture_safe_end_{index}, i8* %output, i64 {cursor}))NVVM",
            fmt::arg("index", index),
            fmt::arg("slot", slot),
            fmt::arg("end_slot", slot + 1U),
            fmt::arg("append", name("append_range")),
            fmt::arg("cursor", cursor)));
      }
      cursor = fmt::format("%replacement_cursor_{}", index);
    }
    output_.emit("{}",
                 fmt::format(R"NVVM(  %replacement_cursor = add i64 {cursor}, 0
  %nonempty_match = icmp ne i64 %match_begin_value, %match_end_value
  br i1 %nonempty_match, label %nonempty, label %empty
nonempty:
  br label %loop
empty:
  %empty_at_end = icmp eq i64 %match_end_value, %size
  br i1 %empty_at_end, label %empty_finish, label %advance_empty
advance_empty:
  %advanced_start = call i64 @{advance}(i8* %data, i64 %size, i64 %match_end_value, i64 1)
  br label %loop
no_match:
  br label %finish
empty_finish:
  br label %finish
finish:
  %tail_begin = phi i64 [ %copied, %no_match ], [ %match_end_value, %empty_finish ]
  %tail_cursor = phi i64 [ %cursor, %no_match ], [ %replacement_cursor, %empty_finish ]
  %result_size = call i64 @{append}(i8* %data, i64 %tail_begin, i64 %size, i8* %output, i64 %tail_cursor)
  ret i64 %result_size
}})NVVM",
                             fmt::arg("cursor", cursor),
                             fmt::arg("advance", name("advance")),
                             fmt::arg("append", name("append_range"))));
    output_.blank();
  }

  /**
   * @brief emits the optional span-store primitive used by split sizing and materialization
   */
  void emit_write_span()
  {
    output_.emit(
      "{}",
      fmt::format(
        R"NVVM(define internal void @{function}(i64* %spans, i64 %index, i64 %begin, i64 %end) alwaysinline nounwind {{
entry:
  %missing = icmp eq i64* %spans, null
  br i1 %missing, label %done, label %write
write:
  %base = shl i64 %index, 1
  %end_index = add i64 %base, 1
  %begin_ptr = getelementptr i64, i64* %spans, i64 %base
  %end_ptr = getelementptr i64, i64* %spans, i64 %end_index
  store i64 %begin, i64* %begin_ptr, align 8
  store i64 %end, i64* %end_ptr, align 8
  br label %done
done:
  ret void
}})NVVM",
        fmt::arg("function", name("write_span"))));
    output_.blank();
  }

  /**
   * @brief emits a sizing-and-span-materialization executor specialized for split
   */
  void emit_split_execute()
  {
    emit_write_span();
    output_.emit(
      "{}",
      fmt::format(R"NVVM(define i64 @{execute}(i8* %data, i64 %size, i64* %spans) nounwind {{
entry:
  %match_begin = alloca i64, align 8
  %match_end = alloca i64, align 8
  br label %loop
loop:
  %search_start = phi i64 [ 0, %entry ], [ %match_end_value, %nonempty ], [ %advanced_start, %advance_empty ]
  %copied = phi i64 [ 0, %entry ], [ %match_end_value, %nonempty ], [ %match_end_value, %advance_empty ]
  %field_count = phi i64 [ 0, %entry ], [ %next_count, %nonempty ], [ %next_count, %advance_empty ]
  %matched = call i1 @{find_from}(i8* %data, i64 %size, i64 %search_start, i64* %match_begin, i64* %match_end, i64* null)
  br i1 %matched, label %found, label %no_match
found:
  %match_begin_value = load i64, i64* %match_begin, align 8
  %match_end_value = load i64, i64* %match_end, align 8
  call void @{write_span}(i64* %spans, i64 %field_count, i64 %copied, i64 %match_begin_value)
  %next_count = add i64 %field_count, 1
  %nonempty_match = icmp ne i64 %match_begin_value, %match_end_value
  br i1 %nonempty_match, label %nonempty, label %empty
nonempty:
  br label %loop
empty:
  %empty_at_end = icmp eq i64 %match_end_value, %size
  br i1 %empty_at_end, label %empty_finish, label %advance_empty
advance_empty:
  %advanced_start = call i64 @{advance}(i8* %data, i64 %size, i64 %match_end_value, i64 1)
  br label %loop
no_match:
  br label %finish
empty_finish:
  br label %finish
finish:
  %tail_begin = phi i64 [ %copied, %no_match ], [ %match_end_value, %empty_finish ]
  %tail_index = phi i64 [ %field_count, %no_match ], [ %next_count, %empty_finish ]
  call void @{write_span}(i64* %spans, i64 %tail_index, i64 %tail_begin, i64 %size)
  %result_count = add i64 %tail_index, 1
  ret i64 %result_count
}})NVVM",
                  fmt::arg("execute", options_.execute_function),
                  fmt::arg("find_from", name("find_from")),
                  fmt::arg("write_span", name("write_span")),
                  fmt::arg("advance", name("advance"))));
    output_.blank();
  }

  instruction_ir const& ir_;
  nvvm_ir_codegen_options const& options_;
  std::optional<deterministic_machine> deterministic_ = std::nullopt;
  std::optional<std::string> ascii_literal_           = std::nullopt;
  std::vector<std::size_t> capture_slots_             = std::vector<std::size_t>{};
  source_buffer output_                               = source_buffer{};
};

std::string_view module_body(std::string_view module)
{
  auto begin = std::string_view::npos;
  for (auto marker :
       {std::string_view{"\n@"}, std::string_view{"\ndefine "}, std::string_view{"\ndeclare "}}) {
    auto candidate = module.find(marker);
    if (candidate != std::string_view::npos) begin = std::min(begin, candidate + 1U);
  }
  if (begin == std::string_view::npos) {
    throw std::invalid_argument("generated alternative module has no declarations");
  }
  auto end = module.find("\n!nvvmir.version", begin);
  return module.substr(begin, end == std::string_view::npos ? end : end - begin);
}

std::string_view module_without_metadata(std::string_view module)
{
  auto end = module.find("\n!nvvmir.version");
  return module.substr(0, end);
}

std::optional<std::string> render_large_boolean_alternation(instruction_ir const& ir,
                                                            nvvm_ir_codegen_options const& options)
{
  if (ir.control.result != result_shape::BOOLEAN || ir.blocks.size() < 80U ||
      ir.entry >= ir.blocks.size()) {
    return std::nullopt;
  }
  auto& entry = ir.blocks[ir.entry];
  if (!entry.instructions.empty() || entry.successors.size() < 2U) return std::nullopt;

  std::vector<instruction_ir> alternatives;
  alternatives.reserve(entry.successors.size());
  for (auto edge : entry.successors) {
    auto branch    = ir;
    branch.entry   = edge.target;
    auto optimized = optimize(std::move(branch));
    if (!optimized) return std::nullopt;
    alternatives.push_back(std::move(*optimized.value));
  }

  std::string result;
  std::vector<std::string> functions;
  functions.reserve(alternatives.size());
  for (std::size_t index = 0; index < alternatives.size(); ++index) {
    auto branch_options = options;
    branch_options.symbol_prefix += fmt::format("_alternative_{}", index);
    branch_options.execute_function += fmt::format("_alternative_{}", index);
    functions.push_back(branch_options.execute_function);
    auto nested = render_large_boolean_alternation(alternatives[index], branch_options);
    auto module = nested.has_value()
                    ? std::move(*nested)
                    : nvvm_ir_renderer(alternatives[index], branch_options).render();
    if (index == 0) {
      result = module_without_metadata(module);
    } else {
      result += '\n';
      result += module_body(module);
    }
  }

  std::string branches;
  for (std::size_t index = 0; index < functions.size(); ++index) {
    auto label = index == 0 ? std::string{"entry"} : fmt::format("alternative_{}", index);
    branches += fmt::format(
      "{}:\n  %matched_{} = call i1 @{}(i8* %data, i64 %size)\n", label, index, functions[index]);
    if (index + 1U < functions.size()) {
      branches +=
        fmt::format("  br i1 %matched_{}, label %yes, label %alternative_{}\n", index, index + 1U);
    } else {
      branches += fmt::format("  ret i1 %matched_{}\n", index);
    }
  }
  result += fmt::format(
    R"NVVM(
define zeroext i1 @{}(i8* %data, i64 %size) nounwind readonly {{
{}yes:
  ret i1 true
}}
)NVVM",
    options.execute_function,
    branches);
  result += "\n!nvvmir.version = !{!0}\n!0 = !{i32 2, i32 0}\n";
  return result;
}

}  // namespace

std::string generate_nvvm_ir(instruction_ir const& ir, nvvm_ir_codegen_options const& options)
{
  if (auto alternatives = render_large_boolean_alternation(ir, options)) {
    return std::move(*alternatives);
  }
  return nvvm_ir_renderer(ir, options).render();
}

}  // namespace regex_ir
