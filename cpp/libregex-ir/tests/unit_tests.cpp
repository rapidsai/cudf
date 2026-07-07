/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>

#include <gtest/gtest.h>
#include <nvJitLink.h>
#include <nvvm.h>
#include <regex_ir.hpp>
#include <regex_ir_boolean_kernel.fatbin.inc>
#include <regex_ir_capture_kernel.fatbin.inc>
#include <regex_ir_count_kernel.fatbin.inc>
#include <regex_ir_find_kernel.fatbin.inc>
#include <regex_ir_replace_kernel.fatbin.inc>
#include <regex_ir_split_kernel.fatbin.inc>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace regex_ir::test {

// source suite for a table-driven boolean case
enum class test_suite : std::uint8_t {
  PROJECT    = 0,
  RE2        = 1,
  RUST_REGEX = 2,
  CPYTHON    = 3,
  SIHLFALL   = 4,
};

struct boolean_test_case {
  std::string_view name    = "";
  test_suite suite         = test_suite::PROJECT;
  std::string_view pattern = "";
  std::string_view input   = "";
  operation_kind operation = operation_kind::CONTAINS;
  compile_options options  = compile_options{};
  bool expected : 1        = false;
};

enum class cudf_regex_operation : std::uint8_t {
  CONTAINS       = 0,
  PREFIX_MATCH   = 1,
  COUNT          = 2,
  FIND           = 3,
  FIND_ALL       = 4,
  EXTRACT        = 5,
  EXTRACT_ALL    = 6,
  EXTRACT_SINGLE = 7,
  REPLACE        = 8,
  SPLIT          = 9,
};

enum class cudf_capture_mode : std::uint8_t {
  CAPTURE     = 0,
  NON_CAPTURE = 1,
};

using maybe_string = std::optional<std::string>;

struct cudf_expected_row {
  bool valid : 1                    = true;
  std::int64_t scalar               = 0;
  std::vector<maybe_string> strings = std::vector<maybe_string>{};
};

struct cudf_regex_case {
  std::string test_name                   = "";
  std::string assertion                   = "";
  std::string pattern                     = "";
  compile_options options                 = compile_options{};
  cudf_regex_operation operation          = cudf_regex_operation::CONTAINS;
  cudf_capture_mode capture_mode          = cudf_capture_mode::CAPTURE;
  std::vector<maybe_string> inputs        = std::vector<maybe_string>{};
  std::vector<cudf_expected_row> expected = std::vector<cudf_expected_row>{};
  std::string replacement                 = "";
  std::size_t max_matches                 = std::numeric_limits<std::size_t>::max();
  std::size_t capture_index               = 0;
  std::size_t expected_columns            = 0;
  bool replacement_has_backrefs : 1       = false;
  bool reverse                  : 1       = false;
  bool table_output             : 1       = false;
  bool expect_operation_error   : 1       = false;
};

struct cudf_compile_case {
  std::string test_name   = "";
  std::string pattern     = "";
  compile_options options = compile_options{};
  bool should_compile : 1 = true;
};

inline constexpr std::size_t cudf_unlimited_matches = std::numeric_limits<std::size_t>::max();

// boolean case tables

namespace {

constexpr compile_options boolean_case_insensitive{.case_insensitive = true};
constexpr compile_options boolean_dot_all{.dot_all = true};
constexpr compile_options boolean_multiline{.multiline = true};
constexpr compile_options boolean_bytes{.characters = character_mode::BYTES};

std::span<boolean_test_case const> boolean_test_cases()
{
  static constexpr auto cases = std::to_array<boolean_test_case>({
    // existing boolean assertions and boolean projections of richer result tests
    {"project_matches_short", test_suite::PROJECT, "abc*", "ab", operation_kind::MATCHES, {}, true},
    {"project_matches_repetition",
     test_suite::PROJECT,
     "abc*",
     "abccc",
     operation_kind::MATCHES,
     {},
     true},
    {"project_matches_rejects_prefix",
     test_suite::PROJECT,
     "abc*",
     "zabccc",
     operation_kind::MATCHES,
     {},
     false},
    {"project_class_matches",
     test_suite::PROJECT,
     "abc[0-9]",
     "abc7",
     operation_kind::MATCHES,
     {},
     true},
    {"project_class_rejects",
     test_suite::PROJECT,
     "abc[0-9]",
     "abcz",
     operation_kind::MATCHES,
     {},
     false},
    {"project_alternation_matches",
     test_suite::PROJECT,
     "^(ab|cd)+$",
     "abcdab",
     operation_kind::MATCHES,
     {},
     true},
    {"project_alternation_rejects",
     test_suite::PROJECT,
     "^(ab|cd)+$",
     "abce",
     operation_kind::MATCHES,
     {},
     false},
    {"project_boundary_contains",
     test_suite::PROJECT,
     R"REGEX(\bcat\b)REGEX",
     "a cat!",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_boundary_rejects",
     test_suite::PROJECT,
     R"REGEX(\bcat\b)REGEX",
     "scatter",
     operation_kind::CONTAINS,
     {},
     false},
    {"project_case_insensitive",
     test_suite::PROJECT,
     "AbC",
     "aBc",
     operation_kind::MATCHES,
     boolean_case_insensitive,
     true},
    {"project_multiline",
     test_suite::PROJECT,
     "^cat$",
     "dog\ncat\nfox",
     operation_kind::CONTAINS,
     boolean_multiline,
     true},
    {"project_unicode_literal",
     test_suite::PROJECT,
     "λ+",
     "xλλ",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_find_shadow",
     test_suite::PROJECT,
     "a+",
     "xxaaay",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_count_shadow",
     test_suite::PROJECT,
     "a+",
     "baacaa",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_extract_shadow",
     test_suite::PROJECT,
     "(a+)(b)",
     "xxaaab",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_split_shadow",
     test_suite::PROJECT,
     ",+",
     "a,,b,",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_empty_count_shadow",
     test_suite::PROJECT,
     "a*",
     "b",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_lazy_find_shadow",
     test_suite::PROJECT,
     "a+?",
     "aaa",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_span_alternative",
     test_suite::PROJECT,
     "ab|cd",
     "xxcdab",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_span_negated_class",
     test_suite::PROJECT,
     "[^x]+",
     "xxxabcdxxx",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_ascii_boundary",
     test_suite::PROJECT,
     R"REGEX(\bfoo\b)REGEX",
     "nofoo foo that",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_ascii_boundary_rejects_inner",
     test_suite::PROJECT,
     R"REGEX(\bfoo\b)REGEX",
     "seafood",
     operation_kind::CONTAINS,
     {},
     false},
    {"project_ascii_non_boundary",
     test_suite::PROJECT,
     R"REGEX(\Bfoo\B)REGEX",
     "xfoox",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_ascii_non_boundary_rejects",
     test_suite::PROJECT,
     R"REGEX(\Bfoo\B)REGEX",
     "foo",
     operation_kind::CONTAINS,
     {},
     false},
    {"project_unicode_neighbors_are_non_word",
     test_suite::PROJECT,
     R"REGEX(\bx\b)REGEX",
     "«x»",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_dot_rejects_newline",
     test_suite::PROJECT,
     "a.*a",
     "aba\naba",
     operation_kind::MATCHES,
     {},
     false},
    {"project_dot_all_accepts_newline",
     test_suite::PROJECT,
     "a.*a",
     "aba\naba",
     operation_kind::MATCHES,
     boolean_dot_all,
     true},
    {"project_utf8_dot", test_suite::PROJECT, "^...$", "日本語", operation_kind::MATCHES, {}, true},
    {"project_byte_dot",
     test_suite::PROJECT,
     "^.........$",
     "日本語",
     operation_kind::MATCHES,
     boolean_bytes,
     true},
    {"project_byte_dot_rejects_codepoints",
     test_suite::PROJECT,
     "^...$",
     "日本語",
     operation_kind::MATCHES,
     boolean_bytes,
     false},
    {"project_replace_anchor_shadow",
     test_suite::PROJECT,
     "^",
     "foo",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_replace_end_shadow",
     test_suite::PROJECT,
     "$",
     "",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_replace_capture_shadow",
     test_suite::PROJECT,
     "(a+)",
     "baaca",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_contains_operation",
     test_suite::PROJECT,
     "a+",
     "xxaa",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_full_literal_rejects",
     test_suite::PROJECT,
     "a",
     "zyzzyva",
     operation_kind::MATCHES,
     {},
     false},
    {"project_full_plus", test_suite::PROJECT, "a+", "aa", operation_kind::MATCHES, {}, true},
    {"project_full_nested_alternation",
     test_suite::PROJECT,
     "(a+|b)+",
     "aaab",
     operation_kind::MATCHES,
     {},
     true},
    {"project_full_alternative",
     test_suite::PROJECT,
     "ab|cd",
     "cd",
     operation_kind::MATCHES,
     {},
     true},
    {"project_full_alternative_rejects_substring",
     test_suite::PROJECT,
     "ab|cd",
     "xabcdx",
     operation_kind::MATCHES,
     {},
     false},
    {"project_full_star_zero", test_suite::PROJECT, "a*b", "b", operation_kind::MATCHES, {}, true},
    {"project_full_star_many",
     test_suite::PROJECT,
     "a*b",
     "aaaaab",
     operation_kind::MATCHES,
     {},
     true},
    {"project_full_bounded_too_few",
     test_suite::PROJECT,
     "a{2,4}",
     "a",
     operation_kind::MATCHES,
     {},
     false},
    {"project_full_bounded_middle",
     test_suite::PROJECT,
     "a{2,4}",
     "aaa",
     operation_kind::MATCHES,
     {},
     true},
    {"project_full_bounded_too_many",
     test_suite::PROJECT,
     "a{2,4}",
     "aaaaa",
     operation_kind::MATCHES,
     {},
     false},
    {"project_full_noncapture_star",
     test_suite::PROJECT,
     "(?:ab)*",
     "abab",
     operation_kind::MATCHES,
     {},
     true},
    {"project_full_noncapture_star_rejects_partial",
     test_suite::PROJECT,
     "(?:ab)*",
     "aba",
     operation_kind::MATCHES,
     {},
     false},
    {"project_full_empty_anchors",
     test_suite::PROJECT,
     "^$",
     "",
     operation_kind::MATCHES,
     {},
     true},
    {"project_full_empty_anchors_reject_text",
     test_suite::PROJECT,
     "^$",
     "x",
     operation_kind::MATCHES,
     {},
     false},
    {"project_full_repeated_anchor",
     test_suite::PROJECT,
     "^^(fo|foo)$",
     "foo",
     operation_kind::MATCHES,
     {},
     true},
    {"project_full_class_alternative",
     test_suite::PROJECT,
     "^(foo|bar|[A-Z])$",
     "X",
     operation_kind::MATCHES,
     {},
     true},
    {"project_full_class_alternative_rejects_pair",
     test_suite::PROJECT,
     "^(foo|bar|[A-Z])$",
     "XY",
     operation_kind::MATCHES,
     {},
     false},
    {"project_span_greedy_plus",
     test_suite::PROJECT,
     "a+",
     "xaaay",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_span_lazy_plus",
     test_suite::PROJECT,
     "a+?",
     "xaaay",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_span_shorter_alternative_first",
     test_suite::PROJECT,
     "fo|foo",
     "foo",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_span_longer_alternative_first",
     test_suite::PROJECT,
     "foo|fo",
     "foo",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_span_digits_before_literal",
     test_suite::PROJECT,
     "[0-9]+7",
     "x1237y",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_ascii_non_boundary_character",
     test_suite::PROJECT,
     R"REGEX(\Bx\B)REGEX",
     "axb",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_ascii_boundary_rejects_character",
     test_suite::PROJECT,
     R"REGEX(\bx\b)REGEX",
     "axb",
     operation_kind::CONTAINS,
     {},
     false},
    {"project_multiline_bar",
     test_suite::PROJECT,
     "^bar$",
     "foo\nbar\nbaz",
     operation_kind::CONTAINS,
     boolean_multiline,
     true},
    {"project_singleline_bar_rejects",
     test_suite::PROJECT,
     "^bar$",
     "foo\nbar\nbaz",
     operation_kind::CONTAINS,
     {},
     false},
    {"project_utf8_dot_mixed",
     test_suite::PROJECT,
     "^...$",
     ".本.",
     operation_kind::MATCHES,
     {},
     true},
    {"project_utf8_find_shadow",
     test_suite::PROJECT,
     "..",
     "xλ",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_capture_alternative_shadow",
     test_suite::PROJECT,
     "(fo|foo)",
     "foo",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_capture_repetition_shadow",
     test_suite::PROJECT,
     "(a|b)+",
     "abba",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_replace_literal_shadow",
     test_suite::PROJECT,
     "b",
     "ababab",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_replace_plus_shadow",
     test_suite::PROJECT,
     "b+",
     "bbbbbb",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_replace_empty_shadow",
     test_suite::PROJECT,
     "b*",
     "aaaaa",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_replace_dot_newline_shadow",
     test_suite::PROJECT,
     "a.*a",
     "aba\naba",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_count_empty_progress_shadow",
     test_suite::PROJECT,
     "a*",
     "bbb",
     operation_kind::CONTAINS,
     {},
     true},
    {"project_split_empty_progress_shadow",
     test_suite::PROJECT,
     "b*",
     "aa",
     operation_kind::CONTAINS,
     {},
     true},

    // direct RE2 FullMatch and PartialMatch expectations
    {"re2_full_single_literal", test_suite::RE2, "h", "h", operation_kind::MATCHES, {}, true},
    {"re2_full_literal_word", test_suite::RE2, "hello", "hello", operation_kind::MATCHES, {}, true},
    {"re2_full_dot_star", test_suite::RE2, "h.*o", "hello", operation_kind::MATCHES, {}, true},
    {"re2_full_is_front_anchored",
     test_suite::RE2,
     "h.*o",
     "othello",
     operation_kind::MATCHES,
     {},
     false},
    {"re2_full_is_end_anchored",
     test_suite::RE2,
     "h.*o",
     "hello!",
     operation_kind::MATCHES,
     {},
     false},
    {"re2_partial_dot_star_prefix",
     test_suite::RE2,
     "h.*o",
     "othello",
     operation_kind::CONTAINS,
     {},
     true},
    {"re2_partial_single_literal", test_suite::RE2, "x", "x", operation_kind::CONTAINS, {}, true},
    {"re2_partial_dot_star_suffix",
     test_suite::RE2,
     "h.*o",
     "hello!",
     operation_kind::CONTAINS,
     {},
     true},
    {"re2_partial_deep_groups",
     test_suite::RE2,
     "((((((((((((((((((((x))))))))))))))))))))",
     "x",
     operation_kind::CONTAINS,
     {},
     true},
    {"re2_partial_zero_args", test_suite::RE2, "e.*o", "hello", operation_kind::CONTAINS, {}, true},
    {"re2_partial_zero_args_rejects",
     test_suite::RE2,
     "a.*o",
     "othello",
     operation_kind::CONTAINS,
     {},
     false},
    {"re2_partial_digits",
     test_suite::RE2,
     R"REGEX((\d+))REGEX",
     "1001 nights",
     operation_kind::CONTAINS,
     {},
     true},
    {"re2_partial_digits_rejects",
     test_suite::RE2,
     R"REGEX((\d+))REGEX",
     "three",
     operation_kind::CONTAINS,
     {},
     false},
    {"re2_partial_multi_capture",
     test_suite::RE2,
     R"REGEX((\d+):(\w+))REGEX",
     "answer: 42:life",
     operation_kind::CONTAINS,
     {},
     true},
    {"re2_full_digits",
     test_suite::RE2,
     R"REGEX(\d+)REGEX",
     "1001",
     operation_kind::MATCHES,
     {},
     true},
    {"re2_full_signed_digits",
     test_suite::RE2,
     R"REGEX((-?\d+))REGEX",
     "-123",
     operation_kind::MATCHES,
     {},
     true},
    {"re2_full_integer_middle",
     test_suite::RE2,
     R"REGEX(1(\d*)4)REGEX",
     "1234",
     operation_kind::MATCHES,
     {},
     true},
    {"re2_full_string_capture",
     test_suite::RE2,
     "h(.*)o",
     "hello",
     operation_kind::MATCHES,
     {},
     true},
    {"re2_full_word_and_digits",
     test_suite::RE2,
     R"REGEX((\w+):(\d+))REGEX",
     "ruby:1234",
     operation_kind::MATCHES,
     {},
     true},
    {"re2_full_word_and_digits_rejects",
     test_suite::RE2,
     R"REGEX((\d+):(\w+))REGEX",
     "hi1",
     operation_kind::MATCHES,
     {},
     false},

    // rust-lang/regex testdata: only syntax whose semantics Regex IR supports
    {"rust_ascii_literal", test_suite::RUST_REGEX, "a", "a", operation_kind::CONTAINS, {}, true},
    {"rust_ascii_literal_not",
     test_suite::RUST_REGEX,
     "a",
     "z",
     operation_kind::CONTAINS,
     {},
     false},
    {"rust_prefix_literal",
     test_suite::RUST_REGEX,
     "^abc",
     "abc",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_prefix_literal_not",
     test_suite::RUST_REGEX,
     "^abc",
     "zabc",
     operation_kind::CONTAINS,
     {},
     false},
    {"rust_incomplete_literal",
     test_suite::RUST_REGEX,
     "abc",
     "xxxxxab",
     operation_kind::CONTAINS,
     {},
     false},
    {"rust_terminates", test_suite::RUST_REGEX, "a$", "a", operation_kind::CONTAINS, {}, true},
    {"rust_suffix_repeated",
     test_suite::RUST_REGEX,
     ".*(?:abcd)+",
     "abcdabcd",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_suffix_with_prefix",
     test_suite::RUST_REGEX,
     ".*x(?:abcd)+",
     "abcdxabcd",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_anchored_greedy",
     test_suite::RUST_REGEX,
     "(abc)+",
     "abcabcabc",
     operation_kind::MATCHES,
     {},
     true},
    {"rust_anchored_nongreedy",
     test_suite::RUST_REGEX,
     "(abc)+?",
     "abcabcabc",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_anchored_no_match_at_start",
     test_suite::RUST_REGEX,
     ".c",
     "abc",
     operation_kind::MATCHES,
     {},
     false},
    {"rust_empty_alternative_left",
     test_suite::RUST_REGEX,
     "|b",
     "abc",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_empty_alternative_right",
     test_suite::RUST_REGEX,
     "b|",
     "abc",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_empty_alternatives",
     test_suite::RUST_REGEX,
     "||",
     "abc",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_empty_group_alternative",
     test_suite::RUST_REGEX,
     "(?:)|b",
     "abc",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_empty_regex_empty_input",
     test_suite::RUST_REGEX,
     "",
     "",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_empty_regex_nonempty_input",
     test_suite::RUST_REGEX,
     "",
     "abc",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_float",
     test_suite::RUST_REGEX,
     R"REGEX([-+]?[0-9]*\.?[0-9]+)REGEX",
     "0.1",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_anchored_float_rejects",
     test_suite::RUST_REGEX,
     R"REGEX(^[-+]?[0-9]*\.?[0-9]+$)REGEX",
     "1.a",
     operation_kind::CONTAINS,
     {},
     false},
    {"rust_email",
     test_suite::RUST_REGEX,
     R"REGEX(\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}\b)REGEX",
     "mine is jam.slam@gmail.com ",
     operation_kind::CONTAINS,
     boolean_case_insensitive,
     true},
    {"rust_email_not",
     test_suite::RUST_REGEX,
     R"REGEX(\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}\b)REGEX",
     "mine is jam.slam@gmail ",
     operation_kind::CONTAINS,
     boolean_case_insensitive,
     false},
    {"rust_date",
     test_suite::RUST_REGEX,
     R"REGEX(^(?:19|20)\d\d[- /.](?:0[1-9]|1[012])[- /.](?:0[1-9]|[12][0-9]|3[01])$)REGEX",
     "1900-01-01",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_date_bad_month_zero",
     test_suite::RUST_REGEX,
     R"REGEX(^(?:19|20)\d\d[- /.](?:0[1-9]|1[012])[- /.](?:0[1-9]|[12][0-9]|3[01])$)REGEX",
     "1900-00-01",
     operation_kind::CONTAINS,
     {},
     false},
    {"rust_date_bad_month_thirteen",
     test_suite::RUST_REGEX,
     R"REGEX(^(?:19|20)\d\d[- /.](?:0[1-9]|1[012])[- /.](?:0[1-9]|[12][0-9]|3[01])$)REGEX",
     "1900-13-01",
     operation_kind::CONTAINS,
     {},
     false},
    {"rust_empty_anchors", test_suite::RUST_REGEX, "^$", "", operation_kind::CONTAINS, {}, true},
    {"rust_reversed_empty_anchors",
     test_suite::RUST_REGEX,
     "$^",
     "",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_negated_class",
     test_suite::RUST_REGEX,
     "[^ac]",
     "acx",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_lazy_nested_repetition",
     test_suite::RUST_REGEX,
     "(?:(?:.*)*?)=",
     "a=b",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_greedy_nested_repetition",
     test_suite::RUST_REGEX,
     "(?:(?:.*){1,2})=",
     "a=b",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_multiline_lines",
     test_suite::RUST_REGEX,
     "^[a-z]+$",
     "abc\ndef\nxyz",
     operation_kind::CONTAINS,
     boolean_multiline,
     true},
    {"rust_multiline_empty_line_rejects",
     test_suite::RUST_REGEX,
     "^$",
     "abc\ndef\nxyz",
     operation_kind::CONTAINS,
     boolean_multiline,
     false},
    {"rust_multiline_character_before_end",
     test_suite::RUST_REGEX,
     "[a-z]$",
     "abc\ndef\nxyz",
     operation_kind::CONTAINS,
     boolean_multiline,
     true},
    {"rust_case_class_underscore",
     test_suite::RUST_REGEX,
     "[a_]+",
     "A_",
     operation_kind::CONTAINS,
     boolean_case_insensitive,
     true},
    {"rust_case_negated_class",
     test_suite::RUST_REGEX,
     "[^x]",
     "X",
     operation_kind::CONTAINS,
     boolean_case_insensitive,
     false},
    {"rust_alternative_with_end",
     test_suite::RUST_REGEX,
     "ab?|$",
     "az",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_leftmost_prefix",
     test_suite::RUST_REGEX,
     "z*azb",
     "azb",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_many_alternatives",
     test_suite::RUST_REGEX,
     "1|2|3|4|5|6|7|8|9|10|int",
     "int",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_partial_anchor",
     test_suite::RUST_REGEX,
     "^a|b",
     "ba",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_partial_anchor_begin_rejects",
     test_suite::RUST_REGEX,
     "^a|z",
     "yyyyya",
     operation_kind::CONTAINS,
     {},
     false},
    {"rust_partial_anchor_end_rejects",
     test_suite::RUST_REGEX,
     "a$|z",
     "ayyyyy",
     operation_kind::CONTAINS,
     {},
     false},
    {"rust_unambiguous_literals",
     test_suite::RUST_REGEX,
     "(?:ABC|CDA|BC)X",
     "CDAX",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_optional_capture_premature_end",
     test_suite::RUST_REGEX,
     "a(b*(X|$))?",
     "abcbX",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_literal_no_match",
     test_suite::RUST_REGEX,
     R"REGEX(typename type\-parameter\-[0-9]+\-[0-9]+::.+)REGEX",
     "test",
     operation_kind::CONTAINS,
     {},
     false},
    {"rust_empty_group_unicode_literal",
     test_suite::RUST_REGEX,
     "(?:)Ј01",
     "zЈ01",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_reverse_suffix",
     test_suite::RUST_REGEX,
     "[0-4][0-4][0-4]000",
     "153.230000",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_reverse_suffix_boundary_rejects",
     test_suite::RUST_REGEX,
     R"REGEX(\w+foobar\b)REGEX",
     "xyzfoobarZ",
     operation_kind::CONTAINS,
     {},
     false},
    {"rust_unicode_literal_snowman",
     test_suite::RUST_REGEX,
     "☃",
     "☃",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_unicode_literal_repetition",
     test_suite::RUST_REGEX,
     "☃+",
     "☃",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_unicode_explicit_class",
     test_suite::RUST_REGEX,
     "[☃Ⅰ]+",
     "☃",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_ascii_digit_before_unicode_space",
     test_suite::RUST_REGEX,
     R"REGEX(\d\b)REGEX",
     "6 ",
     operation_kind::CONTAINS,
     {},
     true},
    {"rust_ascii_word_bytes",
     test_suite::RUST_REGEX,
     R"REGEX(\w+)REGEX",
     "aδ",
     operation_kind::CONTAINS,
     boolean_bytes,
     true},
    {"rust_ascii_space_boundary_bytes",
     test_suite::RUST_REGEX,
     R"REGEX( \b)REGEX",
     " δ",
     operation_kind::CONTAINS,
     boolean_bytes,
     false},
    {"rust_ascii_space_non_boundary_bytes",
     test_suite::RUST_REGEX,
     R"REGEX( \B)REGEX",
     " δ",
     operation_kind::CONTAINS,
     boolean_bytes,
     true},

    // CPython re_tests cases within the shared regular-language syntax
    {"cpython_empty", test_suite::CPYTHON, "", "", operation_kind::CONTAINS, {}, true},
    {"cpython_literal", test_suite::CPYTHON, "abc", "xabcy", operation_kind::CONTAINS, {}, true},
    {"cpython_literal_miss",
     test_suite::CPYTHON,
     "abc",
     "axc",
     operation_kind::CONTAINS,
     {},
     false},
    {"cpython_star_backtracks",
     test_suite::CPYTHON,
     "ab*bc",
     "abbbbc",
     operation_kind::CONTAINS,
     {},
     true},
    {"cpython_plus_requires_one",
     test_suite::CPYTHON,
     "ab+bc",
     "abc",
     operation_kind::CONTAINS,
     {},
     false},
    {"cpython_optional_zero",
     test_suite::CPYTHON,
     "ab?bc",
     "abc",
     operation_kind::CONTAINS,
     {},
     true},
    {"cpython_full_anchor",
     test_suite::CPYTHON,
     "^abc$",
     "abc",
     operation_kind::CONTAINS,
     {},
     true},
    {"cpython_full_anchor_rejects",
     test_suite::CPYTHON,
     "^abc$",
     "aabc",
     operation_kind::CONTAINS,
     {},
     false},
    {"cpython_suffix_anchor",
     test_suite::CPYTHON,
     "abc$",
     "aabc",
     operation_kind::CONTAINS,
     {},
     true},
    {"cpython_dot", test_suite::CPYTHON, "a.c", "axc", operation_kind::CONTAINS, {}, true},
    {"cpython_dot_rejects_lf",
     test_suite::CPYTHON,
     "a.b",
     "a\nb",
     operation_kind::CONTAINS,
     {},
     false},
    {"cpython_dot_accepts_cr",
     test_suite::CPYTHON,
     "a.b",
     "a\rb",
     operation_kind::CONTAINS,
     {},
     true},
    {"cpython_dotall",
     test_suite::CPYTHON,
     "a.*b",
     "acc\nccb",
     operation_kind::CONTAINS,
     boolean_dot_all,
     true},
    {"cpython_class", test_suite::CPYTHON, "a[bc]d", "abd", operation_kind::CONTAINS, {}, true},
    {"cpython_class_range",
     test_suite::CPYTHON,
     "a[b-d]e",
     "ace",
     operation_kind::CONTAINS,
     {},
     true},
    {"cpython_negated_class",
     test_suite::CPYTHON,
     "a[^bc]d",
     "aed",
     operation_kind::CONTAINS,
     {},
     true},
    {"cpython_word_boundary",
     test_suite::CPYTHON,
     R"REGEX(\ba\b)REGEX",
     "-a-",
     operation_kind::CONTAINS,
     {},
     true},
    {"cpython_word_boundary_rejects",
     test_suite::CPYTHON,
     R"REGEX(\by\b)REGEX",
     "xyz",
     operation_kind::CONTAINS,
     {},
     false},
    {"cpython_non_boundary",
     test_suite::CPYTHON,
     R"REGEX(\By\B)REGEX",
     "xyz",
     operation_kind::CONTAINS,
     {},
     true},
    {"cpython_alternation",
     test_suite::CPYTHON,
     "ab|cd",
     "abcde",
     operation_kind::CONTAINS,
     {},
     true},
    {"cpython_nested_alternation",
     test_suite::CPYTHON,
     "(ab|cd)e",
     "abcde",
     operation_kind::CONTAINS,
     {},
     true},
    {"cpython_nullable_alternative",
     test_suite::CPYTHON,
     "(abc|)ef",
     "abcdef",
     operation_kind::CONTAINS,
     {},
     true},
    {"cpython_greedy_capture",
     test_suite::CPYTHON,
     "(.*)c(.*)",
     "abcde",
     operation_kind::CONTAINS,
     {},
     true},
    {"cpython_lazy_repeat",
     test_suite::CPYTHON,
     "a(?:b|c|d)+?(.)",
     "ace",
     operation_kind::CONTAINS,
     {},
     true},
    {"cpython_multiline",
     test_suite::CPYTHON,
     "^abc",
     "jkl\nabc\nxyz",
     operation_kind::CONTAINS,
     boolean_multiline,
     true},
    {"cpython_ignore_case",
     test_suite::CPYTHON,
     "m+",
     "MMM",
     operation_kind::CONTAINS,
     boolean_case_insensitive,
     true},

    // sihlfall's exhaustive RE2-derived categories, projected to booleans
    {"sihlfall_literal", test_suite::SIHLFALL, "a", "a", operation_kind::CONTAINS, {}, true},
    {"sihlfall_literal_miss", test_suite::SIHLFALL, "a", "b", operation_kind::CONTAINS, {}, false},
    {"sihlfall_full_literal",
     test_suite::SIHLFALL,
     "^(?:a)$",
     "a",
     operation_kind::CONTAINS,
     {},
     true},
    {"sihlfall_full_literal_rejects_pair",
     test_suite::SIHLFALL,
     "^(?:a)$",
     "aa",
     operation_kind::CONTAINS,
     {},
     false},
    {"sihlfall_literal_concatenation",
     test_suite::SIHLFALL,
     "(?:a(?:ab))",
     "aab",
     operation_kind::CONTAINS,
     {},
     true},
    {"sihlfall_literal_alternation",
     test_suite::SIHLFALL,
     "(?:a|(?:ab))",
     "ab",
     operation_kind::CONTAINS,
     {},
     true},
    {"sihlfall_dot_concatenation",
     test_suite::SIHLFALL,
     "(?:a(?:a.))",
     "aac",
     operation_kind::CONTAINS,
     {},
     true},
    {"sihlfall_simple_star",
     test_suite::SIHLFALL,
     "^(?:a*)$",
     "aaaa",
     operation_kind::CONTAINS,
     {},
     true},
    {"sihlfall_simple_plus_rejects_empty",
     test_suite::SIHLFALL,
     "^(?:a+)$",
     "",
     operation_kind::CONTAINS,
     {},
     false},
    {"sihlfall_simple_optional",
     test_suite::SIHLFALL,
     "^(?:a?)$",
     "a",
     operation_kind::CONTAINS,
     {},
     true},
    {"sihlfall_capturing_repeat",
     test_suite::SIHLFALL,
     "^((?:a|b)+)$",
     "abba",
     operation_kind::CONTAINS,
     {},
     true},
    {"sihlfall_empty_match",
     test_suite::SIHLFALL,
     "^(?:(?:)*)$",
     "",
     operation_kind::CONTAINS,
     {},
     true},
    {"sihlfall_punctuation_literal",
     test_suite::SIHLFALL,
     R"REGEX(\+\?\.)REGEX",
     "+?.",
     operation_kind::CONTAINS,
     {},
     true},
    {"sihlfall_line_begin",
     test_suite::SIHLFALL,
     "^ab",
     "ab\ncd",
     operation_kind::CONTAINS,
     boolean_multiline,
     true},
    {"sihlfall_line_end",
     test_suite::SIHLFALL,
     "cd$",
     "ab\ncd",
     operation_kind::CONTAINS,
     boolean_multiline,
     true},
    {"sihlfall_character_class",
     test_suite::SIHLFALL,
     "^[a-c]+$",
     "abccba",
     operation_kind::CONTAINS,
     {},
     true},
    {"sihlfall_negated_character_class",
     test_suite::SIHLFALL,
     "^[^c]+$",
     "abba",
     operation_kind::CONTAINS,
     {},
     true},
    {"sihlfall_negated_character_class_rejects",
     test_suite::SIHLFALL,
     "^[^c]+$",
     "abc",
     operation_kind::CONTAINS,
     {},
     false},
    {"sihlfall_utf8_literal",
     test_suite::SIHLFALL,
     "日本",
     "x日本y",
     operation_kind::CONTAINS,
     {},
     true},
    {"sihlfall_utf8_dot", test_suite::SIHLFALL, "^..$", "日本", operation_kind::CONTAINS, {}, true},
    {"sihlfall_utf8_class",
     test_suite::SIHLFALL,
     "^[日語]+$",
     "日日語",
     operation_kind::CONTAINS,
     {},
     true},
  });
  return cases;
}

// scalar cuDF case tables

constexpr compile_options cudf_default{.ascii_classes = false};
constexpr compile_options cudf_ascii{};
constexpr compile_options cudf_multiline{.multiline = true, .ascii_classes = false};
constexpr compile_options cudf_dot_all{.dot_all = true, .ascii_classes = false};
constexpr compile_options cudf_extended{.ascii_classes = false, .extended_newline = true};
constexpr compile_options cudf_extended_multiline{
  .multiline = true, .ascii_classes = false, .extended_newline = true};
constexpr compile_options cudf_extended_dot_all{
  .dot_all = true, .ascii_classes = false, .extended_newline = true};
constexpr compile_options cudf_multiline_dot_all{
  .multiline = true, .dot_all = true, .ascii_classes = false};
constexpr compile_options cudf_case_insensitive{.case_insensitive = true, .ascii_classes = false};

cudf_expected_row scalar_row(std::int64_t value) { return {.scalar = value}; }

cudf_expected_row strings_row(std::vector<maybe_string> values)
{
  return {.strings = std::move(values)};
}

cudf_expected_row null_row() { return {.valid = false}; }

void append_case(std::vector<cudf_regex_case>& cases, cudf_regex_case test)
{
  if (test.inputs.size() != test.expected.size()) {
    throw std::logic_error("cuDF regex case input and expected row counts differ: " +
                           test.test_name + "." + test.assertion);
  }
  for (std::size_t row = 0; row < test.inputs.size(); ++row) {
    if (!test.inputs[row].has_value() && test.expected[row].valid) {
      throw std::logic_error("cuDF regex case failed to propagate null input: " + test.test_name +
                             "." + test.assertion);
    }
  }
  cases.push_back(std::move(test));
}

void append_case(std::vector<cudf_regex_case>& cases,
                 std::string test_name,
                 std::string assertion,
                 std::string pattern,
                 cudf_regex_operation operation,
                 compile_options options,
                 std::vector<maybe_string> inputs,
                 std::vector<std::int64_t> expected)
{
  if (inputs.size() != expected.size()) {
    throw std::logic_error("cuDF scalar case input and expected row counts differ");
  }
  cudf_regex_case test{.test_name = std::move(test_name),
                       .assertion = std::move(assertion),
                       .pattern   = std::move(pattern),
                       .options   = options,
                       .operation = operation,
                       .inputs    = std::move(inputs)};
  test.expected.reserve(expected.size());
  for (std::size_t row = 0; row < expected.size(); ++row) {
    test.expected.push_back(test.inputs[row] ? scalar_row(expected[row]) : null_row());
  }
  append_case(cases, std::move(test));
}

void append_scalar_group(std::vector<cudf_regex_case>& groups,
                         std::string_view test_case,
                         std::string pattern,
                         cudf_regex_operation selected,
                         compile_options options,
                         std::vector<std::string> inputs,
                         std::vector<std::int64_t> expected)
{
  if (inputs.size() != expected.size()) {
    throw std::logic_error("cuDF test inputs and expectations differ in size");
  }
  options.ascii_classes = test_case.find(".ASCII") != std::string_view::npos;
  auto fixture_end      = test_case.find('.');
  auto assertion_begin  = test_case.find('.', fixture_end + 1);
  std::string test_name{test_case.substr(0, assertion_begin)};
  std::vector<maybe_string> nullable_inputs;
  nullable_inputs.reserve(inputs.size());
  for (std::string& input : inputs) {
    nullable_inputs.emplace_back(std::move(input));
  }
  append_case(groups,
              std::move(test_name),
              "cudf_group_" + std::to_string(groups.size()),
              std::move(pattern),
              selected,
              options,
              std::move(nullable_inputs),
              std::move(expected));
}

void append_bit_rows(std::vector<cudf_regex_case>& groups,
                     std::string_view test_case,
                     std::string pattern,
                     cudf_regex_operation selected,
                     compile_options options,
                     std::vector<std::string> const& inputs,
                     std::string_view bits,
                     std::size_t skipped_index = std::numeric_limits<std::size_t>::max())
{
  if (inputs.size() != bits.size()) {
    throw std::logic_error("cuDF test bit string has the wrong size");
  }
  std::vector<std::string> selected_inputs;
  std::vector<std::int64_t> expected;
  selected_inputs.reserve(inputs.size());
  expected.reserve(inputs.size());
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    if (index == skipped_index) continue;
    selected_inputs.push_back(inputs[index]);
    expected.push_back(bits[index] == '1' ? 1U : 0U);
  }
  append_scalar_group(groups,
                      test_case,
                      std::move(pattern),
                      selected,
                      options,
                      std::move(selected_inputs),
                      std::move(expected));
}

std::string hex_escape(unsigned int value)
{
  std::ostringstream output;
  output << R"REGEX(\x)REGEX" << std::hex << std::setfill('0') << std::setw(2) << value;
  return output.str();
}

void append_scalar_cases(std::vector<cudf_regex_case>& groups)
{
  // cuDF ContainsTest: one group per pattern, excluding the null input row
  std::vector<std::string> contains_inputs{"5",
                                           "hej",
                                           "\t \n",
                                           "12345",
                                           R"REGEX(\)REGEX",
                                           "d",
                                           R"REGEX(c:\Tools)REGEX",
                                           "+27",
                                           "1c2",
                                           "1C2",
                                           "0:00:0",
                                           "0:0:00",
                                           "00:0:0",
                                           "00:00:0",
                                           "00:0:00",
                                           "0:00:00",
                                           "00:00:00",
                                           "Hello world !",
                                           "Hello world!   ",
                                           "Hello worldcup  !",
                                           "0123456789",
                                           "1C2",
                                           "Xaa",
                                           "abcdefghxxx",
                                           "ABCDEFGH",
                                           "abcdefgh",
                                           "abc def",
                                           "abc\ndef",
                                           "aa\r\nbb\r\ncc\r\n\r\n",
                                           "abcabc",
                                           "",
                                           ""};
  struct pattern_bits {
    std::string_view pattern = "";
    std::string_view bits    = "";
  };
  std::vector<pattern_bits> contains_patterns{
    {R"REGEX(\d)REGEX", "10010001111111111000110000000000"},
    {R"REGEX(\w+)REGEX", "11010111111111111111111111111100"},
    {R"REGEX(\s)REGEX", "00100000000000000111000000111000"},
    {R"REGEX(\S)REGEX", "11011111111111111111111111111100"},
    {R"REGEX(^.*\\.*$)REGEX", "00001010000000000000000000000000"},
    {"[1-5]+", "10010001110000000000110000000000"},
    {"[a-h]+", "01000110100000000111001101111100"},
    {"[A-H]+", "00000000010000000111010010000000"},
    {"[a-h]*", "11111111111111111111111111111101"},
    {"\n", "00100000000000000000000000011000"},
    {R"REGEX(b.\s*\n)REGEX", "00000000000000000000000000011000"},
    {".*c", "00000010100000000001000101111100"},
    {R"REGEX(\d\d:\d\d:\d\d)REGEX", "00000000000000001000000000000000"},
    {R"REGEX(\d\d?:\d\d?:\d\d?)REGEX", "00000000001111111000000000000000"},
    {"[Hh]ello [Ww]orld", "00000000000000000111000000000000"},
    {R"REGEX(\bworld\b)REGEX", "00000000000000000110000000000000"},
    {".*", "11111111111111111111111111111101"}};
  for (pattern_bits const& item : contains_patterns) {
    append_bit_rows(groups,
                    "StringsContainsTests.ContainsTest",
                    std::string{item.pattern},
                    cudf_regex_operation::CONTAINS,
                    {},
                    contains_inputs,
                    item.bits,
                    30);
  }

  // cuDF matches_re is a prefix operation, not Regex IR's whole-input matches operation
  std::vector<std::string> matches_inputs{
    "The quick brown @fox jumps", "ovér the", "lazy @dog", "1234", "00:0:00", "", ""};
  append_bit_rows(groups,
                  "StringsContainsTests.MatchesTest.lazy",
                  "lazy",
                  cudf_regex_operation::PREFIX_MATCH,
                  {},
                  matches_inputs,
                  "0010000",
                  5);
  append_bit_rows(groups,
                  "StringsContainsTests.MatchesTest.digits",
                  R"REGEX(\d+)REGEX",
                  cudf_regex_operation::PREFIX_MATCH,
                  {},
                  matches_inputs,
                  "0001100",
                  5);
  append_bit_rows(groups,
                  "StringsContainsTests.MatchesTest.word",
                  R"REGEX(@\w+)REGEX",
                  cudf_regex_operation::PREFIX_MATCH,
                  {},
                  matches_inputs,
                  "0000000",
                  5);
  append_bit_rows(groups,
                  "StringsContainsTests.MatchesTest.any",
                  ".*",
                  cudf_regex_operation::PREFIX_MATCH,
                  {},
                  matches_inputs,
                  "1111101",
                  5);

  std::vector<std::string> ipv4_inputs{"5.79.97.178",
                                       "1.2.3.4",
                                       "5",
                                       "5.79",
                                       "5.79.97",
                                       "5.79.97.178.100",
                                       "224.0.0.0",
                                       "239.255.255.255",
                                       "5.79.97.178",
                                       "127.0.0.1"};
  append_bit_rows(
    groups,
    "StringsContainsTests.MatchesIPV4Test.is_ip",
    R"REGEX(^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$)REGEX",
    cudf_regex_operation::PREFIX_MATCH,
    {},
    ipv4_inputs,
    "1100001111");
  append_bit_rows(
    groups,
    "StringsContainsTests.MatchesIPV4Test.is_loopback",
    R"REGEX(^127\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$)REGEX",
    cudf_regex_operation::PREFIX_MATCH,
    {},
    ipv4_inputs,
    "0000000001");
  append_bit_rows(
    groups,
    "StringsContainsTests.MatchesIPV4Test.is_multicast",
    R"REGEX(^(2(2[4-9]|3[0-9]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$)REGEX",
    cudf_regex_operation::PREFIX_MATCH,
    {},
    ipv4_inputs,
    "0000001100");

  // exhaustive cuDF HexTest, including the character-class form
  std::vector<std::string> ascii_inputs;
  ascii_inputs.reserve(127);
  for (unsigned int value = 0; value < 127; ++value) {
    ascii_inputs.emplace_back(1, static_cast<char>(value));
  }
  for (unsigned int value = 0; value < 127; ++value) {
    std::vector<std::int64_t> expected(127, 0);
    expected[value]     = 1U;
    std::string escaped = hex_escape(value);
    append_scalar_group(groups,
                        "StringsContainsTests.HexTest.literal",
                        escaped,
                        cudf_regex_operation::CONTAINS,
                        {},
                        ascii_inputs,
                        expected);
    append_scalar_group(groups,
                        "StringsContainsTests.HexTest.class",
                        "[" + escaped + "]",
                        cudf_regex_operation::CONTAINS,
                        {},
                        ascii_inputs,
                        std::move(expected));
  }

  std::vector<std::string> null_inputs;
  for (char first = 'A'; first <= 'J'; ++first) {
    std::string value;
    value.push_back(first);
    value.push_back('\0');
    value.push_back('B');
    null_inputs.push_back(std::move(value));
  }
  append_bit_rows(groups,
                  "StringsContainsTests.EmbeddedNullCharacter.A",
                  "A",
                  cudf_regex_operation::CONTAINS,
                  {},
                  null_inputs,
                  "1000000000");
  append_bit_rows(groups,
                  "StringsContainsTests.EmbeddedNullCharacter.B",
                  "B",
                  cudf_regex_operation::CONTAINS,
                  {},
                  null_inputs,
                  "1111111111");
  append_bit_rows(groups,
                  "StringsContainsTests.EmbeddedNullCharacter.hex_null",
                  R"REGEX([A-D][\x00]B)REGEX",
                  cudf_regex_operation::CONTAINS,
                  {},
                  null_inputs,
                  "1111000000");

  std::vector<std::string> count_inputs{
    "The quick brown @fox jumps ovér the", "lazy @dog", "1:2:3:4", "00:0:00", ""};
  append_scalar_group(groups,
                      "StringsContainsTests.CountTest.the",
                      "[tT]he",
                      cudf_regex_operation::COUNT,
                      {},
                      count_inputs,
                      {2, 0, 0, 0, 0});
  append_scalar_group(groups,
                      "StringsContainsTests.CountTest.word",
                      R"REGEX(@\w+)REGEX",
                      cudf_regex_operation::COUNT,
                      {},
                      count_inputs,
                      {1, 1, 0, 0, 0});
  append_scalar_group(groups,
                      "StringsContainsTests.CountTest.digits",
                      R"REGEX(\d+:\d+)REGEX",
                      cudf_regex_operation::COUNT,
                      {},
                      count_inputs,
                      {0, 0, 2, 1, 0});

  std::vector<std::string> empty_count_inputs{"hello", "world", "", "abc"};
  for (std::string pattern : {std::string{"a*"},
                              std::string{"X?"},
                              std::string{"b{0,}"},
                              std::string{"()"},
                              std::string{"(?:)"},
                              std::string{"[A-Z]*"}}) {
    append_scalar_group(groups,
                        "StringsContainsTests.CountEmptyMatching",
                        std::move(pattern),
                        cudf_regex_operation::COUNT,
                        {},
                        empty_count_inputs,
                        {6, 6, 1, 4});
  }
  append_scalar_group(groups,
                      "StringsContainsTests.CountEmptyMatching.begin",
                      "^",
                      cudf_regex_operation::COUNT,
                      {},
                      empty_count_inputs,
                      {1, 1, 1, 1});
  append_scalar_group(groups,
                      "StringsContainsTests.CountEmptyMatching.end",
                      "$",
                      cudf_regex_operation::COUNT,
                      {},
                      empty_count_inputs,
                      {1, 1, 1, 1});
  append_scalar_group(groups,
                      "StringsContainsTests.CountEmptyMatching.empty",
                      "^$",
                      cudf_regex_operation::COUNT,
                      {},
                      empty_count_inputs,
                      {0, 0, 1, 0});
  append_scalar_group(groups,
                      "StringsContainsTests.CountEmptyMatching.boundary",
                      R"REGEX(\b)REGEX",
                      cudf_regex_operation::COUNT,
                      {},
                      empty_count_inputs,
                      {2, 2, 0, 2});
  append_scalar_group(groups,
                      "StringsContainsTests.CountEmptyMatching.non_boundary",
                      R"REGEX(\B)REGEX",
                      cudf_regex_operation::COUNT,
                      {},
                      empty_count_inputs,
                      {4, 4, 1, 2});

  std::vector<std::string> fixed_inputs{"a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa"};
  append_scalar_group(groups,
                      "StringsContainsTests.FixedQuantifier.exact",
                      "a{3}",
                      cudf_regex_operation::COUNT,
                      {},
                      fixed_inputs,
                      {0, 0, 1, 1, 1, 2});
  append_scalar_group(groups,
                      "StringsContainsTests.FixedQuantifier.range",
                      "a{3,5}",
                      cudf_regex_operation::COUNT,
                      {},
                      fixed_inputs,
                      {0, 0, 1, 1, 1, 1});
  append_scalar_group(groups,
                      "StringsContainsTests.FixedQuantifier.minimum",
                      "a{2,}",
                      cudf_regex_operation::COUNT,
                      {},
                      fixed_inputs,
                      {0, 1, 1, 1, 1, 1});
  append_scalar_group(groups,
                      "StringsContainsTests.FixedQuantifier.lazy_range",
                      "a{2,4}?",
                      cudf_regex_operation::COUNT,
                      {},
                      fixed_inputs,
                      {0, 1, 1, 2, 2, 3});
  append_scalar_group(groups,
                      "StringsContainsTests.FixedQuantifier.lazy_minimum",
                      "a{1,}?",
                      cudf_regex_operation::COUNT,
                      {},
                      fixed_inputs,
                      {1, 2, 3, 4, 5, 6});
  append_scalar_group(groups,
                      "StringsContainsTests.FixedQuantifier.zero",
                      "aaaa{0}",
                      cudf_regex_operation::COUNT,
                      {},
                      fixed_inputs,
                      {0, 0, 1, 1, 1, 2});

  std::vector<std::string> zero_range_inputs{"a", "", "abc", "XYAZ", "ABC", "ZYXA"};
  append_bit_rows(groups,
                  "StringsContainsTests.ZeroRangeQuantifier.A.contains",
                  "A{0,}",
                  cudf_regex_operation::CONTAINS,
                  {},
                  zero_range_inputs,
                  "111111");
  append_scalar_group(groups,
                      "StringsContainsTests.ZeroRangeQuantifier.A.count",
                      "A{0,}",
                      cudf_regex_operation::COUNT,
                      {},
                      zero_range_inputs,
                      {2, 1, 4, 5, 4, 5});
  append_bit_rows(groups,
                  "StringsContainsTests.ZeroRangeQuantifier.ab.contains",
                  "(?:ab){0,3}",
                  cudf_regex_operation::CONTAINS,
                  {},
                  zero_range_inputs,
                  "111111");
  append_scalar_group(groups,
                      "StringsContainsTests.ZeroRangeQuantifier.ab.count",
                      "(?:ab){0,3}",
                      cudf_regex_operation::COUNT,
                      {},
                      zero_range_inputs,
                      {2, 1, 3, 5, 4, 5});

  append_bit_rows(groups,
                  "StringsContainsTests.NestedQuantifier",
                  R"REGEX((\d{4}\s){4})REGEX",
                  cudf_regex_operation::CONTAINS,
                  {},
                  {"TEST12 1111 2222 3333 4444 5555",
                   "0000 AAAA 9999 BBBB 8888",
                   "7777 6666 4444 3333",
                   "12345 3333 4444 1111 ABCD"},
                  "1001");

  std::vector<std::string> class_inputs{"abcdefg", "defghí", "", "éééééé", "ghijkl"};
  append_scalar_group(groups,
                      "StringsContainsTests.OverlappedClasses.ascii",
                      "[e-gb-da-c]",
                      cudf_regex_operation::COUNT,
                      {},
                      class_inputs,
                      {7, 4, 0, 0, 1});
  append_scalar_group(groups,
                      "StringsContainsTests.OverlappedClasses.unicode",
                      "[á-éê-ú]",
                      cudf_regex_operation::COUNT,
                      {},
                      class_inputs,
                      {0, 1, 0, 6, 0});
  std::vector<std::string> negated_inputs{"abcdefg", "def\tghí", "", "éeé\néeé", "ABC"};
  append_scalar_group(groups,
                      "StringsContainsTests.NegatedClasses.ascii",
                      "[^a-f]",
                      cudf_regex_operation::COUNT,
                      {},
                      negated_inputs,
                      {1, 4, 0, 5, 3});
  append_scalar_group(groups,
                      "StringsContainsTests.NegatedClasses.unicode",
                      "[^a-eá-é]",
                      cudf_regex_operation::COUNT,
                      {},
                      negated_inputs,
                      {2, 5, 0, 1, 3});

  std::vector<std::string> incomplete_inputs{"abc-def", "---", "", "ghijkl", "-wxyz-"};
  struct incomplete_case {
    std::string_view pattern = "";
    std::string_view bits    = "";
  };
  for (incomplete_case item : std::vector<incomplete_case>{{"[a-z]", "10011"},
                                                           {"[a-m-z]", "11011"},
                                                           {"[a-f-q-z]", "11001"},
                                                           {"[g-g-z]", "11011"},
                                                           {"[g-]", "11011"},
                                                           {"[-k]", "11011"},
                                                           {"[-]", "11001"},
                                                           {"[+--]", "11001"},
                                                           {"[a-c-]", "11001"},
                                                           {"[-d-f]", "11001"}}) {
    append_bit_rows(groups,
                    "StringsContainsTests.IncompleteClassesRange",
                    std::string{item.pattern},
                    cudf_regex_operation::CONTAINS,
                    {},
                    incomplete_inputs,
                    item.bits);
  }

  std::vector<std::string> multiline_inputs{
    "abé\nfff\nabé", "fff\nabé\nlll", "abé", "", "abé\n", "abe\nabé\n"};
  append_bit_rows(groups,
                  "StringsContainsTests.MultiLine.contains",
                  "^abé$",
                  cudf_regex_operation::CONTAINS,
                  cudf_multiline,
                  multiline_inputs,
                  "111011");
  append_bit_rows(groups,
                  "StringsContainsTests.MultiLine.contains_default",
                  "^abé$",
                  cudf_regex_operation::CONTAINS,
                  {},
                  multiline_inputs,
                  "001010");
  append_bit_rows(groups,
                  "StringsContainsTests.MultiLine.matches",
                  "^abé$",
                  cudf_regex_operation::PREFIX_MATCH,
                  cudf_multiline,
                  multiline_inputs,
                  "101010");
  append_bit_rows(groups,
                  "StringsContainsTests.MultiLine.matches_default",
                  "^abé$",
                  cudf_regex_operation::PREFIX_MATCH,
                  {},
                  multiline_inputs,
                  "001010");
  append_scalar_group(groups,
                      "StringsContainsTests.MultiLine.count",
                      "^abé$",
                      cudf_regex_operation::COUNT,
                      cudf_multiline,
                      multiline_inputs,
                      {2, 1, 1, 0, 1, 1});
  append_scalar_group(groups,
                      "StringsContainsTests.MultiLine.count_default",
                      "^abé$",
                      cudf_regex_operation::COUNT,
                      {},
                      multiline_inputs,
                      {0, 0, 1, 0, 1, 0});

  std::vector<std::string> dot_inputs{"abc\nfa\nef", "fff\nabbc\nfff", "abcdef", ""};
  append_bit_rows(groups,
                  "StringsContainsTests.DotAll.contains",
                  "a.*f",
                  cudf_regex_operation::CONTAINS,
                  cudf_dot_all,
                  dot_inputs,
                  "1110");
  append_bit_rows(groups,
                  "StringsContainsTests.DotAll.contains_default",
                  "a.*f",
                  cudf_regex_operation::CONTAINS,
                  {},
                  dot_inputs,
                  "0010");
  append_bit_rows(groups,
                  "StringsContainsTests.DotAll.matches",
                  "a.*f",
                  cudf_regex_operation::PREFIX_MATCH,
                  cudf_dot_all,
                  dot_inputs,
                  "1010");
  append_bit_rows(groups,
                  "StringsContainsTests.DotAll.matches_default",
                  "a.*f",
                  cudf_regex_operation::PREFIX_MATCH,
                  {},
                  dot_inputs,
                  "0010");
  append_scalar_group(groups,
                      "StringsContainsTests.DotAll.count",
                      "a.*?f",
                      cudf_regex_operation::COUNT,
                      cudf_dot_all,
                      dot_inputs,
                      {2, 1, 1, 0});
  append_scalar_group(groups,
                      "StringsContainsTests.DotAll.count_default",
                      "a.*?f",
                      cudf_regex_operation::COUNT,
                      {},
                      dot_inputs,
                      {0, 0, 1, 0});

  std::vector<std::string> ascii_flag_inputs{"abc \t\f\r 12", "áé 　❽❽", "aZ ❽4", "XYZ　8"};
  for (std::string pattern : {std::string{R"REGEX(\w+[\s]+\d+)REGEX"},
                              std::string{R"REGEX([\w]+\s+[\d]+)REGEX"},
                              std::string{R"REGEX(\w+\s+\d+)REGEX"}}) {
    append_bit_rows(groups,
                    "StringsContainsTests.ASCII",
                    std::move(pattern),
                    cudf_regex_operation::CONTAINS,
                    {},
                    ascii_flag_inputs,
                    "1000");
  }

  std::vector<std::string> ignore_case_inputs{"abc", "ABC", "aBc", "123áéſ", "ÁÉS123"};
  append_bit_rows(groups,
                  "StringsContainsTests.IgnoreCase.literal",
                  "abc",
                  cudf_regex_operation::CONTAINS,
                  cudf_case_insensitive,
                  ignore_case_inputs,
                  "11100");
  append_bit_rows(groups,
                  "StringsContainsTests.IgnoreCase.class",
                  "[a-c]",
                  cudf_regex_operation::CONTAINS,
                  cudf_case_insensitive,
                  ignore_case_inputs,
                  "11100");

  std::string medium_pattern =
    R"REGEX(hello @abc @def world The quick brown @fox jumps over the lazy @dog hello http://www.world.com)REGEX";
  std::vector<std::string> medium_inputs{
    medium_pattern + " thats all",
    R"INPUT(12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890)INPUT",
    R"INPUT(abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz)INPUT"};
  append_bit_rows(groups,
                  "StringsContainsTests.MediumRegex.contains",
                  medium_pattern,
                  cudf_regex_operation::CONTAINS,
                  {},
                  medium_inputs,
                  "100");
  append_bit_rows(groups,
                  "StringsContainsTests.MediumRegex.matches",
                  medium_pattern,
                  cudf_regex_operation::PREFIX_MATCH,
                  {},
                  medium_inputs,
                  "100");
  append_scalar_group(groups,
                      "StringsContainsTests.MediumRegex.count",
                      medium_pattern,
                      cudf_regex_operation::COUNT,
                      {},
                      medium_inputs,
                      {1, 0, 0});

  std::string large_pattern = medium_pattern + " I'm here @home zzzz";
  std::vector<std::string> large_inputs{large_pattern, medium_inputs[1], medium_inputs[2]};
  append_bit_rows(groups,
                  "StringsContainsTests.LargeRegex.contains",
                  large_pattern,
                  cudf_regex_operation::CONTAINS,
                  {},
                  large_inputs,
                  "100");
  append_bit_rows(groups,
                  "StringsContainsTests.LargeRegex.matches",
                  large_pattern,
                  cudf_regex_operation::PREFIX_MATCH,
                  {},
                  large_inputs,
                  "100");
  append_scalar_group(groups,
                      "StringsContainsTests.LargeRegex.count",
                      large_pattern,
                      cudf_regex_operation::COUNT,
                      {},
                      large_inputs,
                      {1, 0, 0});

  std::string extra_large_pattern(320, '0');
  std::vector<std::string> extra_large_inputs{extra_large_pattern,
                                              extra_large_pattern,
                                              extra_large_pattern,
                                              extra_large_pattern,
                                              extra_large_pattern,
                                              "00"};
  append_bit_rows(groups,
                  "StringsContainsTests.ExtraLargeRegex.contains",
                  extra_large_pattern,
                  cudf_regex_operation::CONTAINS,
                  {},
                  extra_large_inputs,
                  "111110");
  append_bit_rows(groups,
                  "StringsContainsTests.ExtraLargeRegex.matches",
                  extra_large_pattern,
                  cudf_regex_operation::PREFIX_MATCH,
                  {},
                  extra_large_inputs,
                  "111110");
  append_scalar_group(groups,
                      "StringsContainsTests.ExtraLargeRegex.count",
                      extra_large_pattern,
                      cudf_regex_operation::COUNT,
                      {},
                      extra_large_inputs,
                      {1, 1, 1, 1, 1, 0});

  std::vector<std::string> nullable_inputs{
    "ah", "abch", "adeh", "afghh", "abcde", "a", "h", "", "xabchx"};
  append_bit_rows(groups,
                  "StringsContainsTests.AlternationNullableBranch.contains",
                  "a(bc|de|fg|)h",
                  cudf_regex_operation::CONTAINS,
                  {},
                  nullable_inputs,
                  "111100001");
  append_scalar_group(groups,
                      "StringsContainsTests.AlternationNullableBranch.count",
                      "a(bc|de|fg|)h",
                      cudf_regex_operation::COUNT,
                      {},
                      nullable_inputs,
                      {1, 1, 1, 1, 0, 0, 0, 0, 1});
  append_bit_rows(
    groups,
    "StringsContainsTests.BoundedRepetitionGap",
    "ab{0,4}cv",
    cudf_regex_operation::CONTAINS,
    {},
    {"acv", "abcv", "abbcv", "abbbcv", "abbbbcv", "abbbbbcv", "av", "acvx", "xacvx", ""},
    "1111100110");

  append_scalar_group(groups,
                      "StringsContainsTests.AlternationPriorityCount.a",
                      "a|aa",
                      cudf_regex_operation::COUNT,
                      {},
                      {"aaaa", "aaaaaa", "aaab", "a", "b", ""},
                      {4, 6, 3, 1, 0, 0});
  append_scalar_group(groups,
                      "StringsContainsTests.AlternationPriorityCount.foo",
                      "foo|foobar",
                      cudf_regex_operation::COUNT,
                      {},
                      {"foo", "foobar", "foofoo", "bar", ""},
                      {1, 1, 2, 0, 0});
  std::vector<std::string> lazy_inputs{"ab", "abc", "xdefx", "xghix", "jkl", "abc xdefx xghix jkl"};
  append_bit_rows(groups,
                  "StringsContainsTests.LazyQuantifiers.star",
                  "x.*?x",
                  cudf_regex_operation::CONTAINS,
                  {},
                  lazy_inputs,
                  "001101");
  append_bit_rows(groups,
                  "StringsContainsTests.LazyQuantifiers.plus",
                  "x.+?x",
                  cudf_regex_operation::CONTAINS,
                  {},
                  lazy_inputs,
                  "001101");

  // findall tests map directly to count for both the host and boolean CUDA shadow
  append_scalar_group(
    groups,
    "StringsFindallTests.FindallTest",
    R"REGEX(\d+-\w+)REGEX",
    cudf_regex_operation::COUNT,
    {},
    {"3-A", "4-May 5-Day 6-Hay", "12-Dec-2021-Jan", "Feb-March", "4 ABC", "", "25-9000-Hal"},
    {1, 3, 2, 0, 0, 0, 1});
  append_scalar_group(groups,
                      "StringsFindallTests.Multiline",
                      "^abc$",
                      cudf_regex_operation::COUNT,
                      cudf_multiline,
                      {"abc\nfff\nabc", "fff\nabc\nlll", "abc", "", "abc\n"},
                      {2, 1, 1, 0, 1});
  append_scalar_group(groups,
                      "StringsFindallTests.DotAll",
                      "b.*f",
                      cudf_regex_operation::COUNT,
                      cudf_dot_all,
                      {"abc\nfa\nef", "fff\nabbc\nfff", "abcdéf", ""},
                      {1, 1, 1, 0});
  append_scalar_group(groups,
                      "StringsFindallTests.MediumRegex",
                      R"REGEX(\w+ \w+ \d+)REGEX",
                      cudf_regex_operation::COUNT,
                      {},
                      {"first words 1234 and just numbers 9876", "neither"},
                      {2, 0});
  append_scalar_group(groups,
                      "StringsFindallTests.LargeRegex",
                      large_pattern,
                      cudf_regex_operation::COUNT,
                      {},
                      large_inputs,
                      {1, 0, 0});
  append_scalar_group(groups,
                      "StringsFindallTests.NoMatches",
                      "^zzz$",
                      cudf_regex_operation::COUNT,
                      {},
                      {"abc\nfff\nabc", "fff\nabc\nlll", "abc", "", "abc\n"},
                      {0, 0, 0, 0, 0});

  // extraction and replacement groups use contains as the CUDA match-existence shadow
  append_bit_rows(groups,
                  "StringsExtractTests.ExtractTest",
                  R"REGEX((\w+) (\w+))REGEX",
                  cudf_regex_operation::CONTAINS,
                  {},
                  {"First Last", "Joe Schmoe", "John Smith", "Jane Smith", "Beyonce", "Sting", ""},
                  "1111000");
  append_bit_rows(groups,
                  "StringsExtractTests.ExtractDomainTest",
                  R"REGEX(([\w]+[\.].*[^/]|[\-\w]+[\.].*[^/]))REGEX",
                  cudf_regex_operation::CONTAINS,
                  {},
                  {"http://www.google.com",
                   "gmail.com",
                   "github.com",
                   "https://pandas.pydata.org",
                   "http://www.worldbank.org.kg/",
                   "waiterrant.blogspot.com",
                   "http://forums.news.cnn.com.ac/",
                   "ftp://b.cnn.com/",
                   "a.news.uk",
                   "a.news.co.uk",
                   "https://a.news.co.uk",
                   "107-193-100-2.lightspeed.cicril.sbcglobal.net",
                   "a23-44-13-2.deploy.static.akamaitechnologies.com"},
                  "1111111111111");
  append_bit_rows(groups,
                  "StringsReplaceRegexTest.ReplaceRegexTest",
                  R"REGEX((\bthe\b))REGEX",
                  cudf_regex_operation::CONTAINS,
                  {},
                  {"the quick brown fox jumps over the lazy dog",
                   "the fat cat lays next to the other accénted cat",
                   "a slow moving turtlé cannot catch the bird",
                   "which can be composéd together to form a more complete",
                   "thé result does not include the value in the sum in",
                   ""},
                  "111010");
  append_bit_rows(groups,
                  "StringsReplaceRegexTest.Alternation",
                  R"REGEX((^|\s)\d+(\s|$))REGEX",
                  cudf_regex_operation::CONTAINS,
                  {},
                  {"16  6  brr  232323  1  hello  90", "123 ABC 00 2022", "abé123  4567  89xyz"},
                  "111");
  append_bit_rows(groups,
                  "StringsReplaceRegexTest.ZeroLengthMatch",
                  "D*",
                  cudf_regex_operation::CONTAINS,
                  {},
                  {"DD", "zéz", "DsDs", ""},
                  "1111");
}

// structured cuDF case tables

void append_contains_cases(std::vector<cudf_regex_case>& cases)
{
  auto defaults    = cudf_default;
  auto multiline   = cudf_multiline;
  auto extended    = cudf_extended;
  auto extended_ml = cudf_extended_multiline;

  append_case(cases,
              "StringsContainsTests.ContainsTest",
              "null-mask",
              ".*",
              cudf_regex_operation::CONTAINS,
              defaults,
              {std::nullopt},
              {0});
  append_case(cases,
              "StringsContainsTests.MatchesTest",
              "null-mask",
              ".*",
              cudf_regex_operation::PREFIX_MATCH,
              defaults,
              {std::nullopt},
              {0});
  append_case(cases,
              "StringsContainsTests.CountTest",
              "null-mask",
              R"REGEX(\d+)REGEX",
              cudf_regex_operation::COUNT,
              defaults,
              {std::nullopt},
              {0});

  std::vector<maybe_string> octal_inputs{"A3", "B", "CDA3EY", "", "99", "\a\t\r"};
  for (std::string pattern : {std::string{R"REGEX(\101)REGEX"},
                              std::string{R"REGEX(\1013)REGEX"},
                              std::string{R"REGEX(D*\101\063)REGEX"}}) {
    append_case(cases,
                "StringsContainsTests.OctalTest",
                pattern,
                std::move(pattern),
                cudf_regex_operation::CONTAINS,
                defaults,
                octal_inputs,
                {1, 0, 1, 0, 0, 0});
  }
  append_case(cases,
              "StringsContainsTests.OctalTest",
              "two-digit-octal-followed-by-nine",
              R"REGEX(\0719)REGEX",
              cudf_regex_operation::CONTAINS,
              defaults,
              octal_inputs,
              {0, 0, 0, 0, 1, 0});
  append_case(cases,
              "StringsContainsTests.OctalTest",
              "octal-in-classes",
              R"REGEX([\007][\011][\015])REGEX",
              cudf_regex_operation::CONTAINS,
              defaults,
              octal_inputs,
              {0, 0, 0, 0, 0, 1});

  std::vector<maybe_string> nul_inputs;
  for (char first = 'A'; first <= 'J'; ++first) {
    std::string value;
    value.push_back(first);
    value.push_back('\0');
    value.push_back('B');
    nul_inputs.push_back(std::move(value));
  }
  append_case(cases,
              "StringsContainsTests.EmbeddedNullCharacter",
              "octal-null-literal",
              R"REGEX(J\000B)REGEX",
              cudf_regex_operation::CONTAINS,
              defaults,
              nul_inputs,
              {0, 0, 0, 0, 0, 0, 0, 0, 0, 1});
  append_case(cases,
              "StringsContainsTests.EmbeddedNullCharacter",
              "octal-null-class",
              R"REGEX([G-J][\000]B)REGEX",
              cudf_regex_operation::CONTAINS,
              defaults,
              nul_inputs,
              {0, 0, 0, 0, 0, 0, 1, 1, 1, 1});

  append_case(cases,
              "StringsContainsTests.FixedQuantifier",
              "malformed-count-is-literal",
              "aaaa{n,m}",
              cudf_regex_operation::COUNT,
              defaults,
              {"a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa"},
              {0, 0, 0, 0, 0, 0});

  auto next_line           = std::string{"\xC2\x85"};
  auto line_separator      = std::string{"\xE2\x80\xA8"};
  auto paragraph_separator = std::string{"\xE2\x80\xA9"};
  std::vector<maybe_string> special_inputs{
    std::string{"zzé"} + line_separator + "qqq" + next_line + "zzé",
    std::string{"qqq\rzzé"} + line_separator + "lll",
    "zzé",
    "",
    std::string{"zzé"} + paragraph_separator,
    std::string{"abc\nzzé"} + next_line};
  append_case(cases,
              "StringsContainsTests.SpecialNewLines",
              "contains-extended",
              "^zzé$",
              cudf_regex_operation::CONTAINS,
              extended,
              special_inputs,
              {0, 0, 1, 0, 1, 0});
  append_case(cases,
              "StringsContainsTests.SpecialNewLines",
              "contains-extended-multiline",
              "^zzé$",
              cudf_regex_operation::CONTAINS,
              extended_ml,
              special_inputs,
              {1, 1, 1, 0, 1, 1});
  append_case(cases,
              "StringsContainsTests.SpecialNewLines",
              "matches-extended",
              "^zzé$",
              cudf_regex_operation::PREFIX_MATCH,
              extended,
              special_inputs,
              {0, 0, 1, 0, 1, 0});
  append_case(cases,
              "StringsContainsTests.SpecialNewLines",
              "matches-extended-multiline",
              "^zzé$",
              cudf_regex_operation::PREFIX_MATCH,
              extended_ml,
              special_inputs,
              {1, 0, 1, 0, 1, 0});
  append_case(cases,
              "StringsContainsTests.SpecialNewLines",
              "count-extended",
              "^zzé$",
              cudf_regex_operation::COUNT,
              extended,
              special_inputs,
              {0, 0, 1, 0, 1, 0});
  append_case(cases,
              "StringsContainsTests.SpecialNewLines",
              "count-extended-multiline",
              "^zzé$",
              cudf_regex_operation::COUNT,
              extended_ml,
              special_inputs,
              {2, 1, 1, 0, 1, 1});
  append_case(cases,
              "StringsContainsTests.SpecialNewLines",
              "dot-default",
              "q.*l",
              cudf_regex_operation::CONTAINS,
              defaults,
              special_inputs,
              {0, 1, 0, 0, 0, 0});
  append_case(cases,
              "StringsContainsTests.SpecialNewLines",
              "dot-extended",
              "q.*l",
              cudf_regex_operation::CONTAINS,
              extended,
              special_inputs,
              {0, 0, 0, 0, 0, 0});
  append_case(cases,
              "StringsContainsTests.SpecialNewLines",
              "dot-extended-dotall",
              "q.*l",
              cudf_regex_operation::CONTAINS,
              cudf_extended_dot_all,
              special_inputs,
              {0, 1, 0, 0, 0, 0});

  std::vector<maybe_string> line_inputs{
    "abé\nfff\nabé", "fff\nabé\nlll", "abé", "", "abé\n", "abe\nabé\n"};
  for (cudf_regex_operation operation : {cudf_regex_operation::CONTAINS,
                                         cudf_regex_operation::PREFIX_MATCH,
                                         cudf_regex_operation::COUNT}) {
    append_case(cases,
                "StringsContainsTests.EndOfString",
                "strict-anchors-default-" + std::to_string(static_cast<int>(operation)),
                R"REGEX(\Aabé\Z)REGEX",
                operation,
                defaults,
                line_inputs,
                {0, 0, 1, 0, 0, 0});
    append_case(cases,
                "StringsContainsTests.EndOfString",
                "strict-anchors-multiline-" + std::to_string(static_cast<int>(operation)),
                R"REGEX(\Aabé\Z)REGEX",
                operation,
                multiline,
                line_inputs,
                {0, 0, 1, 0, 0, 0});
  }

  append_case(cases,
              "StringsContainsTests.DotAll",
              "count-dotall-and-multiline",
              "a.*?f",
              cudf_regex_operation::COUNT,
              cudf_multiline_dot_all,
              {"abc\nfa\nef", "fff\nabbc\nfff", "abcdef", ""},
              {2, 1, 1, 0});

  std::vector<maybe_string> ascii_inputs{"abc \t\f\r 12", "áé 　❽❽", "aZ ❽4", "XYZ　8"};
  for (std::string pattern : {std::string{R"REGEX(\w+[\s]+\d+)REGEX"},
                              std::string{R"REGEX([^\W]+\s+[^\D]+)REGEX"},
                              std::string{R"REGEX([\w]+[^\S]+[\d]+)REGEX"},
                              std::string{R"REGEX([\w]+\s+[\d]+)REGEX"},
                              std::string{R"REGEX(\w+\s+\d+)REGEX"}}) {
    append_case(cases,
                "StringsContainsTests.ASCII",
                "ascii-" + pattern,
                pattern,
                cudf_regex_operation::CONTAINS,
                cudf_ascii,
                ascii_inputs,
                {1, 0, 0, 0});
    append_case(cases,
                "StringsContainsTests.ASCII",
                "unicode-" + pattern,
                pattern,
                cudf_regex_operation::CONTAINS,
                defaults,
                ascii_inputs,
                {1, 1, 1, 1});
  }

  std::vector<maybe_string> case_inputs{"abc", "ABC", "aBc", "123áéſ", "ÁÉS123"};
  append_case(cases,
              "StringsContainsTests.IgnoreCase",
              "unicode-literal",
              "áéſ",
              cudf_regex_operation::CONTAINS,
              cudf_case_insensitive,
              case_inputs,
              {0, 0, 0, 1, 1});
  append_case(cases,
              "StringsContainsTests.IgnoreCase",
              "unicode-class",
              "[á-é]",
              cudf_regex_operation::CONTAINS,
              cudf_case_insensitive,
              case_inputs,
              {0, 0, 0, 1, 1});

  std::vector<maybe_string> crlf_inputs{"abc\r\n",
                                        "abc\n",
                                        "abc\r",
                                        "abc",
                                        "a\r\nb",
                                        "abc\r\n\r\n",
                                        "",
                                        std::string{"abc"} + next_line,
                                        "a\nb\r\nc",
                                        "\r\n",
                                        "\r\nabc",
                                        "x\n\r",
                                        "a\r\rb",
                                        "a\n\nb"};
  append_case(cases,
              "StringsContainsTests.CrlfLineAnchorExtNewline",
              "contains",
              "^abc$",
              cudf_regex_operation::CONTAINS,
              extended,
              std::vector<maybe_string>(crlf_inputs.begin(), crlf_inputs.begin() + 8),
              {1, 1, 1, 1, 0, 0, 0, 1});
  append_case(cases,
              "StringsContainsTests.CrlfLineAnchorExtNewline",
              "contains-multiline",
              "^abc$",
              cudf_regex_operation::CONTAINS,
              extended_ml,
              std::vector<maybe_string>(crlf_inputs.begin(), crlf_inputs.begin() + 8),
              {1, 1, 1, 1, 0, 1, 0, 1});
  append_case(cases,
              "StringsContainsTests.CrlfBolAnchorExtNewline",
              "begin-line-never-between-crlf",
              "^\n",
              cudf_regex_operation::CONTAINS,
              extended_ml,
              {"abc\r\nDEF", "a\r\nb", "ab\rc", "x\ny"},
              {0, 0, 0, 0});

  struct crlf_scalar_assertion {
    std::string pattern                = "";
    cudf_regex_operation operation     = cudf_regex_operation::CONTAINS;
    compile_options options            = compile_options{};
    std::vector<std::int64_t> expected = std::vector<std::int64_t>{};
  };
  std::vector<crlf_scalar_assertion> edge_assertions{
    {"^abc$", cudf_regex_operation::CONTAINS, extended, {1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0}},
    {"^abc$",
     cudf_regex_operation::CONTAINS,
     extended_ml,
     {1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0}},
    {"abc$",
     cudf_regex_operation::PREFIX_MATCH,
     extended,
     {1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0}},
    {"[a-z]+$", cudf_regex_operation::COUNT, extended, {1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1}},
    {"[a-z]+$",
     cudf_regex_operation::COUNT,
     extended_ml,
     {1, 1, 1, 1, 2, 1, 0, 1, 3, 0, 1, 1, 2, 2}},
    {"^[a-z]+",
     cudf_regex_operation::COUNT,
     extended_ml,
     {1, 1, 1, 1, 2, 1, 0, 1, 3, 0, 1, 1, 2, 2}},
    {R"REGEX(\r$)REGEX",
     cudf_regex_operation::CONTAINS,
     extended_ml,
     {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0}},
    {"^\n",
     cudf_regex_operation::CONTAINS,
     extended_ml,
     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}},
    {"(a$|b)",
     cudf_regex_operation::CONTAINS,
     extended,
     {1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1}},
  };
  for (std::size_t index = 0; index < edge_assertions.size(); ++index) {
    auto& item = edge_assertions[index];
    append_case(cases,
                "StringsContainsTests.CrlfEdgeCasesExtNewline",
                "column-" + std::to_string(index),
                item.pattern,
                item.operation,
                item.options,
                crlf_inputs,
                item.expected);
  }
  append_case(cases,
              "StringsContainsTests.CrlfDefaultLfOnlyNoExtNewline",
              "default-newline-set",
              "^abc$",
              cudf_regex_operation::CONTAINS,
              defaults,
              crlf_inputs,
              {0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

  append_case(cases,
              "StringsContainsTests.AlternationNullableBranch",
              "standard-abcdefgh-row",
              "a(bc|de|fg|)h",
              cudf_regex_operation::CONTAINS,
              defaults,
              {"abcdefgh"},
              {0});
  append_case(cases,
              "StringsContainsTests.AlternationNullableBranch",
              "standard-abcdefgh-count-row",
              "a(bc|de|fg|)h",
              cudf_regex_operation::COUNT,
              defaults,
              {"abcdefgh"},
              {0});

  std::vector<maybe_string> dot_newline_inputs{"axb",
                                               "a\nb",
                                               "a\rb",
                                               std::string{"a"} + next_line + "b",
                                               std::string{"a"} + line_separator + "b",
                                               std::string{"a"} + paragraph_separator + "b",
                                               "abc",
                                               ""};
  append_case(cases,
              "StringsContainsTests.ExtNewlineDotAny",
              "default-dot",
              "a.b",
              cudf_regex_operation::CONTAINS,
              defaults,
              dot_newline_inputs,
              {1, 0, 1, 1, 1, 1, 0, 0});
  append_case(cases,
              "StringsContainsTests.ExtNewlineDotAny",
              "extended-dot",
              "a.b",
              cudf_regex_operation::CONTAINS,
              extended,
              dot_newline_inputs,
              {1, 0, 0, 0, 0, 0, 0, 0});
  append_case(
    cases,
    "StringsContainsTests.ExtNewlineDotAny",
    "extended-only-newlines",
    ".+",
    cudf_regex_operation::CONTAINS,
    extended,
    {"hello", next_line + line_separator + paragraph_separator, "a" + next_line + "b", ""},
    {1, 0, 1, 0});
  append_case(cases,
              "StringsContainsTests.ExtNewlineDotAny",
              "line-separator-literal-regression",
              R"REGEX(\u2028)REGEX",
              cudf_regex_operation::CONTAINS,
              defaults,
              {line_separator, paragraph_separator},
              {1, 0});
  append_case(cases,
              "StringsContainsTests.ExtNewlineDotAny",
              "paragraph-separator-literal-regression",
              R"REGEX(\u2029)REGEX",
              cudf_regex_operation::CONTAINS,
              defaults,
              {line_separator, paragraph_separator},
              {0, 1});
  append_case(cases,
              "StringsContainsTests.ExtNewlineDotAny",
              "paragraph-end-assertion-regression",
              "^zzé$",
              cudf_regex_operation::CONTAINS,
              extended,
              {std::string{"zzé"} + paragraph_separator},
              {1});
}

void append_findall_cases(std::vector<cudf_regex_case>& cases)
{
  auto defaults    = cudf_default;
  auto multiline   = cudf_multiline;
  auto dot_all     = cudf_dot_all;
  auto extended    = cudf_extended;
  auto extended_ml = cudf_extended_multiline;

  append_case(cases,
              {.test_name = "StringsFindallTests.FindallTest",
               .assertion = "matched-strings-and-null-mask",
               .pattern   = R"REGEX(\d+-\w+)REGEX",
               .options   = defaults,
               .operation = cudf_regex_operation::FIND_ALL,
               .inputs    = {"3-A",
                             "4-May 5-Day 6-Hay",
                             "12-Dec-2021-Jan",
                             "Feb-March",
                             "4 ABC",
                             std::nullopt,
                             "",
                             "25-9000-Hal"},
               .expected  = {strings_row({"3-A"}),
                             strings_row({"4-May", "5-Day", "6-Hay"}),
                             strings_row({"12-Dec", "2021-Jan"}),
                             strings_row({}),
                             strings_row({}),
                             null_row(),
                             strings_row({}),
                             strings_row({"25-9000"})}});

  append_case(cases,
              {.test_name = "StringsFindallTests.Multiline",
               .assertion = "matched-lines",
               .pattern   = "^abc$",
               .options   = multiline,
               .operation = cudf_regex_operation::FIND_ALL,
               .inputs    = {"abc\nfff\nabc", "fff\nabc\nlll", "abc", "", "abc\n"},
               .expected  = {strings_row({"abc", "abc"}),
                             strings_row({"abc"}),
                             strings_row({"abc"}),
                             strings_row({}),
                             strings_row({"abc"})}});

  append_case(cases,
              {.test_name = "StringsFindallTests.DotAll",
               .assertion = "greedy-spans",
               .pattern   = "b.*f",
               .options   = dot_all,
               .operation = cudf_regex_operation::FIND_ALL,
               .inputs    = {"abc\nfa\nef", "fff\nabbc\nfff", "abcdéf", ""},
               .expected  = {strings_row({"bc\nfa\nef"}),
                             strings_row({"bbc\nfff"}),
                             strings_row({"bcdéf"}),
                             strings_row({})}});

  auto next_line           = std::string{"\xC2\x85"};
  auto line_separator      = std::string{"\xE2\x80\xA8"};
  auto paragraph_separator = std::string{"\xE2\x80\xA9"};
  std::vector<maybe_string> special_inputs{std::string{"zzé"} + paragraph_separator + "qqq\nzzé",
                                           std::string{"qqq\nzzé"} + paragraph_separator + "lll",
                                           "zzé",
                                           "",
                                           "zzé\r",
                                           std::string{"zzé"} + line_separator + "zzé" + next_line};
  append_case(cases,
              {.test_name = "StringsFindallTests.SpecialNewLines",
               .assertion = "extended",
               .pattern   = "^zzé$",
               .options   = extended,
               .operation = cudf_regex_operation::FIND_ALL,
               .inputs    = special_inputs,
               .expected  = {strings_row({}),
                             strings_row({}),
                             strings_row({"zzé"}),
                             strings_row({}),
                             strings_row({"zzé"}),
                             strings_row({})}});
  append_case(cases,
              {.test_name = "StringsFindallTests.SpecialNewLines",
               .assertion = "extended-multiline",
               .pattern   = "^zzé$",
               .options   = extended_ml,
               .operation = cudf_regex_operation::FIND_ALL,
               .inputs    = special_inputs,
               .expected  = {strings_row({"zzé", "zzé"}),
                             strings_row({"zzé"}),
                             strings_row({"zzé"}),
                             strings_row({}),
                             strings_row({"zzé"}),
                             strings_row({"zzé", "zzé"})}});

  append_case(
    cases,
    {.test_name = "StringsFindallTests.MediumRegex",
     .assertion = "matched-strings",
     .pattern   = R"REGEX(\w+ \w+ \d+)REGEX",
     .options   = defaults,
     .operation = cudf_regex_operation::FIND_ALL,
     .inputs    = {"first words 1234 and just numbers 9876", "neither"},
     .expected  = {strings_row({"first words 1234", "just numbers 9876"}), strings_row({})}});

  auto large_pattern = std::string{
    R"REGEX(hello @abc @def world The quick brown @fox jumps over the lazy @dog hello http://www.world.com I'm here @home zzzz)REGEX"};
  append_case(
    cases,
    {.test_name = "StringsFindallTests.LargeRegex",
     .assertion = "matched-strings",
     .pattern   = large_pattern,
     .options   = defaults,
     .operation = cudf_regex_operation::FIND_ALL,
     .inputs =
       {large_pattern,
        R"INPUT(12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890)INPUT",
        R"INPUT(abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz)INPUT"},
     .expected = {strings_row({large_pattern}), strings_row({}), strings_row({})}});

  append_case(
    cases,
    {.test_name = "StringsFindallTests.NoMatches",
     .assertion = "empty-lists",
     .pattern   = "^zzz$",
     .options   = defaults,
     .operation = cudf_regex_operation::FIND_ALL,
     .inputs    = {"abc\nfff\nabc", "fff\nabc\nlll", "abc", "", "abc\n"},
     .expected  = {
       strings_row({}), strings_row({}), strings_row({}), strings_row({}), strings_row({})}});

  append_case(cases,
              "StringsFindallTests.FindTest",
              "offsets-and-null-mask",
              R"REGEX(\d+)REGEX",
              cudf_regex_operation::FIND,
              defaults,
              {"3A", "May4", "Jan2021", "March", "A9BC", std::nullopt, "", "abcdef ghijklm 12345"},
              {0, 3, 3, -1, 1, 0, -1, 15});

  append_case(cases,
              {.test_name        = "StringsFindallTests.EmptyTest",
               .assertion        = "findall-empty-column-shape",
               .pattern          = R"REGEX(\w+)REGEX",
               .options          = defaults,
               .operation        = cudf_regex_operation::FIND_ALL,
               .expected_columns = 1});
  append_case(cases,
              {.test_name        = "StringsFindallTests.EmptyTest",
               .assertion        = "find-empty-column-shape",
               .pattern          = R"REGEX(\w+)REGEX",
               .options          = defaults,
               .operation        = cudf_regex_operation::FIND,
               .expected_columns = 1});

  std::vector<maybe_string> capture_inputs{"3-A",
                                           "4-May 5-Day 6-Hay",
                                           "12-Dec-2021-Jan",
                                           "Feb-March",
                                           "4 ABC",
                                           std::nullopt,
                                           "",
                                           "25-9000-Hal"};
  append_case(cases,
              {.test_name = "StringsFindallTests.OneCaptureGroup",
               .assertion = "capture-values",
               .pattern   = R"REGEX((\d+)-\w+)REGEX",
               .options   = defaults,
               .operation = cudf_regex_operation::FIND_ALL,
               .inputs    = capture_inputs,
               .expected  = {strings_row({"3"}),
                             strings_row({"4", "5", "6"}),
                             strings_row({"12", "2021"}),
                             strings_row({}),
                             strings_row({}),
                             null_row(),
                             strings_row({}),
                             strings_row({"25"})}});
  append_case(cases,
              {.test_name    = "StringsFindallTests.OneCaptureGroup",
               .assertion    = "non-capture-values",
               .pattern      = R"REGEX((\d+)-\w+)REGEX",
               .options      = defaults,
               .operation    = cudf_regex_operation::FIND_ALL,
               .capture_mode = cudf_capture_mode::NON_CAPTURE,
               .inputs       = capture_inputs,
               .expected     = {strings_row({"3-A"}),
                                strings_row({"4-May", "5-Day", "6-Hay"}),
                                strings_row({"12-Dec", "2021-Jan"}),
                                strings_row({}),
                                strings_row({}),
                                null_row(),
                                strings_row({}),
                                strings_row({"25-9000"})}});

  append_case(cases,
              {.test_name    = "StringsFindallTests.AlternationPriorityFirstWins",
               .assertion    = "first-alternative",
               .pattern      = "foo|foobar",
               .options      = defaults,
               .operation    = cudf_regex_operation::FIND_ALL,
               .capture_mode = cudf_capture_mode::NON_CAPTURE,
               .inputs       = {"foo", "foobar", "foobarbaz", "bar", "xfoobar", ""},
               .expected     = {strings_row({"foo"}),
                                strings_row({"foo"}),
                                strings_row({"foo"}),
                                strings_row({}),
                                strings_row({"foo"}),
                                strings_row({})}});

  append_case(cases,
              {.test_name = "StringsFindallTests.EmptyMatch",
               .assertion = "empty-line-match-is-not-returned",
               .pattern   = "^$",
               .options   = multiline,
               .operation = cudf_regex_operation::FIND_ALL,
               .inputs    = {" ", "hello world", "é\r\ny"},
               .expected  = {strings_row({}), strings_row({}), strings_row({})}});
  for (std::string pattern : {std::string{R"REGEX(\b)REGEX"}, std::string{R"REGEX((\b))REGEX"}}) {
    append_case(cases,
                {.test_name = "StringsFindallTests.EmptyMatch",
                 .assertion = pattern,
                 .pattern   = std::move(pattern),
                 .options   = defaults,
                 .operation = cudf_regex_operation::FIND_ALL,
                 .inputs    = {" ", "hello world", "é\r\ny"},
                 .expected  = {
                   strings_row({}), strings_row({"", "", "", ""}), strings_row({"", "", "", ""})}});
  }

  append_case(cases,
              {.test_name = "StringsFindallTests.Errors",
               .assertion = "multiple-captures-rejected",
               .pattern   = R"REGEX((\d+)-(\w+))REGEX",
               .options   = defaults,
               .operation = cudf_regex_operation::FIND_ALL,
               .inputs    = {"1 One", "2 Two", "3 Three 4 Four", ""},
               .expected  = {scalar_row(0), scalar_row(0), scalar_row(0), scalar_row(0)},
               .expect_operation_error = true});
}

void append_extract_cases(std::vector<cudf_regex_case>& cases)
{
  auto defaults       = cudf_default;
  auto multiline      = cudf_multiline;
  auto dot_all        = cudf_dot_all;
  auto extended       = cudf_extended;
  auto extended_ml    = cudf_extended_multiline;
  auto append_extract = [&](std::string test_name,
                            std::string assertion,
                            std::string pattern,
                            compile_options options,
                            std::vector<maybe_string> inputs,
                            std::vector<cudf_expected_row> expected) {
    append_case(cases,
                {.test_name = std::move(test_name),
                 .assertion = std::move(assertion),
                 .pattern   = std::move(pattern),
                 .options   = options,
                 .operation = cudf_regex_operation::EXTRACT,
                 .inputs    = std::move(inputs),
                 .expected  = std::move(expected)});
  };

  append_extract(
    "StringsExtractTests.ExtractTest",
    "captures-and-null-mask",
    R"REGEX((\w+) (\w+))REGEX",
    defaults,
    {"First Last", "Joe Schmoe", "John Smith", "Jane Smith", "Beyonce", "Sting", std::nullopt, ""},
    {strings_row({"First", "Last"}),
     strings_row({"Joe", "Schmoe"}),
     strings_row({"John", "Smith"}),
     strings_row({"Jane", "Smith"}),
     strings_row({std::nullopt, std::nullopt}),
     strings_row({std::nullopt, std::nullopt}),
     null_row(),
     strings_row({std::nullopt, std::nullopt})});

  std::vector<maybe_string> domains{"http://www.google.com",
                                    "gmail.com",
                                    "github.com",
                                    "https://pandas.pydata.org",
                                    "http://www.worldbank.org.kg/",
                                    "waiterrant.blogspot.com",
                                    "http://forums.news.cnn.com.ac/",
                                    "http://forums.news.cnn.com.ac/",
                                    "ftp://b.cnn.com/",
                                    "a.news.uk",
                                    "a.news.co.uk",
                                    "https://a.news.co.uk",
                                    "107-193-100-2.lightspeed.cicril.sbcglobal.net",
                                    "a23-44-13-2.deploy.static.akamaitechnologies.com"};
  std::vector<std::string> domain_values{"www.google.com",
                                         "gmail.com",
                                         "github.com",
                                         "pandas.pydata.org",
                                         "www.worldbank.org.kg",
                                         "waiterrant.blogspot.com",
                                         "forums.news.cnn.com.ac",
                                         "forums.news.cnn.com.ac",
                                         "b.cnn.com",
                                         "a.news.uk",
                                         "a.news.co.uk",
                                         "a.news.co.uk",
                                         "107-193-100-2.lightspeed.cicril.sbcglobal.net",
                                         "a23-44-13-2.deploy.static.akamaitechnologies.com"};
  std::vector<cudf_expected_row> domain_expected;
  for (std::string const& value : domain_values)
    domain_expected.push_back(strings_row({value}));
  append_extract("StringsExtractTests.ExtractDomainTest",
                 "domain-capture",
                 R"REGEX(([\w]+[\.].*[^/]|[\-\w]+[\.].*[^/]))REGEX",
                 defaults,
                 std::move(domains),
                 std::move(domain_expected));

  auto event = std::string{
    R"EVENT(15162388.26, search_name="Test Search Name", orig_time="1516238826", info_max_time="1566346500.000000000", info_min_time="1566345300.000000000", info_search_time="1566305689.361160000", message.description="Test Message Description", message.hostname="msg.test.hostname", message.ip="100.100.100.123", message.user_name="user@test.com", severity="info", urgency="medium"')EVENT"};
  auto event_patterns = std::to_array<std::pair<std::string, std::string>>({
    {R"REGEX((^[0-9]+\.?[0-9]*),)REGEX", "15162388.26"},
    {R"REGEX(search_name="([0-9A-Za-z\s\-\(\)]+))REGEX", "Test Search Name"},
    {R"REGEX(message.ip="([\w\.]+))REGEX", "100.100.100.123"},
    {R"REGEX(message.hostname="([\w\.]+))REGEX", "msg.test.hostname"},
    {R"REGEX(message.user_name="([\w\.\@]+))REGEX", "user@test.com"},
    {R"REGEX(message\.description="([\w\.\s]+))REGEX", "Test Message Description"},
  });
  for (std::size_t index = 0; index < event_patterns.size(); ++index) {
    append_extract("StringsExtractTests.ExtractEventTest",
                   "pattern-" + std::to_string(index),
                   event_patterns[index].first,
                   defaults,
                   {event},
                   {strings_row({event_patterns[index].second})});
  }

  std::vector<maybe_string> line_inputs{
    "abc\nfff\nabc", "fff\nabc\nlll", "abc", "", "abc\n", "abé\nabc\n"};
  append_extract("StringsExtractTests.MultiLine",
                 "multiline",
                 "(^[a-c]+$)",
                 multiline,
                 line_inputs,
                 {strings_row({"abc"}),
                  strings_row({"abc"}),
                  strings_row({"abc"}),
                  strings_row({std::nullopt}),
                  strings_row({"abc"}),
                  strings_row({"abc"})});
  append_extract("StringsExtractTests.MultiLine",
                 "default",
                 "^([a-c]+)$",
                 defaults,
                 line_inputs,
                 {strings_row({std::nullopt}),
                  strings_row({std::nullopt}),
                  strings_row({"abc"}),
                  strings_row({std::nullopt}),
                  strings_row({"abc"}),
                  strings_row({std::nullopt})});

  std::vector<maybe_string> dot_inputs{"abc\nfa\nef", "fff\nabbc\nfff", "abcdef", ""};
  append_extract("StringsExtractTests.DotAll",
                 "dotall",
                 "(a.*f)",
                 dot_all,
                 dot_inputs,
                 {strings_row({"abc\nfa\nef"}),
                  strings_row({"abbc\nfff"}),
                  strings_row({"abcdef"}),
                  strings_row({std::nullopt})});
  append_extract("StringsExtractTests.DotAll",
                 "default",
                 "(a.*f)",
                 defaults,
                 dot_inputs,
                 {strings_row({std::nullopt}),
                  strings_row({std::nullopt}),
                  strings_row({"abcdef"}),
                  strings_row({std::nullopt})});

  auto next_line           = std::string{"\xC2\x85"};
  auto line_separator      = std::string{"\xE2\x80\xA8"};
  auto paragraph_separator = std::string{"\xE2\x80\xA9"};
  std::vector<maybe_string> special_inputs{
    std::string{"zzé"} + next_line + "qqq" + line_separator + "zzé",
    std::string{"qqq"} + line_separator + "zzé\rlll",
    "zzé",
    "",
    std::string{"zzé"} + next_line,
    std::string{"abc"} + paragraph_separator + "zzé\n"};
  append_extract("StringsExtractTests.SpecialNewLines",
                 "extended",
                 "(^zzé$)",
                 extended,
                 special_inputs,
                 {strings_row({std::nullopt}),
                  strings_row({std::nullopt}),
                  strings_row({"zzé"}),
                  strings_row({std::nullopt}),
                  strings_row({"zzé"}),
                  strings_row({std::nullopt})});
  append_extract("StringsExtractTests.SpecialNewLines",
                 "extended-multiline",
                 "^(zzé)$",
                 extended_ml,
                 special_inputs,
                 {strings_row({"zzé"}),
                  strings_row({"zzé"}),
                  strings_row({"zzé"}),
                  strings_row({std::nullopt}),
                  strings_row({"zzé"}),
                  strings_row({"zzé"})});
  append_extract("StringsExtractTests.SpecialNewLines",
                 "default-dot",
                 "q(q.*l)l",
                 defaults,
                 special_inputs,
                 {strings_row({std::nullopt}),
                  strings_row({std::string{"qq"} + line_separator + "zzé\rll"}),
                  strings_row({std::nullopt}),
                  strings_row({std::nullopt}),
                  strings_row({std::nullopt}),
                  strings_row({std::nullopt})});
  append_extract("StringsExtractTests.SpecialNewLines",
                 "extended-dot",
                 "q(q.*l)l",
                 extended,
                 special_inputs,
                 std::vector<cudf_expected_row>(6, strings_row({std::nullopt})));

  append_extract("StringsExtractTests.NestedQuantifier",
                 "last-capture",
                 R"REGEX((\d{4}\s){4})REGEX",
                 defaults,
                 {"TEST12 1111 2222 3333 4444 5555",
                  "0000 AAAA 9999 BBBB 8888",
                  "7777 6666 4444 3333",
                  "12345 3333 4444 1111 ABCD"},
                 {strings_row({"4444 "}),
                  strings_row({std::nullopt}),
                  strings_row({std::nullopt}),
                  strings_row({"1111 "})});

  append_extract("StringsExtractTests.EmptyExtractTest",
                 "empty-capture-and-null-mask",
                 R"REGEX(([^_]*)\Z)REGEX",
                 defaults,
                 {std::nullopt, "AAA", "AAA_A", "AAA_AAA_", "A__", ""},
                 {null_row(),
                  strings_row({"AAA"}),
                  strings_row({"A"}),
                  strings_row({""}),
                  strings_row({""}),
                  strings_row({""})});

  append_case(
    cases,
    {.test_name = "StringsExtractTests.ExtractAllTest",
     .assertion = "flattened-captures-and-validity",
     .pattern   = R"REGEX((\d+) (\w+))REGEX",
     .options   = defaults,
     .operation = cudf_regex_operation::EXTRACT_ALL,
     .inputs =
       {"123 banana 7 eleven", "41 apple", "6 péar 0 pair", std::nullopt, "", "bees", "4 paré"},
     .expected = {strings_row({"123", "banana", "7", "eleven"}),
                  strings_row({"41", "apple"}),
                  strings_row({"6", "péar", "0", "pair"}),
                  null_row(),
                  null_row(),
                  null_row(),
                  strings_row({"4", "paré"})}});

  std::vector<maybe_string> single_inputs{
    "123 banana 7 eleven", "41 apple", "6 péar 0 pair", std::nullopt, "", "bees", "4 paré"};
  append_case(cases,
              {.test_name     = "StringsExtractTests.ExtractSingle",
               .assertion     = "capture-one",
               .pattern       = R"REGEX((\d+) (\w+))REGEX",
               .options       = defaults,
               .operation     = cudf_regex_operation::EXTRACT_SINGLE,
               .inputs        = single_inputs,
               .expected      = {strings_row({"banana"}),
                                 strings_row({"apple"}),
                                 strings_row({"péar"}),
                                 null_row(),
                                 null_row(),
                                 null_row(),
                                 strings_row({"paré"})},
               .capture_index = 1});
  append_case(cases,
              {.test_name     = "StringsExtractTests.ExtractSingle",
               .assertion     = "capture-zero",
               .pattern       = R"REGEX((\d+) (\w+))REGEX",
               .options       = defaults,
               .operation     = cudf_regex_operation::EXTRACT_SINGLE,
               .inputs        = single_inputs,
               .expected      = {strings_row({"123"}),
                                 strings_row({"41"}),
                                 strings_row({"6"}),
                                 null_row(),
                                 null_row(),
                                 null_row(),
                                 strings_row({"4"})},
               .capture_index = 0});
  append_case(cases,
              {.test_name              = "StringsExtractTests.ExtractSingle",
               .assertion              = "capture-index-error",
               .pattern                = R"REGEX((\d+) (\w+))REGEX",
               .options                = defaults,
               .operation              = cudf_regex_operation::EXTRACT_SINGLE,
               .inputs                 = {"123 banana"},
               .expected               = {scalar_row(0)},
               .capture_index          = 2,
               .expect_operation_error = true});

  for (cudf_regex_operation operation :
       {cudf_regex_operation::EXTRACT, cudf_regex_operation::EXTRACT_ALL}) {
    append_case(cases,
                {.test_name = "StringsExtractTests.Errors",
                 .assertion = "no-capture-" + std::to_string(static_cast<int>(operation)),
                 .pattern   = R"REGEX(\w+)REGEX",
                 .options   = defaults,
                 .operation = operation,
                 .inputs    = {"this column intentionally left blank"},
                 .expected  = {scalar_row(0)},
                 .expect_operation_error = true});
  }

  for (cudf_regex_operation operation : {cudf_regex_operation::EXTRACT,
                                         cudf_regex_operation::EXTRACT_ALL,
                                         cudf_regex_operation::EXTRACT_SINGLE}) {
    append_case(cases,
                {.test_name        = "StringsExtractTests.EmptyInput",
                 .assertion        = "empty-column-" + std::to_string(static_cast<int>(operation)),
                 .pattern          = R"REGEX((\w+))REGEX",
                 .options          = defaults,
                 .operation        = operation,
                 .capture_index    = operation == cudf_regex_operation::EXTRACT_SINGLE ? 1U : 0U,
                 .expected_columns = operation == cudf_regex_operation::EXTRACT ? 1U : 0U});
  }

  auto medium_pattern = std::string{
    R"REGEX(hello @abc @def (world) The quick brown @fox jumps over the lazy @dog hello http://www.world.com)REGEX"};
  append_extract(
    "StringsExtractTests.MediumRegex",
    "capture",
    medium_pattern,
    defaults,
    {std::string{
       R"INPUT(hello @abc @def world The quick brown @fox jumps over the lazy @dog hello http://www.world.com thats all)INPUT"},
     R"INPUT(12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890)INPUT",
     R"INPUT(abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz)INPUT"},
    {strings_row({"world"}), strings_row({std::nullopt}), strings_row({std::nullopt})});
  auto large_pattern = std::string{
    R"REGEX(hello @abc @def world The (quick) brown @fox jumps over the lazy @dog hello http://www.world.com I'm here @home zzzz)REGEX"};
  append_extract(
    "StringsExtractTests.LargeRegex",
    "capture",
    large_pattern,
    defaults,
    {std::string{
       R"INPUT(hello @abc @def world The quick brown @fox jumps over the lazy @dog hello http://www.world.com I'm here @home zzzz)INPUT"},
     R"INPUT(12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890)INPUT",
     R"INPUT(abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz)INPUT"},
    {strings_row({"quick"}), strings_row({std::nullopt}), strings_row({std::nullopt})});

  std::vector<maybe_string> crlf_inputs{"abc\r\n",
                                        "abc\n",
                                        "abc\r",
                                        "abc",
                                        "a\r\nb",
                                        "abc\r\n\r\n",
                                        "",
                                        std::string{"abc"} + next_line,
                                        "a\nb\r\nc",
                                        "\r\n",
                                        "\r\nabc",
                                        "x\n\r",
                                        "a\r\rb",
                                        "a\n\nb"};
  append_extract("StringsExtractTests.CrlfLineAnchorExtNewline",
                 "capture-before-line-end",
                 "([a-z]+)$",
                 extended,
                 crlf_inputs,
                 {strings_row({"abc"}),
                  strings_row({"abc"}),
                  strings_row({"abc"}),
                  strings_row({"abc"}),
                  strings_row({"b"}),
                  strings_row({std::nullopt}),
                  strings_row({std::nullopt}),
                  strings_row({"abc"}),
                  strings_row({"c"}),
                  strings_row({std::nullopt}),
                  strings_row({"abc"}),
                  strings_row({std::nullopt}),
                  strings_row({"b"}),
                  strings_row({"b"})});
}

void append_replace_cases(std::vector<cudf_regex_case>& cases)
{
  auto defaults       = cudf_default;
  auto multiline      = cudf_multiline;
  auto extended       = cudf_extended;
  auto extended_ml    = cudf_extended_multiline;
  auto append_replace = [&](std::string test_name,
                            std::string assertion,
                            std::string pattern,
                            compile_options options,
                            std::string replacement,
                            std::vector<maybe_string> inputs,
                            std::vector<maybe_string> expected,
                            bool backrefs           = false,
                            std::size_t max_matches = cudf_unlimited_matches) {
    if (inputs.size() != expected.size()) {
      throw std::logic_error("cuDF replacement input and expected row counts differ");
    }
    std::vector<cudf_expected_row> expected_rows;
    expected_rows.reserve(expected.size());
    for (maybe_string const& value : expected) {
      expected_rows.push_back(value ? strings_row({*value}) : null_row());
    }
    append_case(cases,
                {.test_name                = std::move(test_name),
                 .assertion                = std::move(assertion),
                 .pattern                  = std::move(pattern),
                 .options                  = options,
                 .operation                = cudf_regex_operation::REPLACE,
                 .inputs                   = std::move(inputs),
                 .expected                 = std::move(expected_rows),
                 .replacement              = std::move(replacement),
                 .max_matches              = max_matches,
                 .replacement_has_backrefs = backrefs});
  };

  std::vector<maybe_string> base_inputs{"the quick brown fox jumps over the lazy dog",
                                        "the fat cat lays next to the other accénted cat",
                                        "a slow moving turtlé cannot catch the bird",
                                        "which can be composéd together to form a more complete",
                                        "thé result does not include the value in the sum in",
                                        "",
                                        std::nullopt};
  append_replace("StringsReplaceRegexTest.ReplaceRegexTest",
                 "literal-and-null-mask",
                 R"REGEX((\bthe\b))REGEX",
                 defaults,
                 "=",
                 base_inputs,
                 {"= quick brown fox jumps over = lazy dog",
                  "= fat cat lays next to = other accénted cat",
                  "a slow moving turtlé cannot catch = bird",
                  "which can be composéd together to form a more complete",
                  "thé result does not include = value in = sum in",
                  "",
                  std::nullopt});

  append_replace("StringsReplaceRegexTest.WithEmptyPattern",
                 "empty-pattern-is-no-op",
                 "",
                 defaults,
                 "bbb",
                 {"asd", "xcv"},
                 {"asd", "xcv"});
  append_replace("StringsReplaceRegexTest.MultiReplacement",
                 "limit-two",
                 "aba",
                 defaults,
                 "_",
                 {"aba bcd aba", "abababa abababa"},
                 {"_ bcd _", "_b_ abababa"},
                 false,
                 2);
  append_replace("StringsReplaceRegexTest.MultiReplacement",
                 "limit-zero",
                 "aba",
                 defaults,
                 "_",
                 {"aba bcd aba", "abababa abababa"},
                 {"aba bcd aba", "abababa abababa"},
                 false,
                 0);

  std::vector<maybe_string> boundary_inputs{"aba bcd\naba", "zéz", "A1B2-é3", "e é", "_", "a_b"};
  append_replace("StringsReplaceRegexTest.WordBoundary",
                 "boundary",
                 R"REGEX(\b)REGEX",
                 defaults,
                 "X",
                 boundary_inputs,
                 {"XabaX XbcdX\nXabaX", "XzézX", "XA1B2X-Xé3X", "XeX XéX", "X_X", "Xa_bX"});
  append_replace("StringsReplaceRegexTest.WordBoundary",
                 "non-boundary",
                 R"REGEX(\B)REGEX",
                 defaults,
                 "X",
                 boundary_inputs,
                 {"aXbXa bXcXd\naXbXa", "zXéXz", "AX1XBX2-éX3", "e é", "_", "aX_Xb"});

  std::vector<maybe_string> alternation_inputs{
    "16  6  brr  232323  1  hello  90", "123 ABC 00 2022", "abé123  4567  89xyz"};
  for (std::string pattern :
       {std::string{R"REGEX((^|\s)\d+(\s|$))REGEX"}, std::string{R"REGEX((\s|^)\d+($|\s))REGEX"}}) {
    append_replace("StringsReplaceRegexTest.Alternation",
                   pattern,
                   pattern,
                   defaults,
                   "_",
                   alternation_inputs,
                   {"__ brr __ hello _", "_ABC_2022", "abé123 _ 89xyz"});
  }

  std::vector<maybe_string> zero_inputs{"DD", "zéz", "DsDs", ""};
  append_replace("StringsReplaceRegexTest.ZeroLengthMatch",
                 "star",
                 "D*",
                 defaults,
                 "_",
                 zero_inputs,
                 {"__", "_z_é_z_", "__s__s_", "_"});
  append_replace("StringsReplaceRegexTest.ZeroLengthMatch",
                 "optional",
                 "D?s?",
                 defaults,
                 "_",
                 zero_inputs,
                 {"___", "_z_é_z_", "___", "_"});

  std::vector<maybe_string> zero_range_inputs{"a", "", "123", "XYAZ", "abc", "zéyab"};
  append_replace("StringsReplaceRegexTest.ZeroRangeQuantifier",
                 "literal",
                 "A{0,5}",
                 defaults,
                 "_",
                 zero_range_inputs,
                 {"_a_", "_", "_1_2_3_", "_X_Y__Z_", "_a_b_c_", "_z_é_y_a_b_"});
  append_replace("StringsReplaceRegexTest.ZeroRangeQuantifier",
                 "class",
                 "[a0-9]{0,2}",
                 defaults,
                 "_",
                 zero_range_inputs,
                 {"__", "_", "___", "_X_Y_A_Z_", "__b_c_", "_z_é_y__b_"});
  append_replace("StringsReplaceRegexTest.ZeroRangeQuantifier",
                 "group",
                 "(?:ab){0,3}",
                 defaults,
                 "_",
                 zero_range_inputs,
                 {"_a_", "_", "_1_2_3_", "_X_Y_A_Z_", "__c_", "_z_é_y__"});

  std::vector<maybe_string> multiline_inputs{"bcd\naba\nefg", "aba\naba abab\naba", "aba"};
  append_replace("StringsReplaceRegexTest.Multiline",
                 "literal-multiline",
                 "^aba$",
                 multiline,
                 "_",
                 multiline_inputs,
                 {"bcd\n_\nefg", "_\naba abab\n_", "_"});
  append_replace("StringsReplaceRegexTest.Multiline",
                 "literal-default",
                 "^aba$",
                 defaults,
                 "_",
                 multiline_inputs,
                 {"bcd\naba\nefg", "aba\naba abab\naba", "_"});
  append_replace("StringsReplaceRegexTest.Multiline",
                 "backref-multiline",
                 "(^aba)",
                 multiline,
                 R"REGEX([\1])REGEX",
                 multiline_inputs,
                 {"bcd\n[aba]\nefg", "[aba]\n[aba] abab\n[aba]", "[aba]"},
                 true);
  append_replace("StringsReplaceRegexTest.Multiline",
                 "backref-default",
                 "(^aba)",
                 defaults,
                 R"REGEX([\1])REGEX",
                 multiline_inputs,
                 {"bcd\naba\nefg", "[aba]\naba abab\naba", "[aba]"},
                 true);

  auto next_line           = std::string{"\xC2\x85"};
  auto paragraph_separator = std::string{"\xE2\x80\xA9"};
  std::vector<maybe_string> special_inputs{
    std::string{"zzé"} + next_line + "qqq" + next_line + "zzé",
    std::string{"qqq"} + next_line + "zzé" + next_line + "lll",
    "zzé",
    "",
    std::string{"zzé"} + paragraph_separator,
    "abc\rzzé\r"};
  append_replace("StringsReplaceRegexTest.SpecialNewLines",
                 "literal-extended",
                 "^zzé$",
                 extended,
                 "_",
                 special_inputs,
                 {std::string{"zzé"} + next_line + "qqq" + next_line + "zzé",
                  std::string{"qqq"} + next_line + "zzé" + next_line + "lll",
                  "_",
                  "",
                  "_" + paragraph_separator,
                  "abc\rzzé\r"});
  append_replace("StringsReplaceRegexTest.SpecialNewLines",
                 "literal-extended-multiline",
                 "^zzé$",
                 extended_ml,
                 "_",
                 special_inputs,
                 {"_" + next_line + "qqq" + next_line + "_",
                  "qqq" + next_line + "_" + next_line + "lll",
                  "_",
                  "",
                  "_" + paragraph_separator,
                  "abc\r_\r"});
  append_replace("StringsReplaceRegexTest.SpecialNewLines",
                 "backref-extended-multiline",
                 "(^zzé$)",
                 extended_ml,
                 R"REGEX([\1])REGEX",
                 special_inputs,
                 {"[zzé]" + next_line + "qqq" + next_line + "[zzé]",
                  "qqq" + next_line + "[zzé]" + next_line + "lll",
                  "[zzé]",
                  "",
                  "[zzé]" + paragraph_separator,
                  "abc\r[zzé]\r"},
                 true);

  append_replace("StringsReplaceRegexTest.ReplaceBackrefsRegexTest",
                 "capture-template-and-null-mask",
                 R"REGEX((\w) (\w))REGEX",
                 defaults,
                 R"REGEX(\1-\2)REGEX",
                 base_inputs,
                 {"the-quick-brown-fox-jumps-over-the-lazy-dog",
                  "the-fat-cat-lays-next-to-the-other-accénted-cat",
                  "a-slow-moving-turtlé-cannot-catch-the-bird",
                  "which-can-be-composéd-together-to-form-a more-complete",
                  "thé-result-does-not-include-the-value-in-the-sum-in",
                  "",
                  std::nullopt},
                 true);
  append_replace("StringsReplaceRegexTest.ReplaceBackrefsRegexAltIndexPatternTest",
                 "braced-indices",
                 R"REGEX((\d+)-(\d+))REGEX",
                 defaults,
                 "${2} X ${1}0",
                 {"12-3 34-5 67-89", "0-99: 777-888:: 5673-0"},
                 {"3 X 120 5 X 340 89 X 670", "99 X 00: 888 X 7770:: 0 X 56730"},
                 true);
  append_replace(
    "StringsReplaceRegexTest.ReplaceBackrefsRegexReversedTest",
    "reversed-captures",
    "([a-z])-([a-zé])",
    defaults,
    R"REGEX(X\2+\1Z)REGEX",
    {"A543", "Z756", "", "tést-string", "two-thréé four-fivé", "abcd-éfgh", "tést-string-again"},
    {"A543",
     "Z756",
     "",
     "tésXs+tZtring",
     "twXt+oZhréé fouXf+rZivé",
     "abcXé+dZfgh",
     "tésXs+tZtrinXa+gZgain"},
    true);

  std::vector<maybe_string> html_inputs{"<h1>title</h1><h2>ABC</h2>",
                                        "<h1>1234567</h1><h2>XYZ</h2>"};
  std::vector<maybe_string> html_expected{"<h2>title</h2><p>ABC</p>", "<h2>1234567</h2><p>XYZ</p>"};
  for (std::string pattern : {std::string{"<h1>(.*)</h1><h2>(.*)</h2>"},
                              std::string{R"REGEX(<h1>([a-z\d]+)</h1><h2>([A-Z]+)</h2>)REGEX"}}) {
    append_replace("StringsReplaceRegexTest.BackrefWithGreedyQuantifier",
                   pattern,
                   pattern,
                   defaults,
                   R"REGEX(<h2>\1</h2><p>\2</p>)REGEX",
                   html_inputs,
                   html_expected,
                   true);
  }

  append_replace("StringsReplaceRegexTest.ReplaceBackrefsRegexZeroIndexTest",
                 "whole-match-index",
                 R"REGEX((TEST)(\d+))REGEX",
                 defaults,
                 "${0}: ${1}, ${2}; ",
                 {"TEST123", "TEST1TEST2", "TEST2-TEST1122", "TEST1-TEST-T", "TES3"},
                 {"TEST123: TEST, 123; ",
                  "TEST1: TEST, 1; TEST2: TEST, 2; ",
                  "TEST2: TEST, 2; -TEST1122: TEST, 1122; ",
                  "TEST1: TEST, 1; -TEST-T",
                  "TES3"},
                 true);
  append_replace("StringsReplaceRegexTest.ReplaceBackrefsWithEmptyCapture",
                 "optional-line-ending",
                 R"REGEX((\r\n|\r)?$)REGEX",
                 defaults,
                 R"REGEX([\1])REGEX",
                 {"one\ntwo", "three\n\n", "four\r\n"},
                 {"one\ntwo[]", "three\n[]\n[]", "four[\r\n][]"},
                 true);
  append_replace("StringsReplaceRegexTest.ReplaceBackrefsWithEmptyCapture",
                 "empty-optional-capture",
                 "^(a?)",
                 defaults,
                 R"REGEX([\1])REGEX",
                 {"one\ntwo", "three\n\n", "four\r\n"},
                 {"[]one\ntwo", "[]three\n\n", "[]four\r\n"},
                 true);

  struct replacement_error {
    std::string pattern     = "";
    std::string replacement = "";
  };
  for (replacement_error item :
       std::to_array<replacement_error>({{R"REGEX((\w).(\w))REGEX", R"REGEX(\3)REGEX"},
                                         {"", R"REGEX(\1)REGEX"},
                                         {R"REGEX((\w))REGEX", ""}})) {
    auto assertion = item.pattern + "/" + item.replacement;
    append_case(cases,
                {.test_name   = "StringsReplaceRegexTest.ReplaceBackrefsRegexErrorTest",
                 .assertion   = std::move(assertion),
                 .pattern     = std::move(item.pattern),
                 .options     = defaults,
                 .operation   = cudf_regex_operation::REPLACE,
                 .inputs      = {"this string left intentionally blank"},
                 .expected    = {scalar_row(0)},
                 .replacement = std::move(item.replacement),
                 .replacement_has_backrefs = true,
                 .expect_operation_error   = true});
  }

  auto medium_pattern = std::string{
    R"REGEX(hello @abc @def world The quick brown @fox jumps over the lazy @dog hello http://www.world.com)REGEX"};
  append_replace(
    "StringsReplaceRegexTest.MediumReplaceRegex",
    "empty-replacement",
    medium_pattern,
    defaults,
    "",
    {medium_pattern + " thats all", "12345678901234567890", "abcdefghijklmnopqrstuvwxyz"},
    {" thats all", "12345678901234567890", "abcdefghijklmnopqrstuvwxyz"});
  auto large_pattern = std::string{
    R"REGEX(hello @abc @def world The (quick) brown @fox jumps over the lazy @dog hello http://www.world.com I'm here @home zzzz)REGEX"};
  append_replace(
    "StringsReplaceRegexTest.LargeReplaceRegex",
    "empty-replacement",
    large_pattern,
    defaults,
    "",
    {std::string{
       R"INPUT(zzzz hello @abc @def world The quick brown @fox jumps over the lazy @dog hello http://www.world.com I'm here @home zzzz)INPUT"},
     "12345678901234567890",
     "abcdefghijklmnopqrstuvwxyz"},
    {"zzzz ", "12345678901234567890", "abcdefghijklmnopqrstuvwxyz"});

  std::vector<maybe_string> crlf_inputs{"abc\r\n",
                                        "abc\n",
                                        "abc\r",
                                        "abc",
                                        "a\r\nb",
                                        "abc\r\n\r\n",
                                        "",
                                        std::string{"abc"} + next_line,
                                        "a\nb\r\nc",
                                        "\r\n",
                                        "\r\nabc",
                                        "x\n\r",
                                        "a\r\rb",
                                        "a\n\nb"};
  std::vector<maybe_string> crlf_literal_expected{"[X]\r\n",
                                                  "[X]\n",
                                                  "[X]\r",
                                                  "[X]",
                                                  "a\r\nb",
                                                  "abc\r\n\r\n",
                                                  "",
                                                  "[X]" + next_line,
                                                  "a\nb\r\nc",
                                                  "\r\n",
                                                  "\r\n[X]",
                                                  "x\n\r",
                                                  "a\r\rb",
                                                  "a\n\nb"};
  append_replace(
    "StringsReplaceRegexTest.CrlfLineAnchorExtNewline",
    "literal",
    "abc$",
    extended,
    "[X]",
    std::vector<maybe_string>(crlf_inputs.begin(), crlf_inputs.begin() + 8),
    std::vector<maybe_string>(crlf_literal_expected.begin(), crlf_literal_expected.begin() + 8));
  append_replace("StringsReplaceRegexTest.CrlfEdgeCasesExtNewline",
                 "literal",
                 "abc$",
                 extended,
                 "[X]",
                 crlf_inputs,
                 crlf_literal_expected);
  std::vector<maybe_string> crlf_backref_expected{"[abc]\r\n",
                                                  "[abc]\n",
                                                  "[abc]\r",
                                                  "[abc]",
                                                  "a\r\nb",
                                                  "abc\r\n\r\n",
                                                  "",
                                                  "[abc]" + next_line,
                                                  "a\nb\r\nc",
                                                  "\r\n",
                                                  "\r\n[abc]",
                                                  "x\n\r",
                                                  "a\r\rb",
                                                  "a\n\nb"};
  append_replace("StringsReplaceRegexTest.CrlfEdgeCasesExtNewline",
                 "backref",
                 "(abc)$",
                 extended,
                 R"REGEX([\1])REGEX",
                 crlf_inputs,
                 crlf_backref_expected,
                 true);

  append_replace("StringsReplaceRegexTest.AlternationPriorityFirstWins",
                 "foo-first",
                 "foo|foobar",
                 defaults,
                 "X",
                 {"foo", "foobar", "foobarbaz", "bar", "xfoobar", ""},
                 {"X", "Xbar", "Xbarbaz", "bar", "xXbar", ""});
  append_replace("StringsReplaceRegexTest.AlternationPriorityFirstWins",
                 "cat-first",
                 "cat|catch",
                 defaults,
                 "X",
                 {"cat", "catch", "catfish", "dog", ""},
                 {"X", "Xch", "Xfish", "dog", ""});
}

void append_split_cases(std::vector<cudf_regex_case>& cases)
{
  auto defaults     = cudf_default;
  auto append_split = [&](std::string test_name,
                          std::string assertion,
                          std::string pattern,
                          std::vector<maybe_string> inputs,
                          std::vector<cudf_expected_row> expected,
                          bool table_output,
                          bool reverse                 = false,
                          std::size_t max_matches      = cudf_unlimited_matches,
                          std::size_t expected_columns = 0) {
    append_case(cases,
                {.test_name        = std::move(test_name),
                 .assertion        = std::move(assertion),
                 .pattern          = std::move(pattern),
                 .options          = defaults,
                 .operation        = cudf_regex_operation::SPLIT,
                 .inputs           = std::move(inputs),
                 .expected         = std::move(expected),
                 .max_matches      = max_matches,
                 .expected_columns = expected_columns,
                 .reverse          = reverse,
                 .table_output     = table_output});
  };

  std::vector<maybe_string> inputs{" Héllo thesé", std::nullopt, "are some  ", "tést String", ""};
  std::vector<cudf_expected_row> whitespace_table{strings_row({"", "Héllo", "thesé"}),
                                                  null_row(),
                                                  strings_row({"are", "some", ""}),
                                                  strings_row({"tést", "String", std::nullopt}),
                                                  strings_row({"", std::nullopt, std::nullopt})};
  std::vector<cudf_expected_row> whitespace_records{strings_row({"", "Héllo", "thesé"}),
                                                    null_row(),
                                                    strings_row({"are", "some", ""}),
                                                    strings_row({"tést", "String"}),
                                                    strings_row({""})};
  append_split("StringsSplitTest.SplitRegex",
               "split-whitespace-table",
               R"REGEX(\s+)REGEX",
               inputs,
               whitespace_table,
               true,
               false,
               cudf_unlimited_matches,
               3);
  append_split("StringsSplitTest.SplitRegex",
               "rsplit-whitespace-table",
               R"REGEX(\s+)REGEX",
               inputs,
               whitespace_table,
               true,
               true,
               cudf_unlimited_matches,
               3);
  append_split("StringsSplitTest.SplitRecordRegex",
               "split-whitespace-record",
               R"REGEX(\s+)REGEX",
               inputs,
               whitespace_records,
               false);
  append_split("StringsSplitTest.SplitRecordRegex",
               "rsplit-whitespace-record",
               R"REGEX(\s+)REGEX",
               inputs,
               whitespace_records,
               false,
               true);

  std::vector<cudf_expected_row> letter_table{
    strings_row({" H", "llo th", "s", ""}),
    null_row(),
    strings_row({"ar", " som", "  ", std::nullopt}),
    strings_row({"t", "st String", std::nullopt, std::nullopt}),
    strings_row({"", std::nullopt, std::nullopt, std::nullopt})};
  std::vector<cudf_expected_row> letter_records{strings_row({" H", "llo th", "s", ""}),
                                                null_row(),
                                                strings_row({"ar", " som", "  "}),
                                                strings_row({"t", "st String"}),
                                                strings_row({""})};
  append_split("StringsSplitTest.SplitRegex",
               "split-letter-table",
               "[eé]",
               inputs,
               letter_table,
               true,
               false,
               cudf_unlimited_matches,
               4);
  append_split("StringsSplitTest.SplitRegex",
               "rsplit-letter-table",
               "[eé]",
               inputs,
               letter_table,
               true,
               true,
               cudf_unlimited_matches,
               4);
  append_split("StringsSplitTest.SplitRecordRegex",
               "split-letter-record",
               "[eé]",
               inputs,
               letter_records,
               false);
  append_split("StringsSplitTest.SplitRecordRegex",
               "rsplit-letter-record",
               "[eé]",
               inputs,
               letter_records,
               false,
               true);

  std::vector<maybe_string> max_inputs{
    " Héllo\tthesé", std::nullopt, "are\nsome  ", "tést\rString", ""};
  append_split("StringsSplitTest.SplitRegexWithMaxSplit",
               "table-max-one",
               R"REGEX(\s+)REGEX",
               max_inputs,
               {strings_row({"", "Héllo\tthesé"}),
                null_row(),
                strings_row({"are", "some  "}),
                strings_row({"tést", "String"}),
                strings_row({"", std::nullopt})},
               true,
               false,
               1,
               2);
  auto max_table_all =
    std::vector<cudf_expected_row>{strings_row({"", "Héllo", "thesé"}),
                                   null_row(),
                                   strings_row({"are", "some", ""}),
                                   strings_row({"tést", "String", std::nullopt}),
                                   strings_row({"", std::nullopt, std::nullopt})};
  for (std::size_t limit : {2U, 3U}) {
    append_split("StringsSplitTest.SplitRegexWithMaxSplit",
                 "table-max-" + std::to_string(limit),
                 R"REGEX(\s+)REGEX",
                 max_inputs,
                 max_table_all,
                 true,
                 false,
                 limit,
                 3);
  }

  append_split("StringsSplitTest.SplitRegexWithMaxSplit",
               "record-single-space-max-one",
               R"REGEX(\s)REGEX",
               max_inputs,
               {strings_row({"", "Héllo\tthesé"}),
                null_row(),
                strings_row({"are", "some  "}),
                strings_row({"tést", "String"}),
                strings_row({""})},
               false,
               false,
               1);
  append_split("StringsSplitTest.SplitRegexWithMaxSplit",
               "record-single-space-max-two",
               R"REGEX(\s)REGEX",
               max_inputs,
               {strings_row({"", "Héllo", "thesé"}),
                null_row(),
                strings_row({"are", "some", " "}),
                strings_row({"tést", "String"}),
                strings_row({""})},
               false,
               false,
               2);
  auto single_space_all = std::vector<cudf_expected_row>{strings_row({"", "Héllo", "thesé"}),
                                                         null_row(),
                                                         strings_row({"are", "some", "", ""}),
                                                         strings_row({"tést", "String"}),
                                                         strings_row({""})};
  append_split("StringsSplitTest.SplitRegexWithMaxSplit",
               "record-single-space-max-three",
               R"REGEX(\s)REGEX",
               max_inputs,
               single_space_all,
               false,
               false,
               3);
  append_split("StringsSplitTest.SplitRegexWithMaxSplit",
               "record-single-space-max-three-repeat",
               R"REGEX(\s)REGEX",
               max_inputs,
               single_space_all,
               false,
               false,
               3);

  std::vector<maybe_string> boundary_inputs{"a", "ab", "-+", "e\né"};
  append_split("StringsSplitTest.SplitRegexWordBoundary",
               "boundary-table",
               R"REGEX(\b)REGEX",
               boundary_inputs,
               {strings_row({"", "a", "", std::nullopt, std::nullopt}),
                strings_row({"", "ab", "", std::nullopt, std::nullopt}),
                strings_row({"-+", std::nullopt, std::nullopt, std::nullopt, std::nullopt}),
                strings_row({"", "e", "\n", "é", ""})},
               true,
               false,
               cudf_unlimited_matches,
               5);
  append_split("StringsSplitTest.SplitRegexWordBoundary",
               "non-boundary-record",
               R"REGEX(\B)REGEX",
               boundary_inputs,
               {strings_row({"a"}),
                strings_row({"a", "b"}),
                strings_row({"", "-", "+", ""}),
                strings_row({"e\né"})},
               false);

  auto empty_inputs = std::vector<maybe_string>{"", "", "", ""};
  auto empty_rows   = std::vector<cudf_expected_row>{
    strings_row({""}), strings_row({""}), strings_row({""}), strings_row({""})};
  for (bool table : {true, false}) {
    for (bool reverse : {false, true}) {
      append_split("StringsSplitTest.SplitRegexAllEmpty",
                   std::string{table ? "table" : "record"} + (reverse ? "-reverse" : "-forward"),
                   "[ _]",
                   empty_inputs,
                   empty_rows,
                   table,
                   reverse,
                   cudf_unlimited_matches,
                   table ? 1 : 0);
    }
  }

  std::vector<maybe_string> reverse_inputs{
    " Héllo\tthesé", std::nullopt, "are some\n ", "tést\rString", ""};
  append_split("StringsSplitTest.RSplitRegexWithMaxSplit",
               "table-max-one",
               R"REGEX(\s+)REGEX",
               reverse_inputs,
               {strings_row({" Héllo", "thesé"}),
                null_row(),
                strings_row({"are some", ""}),
                strings_row({"tést", "String"}),
                strings_row({"", std::nullopt})},
               true,
               true,
               1,
               2);
  append_split("StringsSplitTest.RSplitRegexWithMaxSplit",
               "record-max-one",
               R"REGEX(\s+)REGEX",
               reverse_inputs,
               {strings_row({" Héllo", "thesé"}),
                null_row(),
                strings_row({"are some", ""}),
                strings_row({"tést", "String"}),
                strings_row({""})},
               false,
               true,
               1);
  auto reverse_all = std::vector<cudf_expected_row>{strings_row({"", "Héllo", "thesé"}),
                                                    null_row(),
                                                    strings_row({"are", "some", ""}),
                                                    strings_row({"tést", "String"}),
                                                    strings_row({""})};
  append_split("StringsSplitTest.RSplitRegexWithMaxSplit",
               "record-max-three",
               R"REGEX(\s+)REGEX",
               reverse_inputs,
               reverse_all,
               false,
               true,
               3);
  append_split("StringsSplitTest.RSplitRegexWithMaxSplit",
               "record-max-three-repeat",
               R"REGEX(\s+)REGEX",
               reverse_inputs,
               reverse_all,
               false,
               true,
               3);

  for (bool table : {true, false}) {
    for (bool reverse : {false, true}) {
      append_split("StringsSplitTest.SplitZeroSizeStringsColumns",
                   std::string{table ? "table" : "record"} + (reverse ? "-reverse" : "-forward"),
                   R"REGEX(\s)REGEX",
                   {},
                   {},
                   table,
                   reverse,
                   cudf_unlimited_matches,
                   table ? 1 : 0);
    }
  }

  for (bool record : {false, true}) {
    for (bool reverse : {false, true}) {
      append_case(cases,
                  {.test_name = "StringsSplitTest.InvalidParameter",
                   .assertion =
                     std::string{record ? "record" : "table"} + (reverse ? "-reverse" : "-forward"),
                   .pattern                = "",
                   .options                = defaults,
                   .operation              = cudf_regex_operation::SPLIT,
                   .inputs                 = {"string left intentionally blank"},
                   .expected               = {scalar_row(0)},
                   .reverse                = reverse,
                   .table_output           = !record,
                   .expect_operation_error = true});
    }
  }
}

std::vector<cudf_regex_case> make_regex_cases()
{
  std::vector<cudf_regex_case> cases;
  append_scalar_cases(cases);
  append_contains_cases(cases);
  append_findall_cases(cases);
  append_extract_cases(cases);
  append_replace_cases(cases);
  append_split_cases(cases);
  return cases;
}

std::vector<cudf_compile_case> make_compile_cases()
{
  auto options = cudf_default;
  std::vector<cudf_compile_case> cases;
  auto append = [&](std::string test, std::string pattern, bool should_compile) {
    cases.push_back({.test_name      = std::move(test),
                     .pattern        = std::move(pattern),
                     .options        = options,
                     .should_compile = should_compile});
  };

  for (std::string pattern : {std::string{"(3?)+"}, std::string{"(?:3?)+"}}) {
    append("StringsContainsTests.Errors", std::move(pattern), true);
  }
  for (std::string pattern : {std::string{"3?+"},
                              std::string{"{3}a"},
                              std::string{"aaaa{1234,5678}"},
                              std::string{"aaaa{123,5678}"},
                              std::string{"[a-C]"}}) {
    append("StringsContainsTests.Errors", std::move(pattern), false);
  }
  for (std::string pattern : {std::string{"a{0}"},
                              std::string{"a{0,1}"},
                              std::string{"a{0,}"},
                              std::string{"(ab){0}"},
                              std::string{"(ab){0,1}"},
                              std::string{"(ab){0,}"}}) {
    append("StringsContainsTests.ZeroRangeQuantifier", std::move(pattern), true);
  }
  for (std::string pattern : {std::string{"^+"},
                              std::string{"$+"},
                              std::string{"(^)+"},
                              std::string{"($)+"},
                              std::string{R"REGEX(\A+)REGEX"},
                              std::string{R"REGEX(\Z+)REGEX"},
                              std::string{R"REGEX((\A)+)REGEX"},
                              std::string{R"REGEX((\Z)+)REGEX"},
                              std::string{"(^($))+"}}) {
    append("StringsContainsTests.QuantifierErrors", std::move(pattern), false);
  }
  append("StringsContainsTests.QuantifierErrors", "(^a($))+", true);
  append("StringsContainsTests.QuantifierErrors", "(^(a$))+", true);

  append("StringsReplaceRegexTest.InvalidRegex", "|", true);
  for (std::string pattern : {std::string{"*"},
                              std::string{"+"},
                              std::string{"ab(*)"},
                              std::string{R"REGEX(\)REGEX"},
                              std::string{R"REGEX(\p)REGEX"}}) {
    append("StringsReplaceRegexTest.InvalidRegex", std::move(pattern), false);
  }
  return cases;
}

std::span<cudf_regex_case const> cudf_regex_cases()
{
  static std::vector<cudf_regex_case> const cases = make_regex_cases();
  return cases;
}

std::span<cudf_compile_case const> cudf_compile_cases()
{
  static std::vector<cudf_compile_case> const cases = make_compile_cases();
  return cases;
}

// cuDF result normalization

using testing::execution_result;
using testing::match_span;

using maybe_span = std::optional<match_span>;

std::optional<std::string> span_text(std::string_view input, maybe_span const& span)
{
  if (!span) return std::nullopt;
  return std::string{input.substr(span->begin, span->end - span->begin)};
}

std::optional<std::size_t> replacement_reference_count(std::string_view replacement,
                                                       std::uint32_t capture_count)
{
  if (replacement.empty()) return std::nullopt;
  std::size_t references = 0;
  for (std::size_t position = 0; position < replacement.size();) {
    if (replacement[position] == '\\') {
      ++position;
      if (position >= replacement.size() ||
          std::isdigit(static_cast<unsigned char>(replacement[position])) == 0) {
        return std::nullopt;
      }
      auto index = static_cast<std::uint32_t>(replacement[position++] - '0');
      if (index > capture_count) return std::nullopt;
      ++references;
    } else if (replacement[position] == '$' && position + 1 < replacement.size() &&
               replacement[position + 1] == '{') {
      position += 2;
      std::uint32_t index = 0;
      bool digits         = false;
      while (position < replacement.size() &&
             std::isdigit(static_cast<unsigned char>(replacement[position])) != 0) {
        digits = true;
        index  = index * 10U + static_cast<std::uint32_t>(replacement[position++] - '0');
      }
      if (!digits || position >= replacement.size() || replacement[position] != '}' ||
          index > capture_count) {
        return std::nullopt;
      }
      ++position;
      ++references;
    } else {
      ++position;
    }
  }
  return references;
}

void append_replacement(std::string& output,
                        std::string_view replacement,
                        std::string_view input,
                        std::vector<maybe_span> const& captures)
{
  for (std::size_t position = 0; position < replacement.size();) {
    std::optional<std::uint32_t> capture;
    if (replacement[position] == '\\') {
      ++position;
      capture = static_cast<std::uint32_t>(replacement[position++] - '0');
    } else if (replacement[position] == '$' && position + 1 < replacement.size() &&
               replacement[position + 1] == '{') {
      position += 2;
      std::uint32_t index = 0;
      while (replacement[position] != '}') {
        index = index * 10U + static_cast<std::uint32_t>(replacement[position++] - '0');
      }
      ++position;
      capture = index;
    }
    if (capture) {
      if (*capture < captures.size() && captures[*capture]) {
        auto span = *captures[*capture];
        output.append(input.substr(span.begin, span.end - span.begin));
      }
    } else {
      output.push_back(replacement[position++]);
    }
  }
}

cudf_expected_row evaluate_row(cudf_regex_case const& test,
                               instruction_ir const& ir,
                               std::string const& input,
                               execution_result const& matches)
{
  cudf_expected_row output;
  switch (test.operation) {
    case cudf_regex_operation::CONTAINS: output.scalar = matches.matched ? 1 : 0; break;
    case cudf_regex_operation::PREFIX_MATCH:
      output.scalar = matches.matched && matches.matches.front().begin == 0 ? 1 : 0;
      break;
    case cudf_regex_operation::COUNT:
      output.scalar = static_cast<std::int64_t>(matches.count);
      break;
    case cudf_regex_operation::FIND:
      output.scalar =
        matches.matched ? static_cast<std::int64_t>(matches.matches.front().begin) : -1;
      break;
    case cudf_regex_operation::FIND_ALL: {
      auto selected_capture =
        test.capture_mode == cudf_capture_mode::CAPTURE && ir.capture_count == 1 ? 1U : 0U;
      for (std::vector<maybe_span> const& captures : matches.capture_matches) {
        output.strings.push_back(span_text(input, captures[selected_capture]));
      }
      break;
    }
    case cudf_regex_operation::EXTRACT:
      for (std::uint32_t capture = 1; capture <= ir.capture_count; ++capture) {
        output.strings.push_back(matches.matched
                                   ? span_text(input, matches.capture_matches.front()[capture])
                                   : std::nullopt);
      }
      break;
    case cudf_regex_operation::EXTRACT_ALL:
      if (!matches.matched) {
        output.valid = false;
        break;
      }
      for (std::vector<maybe_span> const& captures : matches.capture_matches) {
        for (std::uint32_t capture = 1; capture <= ir.capture_count; ++capture) {
          output.strings.push_back(span_text(input, captures[capture]));
        }
      }
      break;
    case cudf_regex_operation::EXTRACT_SINGLE:
      if (!matches.matched ||
          !matches.capture_matches.front()[test.capture_index + 1U].has_value()) {
        output.valid = false;
      } else {
        output.strings.push_back(
          span_text(input, matches.capture_matches.front()[test.capture_index + 1U]));
      }
      break;
    case cudf_regex_operation::REPLACE: {
      if (test.pattern.empty() && !test.replacement_has_backrefs) {
        output.strings.push_back(input);
        break;
      }
      std::string replaced;
      std::size_t copied = 0;
      auto limit         = std::min(test.max_matches, matches.matches.size());
      for (std::size_t index = 0; index < limit; ++index) {
        match_span span = matches.matches[index];
        replaced.append(input.substr(copied, span.begin - copied));
        if (test.replacement_has_backrefs) {
          append_replacement(replaced, test.replacement, input, matches.capture_matches[index]);
        } else {
          replaced += test.replacement;
        }
        copied = span.end;
      }
      replaced.append(input.substr(copied));
      output.strings.push_back(std::move(replaced));
      break;
    }
    case cudf_regex_operation::SPLIT: {
      std::size_t begin_index = 0;
      auto end_index          = matches.matches.size();
      if (test.max_matches != cudf_unlimited_matches) {
        if (test.reverse) {
          begin_index = end_index > test.max_matches ? end_index - test.max_matches : 0;
        } else {
          end_index = std::min(end_index, test.max_matches);
        }
      }
      std::size_t copied = 0;
      for (std::size_t index = begin_index; index < end_index; ++index) {
        match_span span = matches.matches[index];
        output.strings.emplace_back(input.substr(copied, span.begin - copied));
        copied = span.end;
      }
      output.strings.emplace_back(input.substr(copied));
      break;
    }
  }
  return output;
}

bool cudf_operation_is_valid(cudf_regex_case const& test, instruction_ir const& ir)
{
  switch (test.operation) {
    case cudf_regex_operation::FIND_ALL:
      return test.capture_mode != cudf_capture_mode::CAPTURE || ir.capture_count <= 1;
    case cudf_regex_operation::EXTRACT:
    case cudf_regex_operation::EXTRACT_ALL: return ir.capture_count != 0;
    case cudf_regex_operation::EXTRACT_SINGLE:
      return test.inputs.empty() || test.capture_index < ir.capture_count;
    case cudf_regex_operation::REPLACE:
      if (test.replacement_has_backrefs) {
        auto references = replacement_reference_count(test.replacement, ir.capture_count);
        return references.has_value() && *references != 0;
      }
      return true;
    case cudf_regex_operation::SPLIT: return !test.pattern.empty();
    default: return true;
  }
}

std::vector<cudf_expected_row> evaluate_cudf_case(cudf_regex_case const& test,
                                                  instruction_ir const& ir,
                                                  std::span<execution_result const> matches)
{
  if (matches.size() != test.inputs.size()) {
    throw std::invalid_argument("backend match row count differs from cuDF case input count");
  }
  std::vector<cudf_expected_row> output;
  output.reserve(test.inputs.size());
  for (std::size_t row = 0; row < test.inputs.size(); ++row) {
    if (!test.inputs[row]) {
      output.push_back(null_row());
    } else {
      output.push_back(evaluate_row(test, ir, *test.inputs[row], matches[row]));
    }
  }
  if (test.operation == cudf_regex_operation::SPLIT && test.table_output) {
    std::size_t columns = test.inputs.empty() ? 1 : 0;
    for (cudf_expected_row const& row : output) {
      if (row.valid) columns = std::max(columns, row.strings.size());
    }
    for (cudf_expected_row& row : output) {
      if (row.valid) row.strings.resize(columns, std::nullopt);
    }
  }
  return output;
}

std::size_t cudf_output_columns(cudf_regex_case const& test,
                                instruction_ir const& ir,
                                std::span<cudf_expected_row const> rows)
{
  if (test.operation == cudf_regex_operation::EXTRACT) return ir.capture_count;
  if (test.operation == cudf_regex_operation::SPLIT && test.table_output) {
    if (rows.empty()) return 1;
    std::size_t columns = 0;
    for (cudf_expected_row const& row : rows)
      columns = std::max(columns, row.strings.size());
    return columns;
  }
  return 1;
}

// backend compilation and execution

bool print_ir = false;

std::string diagnostics_text(std::vector<regex_ir::diagnostic> const& diagnostics)
{
  std::ostringstream output;
  for (regex_ir::diagnostic const& diagnostic : diagnostics) {
    output << diagnostic.message << "; ";
  }
  return output.str();
}

regex_ir::instruction_ir compile_ok(std::string_view pattern,
                                    regex_ir::operation const& operation,
                                    regex_ir::compile_options const& options = {})
{
  if (print_ir) {
    std::cout << "\n=== regex: " << pattern << " ===\n";
    auto automata = regex_ir::compile_automata(pattern, options);
    if (automata) std::cout << regex_ir::to_string(*automata.value);
  }
  auto compiled = regex_ir::compile(pattern, operation, options);
  if (!compiled) {
    ADD_FAILURE() << "compilation failed for " << pattern << ": "
                  << diagnostics_text(compiled.diagnostics);
    return {};
  }
  if (print_ir) std::cout << regex_ir::to_string(*compiled.value) << std::flush;
  return std::move(*compiled.value);
}

std::string cuda_error(CUresult result)
{
  char const* name    = nullptr;
  char const* message = nullptr;
  static_cast<void>(cuGetErrorName(result, &name));
  static_cast<void>(cuGetErrorString(result, &message));
  return std::string{name == nullptr ? "CUDA_ERROR_UNKNOWN" : name} + ": " +
         (message == nullptr ? "unknown error" : message);
}

void check_cuda(CUresult result, std::string_view operation)
{
  if (result != CUDA_SUCCESS) {
    throw std::runtime_error(std::string{operation} + " failed: " + cuda_error(result));
  }
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
    throw std::runtime_error(std::string{operation} + " failed: " + nvvm_log(program));
  }
}

std::vector<char> compile_nvvm_lto_ir(std::string const& source, std::string const& architecture)
{
  nvvmProgram program = nullptr;
  if (nvvmCreateProgram(&program) != NVVM_SUCCESS) {
    throw std::runtime_error("nvvmCreateProgram failed");
  }
  try {
    check_nvvm(
      nvvmAddModuleToProgram(program, source.data(), source.size(), "generated_regex.nvvm"),
      program,
      "nvvmAddModuleToProgram");
    std::string architecture_option = "-arch=" + architecture;
    std::vector<char const*> verify_options;
    std::vector<char const*> compile_options;
    if (!architecture.empty()) {
      verify_options.push_back(architecture_option.c_str());
      compile_options.push_back(architecture_option.c_str());
    }
    compile_options.push_back("-opt=3");
    compile_options.push_back("-gen-lto");
    check_nvvm(
      nvvmVerifyProgram(program, static_cast<int>(verify_options.size()), verify_options.data()),
      program,
      "nvvmVerifyProgram");
    check_nvvm(
      nvvmCompileProgram(program, static_cast<int>(compile_options.size()), compile_options.data()),
      program,
      "nvvmCompileProgram");
    std::size_t size = 0;
    check_nvvm(nvvmGetCompiledResultSize(program, &size), program, "nvvmGetCompiledResultSize");
    std::vector<char> lto_ir(size);
    check_nvvm(nvvmGetCompiledResult(program, lto_ir.data()), program, "nvvmGetCompiledResult");
    static_cast<void>(nvvmDestroyProgram(&program));
    return lto_ir;
  } catch (...) {
    static_cast<void>(nvvmDestroyProgram(&program));
    throw;
  }
}

std::string jitlink_log(nvJitLinkHandle linker)
{
  std::size_t size = 0;
  if (nvJitLinkGetErrorLogSize(linker, &size) != NVJITLINK_SUCCESS || size == 0) return {};
  std::string log(size, '\0');
  if (nvJitLinkGetErrorLog(linker, log.data()) != NVJITLINK_SUCCESS) return {};
  return log;
}

void check_jitlink(nvJitLinkResult result, nvJitLinkHandle linker, std::string_view operation)
{
  if (result != NVJITLINK_SUCCESS) {
    throw std::runtime_error(std::string{operation} + " failed: " + jitlink_log(linker));
  }
}

std::vector<char> link_lto_ir(std::span<char const> generated,
                              std::span<unsigned char const> kernel_fatbin,
                              std::string const& architecture)
{
  std::string architecture_option = "-arch=" + architecture;
  std::array<char const*, 3> options{architecture_option.c_str(), "-lto", "-O3"};
  nvJitLinkHandle linker = nullptr;
  if (nvJitLinkCreate(&linker, options.size(), options.data()) != NVJITLINK_SUCCESS) {
    throw std::runtime_error("nvJitLinkCreate failed");
  }
  try {
    check_jitlink(nvJitLinkAddData(linker,
                                   NVJITLINK_INPUT_LTOIR,
                                   const_cast<char*>(generated.data()),
                                   generated.size(),
                                   "generated_regex.ltoir"),
                  linker,
                  "nvJitLinkAddData(generated)");
    check_jitlink(nvJitLinkAddData(linker,
                                   NVJITLINK_INPUT_FATBIN,
                                   const_cast<unsigned char*>(kernel_fatbin.data()),
                                   kernel_fatbin.size(),
                                   "regex_ir_test_kernel.fatbin"),
                  linker,
                  "nvJitLinkAddData(kernel)");
    check_jitlink(nvJitLinkComplete(linker), linker, "nvJitLinkComplete");
    std::size_t size = 0;
    check_jitlink(
      nvJitLinkGetLinkedCubinSize(linker, &size), linker, "nvJitLinkGetLinkedCubinSize");
    std::vector<char> cubin(size);
    check_jitlink(nvJitLinkGetLinkedCubin(linker, cubin.data()), linker, "nvJitLinkGetLinkedCubin");
    static_cast<void>(nvJitLinkDestroy(&linker));
    return cubin;
  } catch (...) {
    static_cast<void>(nvJitLinkDestroy(&linker));
    throw;
  }
}

class device_allocation {
 public:
  explicit device_allocation(std::size_t bytes)
  {
    check_cuda(cuMemAlloc(&address_, std::max<std::size_t>(bytes, 1)), "cuMemAlloc");
  }

  device_allocation(device_allocation const&)            = delete;
  device_allocation& operator=(device_allocation const&) = delete;

  ~device_allocation()
  {
    if (address_ != 0) static_cast<void>(cuMemFree(address_));
  }

  [[nodiscard]] CUdeviceptr address() const { return address_; }

 private:
  CUdeviceptr address_ = 0;
};

class loaded_module {
 public:
  explicit loaded_module(std::vector<char> const& cubin)
  {
    check_cuda(cuModuleLoadData(&module_, cubin.data()), "cuModuleLoadData");
  }

  loaded_module(loaded_module const&)            = delete;
  loaded_module& operator=(loaded_module const&) = delete;

  ~loaded_module()
  {
    if (module_ != nullptr) static_cast<void>(cuModuleUnload(module_));
  }

  [[nodiscard]] CUfunction function(char const* name) const
  {
    CUfunction result = nullptr;
    check_cuda(cuModuleGetFunction(&result, module_, name), "cuModuleGetFunction");
    return result;
  }

 private:
  CUmodule module_ = nullptr;
};

enum class jit_kernel_kind : std::uint8_t {
  Boolean = 0,
  Extract = 1,
  Find    = 2,
  Count   = 3,
  Replace = 4,
  Split   = 5,
};

class gpu_runtime {
 public:
  gpu_runtime()
  {
    CUresult initialized = cuInit(0);
    if (initialized != CUDA_SUCCESS) {
      unavailable_reason_ = cuda_error(initialized);
      return;
    }
    int count = 0;
    check_cuda(cuDeviceGetCount(&count), "cuDeviceGetCount");
    if (count == 0) {
      unavailable_reason_ = "no CUDA device is visible";
      return;
    }
    check_cuda(cuDeviceGet(&device_, 0), "cuDeviceGet");
    check_cuda(cuCtxGetCurrent(&previous_), "cuCtxGetCurrent");
    check_cuda(cuDevicePrimaryCtxRetain(&context_, device_), "cuDevicePrimaryCtxRetain");
    check_cuda(cuCtxSetCurrent(context_), "cuCtxSetCurrent");
    check_cuda(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 64U * 1024U), "cuCtxSetLimit");
    int major = 0;
    int minor = 0;
    check_cuda(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_),
               "cuDeviceGetAttribute(major)");
    check_cuda(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_),
               "cuDeviceGetAttribute(minor)");
    compute_architecture_ = "compute_" + std::to_string(major) + std::to_string(minor);
    sm_architecture_      = "sm_" + std::to_string(major) + std::to_string(minor);
    available_            = true;
  }

  gpu_runtime(gpu_runtime const&)            = delete;
  gpu_runtime& operator=(gpu_runtime const&) = delete;

  ~gpu_runtime()
  {
    if (context_ != nullptr) {
      static_cast<void>(cuCtxSetCurrent(previous_));
      static_cast<void>(cuDevicePrimaryCtxRelease(device_));
    }
  }

  [[nodiscard]] bool available() const { return available_; }
  [[nodiscard]] std::string const& unavailable_reason() const { return unavailable_reason_; }

  [[nodiscard]] bool execute_boolean(regex_ir::instruction_ir const& ir, std::string_view input)
  {
    regex_ir::nvvm_ir_codegen_options options{
      .symbol_prefix    = "unit_boolean_" + std::to_string(next_id_++),
      .execute_function = "regex_ir_test_execute"};
    auto cubin = compile_and_link(ir, options, jit_kernel_kind::Boolean);
    loaded_module module{cubin};
    CUfunction kernel = module.function("regex_ir_boolean_kernel");
    device_allocation device_input{input.size()};
    device_allocation device_result{sizeof(int)};
    if (!input.empty()) {
      check_cuda(cuMemcpyHtoD(device_input.address(), input.data(), input.size()), "cuMemcpyHtoD");
    }
    int result = 0;
    check_cuda(cuMemcpyHtoD(device_result.address(), &result, sizeof(result)), "cuMemcpyHtoD");
    CUdeviceptr input_address  = device_input.address();
    std::size_t input_size     = input.size();
    CUdeviceptr result_address = device_result.address();
    std::array<void*, 3> arguments{&input_address, &input_size, &result_address};
    check_cuda(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, arguments.data(), nullptr),
               "cuLaunchKernel(boolean)");
    check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(boolean)");
    check_cuda(cuMemcpyDtoH(&result, device_result.address(), sizeof(result)), "cuMemcpyDtoH");
    return result != 0;
  }

  [[nodiscard]] std::vector<regex_ir::testing::execution_result> enumerate(
    regex_ir::instruction_ir const& ir, std::span<maybe_string const> inputs)
  {
    if (inputs.empty()) return {};
    regex_ir::nvvm_ir_codegen_options options{
      .symbol_prefix    = "unit_capture_" + std::to_string(next_id_++),
      .execute_function = "regex_ir_test_find_execute"};
    auto cubin = compile_and_link(ir, options, jit_kernel_kind::Extract);
    return execute_capture(cubin, inputs, ir.capture_count);
  }

  [[nodiscard]] regex_ir::testing::execution_result find(regex_ir::instruction_ir const& ir,
                                                         std::string_view input)
  {
    regex_ir::nvvm_ir_codegen_options options{
      .symbol_prefix    = "unit_find_" + std::to_string(next_id_++),
      .execute_function = "regex_ir_test_find"};
    auto cubin = compile_and_link(ir, options, jit_kernel_kind::Find);
    loaded_module module{cubin};
    CUfunction kernel = module.function("regex_ir_find_kernel");
    device_allocation device_input{input.size()};
    device_allocation device_span{2U * sizeof(std::uint64_t)};
    device_allocation device_result{sizeof(int)};
    if (!input.empty()) {
      check_cuda(cuMemcpyHtoD(device_input.address(), input.data(), input.size()), "cuMemcpyHtoD");
    }
    std::array<std::uint64_t, 2> span{};
    int matched = 0;
    check_cuda(cuMemcpyHtoD(device_span.address(), span.data(), sizeof(span)),
               "cuMemcpyHtoD(span)");
    check_cuda(cuMemcpyHtoD(device_result.address(), &matched, sizeof(matched)),
               "cuMemcpyHtoD(result)");
    CUdeviceptr input_address  = device_input.address();
    std::size_t input_size     = input.size();
    CUdeviceptr span_address   = device_span.address();
    CUdeviceptr result_address = device_result.address();
    std::array<void*, 4> arguments{&input_address, &input_size, &span_address, &result_address};
    check_cuda(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, arguments.data(), nullptr),
               "cuLaunchKernel(find)");
    check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(find)");
    check_cuda(cuMemcpyDtoH(&matched, device_result.address(), sizeof(matched)),
               "cuMemcpyDtoH(result)");
    check_cuda(cuMemcpyDtoH(span.data(), device_span.address(), sizeof(span)),
               "cuMemcpyDtoH(span)");
    regex_ir::testing::execution_result result;
    result.matched = matched != 0;
    result.count   = result.matched ? 1U : 0U;
    if (result.matched) result.matches.push_back({span[0], span[1]});
    return result;
  }

  [[nodiscard]] regex_ir::testing::execution_result count(regex_ir::instruction_ir const& ir,
                                                          std::string_view input)
  {
    regex_ir::nvvm_ir_codegen_options options{
      .symbol_prefix    = "unit_count_" + std::to_string(next_id_++),
      .execute_function = "regex_ir_test_count"};
    auto cubin = compile_and_link(ir, options, jit_kernel_kind::Count);
    loaded_module module{cubin};
    CUfunction kernel = module.function("regex_ir_count_kernel");
    device_allocation device_input{input.size()};
    device_allocation device_result{sizeof(std::uint64_t)};
    if (!input.empty()) {
      check_cuda(cuMemcpyHtoD(device_input.address(), input.data(), input.size()), "cuMemcpyHtoD");
    }
    CUdeviceptr input_address  = device_input.address();
    std::size_t input_size     = input.size();
    CUdeviceptr result_address = device_result.address();
    std::array<void*, 3> arguments{&input_address, &input_size, &result_address};
    check_cuda(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, arguments.data(), nullptr),
               "cuLaunchKernel(count)");
    check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(count)");
    regex_ir::testing::execution_result result;
    check_cuda(cuMemcpyDtoH(&result.count, device_result.address(), sizeof(result.count)),
               "cuMemcpyDtoH(count)");
    result.matched = result.count != 0;
    return result;
  }

  [[nodiscard]] std::string replace(regex_ir::instruction_ir const& ir, std::string_view input)
  {
    regex_ir::nvvm_ir_codegen_options options{
      .symbol_prefix    = "unit_replace_" + std::to_string(next_id_++),
      .execute_function = "regex_ir_test_replace"};
    auto cubin = compile_and_link(ir, options, jit_kernel_kind::Replace);
    loaded_module module{cubin};
    CUfunction kernel = module.function("regex_ir_replace_kernel");
    device_allocation device_input{input.size()};
    device_allocation device_size{sizeof(std::uint64_t)};
    if (!input.empty()) {
      check_cuda(cuMemcpyHtoD(device_input.address(), input.data(), input.size()), "cuMemcpyHtoD");
    }
    CUdeviceptr input_address = device_input.address();
    std::size_t input_size    = input.size();
    CUdeviceptr size_address  = device_size.address();
    auto launch               = [&](CUdeviceptr output_address) {
      std::array<void*, 4> arguments{&input_address, &input_size, &output_address, &size_address};
      check_cuda(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, arguments.data(), nullptr),
                 "cuLaunchKernel(replace)");
      check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(replace)");
    };
    launch(0);
    std::uint64_t result_size = 0;
    check_cuda(cuMemcpyDtoH(&result_size, device_size.address(), sizeof(result_size)),
               "cuMemcpyDtoH(replacement size)");
    device_allocation device_output{result_size};
    launch(device_output.address());
    std::string result(result_size, '\0');
    if (!result.empty()) {
      check_cuda(cuMemcpyDtoH(result.data(), device_output.address(), result.size()),
                 "cuMemcpyDtoH(replacement)");
    }
    return result;
  }

  [[nodiscard]] std::vector<std::string> split(regex_ir::instruction_ir const& ir,
                                               std::string_view input)
  {
    regex_ir::nvvm_ir_codegen_options options{
      .symbol_prefix    = "unit_split_" + std::to_string(next_id_++),
      .execute_function = "regex_ir_test_split"};
    auto cubin = compile_and_link(ir, options, jit_kernel_kind::Split);
    loaded_module module{cubin};
    CUfunction kernel = module.function("regex_ir_split_kernel");
    device_allocation device_input{input.size()};
    device_allocation device_count{sizeof(std::uint64_t)};
    if (!input.empty()) {
      check_cuda(cuMemcpyHtoD(device_input.address(), input.data(), input.size()), "cuMemcpyHtoD");
    }
    CUdeviceptr input_address = device_input.address();
    std::size_t input_size    = input.size();
    CUdeviceptr count_address = device_count.address();
    auto launch               = [&](CUdeviceptr span_address) {
      std::array<void*, 4> arguments{&input_address, &input_size, &span_address, &count_address};
      check_cuda(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, arguments.data(), nullptr),
                 "cuLaunchKernel(split)");
      check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(split)");
    };
    launch(0);
    std::uint64_t field_count = 0;
    check_cuda(cuMemcpyDtoH(&field_count, device_count.address(), sizeof(field_count)),
               "cuMemcpyDtoH(field count)");
    std::vector<std::uint64_t> spans(field_count * 2U);
    device_allocation device_spans{spans.size() * sizeof(std::uint64_t)};
    launch(device_spans.address());
    if (!spans.empty()) {
      check_cuda(
        cuMemcpyDtoH(spans.data(), device_spans.address(), spans.size() * sizeof(spans[0])),
        "cuMemcpyDtoH(split spans)");
    }
    std::vector<std::string> result;
    result.reserve(field_count);
    for (std::size_t field = 0; field < field_count; ++field) {
      auto begin = spans[field * 2U];
      auto end   = spans[field * 2U + 1U];
      result.emplace_back(input.substr(begin, end - begin));
    }
    return result;
  }

 private:
  [[nodiscard]] std::vector<char> compile_and_link(regex_ir::instruction_ir const& ir,
                                                   regex_ir::nvvm_ir_codegen_options const& options,
                                                   jit_kernel_kind kind) const
  {
    std::string source = regex_ir::generate_nvvm_ir(ir, options);
    auto lto_ir        = compile_nvvm_lto_ir(source, compute_architecture_);
    std::span<unsigned char const> kernel_fatbin;
    switch (kind) {
      case jit_kernel_kind::Boolean:
        kernel_fatbin = {regex_ir_boolean_kernel_fatbin, regex_ir_boolean_kernel_fatbinLength};
        break;
      case jit_kernel_kind::Extract:
        kernel_fatbin = {regex_ir_capture_kernel_fatbin, regex_ir_capture_kernel_fatbinLength};
        break;
      case jit_kernel_kind::Find:
        kernel_fatbin = {regex_ir_find_kernel_fatbin, regex_ir_find_kernel_fatbinLength};
        break;
      case jit_kernel_kind::Count:
        kernel_fatbin = {regex_ir_count_kernel_fatbin, regex_ir_count_kernel_fatbinLength};
        break;
      case jit_kernel_kind::Replace:
        kernel_fatbin = {regex_ir_replace_kernel_fatbin, regex_ir_replace_kernel_fatbinLength};
        break;
      case jit_kernel_kind::Split:
        kernel_fatbin = {regex_ir_split_kernel_fatbin, regex_ir_split_kernel_fatbinLength};
        break;
    }
    return link_lto_ir(lto_ir, kernel_fatbin, sm_architecture_);
  }

  [[nodiscard]] std::vector<regex_ir::testing::execution_result> execute_capture(
    std::vector<char> const& cubin,
    std::span<maybe_string const> inputs,
    std::uint32_t capture_count) const
  {
    std::vector<char> characters;
    std::vector<std::size_t> offsets;
    offsets.reserve(inputs.size() + 1);
    offsets.push_back(0);
    std::size_t max_matches = 1;
    for (maybe_string const& input : inputs) {
      if (input) {
        characters.insert(characters.end(), input->begin(), input->end());
        max_matches = std::max(max_matches, input->size() + 1U);
      }
      offsets.push_back(characters.size());
    }

    auto capture_slots  = static_cast<std::size_t>(capture_count + 1U) * 2U;
    auto capture_values = inputs.size() * max_matches * capture_slots;
    loaded_module module{cubin};
    CUfunction kernel = module.function("regex_ir_capture_kernel");
    device_allocation device_characters{characters.size()};
    device_allocation device_offsets{offsets.size() * sizeof(std::size_t)};
    device_allocation device_captures{capture_values * sizeof(std::size_t)};
    device_allocation device_counts{inputs.size() * sizeof(std::size_t)};
    device_allocation device_overflows{inputs.size() * sizeof(int)};
    if (!characters.empty()) {
      check_cuda(cuMemcpyHtoD(device_characters.address(), characters.data(), characters.size()),
                 "cuMemcpyHtoD(characters)");
    }
    check_cuda(
      cuMemcpyHtoD(device_offsets.address(), offsets.data(), offsets.size() * sizeof(std::size_t)),
      "cuMemcpyHtoD(offsets)");
    std::vector<std::size_t> capture_storage(capture_values,
                                             std::numeric_limits<std::size_t>::max());
    std::vector<std::size_t> counts(inputs.size(), 0);
    std::vector<int> overflows(inputs.size(), 0);
    check_cuda(cuMemcpyHtoD(device_captures.address(),
                            capture_storage.data(),
                            capture_storage.size() * sizeof(std::size_t)),
               "cuMemcpyHtoD(captures)");
    check_cuda(
      cuMemcpyHtoD(device_counts.address(), counts.data(), counts.size() * sizeof(std::size_t)),
      "cuMemcpyHtoD(counts)");
    check_cuda(
      cuMemcpyHtoD(device_overflows.address(), overflows.data(), overflows.size() * sizeof(int)),
      "cuMemcpyHtoD(overflows)");

    CUdeviceptr character_address = device_characters.address();
    CUdeviceptr offset_address    = device_offsets.address();
    std::size_t row_count         = inputs.size();
    CUdeviceptr capture_address   = device_captures.address();
    CUdeviceptr count_address     = device_counts.address();
    CUdeviceptr overflow_address  = device_overflows.address();
    std::array<void*, 8> arguments{&character_address,
                                   &offset_address,
                                   &row_count,
                                   &capture_slots,
                                   &max_matches,
                                   &capture_address,
                                   &count_address,
                                   &overflow_address};
    auto grid_size = static_cast<unsigned int>(row_count);
    check_cuda(
      cuLaunchKernel(kernel, grid_size, 1, 1, 1, 1, 1, 0, nullptr, arguments.data(), nullptr),
      "cuLaunchKernel(captures)");
    check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(captures)");
    check_cuda(cuMemcpyDtoH(capture_storage.data(),
                            device_captures.address(),
                            capture_storage.size() * sizeof(std::size_t)),
               "cuMemcpyDtoH(captures)");
    check_cuda(
      cuMemcpyDtoH(counts.data(), device_counts.address(), counts.size() * sizeof(std::size_t)),
      "cuMemcpyDtoH(counts)");
    check_cuda(
      cuMemcpyDtoH(overflows.data(), device_overflows.address(), overflows.size() * sizeof(int)),
      "cuMemcpyDtoH(overflows)");

    std::vector<regex_ir::testing::execution_result> results(inputs.size());
    for (std::size_t row = 0; row < inputs.size(); ++row) {
      if (overflows[row] != 0) throw std::runtime_error("capture capacity exceeded");
      auto& result   = results[row];
      result.matched = counts[row] != 0;
      result.count   = counts[row];
      for (std::size_t match = 0; match < counts[row]; ++match) {
        auto base = (row * max_matches + match) * capture_slots;
        result.matches.push_back({capture_storage[base], capture_storage[base + 1]});
        std::vector<std::optional<regex_ir::testing::match_span>> captures;
        captures.reserve(capture_count + 1U);
        for (std::uint32_t capture = 0; capture <= capture_count; ++capture) {
          auto begin = capture_storage[base + static_cast<std::size_t>(capture) * 2U];
          auto end   = capture_storage[base + static_cast<std::size_t>(capture) * 2U + 1U];
          if (begin == std::numeric_limits<std::size_t>::max() ||
              end == std::numeric_limits<std::size_t>::max()) {
            captures.push_back(std::nullopt);
          } else {
            captures.push_back(regex_ir::testing::match_span{begin, end});
          }
        }
        result.capture_matches.push_back(std::move(captures));
      }
      if (!result.capture_matches.empty()) result.captures = result.capture_matches.front();
    }
    return results;
  }

  bool available_                   = false;
  CUdevice device_                  = 0;
  CUcontext context_                = nullptr;
  CUcontext previous_               = nullptr;
  std::string compute_architecture_ = "";
  std::string sm_architecture_      = "";
  std::string unavailable_reason_   = "CUDA support was not initialized";
  std::size_t next_id_              = 0;
};

gpu_runtime& gpu()
{
  static gpu_runtime runtime;
  return runtime;
}

}  // namespace

// shared fixture and backend assertions

void expect_results_equal(regex_ir::testing::execution_result const& left,
                          regex_ir::testing::execution_result const& right)
{
  EXPECT_EQ(left.matched, right.matched);
  EXPECT_EQ(left.count, right.count);
  EXPECT_EQ(left.matches, right.matches);
  EXPECT_EQ(left.captures, right.captures);
  EXPECT_EQ(left.capture_matches, right.capture_matches);
}

void expect_rows_equal(std::span<cudf_expected_row const> actual,
                       std::span<cudf_expected_row const> expected)
{
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t row = 0; row < actual.size(); ++row) {
    SCOPED_TRACE("row " + std::to_string(row));
    EXPECT_EQ(actual[row].valid, expected[row].valid);
    EXPECT_EQ(actual[row].scalar, expected[row].scalar);
    EXPECT_EQ(actual[row].strings, expected[row].strings);
  }
}

template <typename Case, typename Predicate, typename Action>
std::size_t run_cases(std::span<Case const> cases, Predicate predicate, Action action)
{
  std::size_t count = 0;
  for (Case const& test : cases) {
    if (!predicate(test)) continue;
    action(test);
    ++count;
  }
  return count;
}

class RegexTest : public ::testing::Test {
 protected:
  bool execute_boolean_both(std::string_view pattern,
                            std::string_view input,
                            regex_ir::operation_kind kind,
                            regex_ir::compile_options const& options)
  {
    regex_ir::operation selected = kind == regex_ir::operation_kind::MATCHES
                                     ? regex_ir::operation::matches()
                                     : regex_ir::operation::contains();
    auto compiled                = regex_ir::compile(pattern, selected, options);
    if (!compiled) {
      ADD_FAILURE() << diagnostics_text(compiled.diagnostics);
      return false;
    }
    bool cpu = regex_ir::testing::execute(*compiled.value, input).matched;
    if (gpu().available()) {
      EXPECT_EQ(gpu().execute_boolean(*compiled.value, input), cpu) << "input: " << input;
    }
    return cpu;
  }

  void run_boolean_case(boolean_test_case const& test)
  {
    SCOPED_TRACE(std::string{test.name} + " / " + std::string{test.pattern});
    EXPECT_EQ(execute_boolean_both(test.pattern, test.input, test.operation, test.options),
              test.expected)
      << "input: " << test.input;
  }

  void run_boolean_suite(test_suite suite)
  {
    auto count = run_cases(
      boolean_test_cases(),
      [suite](boolean_test_case const& test) { return test.suite == suite; },
      [this](boolean_test_case const& test) { run_boolean_case(test); });
    EXPECT_NE(count, 0U);
  }

  std::pair<regex_ir::instruction_ir, std::vector<regex_ir::testing::execution_result>>
  enumerate_both(std::string_view pattern,
                 regex_ir::compile_options const& options,
                 std::span<maybe_string const> inputs)
  {
    auto compiled = regex_ir::compile(pattern, regex_ir::operation::extract(), options);
    EXPECT_TRUE(compiled) << diagnostics_text(compiled.diagnostics);
    if (!compiled) return {};

    std::vector<regex_ir::testing::execution_result> cpu;
    cpu.reserve(inputs.size());
    for (maybe_string const& input : inputs) {
      cpu.push_back(input ? regex_ir::testing::enumerate(*compiled.value, *input)
                          : regex_ir::testing::execution_result{});
    }
    if (gpu().available()) {
      auto device = gpu().enumerate(*compiled.value, inputs);
      EXPECT_EQ(device.size(), cpu.size());
      if (device.size() != cpu.size()) return {std::move(*compiled.value), std::move(cpu)};
      for (std::size_t row = 0; row < cpu.size(); ++row) {
        if (!inputs[row]) continue;
        SCOPED_TRACE("backend row " + std::to_string(row));
        expect_results_equal(device[row], cpu[row]);
      }
    }
    return {std::move(*compiled.value), std::move(cpu)};
  }

  void run_cudf_test(std::string_view name)
  {
    auto compile_count = run_cases(
      cudf_compile_cases(),
      [name](cudf_compile_case const& test) { return test.test_name == name; },
      [](cudf_compile_case const& test) {
        SCOPED_TRACE(test.pattern);
        auto compiled =
          regex_ir::compile(test.pattern, regex_ir::operation::extract(), test.options);
        EXPECT_EQ(static_cast<bool>(compiled), test.should_compile)
          << diagnostics_text(compiled.diagnostics);
      });
    auto execution_count = run_cases(
      cudf_regex_cases(),
      [name](cudf_regex_case const& test) { return test.test_name == name; },
      [this](cudf_regex_case const& test) {
        SCOPED_TRACE(test.assertion + " / " + test.pattern);
        auto [ir, cpu_matches] = enumerate_both(test.pattern, test.options, test.inputs);
        if (ir.blocks.empty()) return;
        bool valid = cudf_operation_is_valid(test, ir);
        EXPECT_EQ(valid, !test.expect_operation_error);
        if (!valid) return;
        auto cpu = evaluate_cudf_case(test, ir, cpu_matches);
        expect_rows_equal(cpu, test.expected);
        if (test.expected_columns != 0) {
          EXPECT_EQ(cudf_output_columns(test, ir, cpu), test.expected_columns);
        }
      });
    EXPECT_NE(compile_count + execution_count, 0U) << name;
  }

  void expect_boolean(std::string_view pattern,
                      std::string_view input,
                      bool expected,
                      regex_ir::operation_kind kind            = regex_ir::operation_kind::CONTAINS,
                      regex_ir::compile_options const& options = {})
  {
    boolean_test_case test{"inline", test_suite::PROJECT, pattern, input, kind, options, expected};
    run_boolean_case(test);
  }

  void expect_backends_agree(std::string_view pattern,
                             std::string_view input,
                             regex_ir::compile_options const& options = {})
  {
    static_cast<void>(
      execute_boolean_both(pattern, input, regex_ir::operation_kind::CONTAINS, options));
  }

  regex_ir::testing::execution_result expect_enumeration(
    std::string_view pattern, std::string_view input, regex_ir::compile_options const& options = {})
  {
    std::array<maybe_string, 1> inputs{std::string{input}};
    auto [ir, results] = enumerate_both(pattern, options, inputs);
    static_cast<void>(ir);
    EXPECT_EQ(results.size(), 1U);
    return results.empty() ? regex_ir::testing::execution_result{} : std::move(results.front());
  }
};

class Project : public RegexTest {};
class Re2 : public RegexTest {};
class RustRegex : public RegexTest {};
class CPython : public RegexTest {};
class Sihlfall : public RegexTest {};
class Cudf : public RegexTest {};
class Nvvm : public RegexTest {};

// source-suite registrations

TEST_F(Project, ImportedCases) { run_boolean_suite(test_suite::PROJECT); }

TEST_F(Re2, ImportedCases) { run_boolean_suite(test_suite::RE2); }

TEST_F(RustRegex, ImportedCases) { run_boolean_suite(test_suite::RUST_REGEX); }

TEST_F(CPython, ImportedCases) { run_boolean_suite(test_suite::CPYTHON); }

TEST_F(Sihlfall, ImportedCases) { run_boolean_suite(test_suite::SIHLFALL); }

#define REGEX_IR_CUDF_TEST(Name, TestName) \
  TEST_F(Cudf, Name) { run_cudf_test(TestName); }

REGEX_IR_CUDF_TEST(Contains_ContainsTest, "StringsContainsTests.ContainsTest")
REGEX_IR_CUDF_TEST(Contains_MatchesTest, "StringsContainsTests.MatchesTest")
REGEX_IR_CUDF_TEST(Contains_MatchesIPV4Test, "StringsContainsTests.MatchesIPV4Test")
REGEX_IR_CUDF_TEST(Contains_OctalTest, "StringsContainsTests.OctalTest")
REGEX_IR_CUDF_TEST(Contains_HexTest, "StringsContainsTests.HexTest")
REGEX_IR_CUDF_TEST(Contains_EmbeddedNullCharacter, "StringsContainsTests.EmbeddedNullCharacter")
REGEX_IR_CUDF_TEST(Contains_Errors, "StringsContainsTests.Errors")
REGEX_IR_CUDF_TEST(Contains_CountTest, "StringsContainsTests.CountTest")
REGEX_IR_CUDF_TEST(Contains_CountEmptyMatching, "StringsContainsTests.CountEmptyMatching")
REGEX_IR_CUDF_TEST(Contains_FixedQuantifier, "StringsContainsTests.FixedQuantifier")
REGEX_IR_CUDF_TEST(Contains_ZeroRangeQuantifier, "StringsContainsTests.ZeroRangeQuantifier")
REGEX_IR_CUDF_TEST(Contains_NestedQuantifier, "StringsContainsTests.NestedQuantifier")
REGEX_IR_CUDF_TEST(Contains_QuantifierErrors, "StringsContainsTests.QuantifierErrors")
REGEX_IR_CUDF_TEST(Contains_OverlappedClasses, "StringsContainsTests.OverlappedClasses")
REGEX_IR_CUDF_TEST(Contains_NegatedClasses, "StringsContainsTests.NegatedClasses")
REGEX_IR_CUDF_TEST(Contains_IncompleteClassesRange, "StringsContainsTests.IncompleteClassesRange")
REGEX_IR_CUDF_TEST(Contains_MultiLine, "StringsContainsTests.MultiLine")
REGEX_IR_CUDF_TEST(Contains_SpecialNewLines, "StringsContainsTests.SpecialNewLines")
REGEX_IR_CUDF_TEST(Contains_EndOfString, "StringsContainsTests.EndOfString")
REGEX_IR_CUDF_TEST(Contains_DotAll, "StringsContainsTests.DotAll")
REGEX_IR_CUDF_TEST(Contains_ASCII, "StringsContainsTests.ASCII")
REGEX_IR_CUDF_TEST(Contains_IgnoreCase, "StringsContainsTests.IgnoreCase")
REGEX_IR_CUDF_TEST(Contains_MediumRegex, "StringsContainsTests.MediumRegex")
REGEX_IR_CUDF_TEST(Contains_LargeRegex, "StringsContainsTests.LargeRegex")
REGEX_IR_CUDF_TEST(Contains_ExtraLargeRegex, "StringsContainsTests.ExtraLargeRegex")
REGEX_IR_CUDF_TEST(Contains_CrlfLineAnchorExtNewline,
                   "StringsContainsTests.CrlfLineAnchorExtNewline")
REGEX_IR_CUDF_TEST(Contains_CrlfBolAnchorExtNewline, "StringsContainsTests.CrlfBolAnchorExtNewline")
REGEX_IR_CUDF_TEST(Contains_CrlfEdgeCasesExtNewline, "StringsContainsTests.CrlfEdgeCasesExtNewline")
REGEX_IR_CUDF_TEST(Contains_CrlfDefaultLfOnlyNoExtNewline,
                   "StringsContainsTests.CrlfDefaultLfOnlyNoExtNewline")
REGEX_IR_CUDF_TEST(Contains_AlternationNullableBranch,
                   "StringsContainsTests.AlternationNullableBranch")
REGEX_IR_CUDF_TEST(Contains_BoundedRepetitionGap, "StringsContainsTests.BoundedRepetitionGap")
REGEX_IR_CUDF_TEST(Contains_ExtNewlineDotAny, "StringsContainsTests.ExtNewlineDotAny")
REGEX_IR_CUDF_TEST(Contains_AlternationPriorityCount,
                   "StringsContainsTests.AlternationPriorityCount")
REGEX_IR_CUDF_TEST(Contains_LazyQuantifiers, "StringsContainsTests.LazyQuantifiers")

REGEX_IR_CUDF_TEST(Findall_FindallTest, "StringsFindallTests.FindallTest")
REGEX_IR_CUDF_TEST(Findall_Multiline, "StringsFindallTests.Multiline")
REGEX_IR_CUDF_TEST(Findall_DotAll, "StringsFindallTests.DotAll")
REGEX_IR_CUDF_TEST(Findall_SpecialNewLines, "StringsFindallTests.SpecialNewLines")
REGEX_IR_CUDF_TEST(Findall_MediumRegex, "StringsFindallTests.MediumRegex")
REGEX_IR_CUDF_TEST(Findall_LargeRegex, "StringsFindallTests.LargeRegex")
REGEX_IR_CUDF_TEST(Findall_FindTest, "StringsFindallTests.FindTest")
REGEX_IR_CUDF_TEST(Findall_NoMatches, "StringsFindallTests.NoMatches")
REGEX_IR_CUDF_TEST(Findall_EmptyTest, "StringsFindallTests.EmptyTest")
REGEX_IR_CUDF_TEST(Findall_OneCaptureGroup, "StringsFindallTests.OneCaptureGroup")
REGEX_IR_CUDF_TEST(Findall_AlternationPriorityFirstWins,
                   "StringsFindallTests.AlternationPriorityFirstWins")
REGEX_IR_CUDF_TEST(Findall_EmptyMatch, "StringsFindallTests.EmptyMatch")
REGEX_IR_CUDF_TEST(Findall_Errors, "StringsFindallTests.Errors")

REGEX_IR_CUDF_TEST(Extract_ExtractTest, "StringsExtractTests.ExtractTest")
REGEX_IR_CUDF_TEST(Extract_ExtractDomainTest, "StringsExtractTests.ExtractDomainTest")
REGEX_IR_CUDF_TEST(Extract_ExtractEventTest, "StringsExtractTests.ExtractEventTest")
REGEX_IR_CUDF_TEST(Extract_MultiLine, "StringsExtractTests.MultiLine")
REGEX_IR_CUDF_TEST(Extract_DotAll, "StringsExtractTests.DotAll")
REGEX_IR_CUDF_TEST(Extract_SpecialNewLines, "StringsExtractTests.SpecialNewLines")
REGEX_IR_CUDF_TEST(Extract_NestedQuantifier, "StringsExtractTests.NestedQuantifier")
REGEX_IR_CUDF_TEST(Extract_EmptyExtractTest, "StringsExtractTests.EmptyExtractTest")
REGEX_IR_CUDF_TEST(Extract_ExtractAllTest, "StringsExtractTests.ExtractAllTest")
REGEX_IR_CUDF_TEST(Extract_ExtractSingle, "StringsExtractTests.ExtractSingle")
REGEX_IR_CUDF_TEST(Extract_Errors, "StringsExtractTests.Errors")
REGEX_IR_CUDF_TEST(Extract_EmptyInput, "StringsExtractTests.EmptyInput")
REGEX_IR_CUDF_TEST(Extract_MediumRegex, "StringsExtractTests.MediumRegex")
REGEX_IR_CUDF_TEST(Extract_LargeRegex, "StringsExtractTests.LargeRegex")
REGEX_IR_CUDF_TEST(Extract_CrlfLineAnchorExtNewline, "StringsExtractTests.CrlfLineAnchorExtNewline")

REGEX_IR_CUDF_TEST(Replace_ReplaceRegexTest, "StringsReplaceRegexTest.ReplaceRegexTest")
REGEX_IR_CUDF_TEST(Replace_InvalidRegex, "StringsReplaceRegexTest.InvalidRegex")
REGEX_IR_CUDF_TEST(Replace_WithEmptyPattern, "StringsReplaceRegexTest.WithEmptyPattern")
REGEX_IR_CUDF_TEST(Replace_MultiReplacement, "StringsReplaceRegexTest.MultiReplacement")
REGEX_IR_CUDF_TEST(Replace_WordBoundary, "StringsReplaceRegexTest.WordBoundary")
REGEX_IR_CUDF_TEST(Replace_Alternation, "StringsReplaceRegexTest.Alternation")
REGEX_IR_CUDF_TEST(Replace_ZeroLengthMatch, "StringsReplaceRegexTest.ZeroLengthMatch")
REGEX_IR_CUDF_TEST(Replace_ZeroRangeQuantifier, "StringsReplaceRegexTest.ZeroRangeQuantifier")
REGEX_IR_CUDF_TEST(Replace_Multiline, "StringsReplaceRegexTest.Multiline")
REGEX_IR_CUDF_TEST(Replace_SpecialNewLines, "StringsReplaceRegexTest.SpecialNewLines")
REGEX_IR_CUDF_TEST(Replace_ReplaceBackrefsRegexTest,
                   "StringsReplaceRegexTest.ReplaceBackrefsRegexTest")
REGEX_IR_CUDF_TEST(Replace_ReplaceBackrefsRegexAltIndexPatternTest,
                   "StringsReplaceRegexTest.ReplaceBackrefsRegexAltIndexPatternTest")
REGEX_IR_CUDF_TEST(Replace_ReplaceBackrefsRegexReversedTest,
                   "StringsReplaceRegexTest.ReplaceBackrefsRegexReversedTest")
REGEX_IR_CUDF_TEST(Replace_BackrefWithGreedyQuantifier,
                   "StringsReplaceRegexTest.BackrefWithGreedyQuantifier")
REGEX_IR_CUDF_TEST(Replace_ReplaceBackrefsRegexZeroIndexTest,
                   "StringsReplaceRegexTest.ReplaceBackrefsRegexZeroIndexTest")
REGEX_IR_CUDF_TEST(Replace_ReplaceBackrefsWithEmptyCapture,
                   "StringsReplaceRegexTest.ReplaceBackrefsWithEmptyCapture")
REGEX_IR_CUDF_TEST(Replace_ReplaceBackrefsRegexErrorTest,
                   "StringsReplaceRegexTest.ReplaceBackrefsRegexErrorTest")
REGEX_IR_CUDF_TEST(Replace_MediumReplaceRegex, "StringsReplaceRegexTest.MediumReplaceRegex")
REGEX_IR_CUDF_TEST(Replace_LargeReplaceRegex, "StringsReplaceRegexTest.LargeReplaceRegex")
REGEX_IR_CUDF_TEST(Replace_CrlfLineAnchorExtNewline,
                   "StringsReplaceRegexTest.CrlfLineAnchorExtNewline")
REGEX_IR_CUDF_TEST(Replace_CrlfEdgeCasesExtNewline,
                   "StringsReplaceRegexTest.CrlfEdgeCasesExtNewline")
REGEX_IR_CUDF_TEST(Replace_AlternationPriorityFirstWins,
                   "StringsReplaceRegexTest.AlternationPriorityFirstWins")

REGEX_IR_CUDF_TEST(Split_SplitRegex, "StringsSplitTest.SplitRegex")
REGEX_IR_CUDF_TEST(Split_SplitRecordRegex, "StringsSplitTest.SplitRecordRegex")
REGEX_IR_CUDF_TEST(Split_SplitRegexWithMaxSplit, "StringsSplitTest.SplitRegexWithMaxSplit")
REGEX_IR_CUDF_TEST(Split_SplitRegexWordBoundary, "StringsSplitTest.SplitRegexWordBoundary")
REGEX_IR_CUDF_TEST(Split_SplitRegexAllEmpty, "StringsSplitTest.SplitRegexAllEmpty")
REGEX_IR_CUDF_TEST(Split_RSplitRegexWithMaxSplit, "StringsSplitTest.RSplitRegexWithMaxSplit")
REGEX_IR_CUDF_TEST(Split_SplitZeroSizeStringsColumns,
                   "StringsSplitTest.SplitZeroSizeStringsColumns")
REGEX_IR_CUDF_TEST(Split_InvalidParameter, "StringsSplitTest.InvalidParameter")

#undef REGEX_IR_CUDF_TEST

// focused semantic and code-generation checks

TEST_F(Cudf, CompleteInventory)
{
  std::set<std::string> represented;
  std::size_t rows = 0;
  for (cudf_regex_case const& test : cudf_regex_cases()) {
    represented.insert(test.test_name);
    rows += test.inputs.size();
  }
  for (cudf_compile_case const& test : cudf_compile_cases()) {
    represented.insert(test.test_name);
  }
  EXPECT_EQ(cudf_regex_cases().size(), 543U);
  EXPECT_EQ(cudf_compile_cases().size(), 30U);
  EXPECT_EQ(rows, 34185U);
  EXPECT_EQ(represented.size(), 92U);
}

TEST_F(Project, Printers)
{
  auto automata = regex_ir::compile_automata("a(b|c)+");
  ASSERT_TRUE(automata);
  std::string automata_text = regex_ir::to_string(*automata.value);
  EXPECT_NE(automata_text.find("automata pattern="), std::string::npos);
  EXPECT_NE(automata_text.find("branch"), std::string::npos);

  auto instructions = regex_ir::compile("a(b|c)+", regex_ir::operation::find());
  ASSERT_TRUE(instructions);
  std::string instruction_text = regex_ir::to_string(*instructions.value);
  EXPECT_NE(instruction_text.find("instruction_ir operation=find"), std::string::npos);
  EXPECT_NE(instruction_text.find("metrics blocks="), std::string::npos);
  expect_boolean("a(b|c)+", "xxabcb", true);
}

TEST_F(Project, MatchingAndOptimization)
{
  expect_boolean("abc*", "ab", true, regex_ir::operation_kind::MATCHES);
  expect_boolean("abc*", "abccc", true, regex_ir::operation_kind::MATCHES);
  expect_boolean("abc*", "zabccc", false, regex_ir::operation_kind::MATCHES);
  expect_boolean("abc[0-9]", "abc7", true, regex_ir::operation_kind::MATCHES);
  expect_boolean("abc[0-9]", "abcz", false, regex_ir::operation_kind::MATCHES);
  expect_boolean("abcdefghijklmnop", "xxxxxxxabcdefghijklmnopy", true);
  expect_boolean("abcdefghijklmnop", "aaaaaaaabcdefghijklmnox", false);

  auto ir = compile_ok("abc*", regex_ir::operation::matches());
  EXPECT_TRUE(regex_ir::verify(ir).empty());
  EXPECT_NE(regex_ir::to_string(ir).find("match_literal"), std::string::npos);
}

TEST_F(Project, SyntaxPriorityAndSpans)
{
  expect_boolean("^(ab|cd)+$", "abcdab", true, regex_ir::operation_kind::MATCHES);
  expect_boolean("^(ab|cd)+$", "abce", false, regex_ir::operation_kind::MATCHES);

  auto lazy = expect_enumeration("a+?", "aaa");
  ASSERT_FALSE(lazy.matches.empty());
  EXPECT_EQ(lazy.matches.front(), (regex_ir::testing::match_span{0, 1}));
  expect_boolean(R"REGEX(\bcat\b)REGEX", "a cat!", true);
  expect_boolean(R"REGEX(\bcat\b)REGEX", "scatter", false);
}

TEST_F(Project, AllOperations)
{
  expect_boolean("a+", "xxaa", true);

  auto found = expect_enumeration("a+", "xxaaay");
  ASSERT_FALSE(found.matches.empty());
  EXPECT_EQ(found.matches.front(), (regex_ir::testing::match_span{2, 5}));

  auto counted = expect_enumeration("a+", "baacaa");
  EXPECT_EQ(counted.count, 2U);
  auto count_ir = compile_ok("a+", regex_ir::operation::count());
  EXPECT_EQ(regex_ir::testing::execute(count_ir, "baacaa").count, 2U);

  auto extracted = expect_enumeration("(a+)(b)", "xxaaab");
  ASSERT_EQ(extracted.captures.size(), 3U);
  EXPECT_EQ(extracted.captures[1], (regex_ir::testing::match_span{2, 5}));
  EXPECT_EQ(extracted.captures[2], (regex_ir::testing::match_span{5, 6}));

  static_cast<void>(expect_enumeration("(a+)", "baaca"));
  auto replace_ir = compile_ok("(a+)", regex_ir::operation::replace("<$1>"));
  EXPECT_EQ(regex_ir::testing::execute(replace_ir, "baaca").replaced, "b<aa>c<a>");

  static_cast<void>(expect_enumeration(",+", "a,,b,"));
  auto split_ir = compile_ok(",+", regex_ir::operation::split());
  EXPECT_EQ(regex_ir::testing::execute(split_ir, "a,,b,").pieces,
            (std::vector<std::string>{"a", "b", ""}));

  auto empty = expect_enumeration("a*", "b");
  EXPECT_EQ(empty.count, 2U);

  auto restarted = expect_enumeration("[a-z]+Z", "aaaaYbbbbZ");
  ASSERT_FALSE(restarted.matches.empty());
  EXPECT_EQ(restarted.matches.front(), (regex_ir::testing::match_span{5, 10}));

  auto restarted_classes = expect_enumeration("[a-z]+[A-Z]+", "aaaa1bbbXYZ");
  ASSERT_FALSE(restarted_classes.matches.empty());
  EXPECT_EQ(restarted_classes.matches.front(), (regex_ir::testing::match_span{5, 11}));

  auto restarted_capture = expect_enumeration("([a-z]+Z)", "aaaaYbbbbZ");
  ASSERT_EQ(restarted_capture.captures.size(), 2U);
  EXPECT_EQ(restarted_capture.captures[1], (regex_ir::testing::match_span{5, 10}));
}

TEST_F(Project, OptionsAndUnicode)
{
  regex_ir::compile_options insensitive{.case_insensitive = true};
  expect_boolean("AbC", "aBc", true, regex_ir::operation_kind::MATCHES, insensitive);

  auto unicode = expect_enumeration("λ+", "xλλ");
  ASSERT_FALSE(unicode.matches.empty());
  EXPECT_EQ(unicode.matches.front(), (regex_ir::testing::match_span{1, 5}));

  regex_ir::compile_options multiline{.multiline = true};
  expect_boolean("^cat$", "dog\ncat\nfox", true, regex_ir::operation_kind::CONTAINS, multiline);

  expect_boolean(R"REGEX(\x41)REGEX", "A", true, regex_ir::operation_kind::MATCHES);
  expect_boolean(R"REGEX(\x41)REGEX", "a", false, regex_ir::operation_kind::MATCHES);
  expect_boolean(R"REGEX(\x41)REGEX", "a", true, regex_ir::operation_kind::MATCHES, insensitive);
  expect_boolean("[[:alpha:]]+", "Mark", true, regex_ir::operation_kind::MATCHES);
  expect_boolean("[[:alpha:]]+", "Mark7", false, regex_ir::operation_kind::MATCHES);
  expect_boolean(R"REGEX(\p{Sm})REGEX", "=", true, regex_ir::operation_kind::MATCHES);
  expect_boolean(R"REGEX(\p{Sm})REGEX", "∞", true, regex_ir::operation_kind::MATCHES);
  expect_boolean(R"REGEX(\p{Sm})REGEX", "a", false, regex_ir::operation_kind::MATCHES);
}

TEST_F(Project, DiagnosticsAndLimits)
{
  auto backreference = regex_ir::compile("(a)\\1", regex_ir::operation::matches());
  ASSERT_FALSE(backreference);
  ASSERT_FALSE(backreference.diagnostics.empty());
  EXPECT_EQ(backreference.diagnostics.front().code, regex_ir::diagnostic_code::UNSUPPORTED_FEATURE);

  EXPECT_FALSE(regex_ir::compile("[z-a]", regex_ir::operation::matches()));
  EXPECT_FALSE(regex_ir::compile("(a)", regex_ir::operation::replace("$2")));

  regex_ir::compile_options limited{.limits = {.max_states = 2}};
  auto too_large = regex_ir::compile("abc", regex_ir::operation::matches(), limited);
  ASSERT_FALSE(too_large);
  ASSERT_FALSE(too_large.diagnostics.empty());
  EXPECT_EQ(too_large.diagnostics.front().code, regex_ir::diagnostic_code::RESOURCE_LIMIT);
}

TEST_F(Re2, SupportedParserForms)
{
  constexpr auto patterns = std::to_array<std::string_view>({
    "",       "|",      "|x|",   "a.",   "a.b",  "ab|cd", "(a)",     "(?:ab)*",
    "a{2}",   "a{2,3}", "a{2,}", "a*?",  "a+?",  "a??",   "a{2,3}?", "[ace]",
    "[^a-z]", "\\d+",   "\\D+",  "\\s+", "\\S+", "\\w+",  "\\W+",    "\\|\\(\\)\\*\\+\\?",
  });
  for (std::string_view pattern : patterns) {
    SCOPED_TRACE(pattern);
    auto automata = regex_ir::compile_automata(pattern);
    ASSERT_TRUE(automata);
    EXPECT_TRUE(regex_ir::verify(*automata.value).empty());
    expect_backends_agree(pattern, "abc 123");
  }
}

TEST_F(Re2, FullMatchPrecedenceAndRepetition)
{
  struct case_type {
    std::string_view pattern;
    std::string_view input;
    bool expected : 1;
  };
  constexpr auto cases = std::to_array<case_type>({{"a", "a", true},
                                                   {"a", "zyzzyva", false},
                                                   {"a+", "aa", true},
                                                   {"(a+|b)+", "aaab", true},
                                                   {"ab|cd", "cd", true},
                                                   {"ab|cd", "xabcdx", false},
                                                   {"a*b", "b", true},
                                                   {"a*b", "aaaaab", true},
                                                   {"a{2,4}", "a", false},
                                                   {"a{2,4}", "aaa", true},
                                                   {"a{2,4}", "aaaaa", false},
                                                   {"(?:ab)*", "abab", true},
                                                   {"(?:ab)*", "aba", false},
                                                   {"^$", "", true},
                                                   {"^$", "x", false},
                                                   {"^^(fo|foo)$", "foo", true},
                                                   {"^(foo|bar|[A-Z])$", "X", true},
                                                   {"^(foo|bar|[A-Z])$", "XY", false}});
  for (case_type const& test : cases) {
    expect_boolean(test.pattern, test.input, test.expected, regex_ir::operation_kind::MATCHES);
  }
}

TEST_F(Re2, SearchAndPrioritySpans)
{
  struct case_type {
    std::string_view pattern;
    std::string_view input;
    regex_ir::testing::match_span expected;
  };
  constexpr auto cases = std::to_array<case_type>({{"ab|cd", "xxcdab", {2, 4}},
                                                   {"a+", "xaaay", {1, 4}},
                                                   {"a+?", "xaaay", {1, 2}},
                                                   {"fo|foo", "foo", {0, 2}},
                                                   {"foo|fo", "foo", {0, 3}},
                                                   {"[0-9]+7", "x1237y", {1, 5}},
                                                   {"[^x]+", "xxxabcdxxx", {3, 7}}});
  for (case_type const& test : cases) {
    auto result = expect_enumeration(test.pattern, test.input);
    ASSERT_FALSE(result.matches.empty());
    EXPECT_EQ(result.matches.front(), test.expected);
  }
}

TEST_F(Re2, WordBoundaries)
{
  expect_boolean(R"REGEX(\bfoo\b)REGEX", "nofoo foo that", true);
  expect_boolean(R"REGEX(\bfoo\b)REGEX", "seafood", false);
  expect_boolean(R"REGEX(\Bfoo\B)REGEX", "xfoox", true);
  expect_boolean(R"REGEX(\Bfoo\B)REGEX", "foo", false);
  expect_boolean(R"REGEX(\bx\b)REGEX", "«x»", true);
  expect_boolean(R"REGEX(\Bx\B)REGEX", "axb", true);
  expect_boolean(R"REGEX(\bx\b)REGEX", "axb", false);
}

TEST_F(Re2, DotNewlineAndMultiline)
{
  expect_boolean("a.*a", "aba\naba", false, regex_ir::operation_kind::MATCHES);
  regex_ir::compile_options dot_all{.dot_all = true};
  expect_boolean("a.*a", "aba\naba", true, regex_ir::operation_kind::MATCHES, dot_all);

  regex_ir::compile_options multiline{.multiline = true};
  expect_boolean("^bar$", "foo\nbar\nbaz", true, regex_ir::operation_kind::CONTAINS, multiline);
  expect_boolean("^bar$", "foo\nbar\nbaz", false);
}

TEST_F(Re2, Utf8AndByteModes)
{
  expect_boolean("^...$", "日本語", true, regex_ir::operation_kind::MATCHES);
  expect_boolean("^...$", ".本.", true, regex_ir::operation_kind::MATCHES);
  auto found = expect_enumeration("..", "xλ");
  ASSERT_FALSE(found.matches.empty());
  EXPECT_EQ(found.matches.front(), (regex_ir::testing::match_span{0, 3}));

  regex_ir::compile_options bytes{.characters = regex_ir::character_mode::BYTES};
  expect_boolean("^.........$", "日本語", true, regex_ir::operation_kind::MATCHES, bytes);
  expect_boolean("^...$", "日本語", false, regex_ir::operation_kind::MATCHES, bytes);
}

TEST_F(Re2, CapturesFollowPriority)
{
  auto alternative = expect_enumeration("(fo|foo)", "foo");
  ASSERT_EQ(alternative.captures.size(), 2U);
  EXPECT_EQ(alternative.captures[1], (regex_ir::testing::match_span{0, 2}));

  auto repeated = expect_enumeration("(a|b)+", "abba");
  ASSERT_EQ(repeated.captures.size(), 2U);
  EXPECT_EQ(repeated.captures[0], (regex_ir::testing::match_span{0, 4}));
  EXPECT_EQ(repeated.captures[1], (regex_ir::testing::match_span{3, 4}));
}

TEST_F(Re2, GlobalReplacement)
{
  struct case_type {
    std::string_view pattern;
    std::string_view replacement;
    std::string_view input;
    std::string_view expected;
  };
  constexpr auto cases = std::to_array<case_type>({{"^", "(START)", "foo", "(START)foo"},
                                                   {"$", "(END)", "", "(END)"},
                                                   {"b", "bb", "ababab", "abbabbabb"},
                                                   {"b+", "bb", "bbbbbb", "bb"},
                                                   {"b*", "bb", "aaaaa", "bbabbabbabbabbabb"},
                                                   {"(a+)", "<$1>", "baaca", "b<aa>c<a>"},
                                                   {"a.*a", "<$0>", "aba\naba", "<aba>\n<aba>"}});
  for (case_type const& test : cases) {
    static_cast<void>(expect_enumeration(test.pattern, test.input));
    auto ir = compile_ok(test.pattern, regex_ir::operation::replace(std::string{test.replacement}));
    EXPECT_EQ(regex_ir::testing::execute(ir, test.input).replaced, test.expected);
  }
}

TEST_F(Re2, EmptyMatchesMakeProgress)
{
  auto counted = expect_enumeration("a*", "bbb");
  EXPECT_EQ(counted.count, 4U);
  static_cast<void>(expect_enumeration("b*", "aa"));
  auto split = compile_ok("b*", regex_ir::operation::split());
  EXPECT_EQ(regex_ir::testing::execute(split, "aa").pieces,
            (std::vector<std::string>{"", "a", "a", ""}));
}

TEST_F(Nvvm, PrefixesAndEntryNames)
{
  auto ir = compile_ok("abc[0-9]+", regex_ir::operation::contains());
  regex_ir::nvvm_ir_codegen_options options{.symbol_prefix    = "tenant_17",
                                            .execute_function = "regex_contains_17"};
  std::string source = regex_ir::generate_nvvm_ir(ir, options);
  EXPECT_NE(source.find("define zeroext i1 @regex_contains_17(i8* %data, i64 %size)"),
            std::string::npos);
  EXPECT_NE(source.find("@tenant_17_decode_width"), std::string::npos);
  EXPECT_NE(source.find("@tenant_17_dfa_transitions"), std::string::npos);
  EXPECT_EQ(source.find("@tenant_17_run_block"), std::string::npos);
  expect_boolean("abc[0-9]+", "abc123", true);
}

TEST_F(Nvvm, CompilerSelectedLoadsAndDeterministicExecutor)
{
  auto ir            = compile_ok("needle[0-9]+", regex_ir::operation::contains());
  std::string source = regex_ir::generate_nvvm_ir(ir);
  EXPECT_NE(source.find("%value8 = load i8"), std::string::npos);
  EXPECT_EQ(source.find("ld.global."), std::string::npos);
  EXPECT_NE(source.find("; executor: deterministic table"), std::string::npos);
  EXPECT_EQ(source.find("run_block"), std::string::npos);
  expect_boolean("needle[0-9]+", "xxneedle42", true);
}

TEST_F(Nvvm, AssertionDeterminization)
{
  auto prefix_ir     = compile_ok("^needle", regex_ir::operation::contains());
  auto prefix_source = regex_ir::generate_nvvm_ir(prefix_ir);
  EXPECT_NE(prefix_source.find("; executor: deterministic table"), std::string::npos);
  EXPECT_EQ(prefix_source.find("_dfa_boundary_classify"), std::string::npos);
  expect_boolean("^needle", "needle here", true);
  expect_boolean("^needle", "a needle", false);

  auto anchored_ir            = compile_ok("^needle$", regex_ir::operation::contains());
  std::string anchored_source = regex_ir::generate_nvvm_ir(anchored_ir);
  EXPECT_NE(anchored_source.find("; executor: assertion-aware deterministic table"),
            std::string::npos);
  EXPECT_EQ(anchored_source.find("@regex_ir_generated_run_block"), std::string::npos);
  expect_boolean("^needle$", "needle", true);

  auto boundary_ir = compile_ok(R"REGEX(\bneedle\b)REGEX", regex_ir::operation::contains());
  std::string boundary_source = regex_ir::generate_nvvm_ir(boundary_ir);
  EXPECT_NE(boundary_source.find("; executor: assertion-aware deterministic table"),
            std::string::npos);
  EXPECT_NE(boundary_source.find("_dfa_boundary_classify"), std::string::npos);
  EXPECT_EQ(boundary_source.find("@regex_ir_generated_run_block"), std::string::npos);
  EXPECT_FALSE(compile_nvvm_lto_ir(boundary_source, {}).empty());
  expect_boolean(R"REGEX(\bneedle\b)REGEX", "a needle!", true);
  expect_boolean(R"REGEX(\bneedle\b)REGEX", "needless", false);

  constexpr std::string_view internal_end = R"REGEX(^([0-9]+)(\-| |$)(.*)$)REGEX";
  auto internal_ir     = compile_ok(internal_end, regex_ir::operation::contains());
  auto internal_source = regex_ir::generate_nvvm_ir(internal_ir);
  EXPECT_NE(internal_source.find("; executor: assertion-aware deterministic table"),
            std::string::npos);
  EXPECT_EQ(internal_source.find("@regex_ir_generated_run_block"), std::string::npos);
  EXPECT_FALSE(compile_nvvm_lto_ir(internal_source, {}).empty());
  expect_boolean(internal_end, "100- ftp response", true);
  expect_boolean(internal_end, "100", true);
}

TEST_F(Nvvm, LargeBooleanAlternation)
{
  constexpr std::string_view pattern =
    " (Tom|Sawyer|Huckleberry|Finn).{0,30}river|river.{0,30}(Tom|Sawyer|Huckleberry|Finn)";
  auto ir            = compile_ok(pattern, regex_ir::operation::contains());
  std::string source = regex_ir::generate_nvvm_ir(ir);
  EXPECT_NE(source.find("@regex_ir_execute_alternative_0"), std::string::npos);
  EXPECT_NE(source.find("addrspace(1) constant"), std::string::npos);
  EXPECT_EQ(source.find("!nvvmir.version"), source.rfind("!nvvmir.version"));
  EXPECT_FALSE(compile_nvvm_lto_ir(source, {}).empty());
  expect_boolean(pattern, "A note about Tom walking beside the river today.", true);

  constexpr std::string_view bounded_pattern = "[a-q][^u-z]{13}x";
  auto bounded_ir = compile_ok(bounded_pattern, regex_ir::operation::contains());
  auto bounded    = regex_ir::generate_nvvm_ir(bounded_ir);
  EXPECT_NE(bounded.find("; executor: bit-parallel Glushkov NFA"), std::string::npos);
  EXPECT_NE(bounded.find("; glushkov positions: 15, alphabet classes: 4, shifts: 1, exceptions: 0"),
            std::string::npos);
  EXPECT_NE(bounded.find("@regex_ir_generated_glushkov_follow"), std::string::npos);
  EXPECT_EQ(bounded.find("@regex_ir_generated_dfa_transitions"), std::string::npos);
  EXPECT_EQ(bounded.find("@regex_ir_generated_run_block"), std::string::npos);
  EXPECT_FALSE(compile_nvvm_lto_ir(bounded, {}).empty());
  expect_boolean(bounded_pattern, "xxpABCDEFGHIJKLMxzz", true);
  expect_boolean(bounded_pattern, "xxrABCDEFGHIJKLMxzz", false);
}

TEST_F(Nvvm, GlushkovEligibilityFallbacks)
{
  auto compact =
    regex_ir::generate_nvvm_ir(compile_ok("needle[0-9]+", regex_ir::operation::contains()));
  EXPECT_NE(compact.find("; executor: deterministic table"), std::string::npos);
  EXPECT_EQ(compact.find("Glushkov"), std::string::npos);

  auto assertion = regex_ir::generate_nvvm_ir(
    compile_ok(R"REGEX(\bneedle\b)REGEX", regex_ir::operation::contains()));
  EXPECT_NE(assertion.find("; executor: assertion-aware deterministic table"), std::string::npos);
  EXPECT_EQ(assertion.find("Glushkov"), std::string::npos);

  auto nullable = regex_ir::generate_nvvm_ir(compile_ok("a*", regex_ir::operation::contains()));
  EXPECT_NE(nullable.find("; executor: deterministic table"), std::string::npos);
  EXPECT_EQ(nullable.find("Glushkov"), std::string::npos);

  auto over_limit =
    regex_ir::generate_nvvm_ir(compile_ok("[ab]{65}", regex_ir::operation::contains()));
  EXPECT_NE(over_limit.find("; executor: deterministic table"), std::string::npos);
  EXPECT_EQ(over_limit.find("Glushkov"), std::string::npos);

  constexpr std::string_view flat_alternation =
    R"REGEX(ddd|fff|eee|ggg|hhh|iii|jjj|kkk|[l-n]mm|ooo|ppp|qqq|rrr|sss|ttt|uuu|vvv|www|[x-z]yy)REGEX";
  auto flat =
    regex_ir::generate_nvvm_ir(compile_ok(flat_alternation, regex_ir::operation::contains()));
  EXPECT_NE(flat.find("; executor: bit-parallel Glushkov NFA"), std::string::npos);

  constexpr std::string_view ipv4 =
    R"REGEX((?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]))REGEX";
  auto branching = regex_ir::generate_nvvm_ir(compile_ok(ipv4, regex_ir::operation::contains()));
  EXPECT_NE(branching.find("; executor: deterministic table"), std::string::npos);
  EXPECT_EQ(branching.find("Glushkov"), std::string::npos);
}

TEST_F(Nvvm, GlushkovOverlappingStartsAndUtf8)
{
  constexpr std::string_view pattern = "[a-q].{13}x";
  auto source = regex_ir::generate_nvvm_ir(compile_ok(pattern, regex_ir::operation::contains()));
  EXPECT_NE(source.find("; executor: bit-parallel Glushkov NFA"), std::string::npos);

  expect_boolean(pattern, std::string(15U, 'a') + "x", true);
  expect_boolean(pattern, std::string(15U, 'a') + "y", false);

  std::string unicode = "zzp";
  for (std::size_t index = 0; index < 13U; ++index) {
    unicode += "\xcf\x80";
  }
  unicode += 'x';
  expect_boolean(pattern, unicode, true);

  constexpr std::string_view filtered_pattern = "a[^u-z]{13}x";
  auto filtered =
    regex_ir::generate_nvvm_ir(compile_ok(filtered_pattern, regex_ir::operation::contains()));
  EXPECT_NE(filtered.find("; executor: bit-parallel Glushkov NFA"), std::string::npos);
  expect_boolean(filtered_pattern, "zzzaABCDEFGHIJKLMx", true);
  expect_boolean(filtered_pattern, "zzzbABCDEFGHIJKLMx", false);

  constexpr std::string_view full_pattern = "[ab]{32}x";
  auto full = regex_ir::generate_nvvm_ir(compile_ok(full_pattern, regex_ir::operation::matches()));
  EXPECT_NE(full.find("; executor: bit-parallel Glushkov NFA"), std::string::npos);
  expect_boolean(
    full_pattern, std::string(32U, 'a') + "x", true, regex_ir::operation_kind::MATCHES);
  expect_boolean(full_pattern,
                 std::string{"z"} + std::string(32U, 'a') + "x",
                 false,
                 regex_ir::operation_kind::MATCHES);

  constexpr std::string_view boundary_pattern = "[ab]{63}x";
  auto boundary =
    regex_ir::generate_nvvm_ir(compile_ok(boundary_pattern, regex_ir::operation::matches()));
  EXPECT_NE(boundary.find("; glushkov positions: 64"), std::string::npos);
  expect_boolean(
    boundary_pattern, std::string(63U, 'b') + "x", true, regex_ir::operation_kind::MATCHES);
}

TEST_F(Nvvm, CaptureEnumerationAbi)
{
  auto ir            = compile_ok(R"REGEX((a+)(b?))REGEX", regex_ir::operation::extract());
  std::string source = regex_ir::generate_nvvm_ir(ir);
  EXPECT_NE(
    source.find(
      R"NVVM(define zeroext i1 @regex_ir_execute(i8* %data, i64 %size, i64 %search_start, i64* %captures))NVVM"),
    std::string::npos);
  EXPECT_NE(source.find("i64* %captures"), std::string::npos);
  auto result = expect_enumeration(R"REGEX((a+)(b?))REGEX", "aaab");
  EXPECT_EQ(result.captures.size(), 3U);
}

TEST_F(Nvvm, IndependentPrefixes)
{
  auto ir = compile_ok("a|bc", regex_ir::operation::contains());
  std::string first =
    regex_ir::generate_nvvm_ir(ir, regex_ir::nvvm_ir_codegen_options{"left", "left_execute"});
  std::string second =
    regex_ir::generate_nvvm_ir(ir, regex_ir::nvvm_ir_codegen_options{"right", "right_execute"});
  EXPECT_EQ(first.find("@right_"), std::string::npos);
  EXPECT_EQ(second.find("@left_"), std::string::npos);
  expect_boolean("a|bc", "xxbc", true);
}

TEST_F(Nvvm, RejectsUnsafeNames)
{
  auto boolean_ir = compile_ok("abc", regex_ir::operation::contains());
  EXPECT_THROW(static_cast<void>(regex_ir::generate_nvvm_ir(
                 boolean_ir, regex_ir::nvvm_ir_codegen_options{"bad.name", "execute"})),
               std::invalid_argument);
  expect_boolean("abc", "abc", true);
}

TEST_F(Nvvm, SpecializedOperationAbis)
{
  auto find = regex_ir::generate_nvvm_ir(compile_ok("a+", regex_ir::operation::find()));
  EXPECT_NE(find.find("define zeroext i1 @regex_ir_execute(i8* %data, i64 %size, i64* %span)"),
            std::string::npos);

  auto count = regex_ir::generate_nvvm_ir(compile_ok("a+", regex_ir::operation::count()));
  EXPECT_NE(count.find("define i64 @regex_ir_execute(i8* %data, i64 %size)"), std::string::npos);

  auto replace =
    regex_ir::generate_nvvm_ir(compile_ok("(a+)", regex_ir::operation::replace("<$1>")));
  EXPECT_NE(replace.find("define i64 @regex_ir_execute(i8* %data, i64 %size, i8* %output)"),
            std::string::npos);
  EXPECT_NE(replace.find("; executor: prioritized deterministic table"), std::string::npos);
  EXPECT_EQ(replace.find("%capture_array"), std::string::npos);

  auto captured_replace =
    regex_ir::generate_nvvm_ir(compile_ok("(a+)(b+)", regex_ir::operation::replace("<$2>")));
  EXPECT_NE(captured_replace.find("%capture_array"), std::string::npos);

  auto literal_replace =
    regex_ir::generate_nvvm_ir(compile_ok("(a+)", regex_ir::operation::replace("literal")));
  EXPECT_EQ(literal_replace.find("%capture_array"), std::string::npos);
  EXPECT_EQ(literal_replace.find("_capture_pos_"), std::string::npos);

  auto split = regex_ir::generate_nvvm_ir(compile_ok("a+", regex_ir::operation::split()));
  EXPECT_NE(split.find("define i64 @regex_ir_execute(i8* %data, i64 %size, i64* %spans)"),
            std::string::npos);
}

TEST_F(Nvvm, SingleByteGlobalExecutor)
{
  std::array sources{regex_ir::generate_nvvm_ir(compile_ok(" ", regex_ir::operation::count())),
                     regex_ir::generate_nvvm_ir(compile_ok(" ", regex_ir::operation::replace("_"))),
                     regex_ir::generate_nvvm_ir(compile_ok(" ", regex_ir::operation::split()))};
  for (auto& source : sources) {
    EXPECT_NE(source.find("; executor: single-byte literal scan"), std::string::npos);
    EXPECT_NE(source.find("%matched = icmp eq i32 %value, 32"), std::string::npos);
    EXPECT_EQ(source.find("_dfa_transitions"), std::string::npos);
  }
  EXPECT_FALSE(compile_nvvm_lto_ir(sources.front(), {}).empty());
}

TEST_F(Nvvm, RuntimeAvailability)
{
  if (!gpu().available()) GTEST_SKIP() << gpu().unavailable_reason();
  expect_boolean("a(b|c)+", "abcb", true);
}

TEST_F(Nvvm, SpecializedRuntimeOperations)
{
  if (!gpu().available()) GTEST_SKIP() << gpu().unavailable_reason();

  auto find_ir  = compile_ok("a(b|c)+", regex_ir::operation::find());
  auto find_cpu = regex_ir::testing::execute(find_ir, "xxabcbzz");
  auto find_gpu = gpu().find(find_ir, "xxabcbzz");
  EXPECT_EQ(find_gpu.matched, find_cpu.matched);
  EXPECT_EQ(find_gpu.matches, find_cpu.matches);

  auto count_ir  = compile_ok("a*", regex_ir::operation::count());
  auto count_cpu = regex_ir::testing::execute(count_ir, "bbb");
  auto count_gpu = gpu().count(count_ir, "bbb");
  EXPECT_EQ(count_gpu.count, count_cpu.count);

  constexpr std::array replacement_cases{
    std::array<std::string_view, 4>{"(a+)", "<$1>", "baaca", "b<aa>c<a>"},
    std::array<std::string_view, 4>{"b*", "X", "aa", "XaXaX"},
    std::array<std::string_view, 4>{"a", "$0$0", "aba", "aabaa"}};
  for (auto& test : replacement_cases) {
    auto ir = compile_ok(test[0], regex_ir::operation::replace(std::string{test[1]}));
    EXPECT_EQ(gpu().replace(ir, test[2]), test[3]);
  }

  auto split_ir  = compile_ok("b*", regex_ir::operation::split());
  auto split_cpu = regex_ir::testing::execute(split_ir, "aa");
  EXPECT_EQ(gpu().split(split_ir, "aa"), split_cpu.pieces);

  std::array<maybe_string, 3> tagged_capture_inputs{
    maybe_string{"xx12 34 yy56 78 "}, maybe_string{"no digits"}, std::nullopt};
  auto [tagged_ir, tagged_results] =
    enumerate_both(R"REGEX((\d+) (\d+) )REGEX", {}, tagged_capture_inputs);
  EXPECT_NE(regex_ir::generate_nvvm_ir(tagged_ir).find(
              "; executor: tagged prioritized deterministic table"),
            std::string::npos);
  ASSERT_EQ(tagged_results.size(), tagged_capture_inputs.size());
  EXPECT_EQ(tagged_results[0].count, 2U);
}

TEST_F(Nvvm, PrioritizedRuntimeRegressions)
{
  if (!gpu().available()) GTEST_SKIP() << gpu().unavailable_reason();

  auto expect_count = [&](std::string_view pattern,
                          std::string_view input,
                          std::size_t expected,
                          regex_ir::compile_options const& options = {}) {
    SCOPED_TRACE(std::string{pattern} + " / " + std::string{input});
    auto ir = compile_ok(pattern, regex_ir::operation::count(), options);
    EXPECT_EQ(regex_ir::testing::execute(ir, input).count, expected);
    EXPECT_EQ(gpu().count(ir, input).count, expected);
    return ir;
  };
  auto expect_find = [&](std::string_view pattern,
                         std::string_view input,
                         std::vector<regex_ir::testing::match_span> const& expected,
                         regex_ir::compile_options const& options = {}) {
    SCOPED_TRACE(std::string{pattern} + " / " + std::string{input});
    auto ir     = compile_ok(pattern, regex_ir::operation::find(), options);
    auto cpu    = regex_ir::testing::execute(ir, input);
    auto actual = gpu().find(ir, input);
    EXPECT_EQ(cpu.matches, expected);
    EXPECT_EQ(actual.matched, !expected.empty());
    EXPECT_EQ(actual.matches, expected);
  };
  auto expect_replace = [&](std::string_view pattern,
                            std::string_view input,
                            std::string_view expected,
                            regex_ir::compile_options const& options = {}) {
    SCOPED_TRACE(std::string{pattern} + " / " + std::string{input});
    auto ir = compile_ok(pattern, regex_ir::operation::replace("<$0>"), options);
    EXPECT_EQ(regex_ir::testing::execute(ir, input).replaced, expected);
    EXPECT_EQ(gpu().replace(ir, input), expected);
  };
  auto expect_split = [&](std::string_view pattern,
                          std::string_view input,
                          std::vector<std::string> const& expected,
                          regex_ir::compile_options const& options = {}) {
    SCOPED_TRACE(std::string{pattern} + " / " + std::string{input});
    auto ir = compile_ok(pattern, regex_ir::operation::split(), options);
    EXPECT_EQ(regex_ir::testing::execute(ir, input).pieces, expected);
    EXPECT_EQ(gpu().split(ir, input), expected);
  };

  regex_ir::compile_options byte_dot_all;
  byte_dot_all.case_insensitive = true;
  byte_dot_all.dot_all          = true;
  byte_dot_all.characters       = regex_ir::character_mode::BYTES;

  expect_count(".+?", "abc", 3);
  auto prioritized = expect_count(R"REGEX((?:[ab]{1,3}\w*|.{0,2}a{0,2}\s+))REGEX", "b2a2", 1);
  EXPECT_NE(
    regex_ir::generate_nvvm_ir(prioritized).find("; executor: prioritized deterministic table"),
    std::string::npos);
  expect_find(".+?", "abc", {{0, 1}});
  expect_find(R"REGEX(\d*\w+?.*[^c]*)REGEX", "02", {{0, 2}}, byte_dot_all);
  expect_replace(R"REGEX((?:\w{1,3}|.+?))REGEX", "1 b2", "<1>< ><b2>", byte_dot_all);
  expect_split(
    R"REGEX((?:.?a*\d?|(b)?\d+?))REGEX", "02b_a1ca", {"", "", "", "", "", ""}, byte_dot_all);

  auto filtered        = expect_count("[0-5]+", "éa012z5", 2);
  auto filtered_source = regex_ir::generate_nvvm_ir(filtered);
  EXPECT_NE(filtered_source.find("start_filter_advance:"), std::string::npos);
  EXPECT_NE(filtered_source.find("%start_filter_next = add nuw i64 %start, 1"), std::string::npos);
  auto replacement_source =
    regex_ir::generate_nvvm_ir(compile_ok("[0-5]+", regex_ir::operation::replace("<$0>")));
  auto split_source =
    regex_ir::generate_nvvm_ir(compile_ok("[0-5]+", regex_ir::operation::split()));
  EXPECT_EQ(replacement_source.find("start_filter_advance:"), std::string::npos);
  EXPECT_EQ(split_source.find("start_filter_advance:"), std::string::npos);
  expect_replace("[0-5]+", "éa012z5", "éa<012>z<5>");
  expect_split("[0-5]+", "éa012z5", {"éa", "z", ""});
}

TEST_F(Nvvm, ToolchainCompilation)
{
  std::array operations{regex_ir::operation::contains(),
                        regex_ir::operation::matches(),
                        regex_ir::operation::find(),
                        regex_ir::operation::count(),
                        regex_ir::operation::extract(),
                        regex_ir::operation::replace("<$1>"),
                        regex_ir::operation::split()};
  for (auto& operation : operations) {
    auto ir            = compile_ok("(abc)[0-9]+", operation);
    std::string source = regex_ir::generate_nvvm_ir(ir);
    auto lto_ir        = compile_nvvm_lto_ir(source, {});
    EXPECT_FALSE(lto_ir.empty());
  }
}

}  // namespace regex_ir::test

int main(int argc, char** argv)
{
  int output_index = 1;
  for (int input_index = 1; input_index < argc; ++input_index) {
    if (std::string_view{argv[input_index]} == "--print-ir") {
      regex_ir::test::print_ir = true;
    } else {
      argv[output_index++] = argv[input_index];
    }
  }
  argc = output_index;
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
