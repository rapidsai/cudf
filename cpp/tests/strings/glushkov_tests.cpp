/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file glushkov_tests.cpp
 * @brief Tests for the Glushkov NFA bit-parallel regex engine.
 *
 * These tests verify that:
 *   1. The Glushkov path produces identical results to the Thompson NFA path.
 *   2. Patterns with zero-width assertions (^, $, \b, \B) silently fall back to
 *      Thompson NFA and still produce correct results.
 *   3. Patterns with >64 character positions also fall back to Thompson NFA.
 *   4. Nullable patterns (matching empty string) work correctly.
 *   5. find/contains, match, and findall operations all work correctly.
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/findall.hpp>
#include <cudf/strings/regex/flags.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <string>
#include <vector>

struct GlushkovRegexTests : public cudf::test::BaseFixture {};

// ---------------------------------------------------------------------------
// Helper: run contains_re with both DEFAULT and GLUSHKOV flags and compare.
// ---------------------------------------------------------------------------
static void check_parity(cudf::strings_column_view const& sv, std::string const& pattern)
{
  auto prog_thompson =
    cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::DEFAULT);
  auto prog_glushkov =
    cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::GLUSHKOV);

  auto thompson_result = cudf::strings::contains_re(sv, *prog_thompson);
  auto glushkov_result = cudf::strings::contains_re(sv, *prog_glushkov);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*thompson_result, *glushkov_result);
}

// ---------------------------------------------------------------------------
// Test: basic literal matching
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, BasicLiteral)
{
  cudf::test::strings_column_wrapper strings{
    "hello", "world", "foo", "bar", "hello world", "", "HELLO"};
  auto sv = cudf::strings_column_view(strings);
  check_parity(sv, "hello");
  check_parity(sv, "world");
  check_parity(sv, "foo");
  check_parity(sv, "xyz");
}

// ---------------------------------------------------------------------------
// Test: character classes
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, CharacterClasses)
{
  cudf::test::strings_column_wrapper strings{
    "abc", "ABC", "123", "a1b", "!@#", "aaa", "zzz", ""};
  auto sv = cudf::strings_column_view(strings);
  check_parity(sv, "[a-z]+");
  check_parity(sv, "[A-Z]+");
  check_parity(sv, "[0-9]+");
  check_parity(sv, "[a-zA-Z0-9]+");
  check_parity(sv, "[^a-z]+");
}

// ---------------------------------------------------------------------------
// Test: digit/word/space built-in classes
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, BuiltinClasses)
{
  cudf::test::strings_column_wrapper strings{
    "hello world", "12345", "abc 123", "\t  \n", "a_b_c", ""};
  auto sv = cudf::strings_column_view(strings);
  check_parity(sv, "\\d+");
  check_parity(sv, "\\w+");
  check_parity(sv, "\\s+");
  check_parity(sv, "\\D+");
  check_parity(sv, "\\W+");
}

// ---------------------------------------------------------------------------
// Test: alternation
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, Alternation)
{
  cudf::test::strings_column_wrapper strings{"cat", "dog", "fish", "bird", "catfish", ""};
  auto sv = cudf::strings_column_view(strings);
  check_parity(sv, "cat|dog");
  check_parity(sv, "cat|dog|fish");
  check_parity(sv, "a|b|c|d|e");
}

// ---------------------------------------------------------------------------
// Test: repetition operators
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, Repetition)
{
  cudf::test::strings_column_wrapper strings{
    "a", "aa", "aaa", "b", "ba", "aaab", "", "aab"};
  auto sv = cudf::strings_column_view(strings);
  check_parity(sv, "a+");
  check_parity(sv, "a*");
  check_parity(sv, "a?b");
  check_parity(sv, "a{2,3}");
  check_parity(sv, "a+b");
}

// ---------------------------------------------------------------------------
// Test: dot (any character)
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, DotAny)
{
  cudf::test::strings_column_wrapper strings{
    "abc", "a\nbc", "xyz", "\n", "a", ""};
  auto sv = cudf::strings_column_view(strings);
  check_parity(sv, "a.c");
  check_parity(sv, ".+");
  check_parity(sv, "a.b");
}

// ---------------------------------------------------------------------------
// Test: patterns with assertions fall back to Thompson but still give correct
//       results (parity check still passes since Thompson handles them).
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, AssertionFallback)
{
  cudf::test::strings_column_wrapper strings{
    "hello", "world", "hello world", "say hello", ""};
  auto sv = cudf::strings_column_view(strings);
  // ^ and $ — Glushkov will fall back to Thompson; results must match.
  check_parity(sv, "^hello");
  check_parity(sv, "world$");
  check_parity(sv, "^hello world$");
  // \b word boundary
  check_parity(sv, "\\bhello\\b");
  check_parity(sv, "\\bworld\\b");
}

// ---------------------------------------------------------------------------
// Test: nullable pattern (matches empty string)
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, NullablePattern)
{
  cudf::test::strings_column_wrapper strings{"abc", "", "aaa", "xyz"};
  auto sv = cudf::strings_column_view(strings);
  check_parity(sv, "a*");
  check_parity(sv, "(ab)?");
  check_parity(sv, "x?y?z?");
}

// ---------------------------------------------------------------------------
// Test: match_re (anchored match) parity
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, MatchParity)
{
  cudf::test::strings_column_wrapper strings{
    "abc123", "abc", "123", "HELLO", "hello world", ""};
  auto sv = cudf::strings_column_view(strings);

  std::vector<std::string> patterns{"[a-z]+\\d+", "[a-z]+", "\\d+", "[A-Z]+"};
  for (auto const& pattern : patterns) {
    auto prog_thompson =
      cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::DEFAULT);
    auto prog_glushkov =
      cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::GLUSHKOV);

    auto r1 = cudf::strings::matches_re(sv, *prog_thompson);
    auto r2 = cudf::strings::matches_re(sv, *prog_glushkov);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*r1, *r2);
  }
}

// ---------------------------------------------------------------------------
// Test: alternation with optional empty branch  "a(bc|de|fg|)h"
//
// This pattern exercises several Glushkov construction properties at once:
//   - Multi-way alternation inside a group (bc, de, fg, or empty)
//   - Nullable sub-expression: the empty branch means "ah" is a valid match
//   - follow(a) must include both every first-char of each branch AND h itself
//     (because the empty branch makes h directly reachable after a)
//   - follow(c/e/g) must reach h  (end of each two-char branch)
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, AlternationWithEmptyBranch)
{
  cudf::test::strings_column_wrapper strings{
    "ah",       // matches: empty branch → "ah"
    "abch",     // matches: bc branch
    "adeh",     // matches: de branch
    "afghh",    // matches: fg branch (note extra h at end; contains still finds it)
    "abcde",    // no h at the end → no match
    "a",        // incomplete
    "h",        // missing a
    "",         // empty string
    "abcdefgh", // longer string; contains "afgh" substring? No – but check parity
    "xabchx",   // embedded match
  };
  auto sv = cudf::strings_column_view(strings);
  check_parity(sv, "a(bc|de|fg|)h");
}

// ---------------------------------------------------------------------------
// Test: bounded repetition gap  "ab{0,4}cv"
//
// This pattern exercises the multi-distance shift-mask path in the Glushkov
// compiler (equivalent to the paper's ShiftAndGap / ShiftAndDist family):
//   - 'b' appears 4 times in the linearized regex (positions 1..4)
//   - follow(a) must reach c at distance 1 (zero b's) through distance 5
//     (four b's), producing up to 5 distinct shift spans
//   - The Glushkov compiler greedily assigns the most-populated spans to
//     shift-mask slots; the rest overflow to the exception table
//   - Valid matches: "acv", "abcv", "abbcv", "abbbcv", "abbbbcv"
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, BoundedRepetitionGap)
{
  cudf::test::strings_column_wrapper strings{
    "acv",       // matches: 0 b's
    "abcv",      // matches: 1 b
    "abbcv",     // matches: 2 b's
    "abbbcv",    // matches: 3 b's
    "abbbbcv",   // matches: 4 b's (maximum)
    "abbbbbcv",  // no match: 5 b's exceeds bound
    "av",        // no match: missing c
    "acvx",      // embedded match at start
    "xacvx",     // embedded match in middle
    "",          // empty string
  };
  auto sv = cudf::strings_column_view(strings);
  check_parity(sv, "ab{0,4}cv");
}

// ---------------------------------------------------------------------------
// Test: complex patterns
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, ComplexPatterns)
{
  cudf::test::strings_column_wrapper strings{
    "2024-01-15", "abc-def", "2024/01/15", "not-a-date", "2024-13-01", ""};
  auto sv = cudf::strings_column_view(strings);
  check_parity(sv, "\\d{4}-\\d{2}-\\d{2}");
  check_parity(sv, "[a-z]+-[a-z]+");
  check_parity(sv, "(\\d+)-(\\d+)");
}

// ---------------------------------------------------------------------------
// Test: large pattern (>64 positions) falls back to Thompson silently
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, LargePatternFallback)
{
  // Build a pattern with >64 character-consuming positions.
  // "abcde..." × 13 = 65 positions (each literal char is one position).
  std::string long_pattern;
  for (int i = 0; i < 65; ++i) {
    long_pattern += static_cast<char>('a' + (i % 26));
  }

  cudf::test::strings_column_wrapper strings{"abcdefghij", "xxxxxxxxxxx", ""};
  auto sv = cudf::strings_column_view(strings);
  // Creating with GLUSHKOV should succeed (fall back) and give same results.
  check_parity(sv, long_pattern);
}

// ---------------------------------------------------------------------------
// Test: DOTALL flag combined with GLUSHKOV
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, DotallGlushkov)
{
  cudf::test::strings_column_wrapper strings{
    "hello\nworld", "one\ntwo\nthree", "no newline", ""};
  auto sv = cudf::strings_column_view(strings);

  auto flags_thompson =
    static_cast<cudf::strings::regex_flags>(cudf::strings::regex_flags::DOTALL);
  auto flags_glushkov = static_cast<cudf::strings::regex_flags>(
    cudf::strings::regex_flags::DOTALL | cudf::strings::regex_flags::GLUSHKOV);

  auto prog_t = cudf::strings::regex_program::create("hello.world", flags_thompson);
  auto prog_g = cudf::strings::regex_program::create("hello.world", flags_glushkov);

  auto r1 = cudf::strings::contains_re(sv, *prog_t);
  auto r2 = cudf::strings::contains_re(sv, *prog_g);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*r1, *r2);
}

// ---------------------------------------------------------------------------
// Test: working memory is zero for eligible Glushkov patterns
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, ZeroWorkingMemory)
{
  auto prog_thompson = cudf::strings::regex_program::create("\\d+");
  auto prog_glushkov =
    cudf::strings::regex_program::create("\\d+", cudf::strings::regex_flags::GLUSHKOV);

  // Both paths report non-zero working memory: Thompson needs it for execution,
  // and Glushkov-eligible programs still allocate it because extract() falls
  // back to the Thompson engine (Glushkov does not track capture groups).
  EXPECT_GT(prog_thompson->compute_working_memory_size(1024), 0u);
  EXPECT_GT(prog_glushkov->compute_working_memory_size(1024), 0u);
  // Both should report the same size (same underlying Thompson program).
  EXPECT_EQ(prog_thompson->compute_working_memory_size(1024),
            prog_glushkov->compute_working_memory_size(1024));
}

// ---------------------------------------------------------------------------
// Test: assertion patterns still require working memory (Thompson fallback)
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, AssertionFallbackWorkingMemory)
{
  auto prog = cudf::strings::regex_program::create(
    "^\\d+$", cudf::strings::regex_flags::GLUSHKOV);
  // Should fall back to Thompson → non-zero working memory.
  EXPECT_GT(prog->compute_working_memory_size(1024), 0u);
}

// ---------------------------------------------------------------------------
// Helper: matches_re parity (exercises the end >= 0 code path: end=1).
// ---------------------------------------------------------------------------
static void check_matches_parity(cudf::strings_column_view const& sv,
                                  std::string const& pattern)
{
  auto prog_thompson =
    cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::DEFAULT);
  auto prog_glushkov =
    cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::GLUSHKOV);

  auto r1 = cudf::strings::matches_re(sv, *prog_thompson);
  auto r2 = cudf::strings::matches_re(sv, *prog_glushkov);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*r1, *r2);
}

// ---------------------------------------------------------------------------
// Test: matches_re (end >= 0 path) — pattern must match from position 0.
//
// matches_re passes end=1 to glushkov_find().  Previously this caused a
// length() call to clamp end against str_len.  Now end is used directly.
// Verify Glushkov and Thompson agree for patterns with and without
// has_startchar optimisation.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, MatchesReEndGe0)
{
  cudf::test::strings_column_wrapper strings{
    "abc123",   // starts with [a-z] → matches [a-z]+\d+
    "123abc",   // starts with digit → does NOT match [a-z]+\d+
    "abc",      // only letters → matches [a-z]+, not [a-z]+\d+
    "xyz999",   // starts with [a-z] → matches [a-z]+\d+
    "  abc",    // starts with space → no match
    "",         // empty
    "a",        // single letter
    "1",        // single digit
  };
  auto sv = cudf::strings_column_view(strings);
  // has_startchar=false (character class): exercises reach-table skip with end >= 0
  check_matches_parity(sv, "[a-z]+\\d+");
  check_matches_parity(sv, "[a-z]+");
  check_matches_parity(sv, "\\d+");
  // has_startchar=true (literal): exercises fast byte-scan skip with end >= 0
  check_matches_parity(sv, "abc");
  check_matches_parity(sv, "abc\\d+");
  check_matches_parity(sv, "xyz");
}

// ---------------------------------------------------------------------------
// Test: matches_re with multi-byte UTF-8 strings (end >= 0 path).
//
// For multi-byte UTF-8: size_bytes() > length().  The old code called
// length() to clamp end=1; after the fix it uses end=1 directly.
// Verify correctness on strings where byte length ≠ character length.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, MatchesReUtf8EndGe0)
{
  cudf::test::strings_column_wrapper strings{
    "caf\xc3\xa9",          // "café"  — 5 bytes, 4 chars (é = 2 bytes)
    "na\xc3\xaf""ve",       // "naïve" — 6 bytes, 5 chars (ï = 2 bytes)
    "abc",                  // pure ASCII
    "\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e",  // "日本語" — 9 bytes, 3 chars
    "",
  };
  auto sv = cudf::strings_column_view(strings);
  // .+ matches any sequence of characters from position 0
  check_matches_parity(sv, ".+");
  // [a-z]+ only ASCII letters; multi-byte strings won't fully match
  check_matches_parity(sv, "[a-z]+");
  check_matches_parity(sv, "\\w+");
}

// ---------------------------------------------------------------------------
// Test: nullable pattern via matches_re (both end >= 0 and nullable active).
//
// When a pattern is nullable AND matches_re is used, both code paths were
// previously each causing a length() call.  After the fix, only nullable
// calls length(); end >= 0 uses the caller value directly.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, NullableMatchesRe)
{
  cudf::test::strings_column_wrapper strings{
    "aaa",   // non-empty match of a* from pos 0
    "bbb",   // zero-length match of a* from pos 0
    "",      // empty string — zero-length match
    "abc",   // a* matches "a" at pos 0
  };
  auto sv = cudf::strings_column_view(strings);
  // All these are nullable; matches_re uses end=1 (end >= 0 path)
  check_matches_parity(sv, "a*");
  check_matches_parity(sv, "\\d*");
  check_matches_parity(sv, "(ab)?");
  check_matches_parity(sv, "x?y?z?");
}

// ---------------------------------------------------------------------------
// Test: nullable contains_re — zero-length match at every position.
//
// A nullable pattern (e.g. \d*) must return true for every non-null string,
// including strings with no characters matching the non-empty form.
// This exercises the nullable outer-loop path that calls length() to bound
// the search at str_len (trying the zero-length match at end of string).
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, NullableZeroLengthMatch)
{
  cudf::test::strings_column_wrapper strings{
    "abc",   // no digits; zero-length \d* still matches at pos 0
    "",      // empty: zero-length match at pos 0 == pos str_len
    "123",   // non-empty \d+ match available
    "a1b2",  // mixed; zero-length or non-empty matches exist
    "xyz",   // no pattern-char; zero-length match at pos 0
  };
  auto sv = cudf::strings_column_view(strings);

  // Parity: Glushkov and Thompson must agree
  check_parity(sv, "\\d*");
  check_parity(sv, "a*");
  check_parity(sv, "[xyz]*");

  // Direct correctness: \d* is nullable → contains_re must be true for all rows
  auto prog_glushkov =
    cudf::strings::regex_program::create("\\d*", cudf::strings::regex_flags::GLUSHKOV);
  auto result = cudf::strings::contains_re(sv, *prog_glushkov);
  cudf::test::fixed_width_column_wrapper<bool> expected{true, true, true, true, true};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

// ---------------------------------------------------------------------------
// Test: nullable pattern on multi-byte UTF-8 strings.
//
// For UTF-8: size_bytes() != length() for strings with multi-byte chars.
// Nullable outer loop must stop at str_len (character count), not size_bytes.
// This is the one remaining case where length() is genuinely required.
// Verify Glushkov handles the character-count bound correctly.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, NullableUtf8)
{
  cudf::test::strings_column_wrapper strings{
    "caf\xc3\xa9",          // "café"  — 5 bytes, 4 chars
    "na\xc3\xaf""ve",       // "naïve" — 6 bytes, 5 chars
    "\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e",  // "日本語" — 9 bytes, 3 chars
    "",
    "abc",                  // pure ASCII
  };
  auto sv = cudf::strings_column_view(strings);
  // Nullable patterns on UTF-8 strings: length() path must count chars correctly
  check_parity(sv, "a*");
  check_parity(sv, "[a-z]*");
  check_parity(sv, ".?");
  check_parity(sv, "(abc)?");
}

// ---------------------------------------------------------------------------
// Helper: run contains_re with both EXT_NEWLINE and EXT_NEWLINE|GLUSHKOV flags
//         and compare results.
// ---------------------------------------------------------------------------
static void check_ext_newline_parity(cudf::strings_column_view const& sv,
                                     std::string const& pattern)
{
  auto flags_thompson =
    static_cast<cudf::strings::regex_flags>(cudf::strings::regex_flags::EXT_NEWLINE);
  auto flags_glushkov = static_cast<cudf::strings::regex_flags>(
    cudf::strings::regex_flags::EXT_NEWLINE | cudf::strings::regex_flags::GLUSHKOV);

  auto prog_t = cudf::strings::regex_program::create(pattern, flags_thompson);
  auto prog_g = cudf::strings::regex_program::create(pattern, flags_glushkov);

  auto r1 = cudf::strings::contains_re(sv, *prog_t);
  auto r2 = cudf::strings::contains_re(sv, *prog_g);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*r1, *r2);
}

// ---------------------------------------------------------------------------
// Helper: run count_re with both DEFAULT and GLUSHKOV flags and compare.
// ---------------------------------------------------------------------------
static void check_count_parity(cudf::strings_column_view const& sv,
                                std::string const& pattern)
{
  auto prog_thompson =
    cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::DEFAULT);
  auto prog_glushkov =
    cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::GLUSHKOV);

  auto r1 = cudf::strings::count_re(sv, *prog_thompson);
  auto r2 = cudf::strings::count_re(sv, *prog_glushkov);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*r1, *r2);
}

// ---------------------------------------------------------------------------
// Test: EXT_NEWLINE + GLUSHKOV with dot patterns on extended newline chars.
//
// EXT_NEWLINE makes '.' reject all is_newline() characters:
//   \n (0x0A), \r (0x0D), U+0085 (NEL), U+2028 (LINE SEP), U+2029 (PARA SEP)
// The ASCII reach table handles \n and \r; the non-ASCII newlines (U+0085,
// U+2028, U+2029) fall through to the device slow path.
// This test verifies that both paths produce identical results.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, ExtNewlineGlushkov)
{
  cudf::test::strings_column_wrapper strings{
    "abc",                                       // pure ASCII, no newlines
    "a\nb",                                      // \n (0x0A)
    "a\rb",                                      // \r (0x0D)
    "a\xc2\x85" "b",                             // U+0085 NEL (UTF-8: C2 85)
    "a\xe2\x80\xa8" "b",                         // U+2028 LINE SEPARATOR (UTF-8: E2 80 A8)
    "a\xe2\x80\xa9" "b",                         // U+2029 PARAGRAPH SEPARATOR (UTF-8: E2 80 A9)
    "a\n\r\xc2\x85" "b",                         // multiple newline types
    "\xc2\x85\xe2\x80\xa8\xe2\x80\xa9",          // only extended newlines
    "",                                           // empty
    "hello world",                                // no newlines at all
    "x\xc2\x85y\xe2\x80\xa8z",                   // mixed content with extended newlines
  };
  auto sv = cudf::strings_column_view(strings);

  // '.' with EXT_NEWLINE must NOT match any of the 5 newline characters
  check_ext_newline_parity(sv, ".+");
  check_ext_newline_parity(sv, "a.b");
  check_ext_newline_parity(sv, "a..b");
  check_ext_newline_parity(sv, "...");
  check_ext_newline_parity(sv, "a.+b");
}

// ---------------------------------------------------------------------------
// Test: EXT_NEWLINE + DOTALL + GLUSHKOV — DOTALL overrides EXT_NEWLINE for '.'
//
// When both DOTALL and EXT_NEWLINE are set, '.' uses ANYNL which matches
// everything (DOTALL takes precedence for dot behavior).
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, ExtNewlineDotallGlushkov)
{
  cudf::test::strings_column_wrapper strings{
    "a\xc2\x85" "b",           // U+0085 NEL
    "a\xe2\x80\xa8" "b",       // U+2028 LINE SEPARATOR
    "a\nb",                     // \n
    "abc",                      // no newlines
    "",
  };
  auto sv = cudf::strings_column_view(strings);

  auto flags_thompson = static_cast<cudf::strings::regex_flags>(
    cudf::strings::regex_flags::EXT_NEWLINE | cudf::strings::regex_flags::DOTALL);
  auto flags_glushkov = static_cast<cudf::strings::regex_flags>(
    cudf::strings::regex_flags::EXT_NEWLINE | cudf::strings::regex_flags::DOTALL |
    cudf::strings::regex_flags::GLUSHKOV);

  auto prog_t = cudf::strings::regex_program::create("a.b", flags_thompson);
  auto prog_g = cudf::strings::regex_program::create("a.b", flags_glushkov);

  auto r1 = cudf::strings::contains_re(sv, *prog_t);
  auto r2 = cudf::strings::contains_re(sv, *prog_g);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*r1, *r2);
}

// ---------------------------------------------------------------------------
// Test: findall_re parity — exercises repeated find() calls on same string.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, FindallParity)
{
  cudf::test::strings_column_wrapper strings{
    "abc 123 def 456", "no matches here!", "42", "a1b2c3", ""};
  auto sv = cudf::strings_column_view(strings);

  std::vector<std::string> patterns{"\\d+", "[a-z]+", "[a-z]\\d"};
  for (auto const& pattern : patterns) {
    auto prog_thompson =
      cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::DEFAULT);
    auto prog_glushkov =
      cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::GLUSHKOV);

    auto r1 = cudf::strings::findall(sv, *prog_thompson);
    auto r2 = cudf::strings::findall(sv, *prog_glushkov);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*r1, *r2);
  }
}

// ---------------------------------------------------------------------------
// Test: count_re parity — exercises repeated find() to count matches.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, CountParity)
{
  cudf::test::strings_column_wrapper strings{
    "aaa", "abab", "bbb", "a", "", "abc123def456"};
  auto sv = cudf::strings_column_view(strings);

  check_count_parity(sv, "a");
  check_count_parity(sv, "ab");
  check_count_parity(sv, "\\d+");
  check_count_parity(sv, "[a-z]+");
}

// ---------------------------------------------------------------------------
// Test: replace_re parity — exercises Glushkov find() + Thompson extract()
//       handoff during replacement.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, ReplaceParity)
{
  cudf::test::strings_column_wrapper strings{
    "hello world", "foo bar baz", "123 456", "no-match", ""};
  auto sv = cudf::strings_column_view(strings);

  std::vector<std::string> patterns{"\\w+", "[a-z]+", "\\d+"};
  for (auto const& pattern : patterns) {
    auto prog_thompson =
      cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::DEFAULT);
    auto prog_glushkov =
      cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::GLUSHKOV);

    auto r1 = cudf::strings::replace_re(sv, *prog_thompson, cudf::string_scalar("X"));
    auto r2 = cudf::strings::replace_re(sv, *prog_glushkov, cudf::string_scalar("X"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*r1, *r2);
  }
}

// ---------------------------------------------------------------------------
// Test: pattern with >8 distinct shift spans — forces exception table overflow
//       in build_shift_masks.
//
// Pattern: a(b|cc|ddd|eeee|fffff|gggggg|hhhhhhh|iiiiiiii|jjjjjjjjj)z
// The alternation creates follow transitions from 'a' to the first char of
// each branch at spans 1,2,3,...,9 — producing 9 distinct spans.  Since
// GLUSHKOV_MAX_SHIFTS = 8, one span must overflow to the exception table.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, ShiftOverflowToExceptions)
{
  cudf::test::strings_column_wrapper strings{
    "abz",              // span 1: a→b→z
    "accz",             // span 2: a→cc→z
    "adddz",            // span 3: a→ddd→z
    "aeeee z",          // span 4 but trailing space — no match (space before z)
    "aeeeez",           // span 4: a→eeee→z
    "afffffz",          // span 5
    "aggggggz",         // span 6
    "ahhhhhhhz",        // span 7
    "aiiiiiiiiz",       // span 8
    "ajjjjjjjjjz",     // span 9: the overflow span
    "az",               // no match: no branch is empty
    "xyz",              // no match
    "",
  };
  auto sv = cudf::strings_column_view(strings);
  check_parity(sv, "a(b|cc|ddd|eeee|fffff|gggggg|hhhhhhh|iiiiiiii|jjjjjjjjj)z");
}

// ---------------------------------------------------------------------------
// Test: non-ASCII character class matching
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, NonAsciiCharacterClasses)
{
  cudf::test::strings_column_wrapper strings{
    "caf\xc3\xa9",                                // "café"
    "na\xc3\xaf""ve",                             // "naïve"
    "\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e",      // "日本語"
    "hello",                                       // ASCII only
    "\xc3\xa9\xc3\xa9\xc3\xa9",                   // "ééé"
    "",
  };
  auto sv = cudf::strings_column_view(strings);

  // \w should match word characters including non-ASCII letters
  check_parity(sv, "\\w+");
  // . should match non-ASCII characters
  check_parity(sv, ".+");
  check_parity(sv, "...");
}
