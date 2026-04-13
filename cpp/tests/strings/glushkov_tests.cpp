/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
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
 *   4. find/contains, match, and findall operations all work correctly.
 *
 * Note: Nullable patterns (those matching empty string) are not tested with
 * the GLUSHKOV flag — they fall back to Thompson automatically.
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
#include <cudf/strings/split/split_re.hpp>
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
  cudf::test::strings_column_wrapper strings{"abc", "ABC", "123", "a1b", "!@#", "aaa", "zzz", ""};
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
  cudf::test::strings_column_wrapper strings{"a", "aa", "aaa", "b", "ba", "aaab", "", "aab"};
  auto sv = cudf::strings_column_view(strings);
  check_parity(sv, "a+");
  check_parity(sv, "a?b");
  check_parity(sv, "a{2,3}");
  check_parity(sv, "a+b");
}

// ---------------------------------------------------------------------------
// Test: dot (any character)
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, DotAny)
{
  cudf::test::strings_column_wrapper strings{"abc", "a\nbc", "xyz", "\n", "a", ""};
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
  cudf::test::strings_column_wrapper strings{"hello", "world", "hello world", "say hello", ""};
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
// Test: match_re (anchored match) parity
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, MatchParity)
{
  cudf::test::strings_column_wrapper strings{"abc123", "abc", "123", "HELLO", "hello world", ""};
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
    "ah",        // matches: empty branch → "ah"
    "abch",      // matches: bc branch
    "adeh",      // matches: de branch
    "afghh",     // matches: fg branch (note extra h at end; contains still finds it)
    "abcde",     // no h at the end → no match
    "a",         // incomplete
    "h",         // missing a
    "",          // empty string
    "abcdefgh",  // longer string; contains "afgh" substring? No – but check parity
    "xabchx",    // embedded match
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
  cudf::test::strings_column_wrapper strings{"hello\nworld", "one\ntwo\nthree", "no newline", ""};
  auto sv = cudf::strings_column_view(strings);

  auto flags_thompson = static_cast<cudf::strings::regex_flags>(cudf::strings::regex_flags::DOTALL);
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
  auto prog = cudf::strings::regex_program::create("^\\d+$", cudf::strings::regex_flags::GLUSHKOV);
  // Should fall back to Thompson → non-zero working memory.
  EXPECT_GT(prog->compute_working_memory_size(1024), 0u);
}

// ---------------------------------------------------------------------------
// Helper: matches_re parity (exercises the end >= 0 code path: end=1).
// ---------------------------------------------------------------------------
static void check_matches_parity(cudf::strings_column_view const& sv, std::string const& pattern)
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
    "abc123",  // starts with [a-z] → matches [a-z]+\d+
    "123abc",  // starts with digit → does NOT match [a-z]+\d+
    "abc",     // only letters → matches [a-z]+, not [a-z]+\d+
    "xyz999",  // starts with [a-z] → matches [a-z]+\d+
    "  abc",   // starts with space → no match
    "",        // empty
    "a",       // single letter
    "1",       // single digit
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
    "caf\xc3\xa9",  // "café"  — 5 bytes, 4 chars (é = 2 bytes)
    "na\xc3\xaf"
    "ve",                                    // "naïve" — 6 bytes, 5 chars (ï = 2 bytes)
    "abc",                                   // pure ASCII
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
static void check_count_parity(cudf::strings_column_view const& sv, std::string const& pattern)
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
    "abc",   // pure ASCII, no newlines
    "a\nb",  // \n (0x0A)
    "a\rb",  // \r (0x0D)
    "a\xc2\x85"
    "b",  // U+0085 NEL (UTF-8: C2 85)
    "a\xe2\x80\xa8"
    "b",  // U+2028 LINE SEPARATOR (UTF-8: E2 80 A8)
    "a\xe2\x80\xa9"
    "b",  // U+2029 PARAGRAPH SEPARATOR (UTF-8: E2 80 A9)
    "a\n\r\xc2\x85"
    "b",                                 // multiple newline types
    "\xc2\x85\xe2\x80\xa8\xe2\x80\xa9",  // only extended newlines
    "",                                  // empty
    "hello world",                       // no newlines at all
    "x\xc2\x85y\xe2\x80\xa8z",           // mixed content with extended newlines
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
    "a\xc2\x85"
    "b",  // U+0085 NEL
    "a\xe2\x80\xa8"
    "b",     // U+2028 LINE SEPARATOR
    "a\nb",  // \n
    "abc",   // no newlines
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
  cudf::test::strings_column_wrapper strings{"aaa", "abab", "bbb", "a", "", "abc123def456"};
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
    "abz",          // span 1: a→b→z
    "accz",         // span 2: a→cc→z
    "adddz",        // span 3: a→ddd→z
    "aeeee z",      // span 4 but trailing space — no match (space before z)
    "aeeeez",       // span 4: a→eeee→z
    "afffffz",      // span 5
    "aggggggz",     // span 6
    "ahhhhhhhz",    // span 7
    "aiiiiiiiiz",   // span 8
    "ajjjjjjjjjz",  // span 9: the overflow span
    "az",           // no match: no branch is empty
    "xyz",          // no match
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
    "caf\xc3\xa9",  // "café"
    "na\xc3\xaf"
    "ve",                                    // "naïve"
    "\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e",  // "日本語"
    "hello",                                 // ASCII only
    "\xc3\xa9\xc3\xa9\xc3\xa9",              // "ééé"
    "",
  };
  auto sv = cudf::strings_column_view(strings);

  // \w should match word characters including non-ASCII letters
  check_parity(sv, "\\w+");
  // . should match non-ASCII characters
  check_parity(sv, ".+");
  check_parity(sv, "...");
}

// ===========================================================================
// Priority-kill parity tests
//
// These tests verify that the priority kill in glushkov_find_impl makes the
// Glushkov engine produce the same results as Thompson's NFA for overlapping-
// prefix alternation patterns (a|aa, foo|foobar, cat|catch).
//
// Without priority kill, Glushkov would use leftmost-longest semantics and
// diverge from Thompson's leftmost-first (first-alternative-wins) semantics.
// With priority kill, both engines agree on all operations.
//
// Control patterns that both engines always agreed on: a+|a, a{1,3}, (a|ab)c
// ===========================================================================

// ---------------------------------------------------------------------------
// Helper: run findall with both engines and compare.
// ---------------------------------------------------------------------------
static void check_findall_parity(cudf::strings_column_view const& sv, std::string const& pattern)
{
  auto prog_t = cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::DEFAULT);
  auto prog_g = cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::GLUSHKOV);
  auto r_t    = cudf::strings::findall(sv, *prog_t);
  auto r_g    = cudf::strings::findall(sv, *prog_g);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*r_t, *r_g);
}

// ---------------------------------------------------------------------------
// Helper: run replace_re with both engines and compare.
// ---------------------------------------------------------------------------
static void check_replace_parity(cudf::strings_column_view const& sv,
                                 std::string const& pattern,
                                 std::string const& repl = "X")
{
  auto prog_t = cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::DEFAULT);
  auto prog_g = cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::GLUSHKOV);
  cudf::string_scalar rep(repl);
  auto r_t = cudf::strings::replace_re(sv, *prog_t, rep);
  auto r_g = cudf::strings::replace_re(sv, *prog_g, rep);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*r_t, *r_g);
}

// ---------------------------------------------------------------------------
// Helper: run split_record_re with both engines and compare.
// ---------------------------------------------------------------------------
static void check_split_parity(cudf::strings_column_view const& sv, std::string const& pattern)
{
  auto prog_t = cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::DEFAULT);
  auto prog_g = cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::GLUSHKOV);
  auto r_t    = cudf::strings::split_record_re(sv, *prog_t);
  auto r_g    = cudf::strings::split_record_re(sv, *prog_g);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*r_t, *r_g);
}

// ---------------------------------------------------------------------------
// Test: foo|foobar — longer literal as second alternation branch.
//
// Priority kill ensures "foo" wins (first alternative), so "foobar" is split
// as "foo"+"bar" rather than consumed as one unit.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, DivergenceAlternationFooOrFoobar)
{
  cudf::test::strings_column_wrapper strings{"foo", "foobar", "foobarbaz", "bar", "xfoobar", ""};
  auto sv = cudf::strings_column_view(strings);

  check_parity(sv, "foo|foobar");
  check_count_parity(sv, "foo|foobar");
  check_replace_parity(sv, "foo|foobar");
  check_findall_parity(sv, "foo|foobar");
}

// ---------------------------------------------------------------------------
// Test: cat|catch — suffix overlap where second branch extends the first.
//
// Priority kill ensures "cat" wins (first alternative), so "catch" is matched
// as "cat" leaving "ch" unconsumed.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, DivergenceAlternationCatOrCatch)
{
  cudf::test::strings_column_wrapper strings{"cat", "catch", "catfish", "dog", ""};
  auto sv = cudf::strings_column_view(strings);

  check_parity(sv, "cat|catch");
  check_count_parity(sv, "cat|catch");
  check_replace_parity(sv, "cat|catch");
  check_findall_parity(sv, "cat|catch");
}

// ---------------------------------------------------------------------------
// Test: Multi-occurrence — a|aa on strings with many consecutive 'a' chars.
//
// Priority kill ensures one 'a' per match (first-alternative wins), so
// "aaaa" yields count=4, matching Thompson's behavior.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, DivergenceCountMultiOccurrence)
{
  cudf::test::strings_column_wrapper strings{"aaaa", "aaaaaa", "aaab", "a", ""};
  auto sv = cudf::strings_column_view(strings);

  check_parity(sv, "a|aa");
  check_count_parity(sv, "a|aa");
  check_replace_parity(sv, "a|aa");
  check_findall_parity(sv, "a|aa");
  check_split_parity(sv, "a|aa");
}

// ---------------------------------------------------------------------------
// Control: a+|a — greedy quantifier listed FIRST.
//
// Both engines consume as many 'a' as possible when the greedy branch is first.
// Expected: PASS on all operations.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, ControlGreedyFirstBranchAgreement)
{
  cudf::test::strings_column_wrapper strings{"a", "aa", "aab", "b", ""};
  auto sv = cudf::strings_column_view(strings);

  check_parity(sv, "a+|a");
  check_count_parity(sv, "a+|a");
  check_replace_parity(sv, "a+|a");
  check_findall_parity(sv, "a+|a");
  check_split_parity(sv, "a+|a");
}

// ---------------------------------------------------------------------------
// Control: a{1,3} — bounded repetition.
//
// Both engines are greedy and agree. Confirms bounded repetition alone does
// not cause divergence.
// Expected: PASS on all operations.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, ControlBoundedRepetitionAgreement)
{
  cudf::test::strings_column_wrapper strings{"a", "aa", "aaa", "aaaa", "aaaab", "b", ""};
  auto sv = cudf::strings_column_view(strings);

  check_parity(sv, "a{1,3}");
  check_count_parity(sv, "a{1,3}");
  check_replace_parity(sv, "a{1,3}");
  check_findall_parity(sv, "a{1,3}");
  check_split_parity(sv, "a{1,3}");
}

// ---------------------------------------------------------------------------
// Control: (a|ab)c — prefix-overlap where longer branch is second.
//
// Thompson's parallel NFA and Glushkov's greedy extension both find the same
// result for this pattern (the 'ab' path is the only one that can reach 'c'
// when 'b' is present, so there is no ambiguity).
// Expected: PASS on all operations.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, ControlPrefixOverlapAgreement)
{
  cudf::test::strings_column_wrapper strings{"ac", "abc", "aac", "aabbc", "xabc", ""};
  auto sv = cudf::strings_column_view(strings);

  check_parity(sv, "(a|ab)c");
  check_count_parity(sv, "(a|ab)c");
  check_replace_parity(sv, "(a|ab)c");
  check_findall_parity(sv, "(a|ab)c");
  check_split_parity(sv, "(a|ab)c");
}

// ---------------------------------------------------------------------------
// Test: (|a)a — nullable sub-expression alternation.
//
// The sub-expression (|a) is nullable (empty branch listed first), but the
// overall pattern requires at least one 'a'.  With leftmost-first semantics
// the empty branch wins inside the group, so the engine effectively matches
// a single 'a' each time.  On "aa" both engines should find two matches
// (count=2), not one "aa" match.
//
// This exercises the priority kill for a nullable sub-expression that feeds
// into a required successor character, verifying that Glushkov and Thompson
// agree across count_re, findall, replace_re, and split_re.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, NullableSubExpressionAlternation)
{
  cudf::test::strings_column_wrapper strings{
    "a",    // one match: (empty)a
    "aa",   // two matches: (empty)a, (empty)a
    "aaa",  // three matches
    "b",    // no match
    "ba",   // one match at pos 1
    "aba",  // two matches: pos 0 and pos 2
    "",     // empty — no match
  };
  auto sv = cudf::strings_column_view(strings);

  check_parity(sv, "(|a)a");
  check_count_parity(sv, "(|a)a");
  check_findall_parity(sv, "(|a)a");
  check_replace_parity(sv, "(|a)a");
  check_split_parity(sv, "(|a)a");
}

// ---------------------------------------------------------------------------
// Test: a(|b) — nullable first alternative in a follow frontier.
//
// After consuming 'a', the sub-expression (|b) has a nullable first branch
// that reaches END.  Rule 1 (ACCEPT before char) forces Glushkov fallback.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, NullableFirstAltInFollow)
{
  cudf::test::strings_column_wrapper strings{
    "a",    // match "a" (empty branch wins inside group)
    "ab",   // match "a" at pos 0 (empty branch wins, 'b' is unconsumed)
    "abc",  // match "a" at pos 0
    "aa",   // two matches: "a" at pos 0, "a" at pos 1
    "b",    // no match (pattern requires leading 'a')
    "",     // no match
  };
  auto sv = cudf::strings_column_view(strings);

  check_parity(sv, "a(|b)");
  check_count_parity(sv, "a(|b)");
  check_findall_parity(sv, "a(|b)");
  check_replace_parity(sv, "a(|b)");
  check_split_parity(sv, "a(|b)");
}

// ---------------------------------------------------------------------------
// Test: (a|b)(|c) — non-nullable leading group + nullable trailing group.
//
// The second group (|c) has a nullable first alternative.  After consuming
// 'a' or 'b', the follow frontier hits ACCEPT before 'c' → Rule 1 fallback.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, NonNullableLeadWithNullableTrail)
{
  cudf::test::strings_column_wrapper strings{
    "a",    // match "a"
    "ac",   // match "a" (empty branch wins in second group)
    "bc",   // match "b"
    "abc",  // match "a" at pos 0
    "ba",   // match "b" at pos 0, "a" at pos 1
    "",     // no match
  };
  auto sv = cudf::strings_column_view(strings);

  check_parity(sv, "(a|b)(|c)");
  check_count_parity(sv, "(a|b)(|c)");
  check_findall_parity(sv, "(a|b)(|c)");
  check_replace_parity(sv, "(a|b)(|c)");
  check_split_parity(sv, "(a|b)(|c)");
}

// ---------------------------------------------------------------------------
// Test: (|.)a — dot inside nullable first-alt group.
//
// The dot overlaps with 'a', so Rule 2 (non-monotone gpos + char overlap)
// forces Glushkov fallback.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, NullableDotAlternation)
{
  cudf::test::strings_column_wrapper strings{
    "a",    // match "a" (empty branch wins)
    "aa",   // two matches
    "ba",   // match "a" at pos 1
    "ab",   // match "a" at pos 0
    "xyz",  // no match (no 'a')
    "",     // no match
  };
  auto sv = cudf::strings_column_view(strings);

  check_parity(sv, "(|.)a");
  check_count_parity(sv, "(|.)a");
  check_findall_parity(sv, "(|.)a");
  check_replace_parity(sv, "(|.)a");
  check_split_parity(sv, "(|.)a");
}

// ---------------------------------------------------------------------------
// Test: .*foo — greedy dot-star followed by literal.
//
// This is a common pattern that must stay on the Glushkov fast path.  The
// quantifier OR has monotone gpos, so no priority conflict is detected.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, DotStarLiteral)
{
  cudf::test::strings_column_wrapper strings{
    "foo",        // match "foo"
    "xfoo",       // match "xfoo"
    "xxfoo",      // match "xxfoo"
    "foobar",     // match "foo" (or "foo" prefix depending on greedy)
    "xfooyfooz",  // match
    "bar",        // no match
    "",           // no match
  };
  auto sv = cudf::strings_column_view(strings);

  check_parity(sv, ".*foo");
  check_count_parity(sv, ".*foo");
  check_findall_parity(sv, ".*foo");
  check_replace_parity(sv, ".*foo");
  check_split_parity(sv, ".*foo");
}

// ---------------------------------------------------------------------------
// Test: (a|)a — nullable SECOND alternative (not first).
//
// Here the empty branch is the second alternative (lower Thompson priority).
// The first alternative 'a' gets lower gpos, matching priority_kill's
// assumption.  Frontier gpos is monotone → stays on Glushkov.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, NullableSecondAlt)
{
  cudf::test::strings_column_wrapper strings{
    "a",    // match "a" (first alt 'a' wins inside group + consumes it; but the outer 'a' needs
            // another)
    "aa",   // match "aa"
    "aaa",  // matches
    "b",    // no match
    "ba",   // no match (need two 'a's in a row for the greedy (a|)a to match "aa")
    "baa",  // match "aa" at pos 1
    "",     // no match
  };
  auto sv = cudf::strings_column_view(strings);

  check_parity(sv, "(a|)a");
  check_count_parity(sv, "(a|)a");
  check_findall_parity(sv, "(a|)a");
  check_replace_parity(sv, "(a|)a");
  check_split_parity(sv, "(a|)a");
}

// ---------------------------------------------------------------------------
// Test: Top-level nullable patterns silently fall back to Thompson under
// the GLUSHKOV flag and still produce correct results.
//
// glushkov_regcomp.cpp bails out with `if (gp->nullable) return nullptr;`
// for any pattern whose overall language includes the empty string.  The
// GLUSHKOV flag therefore transparently uses Thompson for these patterns.
// This test verifies that the fallback pipeline is wired correctly and that
// results under GLUSHKOV match results under DEFAULT for all such patterns.
// ---------------------------------------------------------------------------
TEST_F(GlushkovRegexTests, NullablePatternFallbackParity)
{
  cudf::test::strings_column_wrapper strings{"abc", "", "123", "xyz", "a1b2"};
  auto sv = cudf::strings_column_view(strings);

  // All of these patterns can match the empty string → top-level nullable →
  // Glushkov compiler returns nullptr → engine falls back to Thompson.
  std::vector<std::string> const patterns{"a*", "\\d*", "(ab)?", "x?y?z?", "[a-z]*"};
  for (auto const& pattern : patterns) {
    check_parity(sv, pattern);
    check_count_parity(sv, pattern);
    check_replace_parity(sv, pattern);
  }
}
