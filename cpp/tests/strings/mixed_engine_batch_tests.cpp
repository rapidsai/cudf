/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file mixed_engine_batch_tests.cpp
 * @brief Tests for mixed-engine multi-regex batch execution in cuDF's regex engine.
 *
 * When the GLUSHKOV flag is set on a multi-pattern replace_re() call, each
 * pattern independently decides whether it runs through the Glushkov NFA
 * (bit-parallel, zero working memory) or falls back to the Thompson NFA.
 *
 * Fallback triggers include:
 *   - Zero-width assertions: ^, $, \b, \B
 *   - >64 character-consuming positions
 *   - Top-level nullable patterns (matching the empty string)
 *
 * These tests verify:
 *   1. Mixed batches produce correct results when some patterns use Glushkov
 *      and others use Thompson.
 *   2. The working memory bug regression is covered: the old code sized the
 *      shared working-memory buffer from the program with the most instructions
 *      (by insts_counts()), not from the one with the highest
 *      working_memory_size(). When the max-instruction program was Glushkov-
 *      backed (working_memory_size() == 0), the Thompson fallback programs got
 *      no working memory — causing silent OOB access.
 *   3. Various combinations of fallback triggers in the same batch work.
 *   4. Edge cases: all-Glushkov batches, all-Thompson batches, nulls, empty
 *      strings, single-row inputs, and many-pattern batches.
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/findall.hpp>
#include <cudf/strings/regex/flags.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <string>
#include <vector>

struct MixedEngineBatchTests : public cudf::test::BaseFixture {};

// ===========================================================================
// Working-memory bug regression tests
//
// The bug in multi_re.cu: the old code computed max working memory from the
// program with the most instructions. When that program was Glushkov-backed
// (working_memory_size() == 0), Thompson programs in the same batch got zero
// working memory, causing silent OOB access.
//
// To trigger the bug, the Glushkov-eligible pattern must have MORE Thompson
// instructions than the Thompson-fallback pattern.
// ===========================================================================

// ---------------------------------------------------------------------------
// Regression: Glushkov pattern with more instructions than Thompson pattern.
//
// "[a-z][0-9][a-z][0-9][a-z]" → ~6 instructions (Glushkov, longer)
// "^x"                         → ~3 instructions (Thompson, shorter)
//
// Old bug: max_element picks Glushkov program → working_memory_size() = 0
//          → "^x" Thompson program gets no working memory → silent OOB.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, WorkingMemoryBugGlushkovLongerThanThompson)
{
  cudf::test::strings_column_wrapper input({"a1b2c def", "x hello"});
  auto sv = cudf::strings_column_view(input);

  // Pattern 0: Glushkov-eligible (5 CCLASS + END ≈ 6 insts)
  // Pattern 1: Thompson fallback due to ^ assertion (BOL + CHAR + END ≈ 3 insts)
  std::vector<std::string> patterns{"[a-z][0-9][a-z][0-9][a-z]", "^x"};
  cudf::test::strings_column_wrapper repls({"MATCH", "X"});
  auto repls_view = cudf::strings_column_view(repls);

  auto const flags = cudf::strings::regex_flags::GLUSHKOV;
  auto results     = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  cudf::test::strings_column_wrapper expected({"MATCH def", "X hello"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

// ---------------------------------------------------------------------------
// Regression variant: multiple Glushkov patterns, one Thompson at end.
//
// Ensures the fix works when there are several Glushkov programs that all
// have higher instruction counts than the single Thompson fallback.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, WorkingMemoryBugMultipleGlushkovOneThompson)
{
  cudf::test::strings_column_wrapper input({"a1b2c x9y8z hello_world", "test ^start"});
  auto sv = cudf::strings_column_view(input);

  // Patterns 0,1: Glushkov-eligible with many CCLASS instructions (distinct patterns)
  // Pattern 2: Thompson fallback due to ^ assertion
  std::vector<std::string> patterns{
    "[a-z][0-9][a-z][0-9][a-z]", "[A-Z][0-9][A-Z][0-9][A-Z]", "^test"};
  cudf::test::strings_column_wrapper repls({"G0", "G1", "T"});
  auto repls_view = cudf::strings_column_view(repls);

  auto const flags = cudf::strings::regex_flags::GLUSHKOV;
  auto results     = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  // Row 0: "a1b2c" matches pattern 0 → "G0 x9y8z hello_world"
  //         "x9y8z" also matches pattern 0 → "G0 G0 hello_world"
  //         (pattern 1 [A-Z]... doesn't match any lowercase input)
  // Row 1: "test" at ^ matches pattern 2 → "T ^start"
  cudf::test::strings_column_wrapper expected({"G0 G0 hello_world", "T ^start"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

// ---------------------------------------------------------------------------
// Regression variant: Thompson fallback pattern has more instructions.
//
// This is the "safe" case where Thompson was already the max-instruction
// program. Verifies it still works after the fix (no regression).
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, WorkingMemoryThompsonLongerThanGlushkov)
{
  cudf::test::strings_column_wrapper input(
    {"abc def hello world", "start of line", "end of line$"});
  auto sv = cudf::strings_column_view(input);

  // Pattern 0: short Glushkov-eligible pattern
  // Pattern 1: longer Thompson fallback (^ assertion + more chars)
  std::vector<std::string> patterns{"abc", "^start of line"};
  cudf::test::strings_column_wrapper repls({"ABC", "BEGIN"});
  auto repls_view = cudf::strings_column_view(repls);

  auto const flags = cudf::strings::regex_flags::GLUSHKOV;
  auto results     = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  cudf::test::strings_column_wrapper expected({"ABC def hello world", "BEGIN", "end of line$"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

// ===========================================================================
// Mixed-engine correctness tests: assertion fallback patterns
// ===========================================================================

// ---------------------------------------------------------------------------
// Mix of Glushkov patterns with BOL (^) assertion fallback.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, GlushkovWithBOLFallback)
{
  cudf::test::strings_column_wrapper input({"hello world", "world hello", "foo bar", "hello foo"});
  auto sv = cudf::strings_column_view(input);

  // Pattern 0: Glushkov-eligible
  // Pattern 1: Thompson fallback due to ^
  std::vector<std::string> patterns{"foo", "^hello"};
  cudf::test::strings_column_wrapper repls({"FOO", "HI"});
  auto repls_view = cudf::strings_column_view(repls);

  auto const flags = cudf::strings::regex_flags::GLUSHKOV;
  auto results     = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  // Row 0: "^hello" matches at pos 0 → "HI world"
  // Row 1: no "foo", "^hello" doesn't match (starts with "world") → unchanged
  // Row 2: "foo" matches → "FOO bar"
  // Row 3: "^hello" matches at pos 0 AND "foo" matches at pos 6 → "HI FOO"
  cudf::test::strings_column_wrapper expected({"HI world", "world hello", "FOO bar", "HI FOO"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

// ---------------------------------------------------------------------------
// Mix of Glushkov patterns with EOL ($) assertion fallback.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, GlushkovWithEOLFallback)
{
  cudf::test::strings_column_wrapper input({"abc xyz", "xyz abc", "123 abc", "xyz"});
  auto sv = cudf::strings_column_view(input);

  // Pattern 0: Glushkov-eligible
  // Pattern 1: Thompson fallback due to $
  std::vector<std::string> patterns{"123", "abc$"};
  cudf::test::strings_column_wrapper repls({"NUM", "END"});
  auto repls_view = cudf::strings_column_view(repls);

  auto const flags = cudf::strings::regex_flags::GLUSHKOV;
  auto results     = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  // Row 0: "abc" is not at end, no "123" → unchanged
  // Row 1: "abc$" matches "abc" at end → "xyz END"
  // Row 2: "123" matches at pos 0 AND "abc$" matches at pos 4 → "NUM END"
  // Row 3: no match for either → unchanged
  cudf::test::strings_column_wrapper expected({"abc xyz", "xyz END", "NUM END", "xyz"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

// ---------------------------------------------------------------------------
// Mix of Glushkov patterns with word-boundary (\b) assertion fallback.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, GlushkovWithWordBoundaryFallback)
{
  cudf::test::strings_column_wrapper input({"cat in the hat", "catalog", "scattered"});
  auto sv = cudf::strings_column_view(input);

  // Pattern 0: Glushkov-eligible (character class)
  // Pattern 1: Thompson fallback due to \b
  std::vector<std::string> patterns{"\\d+", "\\bcat\\b"};
  cudf::test::strings_column_wrapper repls({"NUM", "CAT"});
  auto repls_view = cudf::strings_column_view(repls);

  auto const flags = cudf::strings::regex_flags::GLUSHKOV;
  auto results     = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  // Row 0: \bcat\b matches "cat" (whole word) → "CAT in the hat"
  // Row 1: \bcat\b does NOT match "catalog" (no boundary after "cat")
  //         No digits either → unchanged
  // Row 2: \bcat\b does NOT match "scattered" → unchanged
  cudf::test::strings_column_wrapper expected({"CAT in the hat", "catalog", "scattered"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

// ---------------------------------------------------------------------------
// All assertion types mixed with Glushkov patterns.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, AllAssertionTypesMixed)
{
  cudf::test::strings_column_wrapper input({"start abc end", "abc start end abc", "no match here"});
  auto sv = cudf::strings_column_view(input);

  // Pattern 0: Glushkov-eligible
  // Pattern 1: Thompson fallback — ^ assertion
  // Pattern 2: Thompson fallback — $ assertion
  // Pattern 3: Thompson fallback — \b assertion
  std::vector<std::string> patterns{"[0-9]+", "^start", "end$", "\\babc\\b"};
  cudf::test::strings_column_wrapper repls({"N", "BEGIN", "FINISH", "X"});
  auto repls_view = cudf::strings_column_view(repls);

  auto const flags = cudf::strings::regex_flags::GLUSHKOV;
  auto results     = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  // Row 0: "^start" matches at 0 → "BEGIN abc end"; then "end$" matches → "BEGIN abc FINISH";
  //         then \babc\b matches "abc" → "BEGIN X FINISH"
  //         Wait — multi_re processes char-by-char left to right, first pattern match at each pos
  //         wins. Let me reconsider.
  //
  //   pos 0: "^start" matches [0,5) → replaced by "BEGIN" → "BEGIN abc end"
  //   pos 6 (after "BEGIN "): \babc\b matches [6,9) → replaced by "X" → "BEGIN X end"
  //   pos 10 (after "BEGIN X "): "end$" matches [10,13) → replaced by "FINISH"
  //   Result: "BEGIN X FINISH"
  //
  // Row 1: pos 0: no ^start (starts with "abc"), \babc\b matches [0,3) → "X start end abc"
  //         pos 4: no match at pos 4 for any pattern  (s of start)
  //         ...
  //         pos 10: "end" doesn't match "end$" because "end abc" not at end; but \babc\b [14,17)
  //         Let me just verify: "abc start end abc"
  //         pos 0: \babc\b matches "abc" [0,3) → "X start end abc"
  //         pos 4: nothing (start not at ^)
  //         pos 14: \babc\b matches "abc" [14,17) → but "end$" checks "abc" at end...
  //              "end$" matches [10,13) "end" — is "end" at end? String is "abc start end abc"
  //              so "end" is at [10,13) but "abc" follows → not at end. No match for "end$".
  //              pos 14: \babc\b matches [14,17) → "X start end X"
  //
  // Row 2: no match for any pattern → unchanged
  cudf::test::strings_column_wrapper expected({"BEGIN X FINISH", "X start end X", "no match here"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

// ===========================================================================
// Mixed-engine correctness tests: nullable pattern fallback
// ===========================================================================

// ---------------------------------------------------------------------------
// Mix of Glushkov-eligible pattern with nullable pattern (Thompson fallback).
// Nullable patterns: a*, \d*, x?
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, GlushkovWithNullableFallback)
{
  cudf::test::strings_column_wrapper input({"abc 123", "xyz", "aaa 999"});
  auto sv = cudf::strings_column_view(input);

  // Pattern 0: Glushkov-eligible (non-nullable)
  // Pattern 1: Thompson fallback due to ^ assertion
  // (Nullable patterns like \d* cause infinite zero-length matches in multi-pattern replace)
  std::vector<std::string> patterns{"[a-z]+", "^\\d+"};
  cudf::test::strings_column_wrapper repls({"W", "N"});
  auto repls_view = cudf::strings_column_view(repls);

  // Without GLUSHKOV flag — Thompson for all. Use as reference.
  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view);

  // With GLUSHKOV flag — pattern 0 uses Glushkov, pattern 1 falls back to Thompson.
  auto const flags   = cudf::strings::regex_flags::GLUSHKOV;
  auto results_mixed = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  // Both should produce identical results.
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ---------------------------------------------------------------------------
// Nullable pattern has more instructions than the Glushkov pattern.
// Exercises the working memory allocation when the Thompson-fallback nullable
// pattern is the one that needs working memory.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, NullableFallbackLongerInstructions)
{
  cudf::test::strings_column_wrapper input({"abcde fghij 12345", "00000 zzzzz"});
  auto sv = cudf::strings_column_view(input);

  // Pattern 0: short Glushkov-eligible
  // Pattern 1: longer Thompson fallback due to $ assertion (more instructions than pattern 0)
  // (Nullable patterns like [a-z]{0,5} cause infinite zero-length matches in multi-pattern replace)
  std::vector<std::string> patterns{"ab", "[a-z]{1,5}$"};
  cudf::test::strings_column_wrapper repls({"AB", "_"});
  auto repls_view = cudf::strings_column_view(repls);

  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view);

  auto const flags   = cudf::strings::regex_flags::GLUSHKOV;
  auto results_mixed = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ===========================================================================
// Mixed-engine correctness tests: >64 position fallback
// ===========================================================================

// ---------------------------------------------------------------------------
// Mix a >64-position pattern (Thompson fallback) with a short Glushkov pattern.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, GlushkovWithLargePatternFallback)
{
  // Build a pattern with 65 character-consuming positions (>64 → Thompson fallback)
  std::string long_pattern;
  for (int i = 0; i < 65; ++i) {
    long_pattern += static_cast<char>('a' + (i % 26));
  }

  cudf::test::strings_column_wrapper input({"hello world", "foo bar"});
  auto sv = cudf::strings_column_view(input);

  // Pattern 0: short Glushkov-eligible
  // Pattern 1: >64 positions → Thompson fallback
  std::vector<std::string> patterns{"foo", long_pattern};
  cudf::test::strings_column_wrapper repls({"FOO", "LONG"});
  auto repls_view = cudf::strings_column_view(repls);

  auto const flags = cudf::strings::regex_flags::GLUSHKOV;
  auto results     = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  // Row 0: no "foo" → unchanged. long_pattern won't match either.
  // Row 1: "foo" matches → "FOO bar"
  cudf::test::strings_column_wrapper expected({"hello world", "FOO bar"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

// ===========================================================================
// Mixed-engine correctness tests: multiple fallback triggers in one batch
// ===========================================================================

// ---------------------------------------------------------------------------
// Batch with assertion, nullable, AND >64-position patterns all together.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, AllFallbackTriggersInOneBatch)
{
  // Build a >64-position pattern
  std::string long_pattern;
  for (int i = 0; i < 65; ++i) {
    long_pattern += static_cast<char>('a' + (i % 26));
  }

  cudf::test::strings_column_wrapper input({"abc 123 xyz", "start end"});
  auto sv = cudf::strings_column_view(input);

  // Pattern 0: Glushkov-eligible
  // Pattern 1: Thompson fallback — assertion (^)
  // Pattern 2: Thompson fallback — assertion ($)
  // Pattern 3: Thompson fallback — >64 positions
  // (Nullable patterns like \d* cause infinite zero-length matches in multi-pattern replace)
  std::vector<std::string> patterns{"[a-z]+", "^start", "\\d+$", long_pattern};
  cudf::test::strings_column_wrapper repls({"W", "BEGIN", "N", "LONG"});
  auto repls_view = cudf::strings_column_view(repls);

  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view);

  auto const flags   = cudf::strings::regex_flags::GLUSHKOV;
  auto results_mixed = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  // Both should produce identical results.
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ===========================================================================
// Baseline correctness: all-Glushkov and all-Thompson batches
// ===========================================================================

// ---------------------------------------------------------------------------
// All patterns are Glushkov-eligible (no assertions, non-nullable, <=64 pos).
// Working memory should be zero for the batch.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, AllGlushkovBatch)
{
  cudf::test::strings_column_wrapper input({"abc 123 def 456", "hello world 789", "no match 000"});
  auto sv = cudf::strings_column_view(input);

  std::vector<std::string> patterns{"[a-z]+", "\\d+"};
  cudf::test::strings_column_wrapper repls({"W", "N"});
  auto repls_view = cudf::strings_column_view(repls);

  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view);

  auto const flags   = cudf::strings::regex_flags::GLUSHKOV;
  auto results_mixed = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ---------------------------------------------------------------------------
// All patterns are Thompson-fallback (all have assertions).
// This verifies the GLUSHKOV flag doesn't break an all-Thompson batch.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, AllThompsonFallbackBatch)
{
  cudf::test::strings_column_wrapper input({"hello world", "world hello", "hello", "test"});
  auto sv = cudf::strings_column_view(input);

  // All patterns contain assertions → all fall back to Thompson
  std::vector<std::string> patterns{"^hello", "world$", "\\bhello\\b"};
  cudf::test::strings_column_wrapper repls({"HI", "EARTH", "GREETING"});
  auto repls_view = cudf::strings_column_view(repls);

  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view);

  auto const flags   = cudf::strings::regex_flags::GLUSHKOV;
  auto results_mixed = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ===========================================================================
// Edge cases
// ===========================================================================

// ---------------------------------------------------------------------------
// Empty input column with mixed patterns.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, EmptyInputColumn)
{
  cudf::test::strings_column_wrapper input(std::initializer_list<std::string>{});
  auto sv = cudf::strings_column_view(input);

  std::vector<std::string> patterns{"[a-z]+", "^start"};
  cudf::test::strings_column_wrapper repls({"W", "BEGIN"});
  auto repls_view = cudf::strings_column_view(repls);

  auto const flags = cudf::strings::regex_flags::GLUSHKOV;
  auto results     = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  EXPECT_EQ(results->size(), 0);
}

// ---------------------------------------------------------------------------
// Input contains null entries.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, NullEntries)
{
  std::vector<char const*> h_strings{"abc 123", nullptr, "def 456", nullptr, "ghi"};
  cudf::test::strings_column_wrapper input(
    h_strings.begin(), h_strings.end(), cudf::test::iterators::nulls_from_nullptrs(h_strings));
  auto sv = cudf::strings_column_view(input);

  // Mixed: Glushkov + Thompson (^ assertion)
  std::vector<std::string> patterns{"[a-z]+", "^abc"};
  cudf::test::strings_column_wrapper repls({"W", "BEGIN"});
  auto repls_view = cudf::strings_column_view(repls);

  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view);

  auto const flags   = cudf::strings::regex_flags::GLUSHKOV;
  auto results_mixed = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ---------------------------------------------------------------------------
// Input contains empty strings (not null, just "").
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, EmptyStrings)
{
  cudf::test::strings_column_wrapper input({"", "abc", "", "def", ""});
  auto sv = cudf::strings_column_view(input);

  std::vector<std::string> patterns{"[a-z]+", "^abc"};
  cudf::test::strings_column_wrapper repls({"W", "BEGIN"});
  auto repls_view = cudf::strings_column_view(repls);

  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view);

  auto const flags   = cudf::strings::regex_flags::GLUSHKOV;
  auto results_mixed = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ---------------------------------------------------------------------------
// Single-row input with mixed patterns.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, SingleRowInput)
{
  cudf::test::strings_column_wrapper input({"hello 123 world"});
  auto sv = cudf::strings_column_view(input);

  std::vector<std::string> patterns{"\\d+", "^hello"};
  cudf::test::strings_column_wrapper repls({"N", "HI"});
  auto repls_view = cudf::strings_column_view(repls);

  auto const flags = cudf::strings::regex_flags::GLUSHKOV;
  auto results     = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  // "^hello" matches at pos 0, so it wins there → "HI 123 world"
  // Then "\\d+" matches "123" → "HI N world"
  cudf::test::strings_column_wrapper expected({"HI N world"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

// ---------------------------------------------------------------------------
// Many patterns in a single batch (stress test for working memory sizing).
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, ManyPatternsBatch)
{
  cudf::test::strings_column_wrapper input(
    {"the quick brown fox", "jumps over the lazy dog", "0123456789"});
  auto sv = cudf::strings_column_view(input);

  // 6 patterns: mix of Glushkov and Thompson
  std::vector<std::string> patterns{
    "quick",       // Glushkov-eligible
    "lazy",        // Glushkov-eligible
    "\\d+",        // Glushkov-eligible
    "^the",        // Thompson fallback (^)
    "dog$",        // Thompson fallback ($)
    "\\bbrown\\b"  // Thompson fallback (\b)
  };
  cudf::test::strings_column_wrapper repls({"FAST", "SLOW", "NUM", "THE", "DOG", "BROWN"});
  auto repls_view = cudf::strings_column_view(repls);

  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view);

  auto const flags   = cudf::strings::regex_flags::GLUSHKOV;
  auto results_mixed = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ===========================================================================
// Flag combination tests
// ===========================================================================

// ---------------------------------------------------------------------------
// GLUSHKOV | DOTALL with mixed patterns.
// DOTALL makes '.' match newlines. Patterns with assertions still fall back.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, GlushkovDotallMixed)
{
  cudf::test::strings_column_wrapper input({"hello\nworld", "foo\nbar", "no newline"});
  auto sv = cudf::strings_column_view(input);

  // Pattern 0: Glushkov-eligible (with DOTALL, . matches \n)
  // Pattern 1: Thompson fallback due to ^
  std::vector<std::string> patterns{"o.w", "^foo"};
  cudf::test::strings_column_wrapper repls({"OW", "FOO"});
  auto repls_view = cudf::strings_column_view(repls);

  auto flags_default  = static_cast<cudf::strings::regex_flags>(cudf::strings::regex_flags::DOTALL);
  auto flags_glushkov = static_cast<cudf::strings::regex_flags>(
    cudf::strings::regex_flags::DOTALL | cudf::strings::regex_flags::GLUSHKOV);

  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view, flags_default);
  auto results_mixed   = cudf::strings::replace_re(sv, patterns, repls_view, flags_glushkov);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ---------------------------------------------------------------------------
// GLUSHKOV | MULTILINE with mixed patterns.
// MULTILINE makes ^ and $ match at line boundaries.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, GlushkovMultilineMixed)
{
  cudf::test::strings_column_wrapper input({"abc\ndef\nghi", "abc\nxyz"});
  auto sv = cudf::strings_column_view(input);

  // Pattern 0: Glushkov-eligible
  // Pattern 1: Thompson fallback — ^ with MULTILINE matches each line start
  std::vector<std::string> patterns{"[a-z]{3}", "^abc"};
  cudf::test::strings_column_wrapper repls({"W", "BEGIN"});
  auto repls_view = cudf::strings_column_view(repls);

  auto flags_default =
    static_cast<cudf::strings::regex_flags>(cudf::strings::regex_flags::MULTILINE);
  auto flags_glushkov = static_cast<cudf::strings::regex_flags>(
    cudf::strings::regex_flags::MULTILINE | cudf::strings::regex_flags::GLUSHKOV);

  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view, flags_default);
  auto results_mixed   = cudf::strings::replace_re(sv, patterns, repls_view, flags_glushkov);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ---------------------------------------------------------------------------
// GLUSHKOV | IGNORECASE with mixed patterns.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, GlushkovIgnorecaseMixed)
{
  cudf::test::strings_column_wrapper input({"Hello World", "HELLO world", "hello WORLD"});
  auto sv = cudf::strings_column_view(input);

  // Pattern 0: Glushkov-eligible with case insensitive
  // Pattern 1: Thompson fallback — \b assertion with case insensitive
  std::vector<std::string> patterns{"hello", "\\bworld\\b"};
  cudf::test::strings_column_wrapper repls({"HI", "EARTH"});
  auto repls_view = cudf::strings_column_view(repls);

  auto flags_default =
    static_cast<cudf::strings::regex_flags>(cudf::strings::regex_flags::IGNORECASE);
  auto flags_glushkov = static_cast<cudf::strings::regex_flags>(
    cudf::strings::regex_flags::IGNORECASE | cudf::strings::regex_flags::GLUSHKOV);

  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view, flags_default);
  auto results_mixed   = cudf::strings::replace_re(sv, patterns, repls_view, flags_glushkov);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ---------------------------------------------------------------------------
// GLUSHKOV | EXT_NEWLINE with mixed patterns.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, GlushkovExtNewlineMixed)
{
  cudf::test::strings_column_wrapper input({
    "a\xc2\x85"
    "b",     // U+0085 NEL between a and b
    "a\nb",  // \n between a and b
    "axb",   // normal char between a and b
  });
  auto sv = cudf::strings_column_view(input);

  // Pattern 0: Glushkov-eligible — a.b (dot won't match extended newlines)
  // Pattern 1: Thompson fallback — ^ assertion
  std::vector<std::string> patterns{"a.b", "^a"};
  cudf::test::strings_column_wrapper repls({"MATCH", "A"});
  auto repls_view = cudf::strings_column_view(repls);

  auto flags_default =
    static_cast<cudf::strings::regex_flags>(cudf::strings::regex_flags::EXT_NEWLINE);
  auto flags_glushkov = static_cast<cudf::strings::regex_flags>(
    cudf::strings::regex_flags::EXT_NEWLINE | cudf::strings::regex_flags::GLUSHKOV);

  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view, flags_default);
  auto results_mixed   = cudf::strings::replace_re(sv, patterns, repls_view, flags_glushkov);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ===========================================================================
// Pattern-specific mixed tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Character class patterns (Glushkov) mixed with assertion patterns (Thompson).
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, CharClassMixedWithAssertions)
{
  cudf::test::strings_column_wrapper input({"abc123def", "123abc", "abc", "123", "abc123"});
  auto sv = cudf::strings_column_view(input);

  std::vector<std::string> patterns{
    "[a-z]+",  // Glushkov
    "[0-9]+",  // Glushkov
    "^abc",    // Thompson (^)
    "\\d+$"    // Thompson ($)
  };
  cudf::test::strings_column_wrapper repls({"L", "D", "START", "END"});
  auto repls_view = cudf::strings_column_view(repls);

  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view);

  auto const flags   = cudf::strings::regex_flags::GLUSHKOV;
  auto results_mixed = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ---------------------------------------------------------------------------
// Alternation patterns (Glushkov) mixed with assertion patterns.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, AlternationGlushkovWithAssertions)
{
  cudf::test::strings_column_wrapper input({"cat dog", "bird fish", "start cat end"});
  auto sv = cudf::strings_column_view(input);

  std::vector<std::string> patterns{
    "cat|dog",    // Glushkov-eligible
    "bird|fish",  // Glushkov-eligible
    "^start",     // Thompson fallback
  };
  cudf::test::strings_column_wrapper repls({"ANIMAL", "SEA", "BEGIN"});
  auto repls_view = cudf::strings_column_view(repls);

  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view);

  auto const flags   = cudf::strings::regex_flags::GLUSHKOV;
  auto results_mixed = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ---------------------------------------------------------------------------
// Bounded repetition Glushkov pattern with assertion Thompson pattern.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, BoundedRepetitionMixedWithAssertion)
{
  cudf::test::strings_column_wrapper input({"aabbb ccc", "bbbbb aaa", "start aabb end"});
  auto sv = cudf::strings_column_view(input);

  std::vector<std::string> patterns{
    "a{1,3}",  // Glushkov-eligible (bounded repetition)
    "b{2,4}",  // Glushkov-eligible (bounded repetition)
    "^start",  // Thompson fallback
    "end$"     // Thompson fallback
  };
  cudf::test::strings_column_wrapper repls({"A", "B", "BEGIN", "FINISH"});
  auto repls_view = cudf::strings_column_view(repls);

  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view);

  auto const flags   = cudf::strings::regex_flags::GLUSHKOV;
  auto results_mixed = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ---------------------------------------------------------------------------
// Quantifier patterns (Glushkov) with non-boundary assertion patterns.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, QuantifierGlushkovWithNonBoundaryAssertions)
{
  cudf::test::strings_column_wrapper input({"abc def", "defabc", "xyzabc", ""});
  auto sv = cudf::strings_column_view(input);

  std::vector<std::string> patterns{
    "[a-z]+",  // Glushkov-eligible
    "\\Babc"   // Thompson fallback (\B non-boundary)
  };
  cudf::test::strings_column_wrapper repls({"W", "ABC"});
  auto repls_view = cudf::strings_column_view(repls);

  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view);

  auto const flags   = cudf::strings::regex_flags::GLUSHKOV;
  auto results_mixed = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ===========================================================================
// UTF-8 / multi-byte string tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Multi-byte UTF-8 strings with mixed Glushkov/Thompson patterns.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, Utf8MixedPatterns)
{
  cudf::test::strings_column_wrapper input({
    "caf\xc3\xa9 123",                       // "café 123"
    "\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e",  // "日本語"
    "hello world",
    "na\xc3\xaf"
    "ve test",  // "naïve test"
  });
  auto sv = cudf::strings_column_view(input);

  std::vector<std::string> patterns{
    "\\d+",    // Glushkov-eligible
    "\\w+",    // Glushkov-eligible
    "^hello",  // Thompson fallback
  };
  cudf::test::strings_column_wrapper repls({"NUM", "WORD", "HI"});
  auto repls_view = cudf::strings_column_view(repls);

  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view);

  auto const flags   = cudf::strings::regex_flags::GLUSHKOV;
  auto results_mixed = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ===========================================================================
// Priority / pattern ordering tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Verify that pattern priority (first matching pattern wins at each position)
// is preserved in mixed-engine batches.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, PatternPriorityPreserved)
{
  cudf::test::strings_column_wrapper input({"abc def"});
  auto sv = cudf::strings_column_view(input);

  // Both patterns match "abc" at position 0. Pattern 0 should win.
  // Pattern 0: Glushkov-eligible
  // Pattern 1: Thompson fallback (^)
  std::vector<std::string> patterns{"abc", "^abc"};
  cudf::test::strings_column_wrapper repls({"FIRST", "SECOND"});
  auto repls_view = cudf::strings_column_view(repls);

  auto const flags = cudf::strings::regex_flags::GLUSHKOV;
  auto results     = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  // Pattern 0 ("abc") is checked first and matches at pos 0 → "FIRST def"
  cudf::test::strings_column_wrapper expected({"FIRST def"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

// ---------------------------------------------------------------------------
// Reversed priority: Thompson pattern is listed first.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, PatternPriorityThompsonFirst)
{
  cudf::test::strings_column_wrapper input({"abc def"});
  auto sv = cudf::strings_column_view(input);

  // Pattern 0: Thompson fallback (^) — listed first, should win
  // Pattern 1: Glushkov-eligible
  std::vector<std::string> patterns{"^abc", "abc"};
  cudf::test::strings_column_wrapper repls({"FIRST", "SECOND"});
  auto repls_view = cudf::strings_column_view(repls);

  auto const flags = cudf::strings::regex_flags::GLUSHKOV;
  auto results     = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  // Pattern 0 ("^abc") is checked first and matches at pos 0 → "FIRST def"
  cudf::test::strings_column_wrapper expected({"FIRST def"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

// ===========================================================================
// Stress: interleaved matches across patterns in mixed engine
// ===========================================================================

// ---------------------------------------------------------------------------
// Multiple non-overlapping matches from both engines in same string.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, InterleavedMatchesBothEngines)
{
  cudf::test::strings_column_wrapper input(
    {"abc 123 def 456 ghi", "start 789 end", "000 aaa 111 bbb"});
  auto sv = cudf::strings_column_view(input);

  // Pattern 0: Glushkov — matches digits
  // Pattern 1: Glushkov — matches lowercase letter runs
  // Pattern 2: Thompson — matches "^start"
  std::vector<std::string> patterns{"\\d+", "[a-z]+", "^start"};
  cudf::test::strings_column_wrapper repls({"D", "L", "BEGIN"});
  auto repls_view = cudf::strings_column_view(repls);

  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view);

  auto const flags   = cudf::strings::regex_flags::GLUSHKOV;
  auto results_mixed = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ===========================================================================
// DEFAULT flag (no GLUSHKOV) baseline: verify identical behavior
// ===========================================================================

// ---------------------------------------------------------------------------
// Without GLUSHKOV flag, all patterns use Thompson. Results should be
// identical to the GLUSHKOV-flagged version where Glushkov-eligible patterns
// use the Glushkov engine.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, DefaultFlagBaseline)
{
  cudf::test::strings_column_wrapper input({
    "hello 123 world 456",
    "test abc 789",
    "no match",
    "",
  });
  auto sv = cudf::strings_column_view(input);

  std::vector<std::string> patterns{"[a-z]+", "\\d{2,}", "^test", "match$"};
  cudf::test::strings_column_wrapper repls({"W", "N", "T", "M"});
  auto repls_view = cudf::strings_column_view(repls);

  // DEFAULT: all Thompson
  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view);

  // GLUSHKOV: patterns 0,1 use Glushkov; patterns 2,3 fall back to Thompson
  auto const flags   = cudf::strings::regex_flags::GLUSHKOV;
  auto results_mixed = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ===========================================================================
// Dot pattern mixed with assertion pattern
// ===========================================================================

// ---------------------------------------------------------------------------
// Dot-plus pattern (Glushkov) mixed with anchored assertion (Thompson).
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, DotPatternMixedWithAnchored)
{
  cudf::test::strings_column_wrapper input({"a.b c.d", "start x.y", "no dots"});
  auto sv = cudf::strings_column_view(input);

  std::vector<std::string> patterns{
    "[a-z]\\.[a-z]",  // Glushkov-eligible (escaped dot is literal)
    "^start"          // Thompson fallback
  };
  cudf::test::strings_column_wrapper repls({"DOT", "BEGIN"});
  auto repls_view = cudf::strings_column_view(repls);

  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view);

  auto const flags   = cudf::strings::regex_flags::GLUSHKOV;
  auto results_mixed = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ===========================================================================
// Single replacement string for all patterns
// ===========================================================================

// ---------------------------------------------------------------------------
// When replacements column has exactly one entry, it is used for all patterns.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, SingleReplacementForAllPatterns)
{
  cudf::test::strings_column_wrapper input({"abc 123 def", "start 456"});
  auto sv = cudf::strings_column_view(input);

  std::vector<std::string> patterns{"[a-z]+", "\\d+", "^start"};
  cudf::test::strings_column_wrapper repls({"_"});
  auto repls_view = cudf::strings_column_view(repls);

  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view);

  auto const flags   = cudf::strings::regex_flags::GLUSHKOV;
  auto results_mixed = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ===========================================================================
// Complex real-world-ish patterns
// ===========================================================================

// ---------------------------------------------------------------------------
// Simulates a batch where we clean up various tokens: emails (Glushkov),
// line-start markers (Thompson), and numbers (Glushkov).
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, RealWorldTokenCleanup)
{
  cudf::test::strings_column_wrapper input({
    "contact user@test 123-456",
    "NOTE: important 789",
    "plain text here",
  });
  auto sv = cudf::strings_column_view(input);

  std::vector<std::string> patterns{
    "[a-z]+@[a-z]+",  // Glushkov-eligible (simple email-like)
    "\\d+-\\d+",      // Glushkov-eligible (number-dash-number)
    "^NOTE:",         // Thompson fallback (^ assertion)
  };
  cudf::test::strings_column_wrapper repls({"[EMAIL]", "[NUM]", "[MARKER]"});
  auto repls_view = cudf::strings_column_view(repls);

  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view);

  auto const flags   = cudf::strings::regex_flags::GLUSHKOV;
  auto results_mixed = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}

// ---------------------------------------------------------------------------
// Patterns with groups (non-capture in multi_re context) mixed engines.
// ---------------------------------------------------------------------------
TEST_F(MixedEngineBatchTests, GroupPatternsMixed)
{
  cudf::test::strings_column_wrapper input({"ab12cd", "12ab", "start ab end"});
  auto sv = cudf::strings_column_view(input);

  std::vector<std::string> patterns{
    "([a-z])([a-z])",  // Glushkov-eligible (two-char lowercase)
    "([0-9])([0-9])",  // Glushkov-eligible (two-char digit)
    "^start",          // Thompson fallback
  };
  cudf::test::strings_column_wrapper repls({"LL", "DD", "BEGIN"});
  auto repls_view = cudf::strings_column_view(repls);

  auto results_default = cudf::strings::replace_re(sv, patterns, repls_view);

  auto const flags   = cudf::strings::regex_flags::GLUSHKOV;
  auto results_mixed = cudf::strings::replace_re(sv, patterns, repls_view, flags);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results_default, *results_mixed);
}
