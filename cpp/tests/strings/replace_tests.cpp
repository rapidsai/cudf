/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/strings/detail/replace.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <vector>

using algorithm = cudf::strings::detail::replace_algorithm;

struct StringsReplaceTest : public cudf::test::BaseFixture {
  cudf::test::strings_column_wrapper build_corpus()
  {
    std::vector<char const*> h_strings{"the quick brown fox jumps over the lazy dog",
                                       "the fat cat lays next to the other accénted cat",
                                       "a slow moving turtlé cannot catch the bird",
                                       "which can be composéd together to form a more complete",
                                       "The result does not include the value in the sum in",
                                       "",
                                       nullptr};

    return cudf::test::strings_column_wrapper(
      h_strings.begin(),
      h_strings.end(),
      thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  }
};

TEST_F(StringsReplaceTest, Replace)
{
  auto input        = build_corpus();
  auto strings_view = cudf::strings_column_view(input);
  // replace all occurrences of 'the ' with '++++ '
  std::vector<char const*> h_expected{"++++ quick brown fox jumps over ++++ lazy dog",
                                      "++++ fat cat lays next to ++++ other accénted cat",
                                      "a slow moving turtlé cannot catch ++++ bird",
                                      "which can be composéd together to form a more complete",
                                      "The result does not include ++++ value in ++++ sum in",
                                      "",
                                      nullptr};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(), h_expected.end(), cudf::test::iterators::nulls_from_nullptrs(h_expected));

  auto stream = cudf::get_default_stream();
  auto mr     = rmm::mr::get_current_device_resource();

  auto results =
    cudf::strings::replace(strings_view, cudf::string_scalar("the "), cudf::string_scalar("++++ "));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = cudf::strings::detail::replace<algorithm::CHAR_PARALLEL>(
    strings_view, cudf::string_scalar("the "), cudf::string_scalar("++++ "), -1, stream, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = cudf::strings::detail::replace<algorithm::ROW_PARALLEL>(
    strings_view, cudf::string_scalar("the "), cudf::string_scalar("++++ "), -1, stream, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsReplaceTest, ReplaceReplLimit)
{
  auto input        = build_corpus();
  auto strings_view = cudf::strings_column_view(input);
  auto stream       = cudf::get_default_stream();
  auto mr           = rmm::mr::get_current_device_resource();

  // only remove the first occurrence of 'the '
  std::vector<char const*> h_expected{"quick brown fox jumps over the lazy dog",
                                      "fat cat lays next to the other accénted cat",
                                      "a slow moving turtlé cannot catch bird",
                                      "which can be composéd together to form a more complete",
                                      "The result does not include value in the sum in",
                                      "",
                                      nullptr};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(), h_expected.end(), cudf::test::iterators::nulls_from_nullptrs(h_expected));
  auto results =
    cudf::strings::replace(strings_view, cudf::string_scalar("the "), cudf::string_scalar(""), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = cudf::strings::detail::replace<algorithm::CHAR_PARALLEL>(
    strings_view, cudf::string_scalar("the "), cudf::string_scalar(""), 1, stream, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = cudf::strings::detail::replace<algorithm::ROW_PARALLEL>(
    strings_view, cudf::string_scalar("the "), cudf::string_scalar(""), 1, stream, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsReplaceTest, ReplaceReplLimitInputSliced)
{
  auto input = build_corpus();
  // replace first two occurrences of ' ' with '--'
  std::vector<char const*> h_expected{"the--quick--brown fox jumps over the lazy dog",
                                      "the--fat--cat lays next to the other accénted cat",
                                      "a--slow--moving turtlé cannot catch the bird",
                                      "which--can--be composéd together to form a more complete",
                                      "The--result--does not include the value in the sum in",
                                      "",
                                      nullptr};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(), h_expected.end(), cudf::test::iterators::nulls_from_nullptrs(h_expected));
  auto stream = cudf::get_default_stream();
  auto mr     = rmm::mr::get_current_device_resource();
  std::vector<cudf::size_type> slice_indices{0, 2, 2, 3, 3, 7};
  auto sliced_strings  = cudf::slice(input, slice_indices);
  auto sliced_expected = cudf::slice(expected, slice_indices);
  for (size_t i = 0; i < sliced_strings.size(); ++i) {
    auto strings_view = cudf::strings_column_view(sliced_strings[i]);
    auto results =
      cudf::strings::replace(strings_view, cudf::string_scalar(" "), cudf::string_scalar("--"), 2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, sliced_expected[i]);
    results = cudf::strings::detail::replace<algorithm::CHAR_PARALLEL>(
      strings_view, cudf::string_scalar(" "), cudf::string_scalar("--"), 2, stream, mr);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, sliced_expected[i]);
    results = cudf::strings::detail::replace<algorithm::ROW_PARALLEL>(
      strings_view, cudf::string_scalar(" "), cudf::string_scalar("--"), 2, stream, mr);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, sliced_expected[i]);
  }
}

TEST_F(StringsReplaceTest, ReplaceTargetOverlap)
{
  auto corpus      = build_corpus();
  auto corpus_view = cudf::strings_column_view(corpus);
  // replace all occurrences of 'the ' with '+++++++ '
  auto input = cudf::strings::replace(
    corpus_view, cudf::string_scalar("the "), cudf::string_scalar("++++++++ "));
  auto strings_view = cudf::strings_column_view(*input);
  // replace all occurrences of '+++' with 'plus '
  std::vector<char const*> h_expected{
    "plus plus ++ quick brown fox jumps over plus plus ++ lazy dog",
    "plus plus ++ fat cat lays next to plus plus ++ other accénted cat",
    "a slow moving turtlé cannot catch plus plus ++ bird",
    "which can be composéd together to form a more complete",
    "The result does not include plus plus ++ value in plus plus ++ sum in",
    "",
    nullptr};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(), h_expected.end(), cudf::test::iterators::nulls_from_nullptrs(h_expected));

  auto stream = cudf::get_default_stream();
  auto mr     = rmm::mr::get_current_device_resource();

  auto results =
    cudf::strings::replace(strings_view, cudf::string_scalar("+++"), cudf::string_scalar("plus "));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::detail::replace<algorithm::CHAR_PARALLEL>(
    strings_view, cudf::string_scalar("+++"), cudf::string_scalar("plus "), -1, stream, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = cudf::strings::detail::replace<algorithm::ROW_PARALLEL>(
    strings_view, cudf::string_scalar("+++"), cudf::string_scalar("plus "), -1, stream, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsReplaceTest, ReplaceTargetOverlapsStrings)
{
  auto input        = build_corpus();
  auto strings_view = cudf::strings_column_view(input);
  auto stream       = cudf::get_default_stream();
  auto mr           = rmm::mr::get_current_device_resource();

  // replace all occurrences of 'dogthe' with '+'
  // should not replace anything unless it incorrectly matches across a string boundary
  auto results =
    cudf::strings::replace(strings_view, cudf::string_scalar("dogthe"), cudf::string_scalar("+"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, input);
  results = cudf::strings::detail::replace<algorithm::CHAR_PARALLEL>(
    strings_view, cudf::string_scalar("dogthe"), cudf::string_scalar("+"), -1, stream, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, input);
  results = cudf::strings::detail::replace<algorithm::ROW_PARALLEL>(
    strings_view, cudf::string_scalar("dogthe"), cudf::string_scalar("+"), -1, stream, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, input);
}

TEST_F(StringsReplaceTest, ReplaceNullInput)
{
  std::vector<char const*> h_null_strings(128);
  auto input = cudf::test::strings_column_wrapper(
    h_null_strings.begin(), h_null_strings.end(), thrust::make_constant_iterator(false));
  auto strings_view = cudf::strings_column_view(input);
  auto stream       = cudf::get_default_stream();
  auto mr           = rmm::mr::get_current_device_resource();
  // replace all occurrences of '+' with ''
  // should not replace anything as input is all null
  auto results =
    cudf::strings::replace(strings_view, cudf::string_scalar("+"), cudf::string_scalar(""));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, input);
  results = cudf::strings::detail::replace<algorithm::CHAR_PARALLEL>(
    strings_view, cudf::string_scalar("+"), cudf::string_scalar(""), -1, stream, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, input);
  results = cudf::strings::detail::replace<algorithm::ROW_PARALLEL>(
    strings_view, cudf::string_scalar("+"), cudf::string_scalar(""), -1, stream, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, input);
}

TEST_F(StringsReplaceTest, ReplaceEndOfString)
{
  auto input        = build_corpus();
  auto strings_view = cudf::strings_column_view(input);
  auto stream       = cudf::get_default_stream();
  auto mr           = rmm::mr::get_current_device_resource();

  // replace all occurrences of 'in' with  ' '
  std::vector<char const*> h_expected{"the quick brown fox jumps over the lazy dog",
                                      "the fat cat lays next to the other accénted cat",
                                      "a slow mov g turtlé cannot catch the bird",
                                      "which can be composéd together to form a more complete",
                                      "The result does not  clude the value   the sum  ",
                                      "",
                                      nullptr};

  cudf::test::strings_column_wrapper expected(
    h_expected.begin(), h_expected.end(), cudf::test::iterators::nulls_from_nullptrs(h_expected));

  auto results =
    cudf::strings::replace(strings_view, cudf::string_scalar("in"), cudf::string_scalar(" "));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::detail::replace<cudf::strings::detail::replace_algorithm::CHAR_PARALLEL>(
    strings_view, cudf::string_scalar("in"), cudf::string_scalar(" "), -1, stream, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::detail::replace<cudf::strings::detail::replace_algorithm::ROW_PARALLEL>(
    strings_view, cudf::string_scalar("in"), cudf::string_scalar(" "), -1, stream, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsReplaceTest, ReplaceSlice)
{
  std::vector<char const*> h_strings{"Héllo", "thesé", nullptr, "ARE THE", "tést strings", ""};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  {
    auto results = cudf::strings::replace_slice(strings_view, cudf::string_scalar("___"), 2, 3);
    std::vector<char const*> h_expected{
      "Hé___lo", "th___sé", nullptr, "AR___ THE", "té___t strings", "___"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::replace_slice(strings_view, cudf::string_scalar("||"), 3, 3);
    std::vector<char const*> h_expected{
      "Hél||lo", "the||sé", nullptr, "ARE|| THE", "tés||t strings", "||"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::replace_slice(strings_view, cudf::string_scalar("x"), -1, -1);
    std::vector<char const*> h_expected{
      "Héllox", "theséx", nullptr, "ARE THEx", "tést stringsx", "x"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsReplaceTest, ReplaceSliceError)
{
  cudf::test::strings_column_wrapper input({"Héllo", "thesé", "are not", "important", ""});
  EXPECT_THROW(
    cudf::strings::replace_slice(cudf::strings_column_view(input), cudf::string_scalar(""), 4, 1),
    cudf::logic_error);
}

TEST_F(StringsReplaceTest, ReplaceMulti)
{
  auto input        = build_corpus();
  auto strings_view = cudf::strings_column_view(input);

  cudf::test::strings_column_wrapper targets({"the ", "a ", "to "});
  auto targets_view = cudf::strings_column_view(targets);

  {
    cudf::test::strings_column_wrapper repls({"_ ", "A ", "2 "});
    auto repls_view = cudf::strings_column_view(repls);

    auto results = cudf::strings::replace(strings_view, targets_view, repls_view);

    std::vector<char const*> h_expected{"_ quick brown fox jumps over _ lazy dog",
                                        "_ fat cat lays next 2 _ other accénted cat",
                                        "A slow moving turtlé cannot catch _ bird",
                                        "which can be composéd together 2 form A more complete",
                                        "The result does not include _ value in _ sum in",
                                        "",
                                        nullptr};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  {
    cudf::test::strings_column_wrapper repls({"* "});
    auto repls_view = cudf::strings_column_view(repls);

    auto results = cudf::strings::replace(strings_view, targets_view, repls_view);

    std::vector<char const*> h_expected{"* quick brown fox jumps over * lazy dog",
                                        "* fat cat lays next * * other accénted cat",
                                        "* slow moving turtlé cannot catch * bird",
                                        "which can be composéd together * form * more complete",
                                        "The result does not include * value in * sum in",
                                        "",
                                        nullptr};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsReplaceTest, ReplaceMultiLong)
{
  // The length of the strings are to trigger the code path governed by the AVG_CHAR_BYTES_THRESHOLD
  // setting in the multi.cu.
  auto input = cudf::test::strings_column_wrapper(
    {"This string needs to be very long to trigger the long-replace internal functions. "
     "This string needs to be very long to trigger the long-replace internal functions. "
     "This string needs to be very long to trigger the long-replace internal functions. "
     "This string needs to be very long to trigger the long-replace internal functions.",
     "012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012"
     "345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345"
     "678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678"
     "901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901"
     "2345678901234567890123456789",
     "012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012"
     "345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345"
     "678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678"
     "901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901"
     "2345678901234567890123456789",
     "Test string for overlap check: bananaápple bananá ápplebananá banápple ápple bananá "
     "Test string for overlap check: bananaápple bananá ápplebananá banápple ápple bananá "
     "Test string for overlap check: bananaápple bananá ápplebananá banápple ápple bananá "
     "Test string for overlap check: bananaápple bananá ápplebananá banápple ápple bananá "
     "Test string for overlap check: bananaápple bananá ápplebananá banápple ápple bananá",
     "",
     ""},
    {1, 1, 1, 1, 0, 1});
  auto strings_view = cudf::strings_column_view(input);

  auto targets      = cudf::test::strings_column_wrapper({"78901", "bananá", "ápple", "78"});
  auto targets_view = cudf::strings_column_view(targets);

  {
    cudf::test::strings_column_wrapper repls({"x", "PEAR", "avocado", "$$"});
    auto repls_view = cudf::strings_column_view(repls);

    auto results = cudf::strings::replace(strings_view, targets_view, repls_view);

    cudf::test::strings_column_wrapper expected(
      {"This string needs to be very long to trigger the long-replace internal functions. "
       "This string needs to be very long to trigger the long-replace internal functions. "
       "This string needs to be very long to trigger the long-replace internal functions. "
       "This string needs to be very long to trigger the long-replace internal functions.",
       "0123456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456"
       "x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x"
       "23456x23456x23456x23456x23456x23456x23456x23456x23456x23456$$9",
       "0123456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456"
       "x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x"
       "23456x23456x23456x23456x23456x23456x23456x23456x23456x23456$$9",
       "Test string for overlap check: bananaavocado PEAR avocadoPEAR banavocado avocado PEAR "
       "Test string for overlap check: bananaavocado PEAR avocadoPEAR banavocado avocado PEAR "
       "Test string for overlap check: bananaavocado PEAR avocadoPEAR banavocado avocado PEAR "
       "Test string for overlap check: bananaavocado PEAR avocadoPEAR banavocado avocado PEAR "
       "Test string for overlap check: bananaavocado PEAR avocadoPEAR banavocado avocado PEAR",
       "",
       ""},
      {1, 1, 1, 1, 0, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  {
    cudf::test::strings_column_wrapper repls({"*"});
    auto repls_view = cudf::strings_column_view(repls);

    auto results = cudf::strings::replace(strings_view, targets_view, repls_view);

    cudf::test::strings_column_wrapper expected(
      {"This string needs to be very long to trigger the long-replace internal functions. "
       "This string needs to be very long to trigger the long-replace internal functions. "
       "This string needs to be very long to trigger the long-replace internal functions. "
       "This string needs to be very long to trigger the long-replace internal functions.",
       "0123456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*"
       "23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*"
       "23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*9",
       "0123456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*"
       "23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*"
       "23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*9",
       "Test string for overlap check: banana* * ** ban* * * Test string for overlap check: "
       "banana* * ** ban* * * Test string for overlap check: banana* * ** ban* * * Test string for "
       "overlap check: banana* * ** ban* * * Test string for overlap check: banana* * ** ban* * *",
       "",
       ""},
      {1, 1, 1, 1, 0, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  {
    targets =
      cudf::test::strings_column_wrapper({"01234567890123456789012345678901234567890123456789012345"
                                          "6789012345678901234567890123456789012"
                                          "34567890123456789012345678901234567890123456789012345678"
                                          "9012345678901234567890123456789012345"
                                          "67890123456789012345678901234567890123456789012345678901"
                                          "2345678901234567890123456789012345678"
                                          "90123456789012345678901234567890123456789012345678901234"
                                          "5678901234567890123456789012345678901"
                                          "2345678901234567890123456789",
                                          "78"});
    targets_view    = cudf::strings_column_view(targets);
    auto repls      = cudf::test::strings_column_wrapper({""});
    auto repls_view = cudf::strings_column_view(repls);

    auto results = cudf::strings::replace(strings_view, targets_view, repls_view);

    cudf::test::strings_column_wrapper expected(
      {"This string needs to be very long to trigger the long-replace internal functions. "
       "This string needs to be very long to trigger the long-replace internal functions. "
       "This string needs to be very long to trigger the long-replace internal functions. "
       "This string needs to be very long to trigger the long-replace internal functions.",
       "",
       "",
       "Test string for overlap check: bananaápple bananá ápplebananá banápple ápple bananá "
       "Test string for overlap check: bananaápple bananá ápplebananá banápple ápple bananá "
       "Test string for overlap check: bananaápple bananá ápplebananá banápple ápple bananá "
       "Test string for overlap check: bananaápple bananá ápplebananá banápple ápple bananá "
       "Test string for overlap check: bananaápple bananá ápplebananá banápple ápple bananá",
       "",
       ""},
      {1, 1, 1, 1, 0, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsReplaceTest, EmptyStringsColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();

  auto strings_view = cudf::strings_column_view(zero_size_strings_column);
  auto results      = cudf::strings::replace(
    strings_view, cudf::string_scalar("not"), cudf::string_scalar("pertinent"));
  auto view = results->view();
  cudf::test::expect_column_empty(results->view());
}
