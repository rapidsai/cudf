/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "generate_skewed_data.hpp"

#include <benchmarks/common/generate_input.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <array>
#include <string>
#include <string_view>
#include <vector>

constexpr std::string_view skewed_string_target_substring{"0987 5W43"};

namespace {
/**
 * @brief Template strings for skewed string benchmarks.
 *
 * Based on the strings used by `create_string_column`, with the first entry ending in
 * `skewed_string_target_substring`.
 */
constexpr auto template_strings = std::to_array<std::string_view>({
  "123 abc 4567890 DEFGHI 0987 5W43123 abc 4567890 DEFGHI 0987 5W43123 abc 4567890 DEFGHI 0987 "
  "5W43123 abc 4567890 DEFGHI 0987 5W43123 abc 4567890 DEFGHI 0987 5W43123 abc 4567890 DEFGHI 0987 "
  "5W43123 abc 4567890 DEFGHI 0987 5W43123 abc 4567890 DEFGHI 0987 5W43",  // matches both patterns;
  "012345 6789 01234 56789 0123 456012345 6789 01234 56789 0123 456012345 6789 01234 56789 0123 "
  "456012345 6789 01234 56789 0123 456012345 6789 01234 56789 0123 456012345 6789 01234 56789 0123 "
  "456012345 6789 01234 56789 0123 456012345 6789 01234 56789 0123 456",  // the rest do not match
  "abc 4567890 DEFGHI 0987 Wxyz 123abc 4567890 DEFGHI 0987 Wxyz 123abc 4567890 DEFGHI 0987 Wxyz "
  "123abc 4567890 DEFGHI 0987 Wxyz 123abc 4567890 DEFGHI 0987 Wxyz 123abc 4567890 DEFGHI 0987 Wxyz "
  "123abc 4567890 DEFGHI 0987 Wxyz 123abc 4567890 DEFGHI 0987 Wxyz 123",
  "abcdefghijklmnopqrstuvwxyz 01234abcdefghijklmnopqrstuvwxyz 01234abcdefghijklmnopqrstuvwxyz "
  "01234abcdefghijklmnopqrstuvwxyz 01234abcdefghijklmnopqrstuvwxyz 01234abcdefghijklmnopqrstuvwxyz "
  "01234abcdefghijklmnopqrstuvwxyz 01234abcdefghijklmnopqrstuvwxyz 01234",
  "sksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksk"
  "sksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksk"
  "sksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksksk",
  "AbcéDEFGHIJKLMNOPQRSTUVWXYZ 012AbcéDEFGHIJKLMNOPQRSTUVWXYZ 012AbcéDEFGHIJKLMNOPQRSTUVWXYZ "
  "012AbcéDEFGHIJKLMNOPQRSTUVWXYZ 012AbcéDEFGHIJKLMNOPQRSTUVWXYZ 012AbcéDEFGHIJKLMNOPQRSTUVWXYZ "
  "012AbcéDEFGHIJKLMNOPQRSTUVWXYZ 012AbcéDEFGHIJKLMNOPQRSTUVWXYZ 012",
  "9876543210,abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,"
  "abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,"
  "abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU",
  "9876543210,abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,"
  "abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,"
  "abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU",
  "123 édf 4567890 DéFG 0987 X567123 édf 4567890 DéFG 0987 X567123 édf 4567890 DéFG 0987 X567123 "
  "édf "
  "4567890 DéFG 0987 X567123 édf 4567890 DéFG 0987 X567123 édf 4567890 DéFG 0987 X567123 édf "
  "4567890 "
  "DéFG 0987 X567123 édf 4567890 DéFG 0987 X567",
  "111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111"
  "111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111"
  "1111111111111111111111111111111111111111111111111111111111111111",
});

/**
 * @brief Repeat a string until it reaches the given width, then take `str[0:width]`.
 */
std::string repeat_to_width(std::string_view str, std::size_t width)
{
  CUDF_EXPECTS(str.size() == 256, "string must an element of template_strings");
  CUDF_EXPECTS(width % 32 == 0,
               "width must be a multiple of 32 so that this function is Unicode compliant");
  std::string result;
  result.reserve(width);
  while (result.size() < width) {
    result += str.substr(0, std::min(width - result.size(), str.size()));
  }
  return result;
}

/**
 * @brief Build the long template column by repeating each string to `template_width`.
 */
std::unique_ptr<cudf::column> make_long_template_column(cudf::size_type template_width)
{
  std::vector<std::string> repeated;
  repeated.reserve(template_strings.size());
  for (auto const& str : template_strings) {
    repeated.push_back(repeat_to_width(str, template_width));
  }
  return cudf::test::strings_column_wrapper(repeated.begin(), repeated.end()).release();
}

/**
 * @brief Build a column by gathering from the template strings with the given hit rate.
 *
 * This follows the same gather/scatter pattern as `create_string_column`.
 */
std::unique_ptr<cudf::column> gather_with_hit_rate(cudf::column_view const& templates,
                                                   cudf::size_type num_rows,
                                                   int32_t hit_rate)
{
  auto const num_matches = (static_cast<int64_t>(num_rows) * hit_rate) / 100;

  data_profile gather_profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_id::INT32, distribution_id::UNIFORM, 1, templates.size() - 1);
  auto gather_table =
    create_random_table({cudf::type_id::INT32}, row_count{num_rows}, gather_profile);

  if (num_matches > 0) {  // guard against division by zero
    auto const zero_scalar  = cudf::numeric_scalar<int32_t>(0);
    auto const scatter_data = cudf::sequence(
      num_matches, zero_scalar, cudf::numeric_scalar<int32_t>(num_rows / num_matches));
    gather_table = cudf::scatter({zero_scalar}, scatter_data->view(), gather_table->view());
  }

  auto const gather_map = gather_table->view().column(0);
  auto table            = cudf::gather(cudf::table_view({templates}), gather_map);
  return std::move(table->release().front());
}

/**
 * @brief Build a mask indicating which rows should use the cropped short string length.
 */
std::unique_ptr<cudf::column> make_short_row_mask(cudf::size_type num_rows,
                                                  cudf::size_type short_string_pct)
{
  auto const num_short =
    static_cast<cudf::size_type>((static_cast<int64_t>(num_rows) * short_string_pct) / 100);
  auto type_table = create_random_table({cudf::type_id::INT32},
                                        row_count{num_rows},
                                        data_profile_builder().no_validity().distribution(
                                          cudf::type_id::INT32, distribution_id::UNIFORM, 1, 1));

  if (num_short > 0) {  // guard against division by zero
    auto const zero_scalar = cudf::numeric_scalar<int32_t>(0);
    auto const scatter_data =
      cudf::sequence(num_short, zero_scalar, cudf::numeric_scalar<int32_t>(num_rows / num_short));
    type_table = cudf::scatter({zero_scalar}, scatter_data->view(), type_table->view());
  }

  auto const zero_col = cudf::make_column_from_scalar(cudf::numeric_scalar<int32_t>(0), num_rows);
  return cudf::binary_operation(type_table->view().column(0),
                                zero_col->view(),
                                cudf::binary_operator::EQUAL,
                                cudf::data_type{cudf::type_id::BOOL8});
}
}  // namespace

std::unique_ptr<cudf::column> create_skewed_string_column(cudf::size_type num_rows,
                                                          cudf::size_type short_length,
                                                          cudf::size_type long_tail_length,
                                                          int32_t short_string_pct,
                                                          int32_t hit_rate)
{
  CUDF_EXPECTS(num_rows >= 0, "num_rows must be non-negative");
  CUDF_EXPECTS(short_length >= 0, "short_length must be non-negative");
  CUDF_EXPECTS(long_tail_length >= 0, "long_tail_length must be non-negative");
  CUDF_EXPECTS(short_string_pct >= 0 && short_string_pct <= 100,
               "short_string_pct must be in the range [0, 100]");
  CUDF_EXPECTS(hit_rate >= 0 && hit_rate <= 100, "hit_rate must be in the range [0, 100]");
  CUDF_EXPECTS(long_tail_length >= short_length,
               "long_tail_length must be greater than or equal to short_length");

  auto const templates     = make_long_template_column(long_tail_length);
  auto const full_col      = gather_with_hit_rate(templates->view(), num_rows, hit_rate);
  auto const is_short_mask = make_short_row_mask(num_rows, short_string_pct);

  auto const starts =
    cudf::make_column_from_scalar(cudf::numeric_scalar<cudf::size_type>(0), num_rows);
  auto const short_stop = cudf::numeric_scalar<cudf::size_type>(short_length);
  auto const long_stop  = cudf::numeric_scalar<cudf::size_type>(long_tail_length);
  auto const stops      = cudf::copy_if_else(short_stop, long_stop, is_short_mask->view());

  return cudf::strings::slice_strings(
    cudf::strings_column_view(full_col->view()), starts->view(), stops->view());
}
