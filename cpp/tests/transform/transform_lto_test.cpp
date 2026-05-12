/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/transform.hpp>

#include <cudf_test_fragments.hpp>

struct TransformLTOTest : public cudf::test::BaseFixture {};

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

TEST_F(TransformLTOTest, ComputeTpchLineitem)
{
  column_wrapper<double> extended_price{{21168.23, 45983.16, 13309.60, 28955.64}};
  column_wrapper<float> discount{{0.04f, 0.06f, 0.07f, 0.02f}};
  column_wrapper<float> tax{{0.02f, 0.08f, 0.04f, 0.06f}};
  column_wrapper<int32_t> ship_date{{19980901, 19980902, 19980903, 19960115}};

  column_wrapper<int32_t> ship_date_cutoff{{19980902}};

  std::vector<cudf::transform_input> inputs{
    extended_price, discount, tax, ship_date, cudf::scalar_column_view{ship_date_cutoff}};

  std::vector<cudf::transform_output> outputs{
    {cudf::data_type{cudf::type_id::FLOAT64}, cudf::output_nullability::ALL_VALID},
    {cudf::data_type{cudf::type_id::FLOAT64}, cudf::output_nullability::ALL_VALID},
    {cudf::data_type{cudf::type_id::FLOAT64}, cudf::output_nullability::ALL_VALID},
    {cudf::data_type{cudf::type_id::BOOL8}, cudf::output_nullability::ALL_VALID}};

  auto const range =
    cudf_test_fragments::file_ranges[cudf_test_fragments::transform_profit_operator];
  std::span<uint8_t const> udf{cudf_test_fragments::files.subspan(range[0], range[1])};

  auto result = cudf::transform_lto(inputs,
                                    udf,
                                    cudf::lto_binary_type::FATBIN,
                                    outputs,
                                    nullptr,
                                    cudf::null_aware::NO,
                                    std::nullopt,
                                    cudf::test::get_default_stream());

  column_wrapper<double> expected_base_price{{21168.23, 45983.16, 13309.60, 28955.64}};
  column_wrapper<double> expected_disc_price{{21168.23 * (1.0 - static_cast<double>(0.04f)),
                                              45983.16 * (1.0 - static_cast<double>(0.06f)),
                                              13309.60 * (1.0 - static_cast<double>(0.07f)),
                                              28955.64 * (1.0 - static_cast<double>(0.02f))}};
  column_wrapper<double> expected_charge{
    {(21168.23 * (1.0 - static_cast<double>(0.04f))) * (1.0 + static_cast<double>(0.02f)),
     (45983.16 * (1.0 - static_cast<double>(0.06f))) * (1.0 + static_cast<double>(0.08f)),
     (13309.60 * (1.0 - static_cast<double>(0.07f))) * (1.0 + static_cast<double>(0.04f)),
     (28955.64 * (1.0 - static_cast<double>(0.02f))) * (1.0 + static_cast<double>(0.06f))}};
  column_wrapper<bool> expected_before_cutoff{{true, true, false, true}};

  cudf::test::detail::expect_columns_equivalent(
    result->view().column(1), expected_charge, cudf::test::debug_output_level::FIRST_ERROR, 64);
  cudf::test::detail::expect_columns_equivalent(
    result->view().column(2), expected_disc_price, cudf::test::debug_output_level::FIRST_ERROR, 64);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view().column(3), expected_before_cutoff);
}

TEST_F(TransformLTOTest, InvSqrtOperator)
{
  column_wrapper<float> input{{1.0f, 4.0f, 9.0f, 16.0f}};

  std::vector<cudf::transform_input> inputs{input};

  std::vector<cudf::transform_output> outputs{
    {cudf::data_type{cudf::type_id::FLOAT32}, cudf::output_nullability::ALL_VALID}};

  auto const range =
    cudf_test_fragments::file_ranges[cudf_test_fragments::transform_invsqrt_operator];
  std::span<uint8_t const> udf{cudf_test_fragments::files.subspan(range[0], range[1])};

  auto result = cudf::transform_lto(inputs,
                                    udf,
                                    cudf::lto_binary_type::FATBIN,
                                    outputs,
                                    nullptr,
                                    cudf::null_aware::NO,
                                    std::nullopt,
                                    cudf::test::get_default_stream());

  column_wrapper<float> expected{{1.0f, 0.5f, 0.33333334f, 0.25f}};

  cudf::test::detail::expect_columns_equivalent(
    result->view().column(0), expected, cudf::test::debug_output_level::FIRST_ERROR, 64);
}

TEST_F(TransformLTOTest, TPCDS_Q7)
{
  column_wrapper<double> store_sales{{100.0, 20.0, 0.0, 45.5}};
  column_wrapper<double> catalog_sales{{10.0, 5.0, 3.0, 4.5}};
  column_wrapper<double> web_sales{{1.0, 0.0, 7.0, 0.5}};
  column_wrapper<double> store_returns{{4.0, 1.0, 0.0, 5.0}};
  column_wrapper<double> catalog_returns{{0.5, 0.0, 1.0, 0.5}};
  column_wrapper<double> web_returns{{0.5, 0.0, 2.0, 0.0}};
  column_wrapper<double> profit{{30.0, 5.0, -2.0, 8.0}};
  column_wrapper<double> profit_loss{{3.0, 1.0, 4.0, 0.5}};

  std::vector<cudf::transform_input> inputs{store_sales,
                                            catalog_sales,
                                            web_sales,
                                            store_returns,
                                            catalog_returns,
                                            web_returns,
                                            profit,
                                            profit_loss};

  std::vector<cudf::transform_output> outputs{
    {cudf::data_type{cudf::type_id::FLOAT64}, cudf::output_nullability::ALL_VALID},
    {cudf::data_type{cudf::type_id::FLOAT64}, cudf::output_nullability::ALL_VALID}};

  auto const range = cudf_test_fragments::file_ranges[cudf_test_fragments::transform_tpcds_q7];
  std::span<uint8_t const> udf{cudf_test_fragments::files.subspan(range[0], range[1])};

  auto result = cudf::transform_lto(inputs,
                                    udf,
                                    cudf::lto_binary_type::FATBIN,
                                    outputs,
                                    nullptr,
                                    cudf::null_aware::NO,
                                    std::nullopt,
                                    cudf::test::get_default_stream());

  column_wrapper<double> expected_net_profit{{27.0, 4.0, -6.0, 7.5}};
  column_wrapper<double> expected_net_sales{{106.0, 24.0, 7.0, 45.0}};

  cudf::test::detail::expect_columns_equivalent(
    result->view().column(0), expected_net_profit, cudf::test::debug_output_level::FIRST_ERROR, 64);
  cudf::test::detail::expect_columns_equivalent(
    result->view().column(1), expected_net_sales, cudf::test::debug_output_level::FIRST_ERROR, 64);
}
