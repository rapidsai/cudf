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

template <typename T>
using decimal_wrapper = cudf::test::fixed_point_column_wrapper<typename T::rep>;

TEST_F(TransformLTOTest, InvSqrt)
{
  column_wrapper<float> input{{1.0f, 4.0f, 9.0f, 16.0f}};

  cudf::transform_output output{cudf::data_type{cudf::type_id::FLOAT32},
                                cudf::output_nullability::ALL_VALID};

  auto const range = cudf_test_fragments::file_ranges[cudf_test_fragments::invsqrt];
  std::span<uint8_t const> udf{cudf_test_fragments::files.subspan(range[0], range[1])};

  auto result = cudf::unary_op_lto(input,
                                   output,
                                   udf,
                                   cudf::lto_binary_type::FATBIN,
                                   cudf::null_aware::NO,
                                   cudf::test::get_default_stream());

  column_wrapper<float> expected{{1.0f, 0.5f, 0.33333334f, 0.25f}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(TransformLTOTest, SumOfSquares)
{
  column_wrapper<float> lhs{{1.0f, 4.0f, 9.0f, 16.0f}};
  column_wrapper<float> rhs{{1.0f, 2.0f, 2.0f, 10.0f}};

  cudf::transform_output output{cudf::data_type{cudf::type_id::FLOAT32},
                                cudf::output_nullability::ALL_VALID};

  auto const range = cudf_test_fragments::file_ranges[cudf_test_fragments::sum_of_squares];
  std::span<uint8_t const> udf{cudf_test_fragments::files.subspan(range[0], range[1])};

  auto result = cudf::binary_op_lto(lhs,
                                    rhs,
                                    output,
                                    udf,
                                    cudf::lto_binary_type::FATBIN,
                                    cudf::null_aware::NO,
                                    cudf::test::get_default_stream());

  column_wrapper<float> expected{{2.0f, 20.0f, 85.0f, 356.0f}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(TransformLTOTest, Decimal32Square)
{
  auto test_type = []<typename T>() {
    decimal_wrapper<T> input{{1, 2, 3}, numeric::scale_type{-2}};

    cudf::transform_output output{cudf::data_type{cudf::type_to_id<T>(), numeric::scale_type{-4}},
                                  cudf::output_nullability::ALL_VALID};

    auto const range = cudf_test_fragments::file_ranges[cudf_test_fragments::decimal_square];
    std::span<uint8_t const> udf{cudf_test_fragments::files.subspan(range[0], range[1])};

    auto result = cudf::unary_op_lto(input,
                                     output,
                                     udf,
                                     cudf::lto_binary_type::FATBIN,
                                     cudf::null_aware::NO,
                                     cudf::test::get_default_stream());

    decimal_wrapper<T> expected{{1, 4, 9}, numeric::scale_type{-4}};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  };

  test_type.operator()<numeric::decimal32>();
  test_type.operator()<numeric::decimal64>();
  test_type.operator()<numeric::decimal128>();
}
