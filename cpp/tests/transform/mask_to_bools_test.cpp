/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

struct MaskToBools : public cudf::test::BaseFixture {
};

TEST_F(MaskToBools, NullDataWithZeroLength)
{
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({});
  auto out      = cudf::mask_to_bools(nullptr, 0, 0);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, out->view());
}

TEST_F(MaskToBools, NullDataWithNonZeroLength)
{
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({});

  EXPECT_THROW(cudf::mask_to_bools(nullptr, 0, 2), cudf::logic_error);
}

TEST_F(MaskToBools, ImproperBitRange)
{
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({});

  EXPECT_THROW(cudf::mask_to_bools(nullptr, 2, 1), cudf::logic_error);
}

struct MaskToBoolsTest
  : public MaskToBools,
    public ::testing::WithParamInterface<std::tuple<cudf::size_type, cudf::size_type>> {
};

TEST_P(MaskToBoolsTest, LargeDataSizeTest)
{
  auto data                       = std::vector<bool>(10000);
  cudf::size_type const begin_bit = std::get<0>(GetParam());
  cudf::size_type const end_bit   = std::get<1>(GetParam());
  std::transform(data.cbegin(), data.cend(), data.begin(), [](auto val) {
    return rand() % 2 == 0 ? true : false;
  });

  auto col      = cudf::test::fixed_width_column_wrapper<bool>(data.begin(), data.end());
  auto expected = cudf::detail::slice(static_cast<cudf::column_view>(col), begin_bit, end_bit);

  auto mask = cudf::bools_to_mask(col);

  auto out = cudf::mask_to_bools(
    static_cast<const cudf::bitmask_type*>(mask.first->data()), begin_bit, end_bit);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, out->view());
}

INSTANTIATE_TEST_CASE_P(MaskToBools,
                        MaskToBoolsTest,
                        ::testing::Values(std::make_tuple(0, 0),
                                          std::make_tuple(0, 500),
                                          std::make_tuple(500, 7456),
                                          std::make_tuple(7456, 10000),
                                          std::make_tuple(0, 10000)));
