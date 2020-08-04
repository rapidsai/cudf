/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/type_lists.hpp>

struct MaskToBools : public cudf::test::BaseFixture {
  cudf::test::fixed_width_column_wrapper<bool> col_data{{false,
                                                         true,
                                                         false,
                                                         true,
                                                         false,
                                                         true,
                                                         false,
                                                         true,
                                                         true,
                                                         false,
                                                         true,
                                                         false,
                                                         true,
                                                         false,
                                                         true,
                                                         false}};
};

TEST_F(MaskToBools, BasicTest)
{
  auto data  = std::vector<uint8_t>{0b10101010, 0b01010101};
  auto input = rmm::device_buffer(data.data(), sizeof(uint8_t) * data.size());

  auto const& expected = this->col_data;

  auto out = cudf::mask_to_bools(static_cast<const cudf::bitmask_type*>(input.data()), 0, 16);

  cudf::test::expect_columns_equal(expected, out->view());
}

TEST_F(MaskToBools, SlicedTest)
{
  auto data  = std::vector<uint8_t>{0b10101010, 0b01010101};
  auto input = rmm::device_buffer(data.data(), sizeof(uint8_t) * data.size());

  auto const& col = this->col_data;
  auto expected   = cudf::detail::slice(static_cast<cudf::column_view>(col), 5, 12);

  auto out = cudf::mask_to_bools(static_cast<const cudf::bitmask_type*>(input.data()), 5, 12);

  cudf::test::expect_columns_equal(expected, out->view());
}

TEST_F(MaskToBools, NullDataWithZeroLength)
{
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({});
  auto out      = cudf::mask_to_bools(nullptr, 0, 0);

  cudf::test::expect_columns_equal(expected, out->view());
}

TEST_F(MaskToBools, DataWithZeroLength)
{
  auto data  = std::vector<uint8_t>{170, 85};
  auto input = rmm::device_buffer(data.data(), sizeof(uint8_t) * data.size());

  auto const& col = this->col_data;
  auto expected   = cudf::detail::slice(static_cast<cudf::column_view>(col), 0, 0);

  auto out = cudf::mask_to_bools(static_cast<const cudf::bitmask_type*>(input.data()), 0, 0);

  cudf::test::expect_columns_equal(expected, out->view());
}
