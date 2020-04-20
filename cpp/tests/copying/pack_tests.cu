/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>

namespace cudf {
namespace test {


struct PackUnpackTest : public BaseFixture {};

TEST_F(PackUnpackTest, SingleColumnFixedWidth)
{
  fixed_width_column_wrapper<int64_t> col1 ({ 1, 2, 3, 4, 5, 6, 7},
                                            { 1, 1, 1, 0, 1, 0, 1});
  table_view t({col1});

  experimental::packed_table packed = experimental::pack(t);
  experimental::packed_table packed2{
    packed.table_metadata, std::make_unique<rmm::device_buffer>(*packed.table_data)};
  experimental::contiguous_split_result unpacked = experimental::unpack(packed2);

  expect_tables_equal(t, unpacked.table);
}

TEST_F(PackUnpackTest, MultiColumnFixedWidth)
{
  fixed_width_column_wrapper<int16_t> col1 ({ 1, 2, 3, 4, 5, 6, 7},
                                            { 1, 1, 1, 0, 1, 0, 1});
  fixed_width_column_wrapper<float>   col2 ({ 7, 8, 6, 5, 4, 3, 2},
                                            { 1, 0, 1, 1, 1, 1, 1});
  fixed_width_column_wrapper<double>  col3 ({ 8, 4, 2, 0, 7, 1, 9},
                                            { 0, 1, 1, 1, 1, 1, 1});
  table_view t({col1, col2, col3});

  experimental::packed_table packed = experimental::pack(t);
  experimental::contiguous_split_result unpacked = experimental::unpack(packed);

  expect_tables_equal(t, unpacked.table);
}

} // namespace test
} // namespace cudf
