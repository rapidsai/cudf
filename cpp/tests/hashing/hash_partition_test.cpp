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
#include <cudf/hashing.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

using cudf::test::fixed_width_column_wrapper;

template <typename T>
class HashPartition : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(HashPartition, cudf::test::FixedWidthTypes);

TYPED_TEST(HashPartition, NumberOfPartitions)
{
  auto first = fixed_width_column_wrapper<TypeParam>({1, 2, 3, 4, 5, 6});
  auto second = fixed_width_column_wrapper<TypeParam>({7, 8, 9, 10, 11, 12});
  auto input = cudf::table_view({first, second});

  auto columns_to_hash = std::vector<cudf::size_type>({0, 1});

  cudf::size_type const num_partitions = 10;
  auto result = cudf::hash_partition(input, columns_to_hash, num_partitions);
  EXPECT_EQ(size_t{num_partitions}, result.size());
}
