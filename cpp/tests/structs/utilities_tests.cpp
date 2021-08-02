/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>
#include <structs/utilities.hpp>

namespace cudf::test {

/**
 * @brief Round-trip input table through flatten/unflatten,
 *        verify that the table remains equivalent.
 */
void flatten_unflatten_compare(table_view const& input_table)
{
  using namespace cudf::structs::detail;

  auto [flattened, _, __, ___] =
    flatten_nested_columns(input_table, {}, {}, column_nullability::FORCE);
  auto unflattened =
    unflatten_nested_columns(std::make_unique<cudf::table>(flattened), input_table);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_table, unflattened->view());
}

using namespace cudf;
using iterators::null_at;
using strings = strings_column_wrapper;
using structs = structs_column_wrapper;

struct StructUtilitiesTest : BaseFixture {
};

template <typename T>
struct TypedStructUtilitiesTest : StructUtilitiesTest {
};

TYPED_TEST_CASE(TypedStructUtilitiesTest, FixedWidthTypes);

TYPED_TEST(TypedStructUtilitiesTest, ListsAtTopLevelUnsupported)
{
  using T     = TypeParam;
  using lists = lists_column_wrapper<T, int32_t>;
  using nums  = fixed_width_column_wrapper<T, int32_t>;

  auto lists_col = lists{{0, 1}, {22, 333}, {4444, 55555, 666666}};
  auto nums_col  = nums{{0, 1, 2}, null_at(6)};

  EXPECT_THROW(flatten_unflatten_compare(cudf::table_view{{lists_col, nums_col}}),
               cudf::logic_error);
}

TYPED_TEST(TypedStructUtilitiesTest, NestedListsUnsupported)
{
  using T     = TypeParam;
  using lists = lists_column_wrapper<T, int32_t>;
  using nums  = fixed_width_column_wrapper<T, int32_t>;

  auto lists_member = lists{{0, 1}, {22, 333}, {4444, 55555, 666666}};
  auto nums_member  = nums{{0, 1, 2}, null_at(6)};
  auto structs_col  = structs{{nums_member, lists_member}};

  auto nums_col = nums{{0, 1, 2}, null_at(6)};

  EXPECT_THROW(flatten_unflatten_compare(cudf::table_view{{nums_col, structs_col}}),
               cudf::logic_error);
}

TYPED_TEST(TypedStructUtilitiesTest, NoStructs)
{
  using T    = TypeParam;
  using nums = fixed_width_column_wrapper<T, int32_t>;

  auto nums_col        = nums{{0, 1, 22, 333, 4444, 55555, 666666}, null_at(0)};
  auto strings_col     = strings{{"", "1", "22", "333", "4444", "55555", "666666"}, null_at(1)};
  auto nuther_nums_col = nums{{0, 1, 2, 3, 4, 5, 6}, null_at(6)};

  flatten_unflatten_compare(cudf::table_view{{nums_col, strings_col, nuther_nums_col}});
}

TYPED_TEST(TypedStructUtilitiesTest, SingleLevelStruct)
{
  using T    = TypeParam;
  using nums = fixed_width_column_wrapper<T, int32_t>;

  auto nums_member    = nums{{0, 1, 22, 333, 4444, 55555, 666666}, null_at(0)};
  auto strings_member = strings{{"", "1", "22", "333", "4444", "55555", "666666"}, null_at(1)};
  auto structs_col    = structs{{nums_member, strings_member}};
  auto nums_col       = nums{{0, 1, 2, 3, 4, 5, 6}, null_at(6)};

  flatten_unflatten_compare(cudf::table_view{{nums_col, structs_col}});
}

TYPED_TEST(TypedStructUtilitiesTest, SingleLevelStructWithNulls)
{
  using T    = TypeParam;
  using nums = fixed_width_column_wrapper<T, int32_t>;

  auto nums_member    = nums{{0, 1, 22, 333, 4444, 55555, 666666}, null_at(0)};
  auto strings_member = strings{{"", "1", "22", "333", "4444", "55555", "666666"}, null_at(1)};
  auto structs_col    = structs{{nums_member, strings_member}, null_at(2)};
  auto nums_col       = nums{{0, 1, 2, 3, 4, 5, 6}, null_at(6)};

  flatten_unflatten_compare(cudf::table_view{{nums_col, structs_col}});
}

TYPED_TEST(TypedStructUtilitiesTest, StructOfStruct)
{
  using T    = TypeParam;
  using nums = fixed_width_column_wrapper<T, int32_t>;

  auto nums_col = nums{{0, 1, 2, 3, 4, 5, 6}, null_at(6)};

  auto struct_0_nums_member = nums{{0, 1, 22, 333, 4444, 55555, 666666}, null_at(0)};
  auto struct_0_strings_member =
    strings{{"", "1", "22", "333", "4444", "55555", "666666"}, null_at(1)};
  auto structs_1_structs_member = structs{{struct_0_nums_member, struct_0_strings_member}};

  auto struct_1_nums_member  = nums{{0, 1, 22, 333, 4444, 55555, 666666}, null_at(3)};
  auto struct_of_structs_col = structs{{struct_1_nums_member, structs_1_structs_member}};

  flatten_unflatten_compare(cudf::table_view{{nums_col, struct_of_structs_col}});
}

TYPED_TEST(TypedStructUtilitiesTest, StructOfStructWithNullsAtLeafLevel)
{
  using T    = TypeParam;
  using nums = fixed_width_column_wrapper<T, int32_t>;

  auto nums_col = nums{{0, 1, 2, 3, 4, 5, 6}, null_at(6)};

  auto struct_0_nums_member = nums{{0, 1, 22, 333, 4444, 55555, 666666}, null_at(0)};
  auto struct_0_strings_member =
    strings{{"", "1", "22", "333", "4444", "55555", "666666"}, null_at(1)};
  auto structs_1_structs_member =
    structs{{struct_0_nums_member, struct_0_strings_member}, null_at(2)};

  auto struct_1_nums_member  = nums{{0, 1, 22, 333, 4444, 55555, 666666}, null_at(3)};
  auto struct_of_structs_col = structs{{struct_1_nums_member, structs_1_structs_member}};

  flatten_unflatten_compare(cudf::table_view{{nums_col, struct_of_structs_col}});
}

TYPED_TEST(TypedStructUtilitiesTest, StructOfStructWithNullsAtTopLevel)
{
  using T    = TypeParam;
  using nums = fixed_width_column_wrapper<T, int32_t>;

  auto nums_col = nums{{0, 1, 2, 3, 4, 5, 6}, null_at(6)};

  auto struct_0_nums_member = nums{{0, 1, 22, 333, 4444, 55555, 666666}, null_at(0)};
  auto struct_0_strings_member =
    strings{{"", "1", "22", "333", "4444", "55555", "666666"}, null_at(1)};
  auto structs_1_structs_member = structs{{struct_0_nums_member, struct_0_strings_member}};

  auto struct_1_nums_member = nums{{0, 1, 22, 333, 4444, 55555, 666666}, null_at(3)};
  auto struct_of_structs_col =
    structs{{struct_1_nums_member, structs_1_structs_member}, null_at(4)};

  flatten_unflatten_compare(cudf::table_view{{nums_col, struct_of_structs_col}});
}

TYPED_TEST(TypedStructUtilitiesTest, StructOfStructWithNullsAtAllLevels)
{
  using T    = TypeParam;
  using nums = fixed_width_column_wrapper<T, int32_t>;

  auto nums_col = nums{{0, 1, 2, 3, 4, 5, 6}, null_at(6)};

  auto struct_0_nums_member = nums{{0, 1, 22, 333, 4444, 55555, 666666}, null_at(0)};
  auto struct_0_strings_member =
    strings{{"", "1", "22", "333", "4444", "55555", "666666"}, null_at(1)};
  auto structs_1_structs_member =
    structs{{struct_0_nums_member, struct_0_strings_member}, null_at(2)};

  auto struct_1_nums_member = nums{{0, 1, 22, 333, 4444, 55555, 666666}, null_at(3)};
  auto struct_of_structs_col =
    structs{{struct_1_nums_member, structs_1_structs_member}, null_at(4)};

  flatten_unflatten_compare(cudf::table_view{{nums_col, struct_of_structs_col}});
}

}  // namespace cudf::test
