/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>

namespace cudf {
namespace test {

constexpr debug_output_level verbosity{debug_output_level::ALL_ERRORS};
constexpr int32_t null{0};  // Mark for null child elements
constexpr int32_t XXX{0};   // Mark for null struct elements

using structs_col = cudf::test::structs_column_wrapper;

template <typename T>
struct TypedStructScalarScatterTest : public cudf::test::BaseFixture {
};

TYPED_TEST_SUITE(TypedStructScalarScatterTest, FixedWidthTypes);

column scatter_single_scalar(scalar const& slr, column_view scatter_map, column_view target)
{
  auto result = scatter({slr}, scatter_map, table_view{{target}});
  return result->get_column(0);
}

TYPED_TEST(TypedStructScalarScatterTest, Basic)
{
  using fixed_width_wrapper = fixed_width_column_wrapper<TypeParam>;

  // Source scalar
  fixed_width_wrapper slr_f0{777};
  strings_column_wrapper slr_f1{"hello"};
  auto slr = make_struct_scalar(table_view{{slr_f0, slr_f1}});

  // Scatter map
  fixed_width_column_wrapper<size_type> scatter_map{2};

  // Target column
  fixed_width_wrapper field0{11, 11, 22, 22, 33};
  strings_column_wrapper field1{"aa", "aa", "bb", "bb", "cc"};
  structs_col target{field0, field1};

  // Expect column
  fixed_width_wrapper ef0{11, 11, 777, 22, 33};
  strings_column_wrapper ef1{"aa", "aa", "hello", "bb", "cc"};
  structs_col expected{ef0, ef1};

  auto got = scatter_single_scalar(*slr, scatter_map, target);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, got, verbosity);
}

TYPED_TEST(TypedStructScalarScatterTest, FillNulls)
{
  using fixed_width_wrapper = fixed_width_column_wrapper<TypeParam>;

  // Source scalar
  fixed_width_wrapper slr_f0{777};
  auto slr = make_struct_scalar(table_view{{slr_f0}});

  // Scatter map
  fixed_width_column_wrapper<size_type> scatter_map{3, 4};

  // Target column
  fixed_width_wrapper field0({11, 11, 22, null, XXX}, iterators::null_at(3));
  structs_col target({field0}, iterators::null_at(4));

  // Expect column
  fixed_width_wrapper ef0{11, 11, 22, 777, 777};
  structs_col expected{ef0};

  auto got = scatter_single_scalar(*slr, scatter_map, target);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, got, verbosity);
}

TYPED_TEST(TypedStructScalarScatterTest, ScatterNullElements)
{
  using fixed_width_wrapper = fixed_width_column_wrapper<TypeParam>;

  // Source scalar
  fixed_width_wrapper slr_f0{777};
  std::vector<column_view> source_fields{slr_f0};
  auto slr = std::make_unique<struct_scalar>(source_fields, false);

  // Scatter map
  fixed_width_column_wrapper<size_type> scatter_map{0, 3, 4};

  // Target column
  fixed_width_wrapper field0({11, 11, 22, null, XXX}, iterators::null_at(3));
  structs_col target({field0}, iterators::null_at(4));

  // Expect column
  fixed_width_wrapper ef0{XXX, 11, 22, XXX, XXX};
  structs_col expected({ef0}, {false, true, true, false, false});

  auto got = scatter_single_scalar(*slr, scatter_map, target);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got, verbosity);
}

TYPED_TEST(TypedStructScalarScatterTest, ScatterNullFields)
{
  using fixed_width_wrapper = fixed_width_column_wrapper<TypeParam>;

  // Source scalar
  fixed_width_wrapper slr_f0({null}, {false});
  auto slr = make_struct_scalar(table_view{{slr_f0}});

  // Scatter map
  fixed_width_column_wrapper<size_type> scatter_map{0, 2};

  // Target column
  fixed_width_wrapper field0({11, 11, 22, null, XXX}, iterators::null_at(3));
  structs_col target({field0}, iterators::null_at(4));

  // Expect column
  fixed_width_wrapper ef0({null, 11, null, null, XXX}, {false, true, false, false, true});
  structs_col expected({ef0}, iterators::null_at(4));

  auto got = scatter_single_scalar(*slr, scatter_map, target);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got, verbosity);
}

TYPED_TEST(TypedStructScalarScatterTest, NegativeIndices)
{
  using fixed_width_wrapper = fixed_width_column_wrapper<TypeParam>;

  // Source scalar
  fixed_width_wrapper slr_f0{777};
  auto slr = make_struct_scalar(table_view{{slr_f0}});

  // Scatter map
  fixed_width_column_wrapper<size_type> scatter_map{-1, -5};

  // Target column
  fixed_width_wrapper field0({11, 11, 22, null, XXX}, iterators::null_at(3));
  structs_col target({field0}, iterators::null_at(4));

  // Expect column
  fixed_width_wrapper ef0({777, 11, 22, null, 777}, iterators::null_at(3));
  structs_col expected{ef0};

  auto got = scatter_single_scalar(*slr, scatter_map, target);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, got, verbosity);
}

TYPED_TEST(TypedStructScalarScatterTest, EmptyInputTest)
{
  using fixed_width_wrapper = fixed_width_column_wrapper<TypeParam>;

  // Source scalar
  fixed_width_wrapper slr_f0{777};
  auto slr = make_struct_scalar(table_view{{slr_f0}});

  // Scatter map
  fixed_width_column_wrapper<size_type> scatter_map{};

  // Target column
  fixed_width_wrapper field0{};
  structs_col target{field0};

  // Expect column
  fixed_width_wrapper ef0{};
  structs_col expected{ef0};

  auto got = scatter_single_scalar(*slr, scatter_map, target);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, got, verbosity);
}

TYPED_TEST(TypedStructScalarScatterTest, EmptyScatterMapTest)
{
  using fixed_width_wrapper = fixed_width_column_wrapper<TypeParam>;

  // Source scalar
  fixed_width_wrapper slr_f0{777};
  auto slr = make_struct_scalar(table_view{{slr_f0}});

  // Scatter map
  fixed_width_column_wrapper<size_type> scatter_map{};

  // Target column
  fixed_width_wrapper field0({11, 11, 22, null, XXX}, iterators::null_at(3));
  structs_col target({field0}, iterators::null_at(4));

  auto got = scatter_single_scalar(*slr, scatter_map, target);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(target, got, verbosity);
}

TYPED_TEST(TypedStructScalarScatterTest, FixedWidthStringTypes)
{
  using fixed_width_wrapper = fixed_width_column_wrapper<TypeParam>;

  // Source scalar
  fixed_width_wrapper slr_f0{777};
  strings_column_wrapper slr_f1{"hello"};
  auto slr = make_struct_scalar(table_view{{slr_f0, slr_f1}});

  // Scatter map
  fixed_width_column_wrapper<size_type> scatter_map{0, 2, 4};

  // Target column
  fixed_width_wrapper field0({11, 11, 22, null, XXX}, iterators::null_at(3));
  strings_column_wrapper field1({"aa", "null", "ccc", "null", "XXX"},
                                {true, false, true, false, true});
  structs_col target({field0, field1}, iterators::null_at(4));

  // Expect column
  fixed_width_wrapper ef0({777, 11, 777, null, 777}, iterators::null_at(3));
  strings_column_wrapper ef1({"hello", "null", "hello", "null", "hello"},
                             {true, false, true, false, true});
  structs_col expected{ef0, ef1};

  auto got = scatter_single_scalar(*slr, scatter_map, target);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, got, verbosity);
}

TYPED_TEST(TypedStructScalarScatterTest, StructOfLists)
{
  using LCW = lists_column_wrapper<TypeParam, int32_t>;

  // Source scalar
  LCW slr_f0{777};
  auto slr = make_struct_scalar(table_view{{slr_f0}});

  // Scatter map
  fixed_width_column_wrapper<size_type> scatter_map{0, 1, 4};

  // Target column
  LCW field0({LCW{XXX}, LCW{22}, LCW{33, 44}, LCW{null}, LCW{55}}, iterators::null_at(3));
  structs_col target({field0}, iterators::null_at(0));

  // Expect column
  LCW ef0({LCW{777}, LCW{777}, LCW{33, 44}, LCW{null}, LCW{777}}, iterators::null_at(3));
  structs_col expected{ef0};

  auto got = scatter_single_scalar(*slr, scatter_map, target);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, got, verbosity);
}

}  // namespace test
}  // namespace cudf
