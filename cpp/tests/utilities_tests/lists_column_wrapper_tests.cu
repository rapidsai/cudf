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

#include <cudf/lists/lists_column_view.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

struct ListColumnWrapperTest : public cudf::test::BaseFixture {
};
template <typename T>
struct ListColumnWrapperTestTyped : public cudf::test::BaseFixture {
  ListColumnWrapperTestTyped() {}

  auto data_type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::TimestampTypes>;
TYPED_TEST_CASE(ListColumnWrapperTestTyped, FixedWidthTypesNotBool);

TYPED_TEST(ListColumnWrapperTestTyped, List)
{
  using namespace cudf;
  using T = TypeParam;

  // List<T>, 1 row
  //
  // List<T>:
  // Length : 1
  // Offsets : 0, 2
  // Children :
  //    2, 3
  //
  {
    test::lists_column_wrapper<T> list{2, 3};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 2);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 2});
    test::expect_columns_equal(e_offsets, offsets);

    auto data = lcv.child();
    EXPECT_EQ(data.size(), 2);
    test::fixed_width_column_wrapper<T> e_data({2, 3});
    test::expect_columns_equal(e_data, data);
  }

  // List<T>, 1 row
  //
  // List<T>:
  // Length : 1
  // Offsets : 0, 2
  // Children :
  //    2, 3
  //
  {
    test::lists_column_wrapper<T> list{{2, 3}};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 2);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 2});
    test::expect_columns_equal(e_offsets, offsets);

    auto data = lcv.child();
    EXPECT_EQ(data.size(), 2);
    test::fixed_width_column_wrapper<T> e_data({2, 3});
    test::expect_columns_equal(e_data, data);
  }
}

TYPED_TEST(ListColumnWrapperTestTyped, ListWithValidity)
{
  using namespace cudf;
  using T = TypeParam;

  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  // List<T>, 1 row
  //
  // List<T>:
  // Length : 1
  // Offsets : 0, 2
  // Children :
  //    2, NULL
  //
  {
    test::lists_column_wrapper<T> list{{{2, 3}, valids}};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 2);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 2});
    test::expect_columns_equal(e_offsets, offsets);

    auto data = lcv.child();
    EXPECT_EQ(data.size(), 2);
    test::fixed_width_column_wrapper<T> e_data({2, 3}, valids);
    test::expect_columns_equal(e_data, data);
  }

  // List<T>, 3 rows
  //
  // List<T>:
  // Length : 3
  // Offsets : 0, 2, 4, 7
  // Children :
  //    2, NULL, 4, NULL, 6, NULL, 8
  {
    test::lists_column_wrapper<T> list{{{2, 3}, valids}, {{4, 5}, valids}, {{6, 7, 8}, valids}};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 2, 4, 7});
    test::expect_columns_equal(e_offsets, offsets);

    auto data = lcv.child();
    EXPECT_EQ(data.size(), 7);
    test::fixed_width_column_wrapper<T> e_data({2, 3, 4, 5, 6, 7, 8}, valids);
    test::expect_columns_equal(e_data, data);
  }
}

TYPED_TEST(ListColumnWrapperTestTyped, ListFromIterator)
{
  using namespace cudf;
  using T = TypeParam;

  // List<T>, 1 row
  //
  // List<T>:
  // Length : 1
  // Offsets : 0, 5
  // Children :
  //    0, 1, 2, 3, 4
  //
  auto sequence =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return static_cast<T>(i); });

  test::lists_column_wrapper<T> list{sequence, sequence + 5};

  lists_column_view lcv(list);
  EXPECT_EQ(lcv.size(), 1);

  auto offsets = lcv.offsets();
  EXPECT_EQ(offsets.size(), 2);
  test::fixed_width_column_wrapper<size_type> e_offsets({0, 5});
  test::expect_columns_equal(e_offsets, offsets);

  auto data = lcv.child();
  EXPECT_EQ(data.size(), 5);
  test::fixed_width_column_wrapper<T> e_data({0, 1, 2, 3, 4});
  test::expect_columns_equal(e_data, data);
}

TYPED_TEST(ListColumnWrapperTestTyped, ListFromIteratorWithValidity)
{
  using namespace cudf;
  using T = TypeParam;

  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  // List<int>, 1 row
  //
  // List<int32_t>:
  // Length : 1
  // Offsets : 0, 5
  // Children :
  //    0, NULL, 2, NULL, 4
  //
  auto sequence =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return static_cast<T>(i); });

  test::lists_column_wrapper<T> list{sequence, sequence + 5, valids};

  lists_column_view lcv(list);
  EXPECT_EQ(lcv.size(), 1);

  auto offsets = lcv.offsets();
  EXPECT_EQ(offsets.size(), 2);
  test::fixed_width_column_wrapper<size_type> e_offsets({0, 5});
  test::expect_columns_equal(e_offsets, offsets);

  auto data = lcv.child();
  EXPECT_EQ(data.size(), 5);
  test::fixed_width_column_wrapper<T> e_data({0, 0, 2, 0, 4}, valids);
  test::expect_columns_equal(e_data, data);
}

TYPED_TEST(ListColumnWrapperTestTyped, ListOfLists)
{
  using namespace cudf;
  using T = TypeParam;

  // List<List<T>>, 1 row
  //
  // List<List<T>>:
  // Length : 1
  // Offsets : 0, 2
  // Children :
  //    List<T>:
  //    Length : 2
  //    Offsets : 0, 2, 4
  //    Children :
  //      2, 3, 4, 5
  {
    test::lists_column_wrapper<T> list{{{2, 3}, {4, 5}}};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 2);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 2});
    test::expect_columns_equal(e_offsets, offsets);

    auto child = lcv.child();
    lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 2);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 3);
    test::fixed_width_column_wrapper<size_type> e_child_offsets({0, 2, 4});
    test::expect_columns_equal(e_child_offsets, child_offsets);

    auto child_data = childv.child();
    EXPECT_EQ(child_data.size(), 4);
    test::fixed_width_column_wrapper<T> e_child_data({2, 3, 4, 5});
    test::expect_columns_equal(e_child_data, child_data);
  }

  // List<List<T>> 3 rows
  //
  // List<List<T>>:
  // Length : 3
  // Offsets : 0, 2, 5, 6
  // Children :
  //    List<T>:
  //    Length : 6
  //    Offsets : 0, 2, 4, 7, 8, 9, 11
  //    Children :
  //      1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10
  {
    test::lists_column_wrapper<T> list{{{1, 2}, {3, 4}}, {{5, 6, 7}, {0}, {8}}, {{9, 10}}};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 2, 5, 6});
    test::expect_columns_equal(e_offsets, offsets);

    auto child = lcv.child();
    lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 6);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 7);
    test::fixed_width_column_wrapper<size_type> e_child_offsets({0, 2, 4, 7, 8, 9, 11});
    test::expect_columns_equal(e_child_offsets, child_offsets);

    auto child_data = childv.child();
    EXPECT_EQ(child_data.size(), 11);
    test::fixed_width_column_wrapper<T> e_child_data({1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10});
    test::expect_columns_equal(e_child_data, child_data);
  }
}

TYPED_TEST(ListColumnWrapperTestTyped, ListOfListsWithValidity)
{
  using namespace cudf;
  using T = TypeParam;

  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  // List<List<T>>, 1 row
  //
  // List<List<T>>:
  // Length : 1
  // Offsets : 0, 2
  // Children :
  //    List<T>:
  //    Length : 2
  //    Offsets : 0, 2, 4
  //    Children :
  //      2, NULL, 4, NULL
  {
    // equivalent to { {2, NULL}, {4, NULL} }
    test::lists_column_wrapper<T> list{{{{2, 3}, valids}, {{4, 5}, valids}}};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 2);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 2});
    test::expect_columns_equal(e_offsets, offsets);

    auto child = lcv.child();
    lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 2);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 3);
    test::fixed_width_column_wrapper<size_type> e_child_offsets({0, 2, 4});
    test::expect_columns_equal(e_child_offsets, child_offsets);

    auto child_data = childv.child();
    EXPECT_EQ(child_data.size(), 4);
    test::fixed_width_column_wrapper<T> e_child_data({2, 3, 4, 5}, valids);
    test::expect_columns_equal(e_child_data, child_data);
  }

  // List<List<T>> 3 rows
  //
  // List<List<T>>:
  // Length : 3
  // Offsets : 0, 2, 5, 6
  // Children :
  //    List<T>:
  //    Length : 6
  //    Offsets : 0, 2, 2, 5, 5, 6, 8
  //    Null count: 2
  //    110101
  //    Children :
  //      1, 2, 5, 6, 7, 8, 9, 10
  {
    // equivalent to  { {{1, 2}, NULL}, {{5, 6, 7}, NULL, {8}}, {{9, 10}} }
    test::lists_column_wrapper<T> list{
      {{{1, 2}, {3, 4}}, valids}, {{{5, 6, 7}, {0}, {8}}, valids}, {{{9, 10}}, valids}};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 2, 5, 6});
    test::expect_columns_equal(e_offsets, offsets);

    auto child = lcv.child();
    lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 6);
    EXPECT_EQ(childv.null_count(), 2);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 7);
    test::fixed_width_column_wrapper<size_type> e_child_offsets({0, 2, 2, 5, 5, 6, 8});
    test::expect_columns_equal(e_child_offsets, child_offsets);

    auto child_data = childv.child();
    EXPECT_EQ(child_data.size(), 8);
    test::fixed_width_column_wrapper<T> e_child_data({1, 2, 5, 6, 7, 8, 9, 10});
    test::expect_columns_equal(e_child_data, child_data);
  }
}

TYPED_TEST(ListColumnWrapperTestTyped, ListOfListOfListsWithValidity)
{
  using namespace cudf;
  using T = TypeParam;

  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  // List<List<List<T>>>, 2 rows
  //
  // List<List<List<T>>>:
  // Length : 2
  // Offsets : 0, 2, 4
  // Children :
  //    List<List<T>>:
  //    Length : 4
  //    Offsets : 0, 2, 2, 4, 6
  //    Null count: 1
  //    1101
  //    Children :
  //      List<T>:
  //      Length : 6
  //      Offsets : 0, 2, 4, 6, 8, 11, 12
  //      Children :
  //        1, 2, 3, 4, 10, 20, 30, 40, 50, 60, 70, 0
  {
    // equivalent to  { {{{1, 2}, {3, 4}}, NULL}, {{{10, 20}, {30, 40}}, {{50, 60, 70}, {0}}} }
    test::lists_column_wrapper<T> list{{{{{1, 2}, {3, 4}}, {{5, 6, 7}, {0}}}, valids},
                                       {{{10, 20}, {30, 40}}, {{50, 60, 70}, {0}}}};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 2);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 3);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 2, 4});
    test::expect_columns_equal(e_offsets, offsets);

    auto child = lcv.child();
    lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 4);
    EXPECT_EQ(childv.null_count(), 1);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 5);
    test::fixed_width_column_wrapper<size_type> e_child_offsets({0, 2, 2, 4, 6});
    test::expect_columns_equal(e_child_offsets, child_offsets);

    auto child_child = childv.child();
    lists_column_view child_childv(child_child);
    EXPECT_EQ(child_childv.size(), 6);

    auto child_child_offsets = child_childv.offsets();
    EXPECT_EQ(child_child_offsets.size(), 7);
    test::fixed_width_column_wrapper<size_type> e_child_child_offsets({0, 2, 4, 6, 8, 11, 12});
    test::expect_columns_equal(e_child_child_offsets, child_child_offsets);

    auto child_child_data = child_childv.child();
    EXPECT_EQ(child_child_data.size(), 12);
    test::fixed_width_column_wrapper<T> e_child_child_data(
      {1, 2, 3, 4, 10, 20, 30, 40, 50, 60, 70, 0});
    test::expect_columns_equal(e_child_child_data, child_child_data);
  }
}

TYPED_TEST(ListColumnWrapperTestTyped, EmptyLists)
{
  using namespace cudf;
  using T = TypeParam;

  // to disambiguiate between {} == 0 and {} == List{0}
  using LCW = test::lists_column_wrapper<T>;

  // List<T>, empty
  //
  // List<T>:
  // Length : 0
  // Offsets :
  // Children :
  {
    // equivalent to  {}
    test::lists_column_wrapper<T> list{};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 0);
  }

  // List<T>, 1 row
  //
  // List<T>:
  // Length : 1
  // Offsets : 0, 0
  // Children :
  {
    // equivalent to  {}
    test::lists_column_wrapper<T> list{LCW{}};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 2);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 0});
    test::expect_columns_equal(e_offsets, offsets);
  }

  // List<T>, 2 rows
  //
  // List<T>:
  // Length : 2
  // Offsets : 0, 0, 0
  // Children :
  {
    // equivalent to  {}
    test::lists_column_wrapper<T> list{LCW{}, LCW{}};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 2);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 3);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 0, 0});
    test::expect_columns_equal(e_offsets, offsets);
  }

  // List<L>, mixed
  //
  // List<T>:
  // Length : 3
  // Offsets : 0, 2, 2, 4
  // Children :
  //   1, 2, 3, 4
  {
    // equivalent to  {{1, 2}, {}, {3, 4}}

    test::lists_column_wrapper<T> list{{1, 2}, LCW{}, {3, 4}};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 2, 2, 4});
    test::expect_columns_equal(e_offsets, offsets);

    auto child_data = lcv.child();
    EXPECT_EQ(child_data.size(), 4);
    test::fixed_width_column_wrapper<T> e_child_data({1, 2, 3, 4});
    test::expect_columns_equal(e_child_data, child_data);
  }

  // List<List<T>>, mixed
  //
  // List<List<T>>:
  // Length : 3
  // Offsets : 0, 1, 4, 7
  // Children :
  //   List<int32_t>:
  //   Length : 7
  //   Offsets : 0, 0, 2, 2, 4, 4, 8, 8
  //   Children :
  //     1, 2, 3, 4, 5, 6, 7, 8
  {
    // equivalent to  { {{}}, {{1, 2}, {}, {3, 4}}, {{}, {5, 6, 7, 8}, {}} }
    test::lists_column_wrapper<T> list{
      {LCW{}}, {{1, 2}, LCW{}, {3, 4}}, {LCW{}, {5, 6, 7, 8}, LCW{}}};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 1, 4, 7});
    test::expect_columns_equal(e_offsets, offsets);

    auto child = lcv.child();
    lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 7);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 8);
    test::fixed_width_column_wrapper<size_type> e_child_offsets({0, 0, 2, 2, 4, 4, 8, 8});
    test::expect_columns_equal(e_child_offsets, child_offsets);

    auto child_data = childv.child();
    EXPECT_EQ(child_data.size(), 8);
    test::fixed_width_column_wrapper<T> e_child_data({1, 2, 3, 4, 5, 6, 7, 8});
    test::expect_columns_equal(e_child_data, child_data);
  }
}

TYPED_TEST(ListColumnWrapperTestTyped, EmptyListsWithValidity)
{
  using namespace cudf;
  using T = TypeParam;

  // to disambiguiate between {} == 0 and {} == List{0}
  using LCW = test::lists_column_wrapper<T>;

  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  // List<T>, 2 rows
  //
  // List<T>:
  // Length : 2
  // Offsets : 0, 0, 0
  // Null count: 1
  // 01
  // Children :
  {
    // equivalent to  {{}, NULL}
    test::lists_column_wrapper<T> list{{LCW{}, LCW{}}, valids};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 2);
    EXPECT_EQ(lcv.null_count(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 3);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 0, 0});
    test::expect_columns_equal(e_offsets, offsets);
  }

  // List<T>, 3 rows
  //
  // List<T>:
  // Length : 3
  // Offsets : 0, 0, 0, 0
  // Null count: 1
  // 101
  // Children :
  {
    // equivalent to  {{}, NULL, {}}
    test::lists_column_wrapper<T> list{{LCW{}, {1, 2, 3}, LCW{}}, valids};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);
    EXPECT_EQ(lcv.null_count(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 0, 0, 0});
    test::expect_columns_equal(e_offsets, offsets);
  }

  // List<T>, 3 rows
  //
  // List<T>:
  // Length : 3
  // Offsets : 0, 0, 0, 3
  // Null count: 1
  // 101
  // Children :
  //   1, 2, 3
  {
    // equivalent to  {{}, NULL, {1, 2, 3}}
    test::lists_column_wrapper<T> list{{LCW{}, LCW{}, {1, 2, 3}}, valids};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);
    EXPECT_EQ(lcv.null_count(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 0, 0, 3});
    test::expect_columns_equal(e_offsets, offsets);
  }

  // List<List<T>>, mixed
  //
  // List<List<T>>:
  // Length : 3
  // Offsets : 0, 1, 1, 4
  // Null count: 1
  // 101
  // Children :
  //   List<T>:
  //   Length : 4
  //   Offsets : 0, 0, 0, 4, 4
  //   Children :
  //     5, 6, 7, 8
  {
    // equivalent to  { {{}}, NULL, {{}, {5, 6, 7, 8}, {}} }
    test::lists_column_wrapper<T> list{
      {{LCW{}}, {{1, 2}, LCW{}, {3, 4}}, {LCW{}, {5, 6, 7, 8}, LCW{}}}, valids};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 1, 1, 4});
    test::expect_columns_equal(e_offsets, offsets);

    auto child = lcv.child();
    lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 4);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 5);
    test::fixed_width_column_wrapper<size_type> e_child_offsets({0, 0, 0, 4, 4});
    test::expect_columns_equal(e_child_offsets, child_offsets);

    auto child_data = childv.child();
    EXPECT_EQ(child_data.size(), 4);
    test::fixed_width_column_wrapper<T> e_child_data({5, 6, 7, 8});
    test::expect_columns_equal(e_child_data, child_data);
  }
}

TEST_F(ListColumnWrapperTest, ListOfStrings)
{
  using namespace cudf;

  // List<string>, 2 rows
  //
  // List<cudf::string_view>:
  // Length : 2
  // Offsets : 0, 2, 5
  // Children :
  //    one, two, three, four, five
  {
    test::lists_column_wrapper<cudf::string_view> list{{"one", "two"}, {"three", "four", "five"}};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 2);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 3);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 2, 5});
    test::expect_columns_equal(e_offsets, offsets);

    auto data = lcv.child();
    EXPECT_EQ(data.size(), 5);
    test::strings_column_wrapper e_data({"one", "two", "three", "four", "five"});
    test::expect_columns_equal(e_data, data);
  }
}

TEST_F(ListColumnWrapperTest, ListOfListOfStrings)
{
  using namespace cudf;

  // List<List<string>>, 2 rows
  //
  // List<List<cudf::string_view>>:
  // Length : 2
  // Offsets : 0, 2, 4
  // Children :
  //    List<cudf::string_view>:
  //    Length : 4
  //    Offsets : 0, 2, 5, 6, 8
  //    Children :
  //      one, two, three, four, five, eight, nine, ten
  {
    test::lists_column_wrapper<cudf::string_view> list{{{"one", "two"}, {"three", "four", "five"}},
                                                       {{"eight"}, {"nine", "ten"}}};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 2);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 3);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 2, 4});
    test::expect_columns_equal(e_offsets, offsets);

    auto child = lcv.child();
    lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 4);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 5);
    test::fixed_width_column_wrapper<size_type> e_child_offsets({0, 2, 5, 6, 8});
    test::expect_columns_equal(e_child_offsets, child_offsets);

    auto child_data = childv.child();
    EXPECT_EQ(child_data.size(), 8);
    test::strings_column_wrapper e_child_data(
      {"one", "two", "three", "four", "five", "eight", "nine", "ten"});
    test::expect_columns_equal(e_child_data, child_data);
  }
}

TEST_F(ListColumnWrapperTest, ListOfBools)
{
  using namespace cudf;

  // List<bool>, 1 row
  //
  // List<bool>:
  // Length : 1
  // Offsets : 0, 2
  // Children :
  //   1, 0
  //
  {
    test::lists_column_wrapper<bool> list{true, false};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 2);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 2});
    test::expect_columns_equal(e_offsets, offsets);

    auto data = lcv.child();
    EXPECT_EQ(data.size(), 2);
    test::fixed_width_column_wrapper<bool> e_data({true, false});
    test::expect_columns_equal(e_data, data);
  }

  // List<bool>, 1 row
  //
  // List<bool>:
  // Length : 1
  // Offsets : 0, 3
  // Children :
  //   1, 0, 0
  //
  {
    test::lists_column_wrapper<bool> list{{true, false, false}};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 1);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 2);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 3});
    test::expect_columns_equal(e_offsets, offsets);

    auto data = lcv.child();
    EXPECT_EQ(data.size(), 3);
    test::fixed_width_column_wrapper<bool> e_data({true, false, false});
    test::expect_columns_equal(e_data, data);
  }
}

TEST_F(ListColumnWrapperTest, ListOfBoolsWithValidity)
{
  using namespace cudf;

  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  // List<bool>, 3 rows
  //
  // List<bool>:
  // Length : 3
  // Offsets : 0, 2, 4, 7
  // Children :
  //   1, NULL, 0, NULL, 0, NULL, 0
  {
    test::lists_column_wrapper<bool> list{
      {{true, true}, valids}, {{false, true}, valids}, {{false, true, false}, valids}};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 2, 4, 7});
    test::expect_columns_equal(e_offsets, offsets);

    auto data = lcv.child();
    EXPECT_EQ(data.size(), 7);
    test::fixed_width_column_wrapper<bool> e_data({true, true, false, true, false, true, false},
                                                  valids);
    test::expect_columns_equal(e_data, data);
  }
}

TEST_F(ListColumnWrapperTest, ListOfListOfBools)
{
  using namespace cudf;

  using T = int;
  using L = test::lists_column_wrapper<T>;

  // List<List<bool>> 3 rows
  //
  // List<List<bool>>:
  // Length : 3
  // Offsets : 0, 2, 5, 6
  // Children :
  //    List<bool>:
  //    Length : 6
  //    Offsets : 0, 2, 4, 7, 8, 9, 11
  //    Children :
  //      0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1
  {
    test::lists_column_wrapper<bool> list{
      {{false, true}, {true, true}}, {{true, false, true}, {true}, {true}}, {{false, true}}};

    lists_column_view lcv(list);
    EXPECT_EQ(lcv.size(), 3);

    auto offsets = lcv.offsets();
    EXPECT_EQ(offsets.size(), 4);
    test::fixed_width_column_wrapper<size_type> e_offsets({0, 2, 5, 6});
    test::expect_columns_equal(e_offsets, offsets);

    auto child = lcv.child();
    lists_column_view childv(child);
    EXPECT_EQ(childv.size(), 6);

    auto child_offsets = childv.offsets();
    EXPECT_EQ(child_offsets.size(), 7);
    test::fixed_width_column_wrapper<size_type> e_child_offsets({0, 2, 4, 7, 8, 9, 11});
    test::expect_columns_equal(e_child_offsets, child_offsets);

    auto child_child_data = childv.child();
    EXPECT_EQ(child_child_data.size(), 11);
    test::fixed_width_column_wrapper<bool> e_child_child_data(
      {false, true, true, true, true, false, true, true, true, false, true});
    test::expect_columns_equal(e_child_child_data, child_child_data);
  }
}