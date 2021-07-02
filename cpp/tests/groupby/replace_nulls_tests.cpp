
/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

using namespace cudf::test::iterators;

namespace cudf {
namespace test {

using K = int32_t;

template <typename T>
struct GroupbyReplaceNullsFixedWidthTest : public BaseFixture {
};

TYPED_TEST_CASE(GroupbyReplaceNullsFixedWidthTest, FixedWidthTypes);

template <typename K, typename V>
void TestReplaceNullsGroupbySingle(
  K const& key, V const& input, K const& expected_key, V const& expected_val, replace_policy policy)
{
  groupby::groupby gb_obj(table_view({key}));
  std::vector<replace_policy> policies{policy};
  auto p = gb_obj.replace_nulls(table_view({input}), policies);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*p.first, table_view({expected_key}));
  CUDF_TEST_EXPECT_TABLES_EQUAL(*p.second, table_view({expected_val}));
}

TYPED_TEST(GroupbyReplaceNullsFixedWidthTest, PrecedingFill)
{
  // Group 0 value: {42, 24, null}  --> {42, 24, 24}
  // Group 1 value: {7, null, null} --> {7, 7, 7}
  fixed_width_column_wrapper<K> key{0, 1, 0, 1, 0, 1};
  fixed_width_column_wrapper<TypeParam> val({42, 7, 24, 10, 1, 1000}, {1, 1, 1, 0, 0, 0});

  fixed_width_column_wrapper<K> expect_key{0, 0, 0, 1, 1, 1};
  fixed_width_column_wrapper<TypeParam> expect_val({42, 24, 24, 7, 7, 7}, no_nulls());

  TestReplaceNullsGroupbySingle(key, val, expect_key, expect_val, replace_policy::PRECEDING);
}

TYPED_TEST(GroupbyReplaceNullsFixedWidthTest, FollowingFill)
{
  // Group 0 value: {2, null, 32}               --> {2, 32, 32}
  // Group 1 value: {8, null, null, 128, 256}   --> {8, 128, 128, 128, 256}
  fixed_width_column_wrapper<K> key{0, 0, 1, 1, 0, 1, 1, 1};
  fixed_width_column_wrapper<TypeParam> val({2, 4, 8, 16, 32, 64, 128, 256},
                                            {1, 0, 1, 0, 1, 0, 1, 1});

  fixed_width_column_wrapper<K> expect_key{0, 0, 0, 1, 1, 1, 1, 1};
  fixed_width_column_wrapper<TypeParam> expect_val({2, 32, 32, 8, 128, 128, 128, 256}, no_nulls());

  TestReplaceNullsGroupbySingle(key, val, expect_key, expect_val, replace_policy::FOLLOWING);
}

TYPED_TEST(GroupbyReplaceNullsFixedWidthTest, PrecedingFillLeadingNulls)
{
  // Group 0 value: {null, 24, null}    --> {null, 24, 24}
  // Group 1 value: {null, null, null}  --> {null, null, null}
  fixed_width_column_wrapper<K> key{0, 1, 0, 1, 0, 1};
  fixed_width_column_wrapper<TypeParam> val({42, 7, 24, 10, 1, 1000}, {0, 0, 1, 0, 0, 0});

  fixed_width_column_wrapper<K> expect_key{0, 0, 0, 1, 1, 1};
  fixed_width_column_wrapper<TypeParam> expect_val({-1, 24, 24, -1, -1, -1}, {0, 1, 1, 0, 0, 0});

  TestReplaceNullsGroupbySingle(key, val, expect_key, expect_val, replace_policy::PRECEDING);
}

TYPED_TEST(GroupbyReplaceNullsFixedWidthTest, FollowingFillTrailingNulls)
{
  // Group 0 value: {2, null, null}                 --> {2, null, null}
  // Group 1 value: {null, null, 64, null, null}    --> {64, 64, 64, null, null}
  fixed_width_column_wrapper<K> key{0, 0, 1, 1, 0, 1, 1, 1};
  fixed_width_column_wrapper<TypeParam> val({2, 4, 8, 16, 32, 64, 128, 256},
                                            {1, 0, 0, 0, 0, 1, 0, 0});

  fixed_width_column_wrapper<K> expect_key{0, 0, 0, 1, 1, 1, 1, 1};
  fixed_width_column_wrapper<TypeParam> expect_val({2, -1, -1, 64, 64, 64, -1, -1},
                                                   {1, 0, 0, 1, 1, 1, 0, 0});

  TestReplaceNullsGroupbySingle(key, val, expect_key, expect_val, replace_policy::FOLLOWING);
}

struct GroupbyReplaceNullsStringsTest : public BaseFixture {
};

TEST_F(GroupbyReplaceNullsStringsTest, PrecedingFill)
{
  // Group 0 value: {"y" "42"}  --> {"y", "42"}
  // Group 1 value: {"xx" @ "zzz" @ "one"} --> {"xx" "xx" "zzz" "zzz" "one"}
  fixed_width_column_wrapper<K> key{1, 1, 0, 1, 0, 1, 1};
  strings_column_wrapper val({"xx", "", "y", "zzz", "42", "", "one"},
                             {true, false, true, true, true, false, true});

  fixed_width_column_wrapper<K> expect_key{0, 0, 1, 1, 1, 1, 1};
  strings_column_wrapper expect_val({"y", "42", "xx", "xx", "zzz", "zzz", "one"}, no_nulls());

  TestReplaceNullsGroupbySingle(key, val, expect_key, expect_val, replace_policy::PRECEDING);
}

TEST_F(GroupbyReplaceNullsStringsTest, FollowingFill)
{
  // Group 0 value: {@ "42"}  --> {"42", "42"}
  // Group 1 value: {"xx" @ "zzz" @ "one"} --> {"xx" "zzz" "zzz" "one" "one"}
  fixed_width_column_wrapper<K> key{1, 1, 0, 1, 0, 1, 1};
  strings_column_wrapper val({"xx", "", "", "zzz", "42", "", "one"},
                             {true, false, false, true, true, false, true});

  fixed_width_column_wrapper<K> expect_key{0, 0, 1, 1, 1, 1, 1};
  strings_column_wrapper expect_val({"42", "42", "xx", "zzz", "zzz", "one", "one"}, no_nulls());

  TestReplaceNullsGroupbySingle(key, val, expect_key, expect_val, replace_policy::FOLLOWING);
}

TEST_F(GroupbyReplaceNullsStringsTest, PrecedingFillPrecedingNull)
{
  // Group 0 value: {"y" "42"}  --> {"y", "42"}
  // Group 1 value: {@ @ "zzz" "zzz" "zzz"} --> {@ @ "zzz" "zzz" "zzz"}
  fixed_width_column_wrapper<K> key{1, 1, 0, 1, 0, 1, 1};
  strings_column_wrapper val({"", "", "y", "zzz", "42", "", ""},
                             {false, false, true, true, true, false, false});

  fixed_width_column_wrapper<K> expect_key{0, 0, 1, 1, 1, 1, 1};
  strings_column_wrapper expect_val({"y", "42", "", "", "zzz", "zzz", "zzz"},
                                    {true, true, false, false, true, true, true});

  TestReplaceNullsGroupbySingle(key, val, expect_key, expect_val, replace_policy::PRECEDING);
}

TEST_F(GroupbyReplaceNullsStringsTest, FollowingFillTrailingNull)
{
  // Group 0 value: {@ "y"}  --> {"y", "y"}
  // Group 1 value: {"xx" @ "zzz" @ @} --> {"xx" "zzz" "zzz" @ @}
  fixed_width_column_wrapper<K> key{1, 1, 0, 1, 0, 1, 1};
  strings_column_wrapper val({"xx", "", "", "zzz", "y", "", ""},
                             {true, false, false, true, true, false, false});

  fixed_width_column_wrapper<K> expect_key{0, 0, 1, 1, 1, 1, 1};
  strings_column_wrapper expect_val({"y", "y", "xx", "zzz", "zzz", "", ""},
                                    {true, true, true, true, true, false, false});

  TestReplaceNullsGroupbySingle(key, val, expect_key, expect_val, replace_policy::FOLLOWING);
}

template <typename T>
struct GroupbyReplaceNullsListsTest : public BaseFixture {
};

TYPED_TEST_CASE(GroupbyReplaceNullsListsTest, FixedWidthTypes);

TYPED_TEST(GroupbyReplaceNullsListsTest, PrecedingFillNonNested)
{
  using LCW = lists_column_wrapper<TypeParam, int32_t>;
  // Group 0 value: {{1 2 3} @ {4 5} @} --> {{1 2 3} {1 2 3} {4 5} {4 5}}, w/o leading nulls
  // Group 1 value: {@ {} @} --> {@ {} {}}, w/ leading nulls
  fixed_width_column_wrapper<K> key{0, 1, 0, 0, 1, 1, 0};

  std::vector<valid_type> mask{1, 0, 0, 1, 1, 0, 0};
  LCW val({{1, 2, 3}, {}, {}, {4, 5}, {}, {}, {}}, mask.begin());

  fixed_width_column_wrapper<K> expect_key{0, 0, 0, 0, 1, 1, 1};
  std::vector<valid_type> expected_mask{1, 1, 1, 1, 0, 1, 1};
  LCW expect_val({{1, 2, 3}, {1, 2, 3}, {4, 5}, {4, 5}, {-1}, {}, {}}, expected_mask.begin());

  TestReplaceNullsGroupbySingle(key, val, expect_key, expect_val, replace_policy::PRECEDING);
}

TYPED_TEST(GroupbyReplaceNullsListsTest, FollowingFillNonNested)
{
  using LCW = lists_column_wrapper<TypeParam, int32_t>;
  // Group 0 value: {@ {5 6} @ {-1}} --> {{5 6} {5 6} {-1} {-1}}, w/o trailing nulls
  // Group 1 value: {@ {} @} --> {{} {} @}}, w/ trailing nulls
  fixed_width_column_wrapper<K> key{0, 1, 0, 0, 1, 1, 0};

  std::vector<valid_type> mask{0, 0, 1, 0, 1, 0, 1};
  LCW val({{}, {}, {5, 6}, {}, {}, {}, {-1}}, mask.begin());

  fixed_width_column_wrapper<K> expect_key{0, 0, 0, 0, 1, 1, 1};
  std::vector<valid_type> expected_mask{1, 1, 1, 1, 1, 1, 0};
  LCW expect_val({{5, 6}, {5, 6}, {-1}, {-1}, {}, {}, {}}, expected_mask.begin());

  TestReplaceNullsGroupbySingle(key, val, expect_key, expect_val, replace_policy::FOLLOWING);
}

TYPED_TEST(GroupbyReplaceNullsListsTest, PrecedingFillNested)
{
  using LCW    = lists_column_wrapper<TypeParam, int32_t>;
  using Mask_t = std::vector<valid_type>;
  // Group 0 value: {{{1 @ 3} @}
  //                 @
  //                 {{@} {}}}} -->
  //                {{{1 @ 3} @}
  //                 {{1 @ 3} @}
  //                 {{@} {}}}}, w/o leading nulls
  // Group 1 value: {@
  //                 {@ {102 @}}
  //                 @
  //                 {{@ 202} {}}}} -->
  //                {@
  //                 {@ {102 @}}
  //                 {@ {102 @}}
  //                 {{@ 202} {}}}}, w/ leading nulls
  // Only top level nulls are replaced.
  fixed_width_column_wrapper<K> key{1, 0, 1, 1, 0, 0, 1};

  // clang-format off
  LCW val({{},
           LCW({LCW({1, -1, 3}, Mask_t{1, 0, 1}.begin()), {}}, Mask_t{1, 0}.begin()),
           LCW({LCW{}, LCW({102, -1}, Mask_t{1, 0}.begin())}, Mask_t{0, 1}.begin()),
           {},
           {},
           {LCW({{}}, Mask_t{0}.begin()), LCW{}},
           {LCW({-1, 202}, Mask_t{0, 1}.begin()), LCW{}}},
           Mask_t{0, 1, 1, 0, 0, 1, 1}.begin());
  // clang-format on

  fixed_width_column_wrapper<K> expect_key{0, 0, 0, 1, 1, 1, 1};

  // clang-format off
  LCW expect_val({LCW({LCW({1, -1, 3}, Mask_t{1, 0, 1}.begin()), {}}, Mask_t{1, 0}.begin()),
                  LCW({LCW({1, -1, 3}, Mask_t{1, 0, 1}.begin()), {}}, Mask_t{1, 0}.begin()),
                  {LCW({{}}, Mask_t{0}.begin()), LCW{}},
                  {},
                  LCW({LCW{}, LCW({102, -1}, Mask_t{1, 0}.begin())}, Mask_t{0, 1}.begin()),
                  LCW({LCW{}, LCW({102, -1}, Mask_t{1, 0}.begin())}, Mask_t{0, 1}.begin()),
                  {LCW({-1, 202}, Mask_t{0, 1}.begin()), LCW{}}},
           Mask_t{1, 1, 1, 0, 1 ,1 ,1}.begin());
  // clang-format on

  TestReplaceNullsGroupbySingle(key, val, expect_key, expect_val, replace_policy::PRECEDING);
}

TYPED_TEST(GroupbyReplaceNullsListsTest, FollowingFillNested)
{
  using LCW    = lists_column_wrapper<TypeParam, int32_t>;
  using Mask_t = std::vector<valid_type>;
  // Group 0 value: {{{1 @ 3} @}
  //                 @
  //                 {{@} {}}}} -->
  //                {{{1 @ 3} @}
  //                 {{@} {}}}}
  //                 {{@} {}}}}, w/o trailing nulls
  // Group 1 value: {{@ {102 @}}
  //                 @
  //                 {{@ 202} {}}}}
  //                 @ -->
  //                {{@ {102 @}}
  //                 {{@ 202} {}}}
  //                 {{@ 202} {}}}
  //                 @}, w/ trailing nulls
  // Only top level nulls are replaced.
  fixed_width_column_wrapper<K> key{1, 0, 1, 1, 0, 0, 1};

  // clang-format off
  LCW val({LCW({LCW{}, LCW({102, -1}, Mask_t{1, 0}.begin())}, Mask_t{0, 1}.begin()),
           LCW({LCW({1, -1, 3}, Mask_t{1, 0, 1}.begin()), {}}, Mask_t{1, 0}.begin()),
           {},
           {LCW({-1, 202}, Mask_t{0, 1}.begin()), LCW{}},
           {},
           {LCW({{}}, Mask_t{0}.begin()), LCW{}},
           {}},
           Mask_t{1, 1, 0, 1, 0, 1, 0}.begin());
  // clang-format on

  fixed_width_column_wrapper<K> expect_key{0, 0, 0, 1, 1, 1, 1};

  // clang-format off
  LCW expect_val({LCW({LCW({1, -1, 3}, Mask_t{1, 0, 1}.begin()), {}}, Mask_t{1, 0}.begin()),
                 {LCW({{}}, Mask_t{0}.begin()), LCW{}},
                 {LCW({{}}, Mask_t{0}.begin()), LCW{}},
                 LCW({LCW{}, LCW({102, -1}, Mask_t{1, 0}.begin())}, Mask_t{0, 1}.begin()),
                 {LCW({-1, 202}, Mask_t{0, 1}.begin()), LCW{}},
                 {LCW({-1, 202}, Mask_t{0, 1}.begin()), LCW{}},
                 {}},
           Mask_t{1, 1, 1, 1, 1, 1, 0}.begin());
  // clang-format on

  TestReplaceNullsGroupbySingle(key, val, expect_key, expect_val, replace_policy::FOLLOWING);
}

struct GroupbyReplaceNullsStructsTest : public BaseFixture {
  using SCW = structs_column_wrapper;

  SCW data(fixed_width_column_wrapper<int32_t> field0,
           strings_column_wrapper field1,
           lists_column_wrapper<int32_t> field2,
           std::initializer_list<valid_type> mask)
  {
    return SCW({field0, field1, field2}, mask.begin());
  }
};

TEST_F(GroupbyReplaceNullsStructsTest, PrecedingFill)
{
  using LCW    = lists_column_wrapper<int32_t>;
  using Mask_t = std::vector<valid_type>;
  fixed_width_column_wrapper<K> key{1, 0, 0, 1, 0, 1, 1};

  // Only null rows are replaced.

  SCW val =
    this->data({{1, -1, 3, -1, -1, -1, 7}, {1, 0, 1, 0, 0, 0, 1}},
               {{"x", "yy", "", "", "", "zz", ""}, {true, true, false, false, false, true, false}},
               LCW({{1, 2, 3}, {-1}, {}, {}, {42}, {}, {}}, Mask_t{1, 1, 0, 0, 1, 0, 0}.begin()),
               {1, 1, 0, 0, 1, 1, 0});

  fixed_width_column_wrapper<K> expect_key{0, 0, 0, 1, 1, 1, 1};

  SCW expect_val = this->data(
    {{-1, -1, -1, 1, 1, -1, -1}, {0, 0, 0, 1, 1, 0, 0}},
    {{"yy", "yy", "", "x", "x", "zz", "zz"}, {true, true, false, true, true, true, true}},
    LCW({LCW{-1}, {-1}, {42}, {1, 2, 3}, {1, 2, 3}, {}, {}}, Mask_t{1, 1, 1, 1, 1, 0, 0}.begin()),
    {1, 1, 1, 1, 1, 1, 1});

  TestReplaceNullsGroupbySingle(key, val, expect_key, expect_val, replace_policy::PRECEDING);
}

TEST_F(GroupbyReplaceNullsStructsTest, FollowingFill)
{
  using LCW    = lists_column_wrapper<int32_t>;
  using Mask_t = std::vector<valid_type>;
  fixed_width_column_wrapper<K> key{1, 0, 0, 1, 0, 1, 1};

  // Only null rows are replaced.

  SCW val =
    this->data({{1, -1, 3, -1, -1, -1, 7}, {1, 0, 1, 0, 0, 0, 1}},
               {{"x", "yy", "", "", "", "zz", ""}, {true, true, false, false, false, true, false}},
               LCW({{1, 2, 3}, {-1}, {}, {}, {42}, {}, {}}, Mask_t{1, 1, 0, 0, 1, 0, 0}.begin()),
               {1, 1, 0, 0, 1, 1, 0});

  fixed_width_column_wrapper<K> expect_key{0, 0, 0, 1, 1, 1, 1};

  SCW expect_val = this->data(
    {{-1, -1, -1, 1, -1, -1, -1}, {0, 0, 0, 1, 0, 0, 0}},
    {{"yy", "", "", "x", "zz", "zz", ""}, {true, false, false, true, true, true, false}},
    LCW({LCW{-1}, {42}, {42}, {1, 2, 3}, {}, {}, {}}, Mask_t{1, 1, 1, 1, 0, 0, 0}.begin()),
    {1, 1, 1, 1, 1, 1, 0});

  TestReplaceNullsGroupbySingle(key, val, expect_key, expect_val, replace_policy::FOLLOWING);
}

}  // namespace test
}  // namespace cudf
