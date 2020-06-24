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

#include <tests/groupby/groupby_test_util.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>

namespace cudf {
namespace test {

template <typename V>
struct groupby_nth_element_test : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(groupby_nth_element_test, cudf::test::AllTypes);

// clang-format off
TYPED_TEST(groupby_nth_element_test, basic)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::NTH_ELEMENT>;

  fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  fixed_width_column_wrapper<V, int32_t> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  //keys                            {1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
  //vals                            {0, 3, 6, 1, 4, 5, 9, 2, 7, 8};

  fixed_width_column_wrapper<K> expect_keys{1, 2, 3};

  //groupby.first()
  auto agg = cudf::make_nth_element_aggregation(0);
  fixed_width_column_wrapper<R, int32_t> expect_vals0({0, 1, 2});
  test_single_agg(keys, vals, expect_keys, expect_vals0, std::move(agg));

  agg = cudf::make_nth_element_aggregation(1);
  fixed_width_column_wrapper<R, int32_t> expect_vals1({3, 4, 7});
  test_single_agg(keys, vals, expect_keys, expect_vals1, std::move(agg));

  agg = cudf::make_nth_element_aggregation(2);
  fixed_width_column_wrapper<R, int32_t> expect_vals2({6, 5, 8});
  test_single_agg(keys, vals, expect_keys, expect_vals2, std::move(agg));
}

TYPED_TEST(groupby_nth_element_test, empty_cols)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::NTH_ELEMENT>;

  fixed_width_column_wrapper<K> keys{};
  fixed_width_column_wrapper<V> vals{};

  fixed_width_column_wrapper<K> expect_keys{};
  fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_nth_element_aggregation(0);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nth_element_test, basic_out_of_bounds)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::NTH_ELEMENT>;

  fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  fixed_width_column_wrapper<V, int32_t> vals({0, 1, 2, 3, 4, 5, 3, 2, 2, 9});

  fixed_width_column_wrapper<K> expect_keys{1, 2, 3};

  auto agg = cudf::make_nth_element_aggregation(3);
  fixed_width_column_wrapper<R, int32_t> expect_vals({0, 9, 0}, {0, 1, 0});
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nth_element_test, negative)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::NTH_ELEMENT>;

  fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  fixed_width_column_wrapper<V, int32_t> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  //keys                            {1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
  //vals                            {0, 3, 6, 1, 4, 5, 9, 2, 7, 8};

  fixed_width_column_wrapper<K> expect_keys{1, 2, 3};

  //groupby.last()
  auto agg = cudf::make_nth_element_aggregation(-1);
  fixed_width_column_wrapper<R, int32_t> expect_vals0({6, 9, 8});
  test_single_agg(keys, vals, expect_keys, expect_vals0, std::move(agg));

  agg = cudf::make_nth_element_aggregation(-2);
  fixed_width_column_wrapper<R, int32_t> expect_vals1({3, 5, 7});
  test_single_agg(keys, vals, expect_keys, expect_vals1, std::move(agg));

  agg = cudf::make_nth_element_aggregation(-3);
  fixed_width_column_wrapper<R, int32_t> expect_vals2({0, 4, 2});
  test_single_agg(keys, vals, expect_keys, expect_vals2, std::move(agg));
}

TYPED_TEST(groupby_nth_element_test, negative_out_of_bounds)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::NTH_ELEMENT>;

  fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  fixed_width_column_wrapper<V, int32_t> vals({0, 1, 2, 3, 4, 5, 3, 2, 2, 9});

  fixed_width_column_wrapper<K> expect_keys{1, 2, 3};

  auto agg = cudf::make_nth_element_aggregation(-4);
  fixed_width_column_wrapper<R, int32_t> expect_vals({0, 1, 0}, {0, 1, 0});
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nth_element_test, zero_valid_keys)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::NTH_ELEMENT>;

  fixed_width_column_wrapper<K> keys({1, 2, 3}, all_null());
  fixed_width_column_wrapper<V, int32_t> vals({3, 4, 5});

  fixed_width_column_wrapper<K> expect_keys{};
  fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_nth_element_aggregation(0);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nth_element_test, zero_valid_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::NTH_ELEMENT>;

  fixed_width_column_wrapper<K> keys{1, 1, 1};
  fixed_width_column_wrapper<V, int32_t> vals({3, 4, 5}, all_null());

  fixed_width_column_wrapper<K> expect_keys{1};
  fixed_width_column_wrapper<R, int32_t> expect_vals({3}, all_null());

  auto agg = cudf::make_nth_element_aggregation(0);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nth_element_test, null_keys_and_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::NTH_ELEMENT>;

  fixed_width_column_wrapper<K> keys({1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                     {1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
  fixed_width_column_wrapper<V, int32_t> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                              {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, all_valid());
  //keys                                    {1, 1, 1   2,2,2,2    3, 3,    4}
  //vals                                    {-,3,6,    1,4,-,9,  2,8,      -}
  fixed_width_column_wrapper<R, int32_t> expect_vals({-1, 1, 2, -1}, {0, 1, 1, 0});

  auto agg = cudf::make_nth_element_aggregation(0);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nth_element_test, null_keys_and_values_out_of_bounds)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::NTH_ELEMENT>;

  fixed_width_column_wrapper<K> keys({1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                     {1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
  fixed_width_column_wrapper<V, int32_t> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                              {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});
  //                                        {1, 1, 1    2, 2, 2,    3, 3,   4}
  fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, all_valid());
  //                                        {-,3,6,     1,4,-,9,    2,8,    -}
  //                                         value,     null,       out,    out
  fixed_width_column_wrapper<R, int32_t> expect_vals({6, -1, -1, -1}, {1, 0, 0, 0});

  auto agg = cudf::make_nth_element_aggregation(2);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nth_element_test, exclude_nulls)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::NTH_ELEMENT>;

  fixed_width_column_wrapper<K> keys({1, 2, 3, 3, 1, 2, 2, 1, 3, 3, 2, 4, 4, 2},
                                     {1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1});
  fixed_width_column_wrapper<V, int32_t> vals({0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 4, 4, 2},
                                              {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0});

  fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, all_valid());
  //keys                                    {1, 1, 1    2, 2, 2, 2      3, 3, 3    4}
  //vals                                    {-, 3, 6    1, 4, -, 9, -   2, 2, 8,   4,-}
  //                                      0  null,      value,          value,     null
  //                                      1  value,     value,          value,     null
  //                                      2  value,     null,           value,     out
  //null_policy::INCLUDE
  fixed_width_column_wrapper<R, int32_t> expect_nuls0({-1, 1, 2, 4}, {0, 1, 1, 1});
  fixed_width_column_wrapper<R, int32_t> expect_nuls1({3, 4, 2, -1}, {1, 1, 1, 0});
  fixed_width_column_wrapper<R, int32_t> expect_nuls2({6, -1, 8, -1}, {1, 0, 1, 0});

  //null_policy::EXCLUDE
  fixed_width_column_wrapper<R, int32_t> expect_vals0({3, 1, 2, 4});
  fixed_width_column_wrapper<R, int32_t> expect_vals1({6, 4, 2, -1}, {1, 1, 1, 0});
  fixed_width_column_wrapper<R, int32_t> expect_vals2({-1, 9, 8, -1}, {0, 1, 1, 0});

  auto agg = cudf::make_nth_element_aggregation(0, cudf::null_policy::INCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_nuls0, std::move(agg));
  agg = cudf::make_nth_element_aggregation(1, cudf::null_policy::INCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_nuls1, std::move(agg));
  agg = cudf::make_nth_element_aggregation(2, cudf::null_policy::INCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_nuls2, std::move(agg));

  agg = cudf::make_nth_element_aggregation(0, cudf::null_policy::EXCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_vals0, std::move(agg));
  agg = cudf::make_nth_element_aggregation(1, cudf::null_policy::EXCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_vals1, std::move(agg));
  agg = cudf::make_nth_element_aggregation(2, cudf::null_policy::EXCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_vals2, std::move(agg));
}

TYPED_TEST(groupby_nth_element_test, exclude_nulls_negative_index)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::detail::target_type_t<V, aggregation::NTH_ELEMENT>;

  fixed_width_column_wrapper<K> keys({1, 2, 3, 3, 1, 2, 2, 1, 3, 3, 2, 4, 4, 2},
                                     {1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1});
  fixed_width_column_wrapper<V, int32_t> vals({0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 4, 4, 2},
                                              {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0});

  fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, all_valid());
  //keys                                    {1, 1, 1    2, 2, 2,        3, 3,       4}
  //vals                                    {-, 3, 6    1, 4, -, 9, -   2, 2, 8,    4,-}
  //                                      0  null,      value,          value,      value
  //                                      1  value,     value,          value,      null
  //                                      2  value,     null,           value,      out
  //                                      3  out,       value,          out,        out
  //                                      4  out,       null,           out,        out

  //null_policy::INCLUDE
  fixed_width_column_wrapper<R, int32_t> expect_nuls0({6, -1, 8, -1}, {1, 0, 1, 0});
  fixed_width_column_wrapper<R, int32_t> expect_nuls1({3, 9, 2, 4});
  fixed_width_column_wrapper<R, int32_t> expect_nuls2({-1, -1, 2, -1}, {0, 0, 1, 0});

  //null_policy::EXCLUDE
  fixed_width_column_wrapper<R, int32_t> expect_vals0({6, 9, 8, 4});
  fixed_width_column_wrapper<R, int32_t> expect_vals1({3, 4, 2, -1}, {1, 1, 1, 0});
  fixed_width_column_wrapper<R, int32_t> expect_vals2({-1, 1, 2, -1}, {0, 1, 1, 0});

  auto agg = cudf::make_nth_element_aggregation(-1, cudf::null_policy::INCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_nuls0, std::move(agg));
  agg = cudf::make_nth_element_aggregation(-2, cudf::null_policy::INCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_nuls1, std::move(agg));
  agg = cudf::make_nth_element_aggregation(-3, cudf::null_policy::INCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_nuls2, std::move(agg));

  agg = cudf::make_nth_element_aggregation(-1, cudf::null_policy::EXCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_vals0, std::move(agg));
  agg = cudf::make_nth_element_aggregation(-2, cudf::null_policy::EXCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_vals1, std::move(agg));
  agg = cudf::make_nth_element_aggregation(-3, cudf::null_policy::EXCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_vals2, std::move(agg));
}

TYPED_TEST(groupby_nth_element_test, basic_string)
{
  using K = int32_t;
  using V = std::string;
  using R = cudf::detail::target_type_t<V, aggregation::NTH_ELEMENT>;

  fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  strings_column_wrapper vals{"ABCD", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
  //keys                     {1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
  //vals                     {A, 3, 6, 1, 4, 5, 9, 2, 7, 8};

  fixed_width_column_wrapper<K> expect_keys{1, 2, 3};

  //groupby.first()
  auto agg = cudf::make_nth_element_aggregation(0);
  strings_column_wrapper expect_vals0{"ABCD", "1", "2"};
  test_single_agg(keys, vals, expect_keys, expect_vals0, std::move(agg));

  agg = cudf::make_nth_element_aggregation(1);
  strings_column_wrapper expect_vals1{"3", "4", "7"};
  test_single_agg(keys, vals, expect_keys, expect_vals1, std::move(agg));

  agg = cudf::make_nth_element_aggregation(2);
  strings_column_wrapper expect_vals2{"6", "5", "8"};
  test_single_agg(keys, vals, expect_keys, expect_vals2, std::move(agg));

  //+ve out of bounds
  agg = cudf::make_nth_element_aggregation(3);
  strings_column_wrapper expect_vals3{{"", "9", ""}, {0, 1, 0}};
  test_single_agg(keys, vals, expect_keys, expect_vals3, std::move(agg));

  //groupby.last()
  agg = cudf::make_nth_element_aggregation(-1);
  strings_column_wrapper expect_vals4{"6", "9", "8"};
  test_single_agg(keys, vals, expect_keys, expect_vals4, std::move(agg));

  agg = cudf::make_nth_element_aggregation(-2);
  strings_column_wrapper expect_vals5{"3", "5", "7"};
  test_single_agg(keys, vals, expect_keys, expect_vals5, std::move(agg));

  agg = cudf::make_nth_element_aggregation(-3);
  strings_column_wrapper expect_vals6{"ABCD", "4", "2"};
  test_single_agg(keys, vals, expect_keys, expect_vals6, std::move(agg));

  //-ve out of bounds
  agg = cudf::make_nth_element_aggregation(-4);
  strings_column_wrapper expect_vals7{{"", "1", ""}, {0, 1, 0}};
  test_single_agg(keys, vals, expect_keys, expect_vals7, std::move(agg));
}
// clang-format on

}  // namespace test
}  // namespace cudf
