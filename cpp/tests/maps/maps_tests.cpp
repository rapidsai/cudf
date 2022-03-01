/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/maps/maps_column_view.hpp>
#include <cudf/scalar/scalar_factories.hpp>

namespace cudf::test {
using namespace cudf::test::iterators;
using structs = structs_column_wrapper;
template <typename T>
using fwcw = fixed_width_column_wrapper<T, int32_t>;
template <typename T>
using lists   = lists_column_wrapper<T, int32_t>;
using offsets = fwcw<cudf::size_type>;

auto constexpr X = int32_t{0};  // Placeholder for null values.

template <typename T>
constexpr bool always_false_v = false;  // Type dependent falsifier, for use with static_assert.

template <typename T>
auto make_null_scalar_search_key()
{
  auto type = data_type{type_to_id<T>()};
  if constexpr (cudf::is_numeric<T>()) {
    return cudf::make_numeric_scalar(type);
  } else if constexpr (cudf::is_timestamp<T>()) {
    return cudf::make_timestamp_scalar(type);
  } else if constexpr (cudf::is_duration<T>()) {
    return cudf::make_duration_scalar(type);
  } else {
    static_assert(always_false_v<T>, "Unsupported test type.");
  }
}

template <typename T, std::enable_if_t<cudf::is_numeric<T>(), void>* = nullptr>
auto make_scalar_search_key(T const& value)
{
  auto search_key = make_null_scalar_search_key<T>();
  search_key->set_valid_async(true);
  static_cast<scalar_type_t<T>*>(search_key.get())->set_value(value);
  return search_key;
}

template <typename T, std::enable_if_t<cudf::is_timestamp<T>(), void>* = nullptr>
auto make_scalar_search_key(typename T::rep const& value)
{
  auto search_key = make_null_scalar_search_key<T>();
  search_key->set_valid_async(true);
  static_cast<scalar_type_t<typename T::rep>*>(search_key.get())->set_value(value);
  return search_key;
}

template <typename T, std::enable_if_t<cudf::is_duration<T>(), void>* = nullptr>
auto make_scalar_search_key(typename T::rep const& value)
{
  auto search_key = cudf::make_duration_scalar(data_type{type_to_id<T>()});
  search_key->set_valid_async(true);
  static_cast<scalar_type_t<typename T::rep>*>(search_key.get())->set_value(value);
  return search_key;
}

template <typename T>
std::unique_ptr<cudf::column> make_numeric_maps_input()
{
  // Construct maps column with the following contents:
  // [ {0:0, 0:1, 0:2},   // Duplicate keys
  //   {0:0, 1:1, 2:2},
  //   {3:3, 0:4, 4:5},
  //   {5:6, 6:7, 0:8},
  //   {7:9, X:10, 9:11}, // Null key.
  //   {8:8, 0:X, 8:0},   // Null value
  //   {},                // Empty map.
  //   X,                 // Null map.
  //   {9:9, 9:8, 9:7}
  // ]
  // clang-format off
  auto keys = fwcw<T>{{0,0,0, 0,1,2, 3,0,4, 5,6,0, 7,X,9,   8,0,8, 9,9,9}, null_at(13)};
  auto vals = fwcw<T>{{0,1,2, 0,1,2, 3,4,5, 6,7,8, 9,10,11, 8,X,0, 9,8,7}, null_at(16)};
  // clang-format on
  auto key_vals              = structs{keys, vals};
  auto const num_rows        = 9;
  auto const null_mask_begin = null_at(7);
  return cudf::make_lists_column(
    9,
    offsets{0, 3, 6, 9, 12, 15, 18, 18, 18, 21}.release(),
    key_vals.release(),
    1,
    detail::make_null_mask(null_mask_begin, null_mask_begin + num_rows));
}

/**
 * @brief Return sliced column, with the first and last rows sliced off.
 */
cudf::column_view slice_off_ends(cudf::column_view const& col)
{
  return col.size() >= 2 ? cudf::slice(col, {1, col.size() - 1}).front() : col;
}

template <typename T>
struct MapsTest : BaseFixture {
};

using MapsTestTypes = Concat<IntegralTypesNotBool, FloatingPointTypes, ChronoTypes>;

TYPED_TEST_SUITE(MapsTest, MapsTestTypes);

TYPED_TEST(MapsTest, Construction)
{
  using T                        = TypeParam;
  using list                     = lists<T>;
  auto const list_binary_structs = make_numeric_maps_input<T>();
  auto const maps                = cudf::maps_column_view{list_binary_structs->view()};
  auto const expected_keys       = list{{{0, 0, 0},
                                   {0, 1, 2},
                                   {3, 0, 4},
                                   {5, 6, 0},
                                   list{{7, X, 9}, null_at(1)},
                                   {8, 0, 8},
                                   {},
                                   {},  // Null map.
                                   {9, 9, 9}},
                                  null_at(7)};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(maps.keys().parent(), expected_keys);
  auto const expected_values = list{{{0, 1, 2},
                                     {0, 1, 2},
                                     {3, 4, 5},
                                     {6, 7, 8},
                                     {9, 10, 11},
                                     list{{8, X, 0}, null_at(1)},
                                     {},
                                     {},  // Null map.
                                     {9, 8, 7}},
                                    null_at(7)};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(maps.values().parent(), expected_values);

  // Test that construction from a sliced input produces expected results.
  auto const sliced_lists         = slice_off_ends({list_binary_structs->view()});
  auto const sliced_maps          = cudf::maps_column_view{sliced_lists};
  auto const sliced_expected_keys = slice_off_ends(expected_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sliced_maps.keys().parent(), sliced_expected_keys);
  auto const sliced_expected_values = slice_off_ends(expected_values);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sliced_maps.values().parent(), sliced_expected_values);
}

TYPED_TEST(MapsTest, Lookup)
{
  using T = TypeParam;

  auto const list_binary_structs_column = make_numeric_maps_input<T>();
  auto const lists_view                 = list_binary_structs_column->view();
  auto const maps                       = cudf::maps_column_view{lists_view};
  {
    // Lookup keys are a column of '0's.
    auto const result   = maps.get_values_for(fwcw<T>{0, 0, 0, 0, 0, 0, 0, 0, 0});
    auto const expected = fwcw<T>{{2, 0, 4, 8, X, X, X, X, X}, nulls_at({4, 5, 6, 7, 8})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
  }
  {
    // Lookup key is a scalar{0}.
    auto const result   = maps.get_values_for(*make_scalar_search_key<T>(0));
    auto const expected = fwcw<T>{{2, 0, 4, 8, X, X, X, X, X}, nulls_at({4, 5, 6, 7, 8})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
  }
  {
    // Lookup key is a column of different values.
    auto const result   = maps.get_values_for(fwcw<T>{{0, 1, 3, 5, 7, 0, 1, 2, X}, nulls_at({8})});
    auto const expected = fwcw<T>{{2, 1, 3, 6, 9, X, X, X, X}, nulls_at({5, 6, 7, 8})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
  }
  {
    // Lookup keys are a column of nulls.
    auto const result   = maps.get_values_for(fwcw<T>{{X, X, X, X, X, X, X, X, X}, all_nulls()});
    auto const expected = fwcw<T>{{X, X, X, X, X, X, X, X, X}, all_nulls()};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
  }
  {
    // Lookup key is null scalar.
    auto const result   = maps.get_values_for(*make_null_scalar_search_key<T>());
    auto const expected = fwcw<T>{{X, X, X, X, X, X, X, X, X}, all_nulls()};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
  }
  {
    // Input is a sliced column.
    auto const sliced_lists      = slice_off_ends({lists_view});
    auto const sliced_maps       = cudf::maps_column_view{sliced_lists};
    auto const result_col_0      = sliced_maps.get_values_for(fwcw<T>{0, 0, 0, 0, 0, 0, 0});
    auto const expected_lookup_0 = fwcw<T>{{0, 4, 8, X, X, X, X}, nulls_at({3, 4, 5, 6})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result_col_0->view(), expected_lookup_0);

    auto const result_scalar_0 = sliced_maps.get_values_for(*make_scalar_search_key<T>(0));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result_scalar_0->view(), expected_lookup_0);

    auto const result_multi_col      = sliced_maps.get_values_for(fwcw<T>{1, 3, 5, 7, 0, 1, 2});
    auto const expected_lookup_multi = fwcw<T>{{1, 3, 6, 9, X, X, X}, nulls_at({4, 5, 6})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result_multi_col->view(), expected_lookup_multi);
  }
}

}  // namespace cudf::test
