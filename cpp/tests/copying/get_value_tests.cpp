/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <thrust/iterator/counting_iterator.h>

using namespace cudf::test::iterators;

namespace cudf {
namespace test {

template <typename T>
struct FixedWidthGetValueTest : public BaseFixture {
};

TYPED_TEST_SUITE(FixedWidthGetValueTest, FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(FixedWidthGetValueTest, BasicGet)
{
  fixed_width_column_wrapper<TypeParam, int32_t> col({9, 8, 7, 6});
  auto s = get_element(col, 0);

  using ScalarType = scalar_type_t<TypeParam>;
  auto typed_s     = static_cast<ScalarType const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ(cudf::test::make_type_param_scalar<TypeParam>(9), typed_s->value());
}

TYPED_TEST(FixedWidthGetValueTest, GetFromNullable)
{
  fixed_width_column_wrapper<TypeParam, int32_t> col({9, 8, 7, 6}, {0, 1, 0, 1});
  auto s = get_element(col, 1);

  using ScalarType = scalar_type_t<TypeParam>;
  auto typed_s     = static_cast<ScalarType const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ(cudf::test::make_type_param_scalar<TypeParam>(8), typed_s->value());
}

TYPED_TEST(FixedWidthGetValueTest, GetNull)
{
  fixed_width_column_wrapper<TypeParam, int32_t> col({9, 8, 7, 6}, {0, 1, 0, 1});
  auto s = get_element(col, 2);

  EXPECT_FALSE(s->is_valid());
}

TYPED_TEST(FixedWidthGetValueTest, IndexOutOfBounds)
{
  fixed_width_column_wrapper<TypeParam, int32_t> col({9, 8, 7, 6}, {0, 1, 0, 1});

  // Test for out of bounds indexes in both directions.
  EXPECT_THROW(get_element(col, -1), cudf::logic_error);
  EXPECT_THROW(get_element(col, 4), cudf::logic_error);
}

struct StringGetValueTest : public BaseFixture {
};

TEST_F(StringGetValueTest, BasicGet)
{
  strings_column_wrapper col{"this", "is", "a", "test"};
  auto s = get_element(col, 3);

  auto typed_s = static_cast<string_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ("test", typed_s->to_string());
}

TEST_F(StringGetValueTest, GetEmpty)
{
  strings_column_wrapper col{"this", "is", "", "test"};
  auto s = get_element(col, 2);

  auto typed_s = static_cast<string_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ("", typed_s->to_string());
}

TEST_F(StringGetValueTest, GetFromNullable)
{
  strings_column_wrapper col({"this", "is", "a", "test"}, {0, 1, 0, 1});
  auto s = get_element(col, 1);

  auto typed_s = static_cast<string_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ("is", typed_s->to_string());
}

TEST_F(StringGetValueTest, GetNull)
{
  strings_column_wrapper col({"this", "is", "a", "test"}, {0, 1, 0, 1});
  auto s = get_element(col, 2);

  EXPECT_FALSE(s->is_valid());
}

template <typename T>
struct DictionaryGetValueTest : public BaseFixture {
};

TYPED_TEST_SUITE(DictionaryGetValueTest, FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(DictionaryGetValueTest, BasicGet)
{
  fixed_width_column_wrapper<TypeParam, int32_t> keys({6, 7, 8, 9});
  fixed_width_column_wrapper<uint32_t> indices{0, 0, 1, 2, 1, 3, 3, 2};
  auto col = make_dictionary_column(keys, indices);

  auto s = get_element(*col, 2);

  using ScalarType = scalar_type_t<TypeParam>;
  auto typed_s     = static_cast<ScalarType const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ(cudf::test::make_type_param_scalar<TypeParam>(7), typed_s->value());
}

TYPED_TEST(DictionaryGetValueTest, GetFromNullable)
{
  fixed_width_column_wrapper<TypeParam, int32_t> keys({6, 7, 8, 9});
  fixed_width_column_wrapper<uint32_t> indices({0, 0, 1, 2, 1, 3, 3, 2}, {0, 1, 0, 1, 1, 1, 0, 0});
  auto col = make_dictionary_column(keys, indices);

  auto s = get_element(*col, 3);

  using ScalarType = scalar_type_t<TypeParam>;
  auto typed_s     = static_cast<ScalarType const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ(cudf::test::make_type_param_scalar<TypeParam>(8), typed_s->value());
}

TYPED_TEST(DictionaryGetValueTest, GetNull)
{
  fixed_width_column_wrapper<TypeParam, int32_t> keys({6, 7, 8, 9});
  fixed_width_column_wrapper<uint32_t> indices({0, 0, 1, 2, 1, 3, 3, 2}, {0, 1, 0, 1, 1, 1, 0, 0});
  auto col = make_dictionary_column(keys, indices);

  auto s = get_element(*col, 2);

  EXPECT_FALSE(s->is_valid());
}

/*
 * Lists test grid:
 * Dim1 nestedness:          {Nested, Non-nested}
 * Dim2 validity, emptiness: {Null element, Non-null non-empty list, Non-null empty list}
 * Dim3 leaf data type:      {Fixed-width, string, struct}
 */

template <typename T>
struct ListGetFixedWidthValueTest : public BaseFixture {
  auto odds_valid()
  {
    return cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  }
  auto nth_valid(size_type x)
  {
    return cudf::detail::make_counting_transform_iterator(0, [=](auto i) { return x == i; });
  }
};

TYPED_TEST_SUITE(ListGetFixedWidthValueTest, FixedWidthTypes);

TYPED_TEST(ListGetFixedWidthValueTest, NonNestedGetNonNullNonEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  LCW col{LCW({1, 2, 34}, this->odds_valid()), LCW{}, LCW{1}, LCW{}};
  fixed_width_column_wrapper<TypeParam> expected_data({1, 2, 34}, this->odds_valid());
  size_type index = 0;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TYPED_TEST(ListGetFixedWidthValueTest, NonNestedGetNonNullEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  LCW col{LCW{1, 2, 34}, LCW{}, LCW{1}, LCW{}};
  fixed_width_column_wrapper<TypeParam> expected_data{};
  size_type index = 1;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TYPED_TEST(ListGetFixedWidthValueTest, NonNestedGetNull)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  using FCW = cudf::test::fixed_width_column_wrapper<TypeParam>;

  LCW col({LCW{1, 2, 34}, LCW{}, LCW{1}, LCW{}}, this->odds_valid());
  size_type index = 2;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_FALSE(s->is_valid());
  // Test preserve column hierarchy
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(typed_s->view(), FCW{});
}

TYPED_TEST(ListGetFixedWidthValueTest, NestedGetNonNullNonEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // clang-format off
  LCW col{
    LCW{LCW{1, 2}, LCW{34}},
    LCW{},
    LCW{LCW{1}},
    LCW{LCW{42}, LCW{10}}
  };
  // clang-format on
  LCW expected_data{LCW{42}, LCW{10}};

  size_type index = 3;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TYPED_TEST(ListGetFixedWidthValueTest, NestedGetNonNullNonEmptyPreserveNull)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  std::vector<valid_type> valid{0, 1, 1};
  // clang-format off
  LCW col{
    LCW{LCW{1, 2}, LCW{34}},
    LCW{},
    LCW{LCW{1}},
    LCW({LCW{42}, LCW{10}, LCW({1, 3, 2}, this->nth_valid(1))}, valid.begin())
  };
  // clang-format on
  LCW expected_data({LCW{42}, LCW{10}, LCW({1, 3, 2}, this->nth_valid(1))}, valid.begin());
  size_type index = 3;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TYPED_TEST(ListGetFixedWidthValueTest, NestedGetNonNullEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // clang-format off
  LCW col{
    LCW{LCW{1, 2}, LCW{34}},
    LCW{},
    LCW{LCW{1}},
    LCW{LCW{42}, LCW{10}}
  };
  // clang-format on
  LCW expected_data{};
  size_type index = 1;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TYPED_TEST(ListGetFixedWidthValueTest, NestedGetNull)
{
  using LCW      = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  using FCW      = cudf::test::fixed_width_column_wrapper<TypeParam>;
  using offset_t = cudf::test::fixed_width_column_wrapper<offset_type>;

  std::vector<valid_type> valid{1, 0, 1, 0};
  // clang-format off
  LCW col(
    {
      LCW{LCW{1, 2}, LCW{34}},
      LCW{},
      LCW{LCW{1}},
      LCW{LCW{42}, LCW{10}}
    }, valid.begin());
  // clang-format on
  size_type index = 1;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  auto expected_data =
    make_lists_column(0, offset_t{}.release(), FCW{}.release(), 0, rmm::device_buffer{});

  EXPECT_FALSE(s->is_valid());
  // Test preserve column hierarchy
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(typed_s->view(), *expected_data);
}

struct ListGetStringValueTest : public BaseFixture {
  auto odds_valid()
  {
    return cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  }
  auto nth_valid(size_type x)
  {
    return cudf::detail::make_counting_transform_iterator(0, [=](auto i) { return x == i; });
  }
};

TEST_F(ListGetStringValueTest, NonNestedGetNonNullNonEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<string_view>;

  LCW col{LCW({"aaa", "Héllo"}, this->odds_valid()), LCW{}, LCW{""}, LCW{"42"}};
  strings_column_wrapper expected_data({"aaa", "Héllo"}, this->odds_valid());
  size_type index = 0;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TEST_F(ListGetStringValueTest, NonNestedGetNonNullEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<string_view>;

  LCW col{LCW{"aaa", "Héllo"}, LCW{}, LCW{""}, LCW{"42"}};
  strings_column_wrapper expected_data{};
  size_type index = 1;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TEST_F(ListGetStringValueTest, NonNestedGetNull)
{
  using LCW      = cudf::test::lists_column_wrapper<string_view>;
  using StringCW = strings_column_wrapper;

  std::vector<valid_type> valid{1, 0, 0, 1};
  LCW col({LCW{"aaa", "Héllo"}, LCW{}, LCW{""}, LCW{"42"}}, valid.begin());
  size_type index = 2;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_FALSE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(typed_s->view(), StringCW{});
}

TEST_F(ListGetStringValueTest, NestedGetNonNullNonEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<string_view>;

  // clang-format off
  LCW col{
    LCW{LCW{"aaa", "Héllo"}},
    LCW{},
    LCW{LCW{""}, LCW({"string", "str2", "xyz"}, this->nth_valid(0))},
    LCW{LCW{"42"}, LCW{"21"}}
  };
  // clang-format on
  LCW expected_data{LCW{""}, LCW({"string", "str2", "xyz"}, this->nth_valid(0))};
  size_type index = 2;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TEST_F(ListGetStringValueTest, NestedGetNonNullNonEmptyPreserveNull)
{
  using LCW = cudf::test::lists_column_wrapper<string_view>;

  std::vector<valid_type> valid{0, 1, 1};
  // clang-format off
  LCW col{
    LCW{LCW{"aaa", "Héllo"}},
    LCW{},
    LCW({LCW{""}, LCW{"cc"}, LCW({"string", "str2", "xyz"}, this->nth_valid(0))}, valid.begin()),
    LCW{LCW{"42"}, LCW{"21"}}
  };
  // clang-format on
  LCW expected_data({LCW{""}, LCW{"cc"}, LCW({"string", "str2", "xyz"}, this->nth_valid(0))},
                    valid.begin());
  size_type index = 2;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TEST_F(ListGetStringValueTest, NestedGetNonNullEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<string_view>;

  // clang-format off
  LCW col{
    LCW{LCW{"aaa", "Héllo"}},
    LCW{LCW{""}},
    LCW{LCW{"42"}, LCW{"21"}},
    LCW{}
  };
  // clang-format on
  LCW expected_data{};
  size_type index = 3;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  // Relax to equivalent. `expected_data` leaf string column does not
  // allocate offset and byte array, but `typed_s` does.
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_data, typed_s->view());
}

TEST_F(ListGetStringValueTest, NestedGetNull)
{
  using LCW      = cudf::test::lists_column_wrapper<string_view>;
  using offset_t = cudf::test::fixed_width_column_wrapper<offset_type>;
  using StringCW = cudf::test::strings_column_wrapper;

  std::vector<valid_type> valid{0, 0, 1, 1};
  // clang-format off
  LCW col(
    {
      LCW{LCW{"aaa", "Héllo"}},
      LCW{LCW{""}},
      LCW{LCW{"42"}, LCW{"21"}},
      LCW{}
    }, valid.begin());
  // clang-format on
  size_type index = 0;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  auto expected_data =
    make_lists_column(0, offset_t{}.release(), StringCW{}.release(), 0, rmm::device_buffer{});

  EXPECT_FALSE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected_data, typed_s->view());
}

/**
 * @brief Some shared helper functions used by lists of structs test.
 */
template <typename T>
struct ListGetStructValueTest : public BaseFixture {
  using SCW        = structs_column_wrapper;
  using LCWinner_t = cudf::test::lists_column_wrapper<T, int32_t>;

  /**
   * @brief Create a lists column
   *
   * @note Different from `cudf::make_lists_column`, this allows setting the `null_mask`
   * in `initializer_list`. However this is an expensive function because it repeatedly
   * calls `cudf::set_null_mask` for each row.
   */
  std::unique_ptr<cudf::column> make_test_lists_column(
    size_type num_lists,
    fixed_width_column_wrapper<offset_type> offsets,
    std::unique_ptr<cudf::column> child,
    std::initializer_list<valid_type> null_mask)
  {
    size_type null_count = num_lists - std::accumulate(null_mask.begin(), null_mask.end(), 0);
    auto d_null_mask     = cudf::create_null_mask(
      num_lists, null_count == 0 ? cudf::mask_state::UNALLOCATED : cudf::mask_state::ALL_NULL);
    if (null_count > 0) {
      std::for_each(
        thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_lists), [&](auto i) {
          if (*(null_mask.begin() + i)) {
            set_null_mask(static_cast<bitmask_type*>(d_null_mask.data()), i, i + 1, true);
          }
        });
    }
    return cudf::make_lists_column(
      num_lists, offsets.release(), std::move(child), null_count, std::move(d_null_mask));
  }

  /**
   * @brief Create a structs column that contains 3 fields: int, string, List<int>
   */
  template <typename MaskIterator>
  SCW make_test_structs_column(fixed_width_column_wrapper<T> field1,
                               strings_column_wrapper field2,
                               lists_column_wrapper<T, int32_t> field3,
                               MaskIterator mask)
  {
    return SCW{{field1, field2, field3}, mask};
  }

  /**
   * @brief Create a 0-length structs column
   */
  SCW zero_length_struct() { return SCW{}; }

  /**
   * @brief Concatenate structs columns, allow specifying inputs in `initializer_list`
   */
  std::unique_ptr<cudf::column> concat(std::initializer_list<SCW> rows)
  {
    std::vector<column_view> views;
    std::transform(
      rows.begin(), rows.end(), std::back_inserter(views), [](auto& r) { return column_view(r); });
    return cudf::concatenate(views);
  }

  /**
   * @brief Test data setup: row 0 of structs column
   */
  SCW row0()
  {
    // {int: 1, string: NULL, list: NULL}
    return this->make_test_structs_column({{1}, {1}},
                                          strings_column_wrapper({"aa"}, {false}),
                                          LCWinner_t({{}}, all_nulls()),
                                          no_nulls());
  }

  /**
   * @brief Test data setup: row 1 of structs column
   */
  SCW row1()
  {
    // NULL
    return this->make_test_structs_column({-1}, {""}, LCWinner_t{-1}, all_nulls());
  }

  /**
   * @brief Test data setup: row 2 of structs column
   */
  SCW row2()
  {
    // {int: 3, string: "xyz", list: [3, 8, 4]}
    return this->make_test_structs_column({{3}, {1}},
                                          strings_column_wrapper({"xyz"}, {true}),
                                          LCWinner_t({{3, 8, 4}}, no_nulls()),
                                          no_nulls());
  }

  /**
   * @brief Test data setup: a 3-row structs column
   */
  std::unique_ptr<cudf::column> leaf_data()
  {
    // 3 rows:
    // {int: 1, string: NULL, list: NULL}
    // NULL
    // {int: 3, string: "xyz", list: [3, 8, 4]}
    return this->concat({row0(), row1(), row2()});
  }
};

TYPED_TEST_SUITE(ListGetStructValueTest, FixedWidthTypes);

TYPED_TEST(ListGetStructValueTest, NonNestedGetNonNullNonEmpty)
{
  // 2-rows
  // [{1, NULL, NULL}, NULL]
  // [{3, "xyz", [3, 8, 4]}] <- get_element(1)

  auto list_column   = this->make_test_lists_column(2, {0, 2, 3}, this->leaf_data(), {1, 1});
  size_type index    = 1;
  auto expected_data = this->row2();

  auto s       = get_element(list_column->view(), index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  // Relax to equivalent. The nested list column in struct allocates `null_mask`.
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_data, typed_s->view());
}

TYPED_TEST(ListGetStructValueTest, NonNestedGetNonNullNonEmpty2)
{
  // 2-rows
  // [{1, NULL, NULL}, NULL] <- get_element(0)
  // [{3, "xyz", [3, 8, 4]}]

  auto list_column   = this->make_test_lists_column(2, {0, 2, 3}, this->leaf_data(), {1, 1});
  size_type index    = 0;
  auto expected_data = this->concat({this->row0(), this->row1()});

  auto s       = get_element(list_column->view(), index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_data, typed_s->view());
}

TYPED_TEST(ListGetStructValueTest, NonNestedGetNonNullEmpty)
{
  // 3-rows
  // [{1, NULL, NULL}, NULL]
  // [{3, "xyz", [3, 8, 4]}]
  // []                      <- get_element(2)

  auto list_column = this->make_test_lists_column(3, {0, 2, 3, 3}, this->leaf_data(), {1, 1, 1});
  size_type index  = 2;
  // For well-formed list column, an empty list still holds the complete structure of
  // a 0-length structs column
  auto expected_data = this->zero_length_struct();

  auto s       = get_element(list_column->view(), index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  // Relax to equivalent. The nested list column in struct allocates `null_mask`.
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_data, typed_s->view());
}

TYPED_TEST(ListGetStructValueTest, NonNestedGetNull)
{
  // 2-rows
  // NULL                    <- get_element(0)
  // [{3, "xyz", [3, 8, 4]}]

  using valid_t = std::vector<valid_type>;

  auto list_column = this->make_test_lists_column(2, {0, 2, 3}, this->leaf_data(), {0, 1});
  size_type index  = 0;

  auto s       = get_element(list_column->view(), index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  auto expected_data = this->make_test_structs_column({}, {}, {}, valid_t{}.begin());

  EXPECT_FALSE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(typed_s->view(), expected_data);
}

TYPED_TEST(ListGetStructValueTest, NestedGetNonNullNonEmpty)
{
  // 2-rows
  // [[{1, NULL, NULL}, NULL], [{3, "xyz", [3, 8, 4]}]] <- get_element(0)
  // []

  auto list_column   = this->make_test_lists_column(2, {0, 2, 3}, this->leaf_data(), {1, 1});
  auto expected_data = std::make_unique<cudf::column>(*list_column);

  auto list_column_nested =
    this->make_test_lists_column(2, {0, 2, 2}, std::move(list_column), {1, 1});

  size_type index = 0;
  auto s          = get_element(list_column_nested->view(), index);
  auto typed_s    = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_data, typed_s->view());
}

TYPED_TEST(ListGetStructValueTest, NestedGetNonNullNonEmpty2)
{
  // 2-rows
  // [[{1, NULL, NULL}, NULL]] <- get_element(0)
  // [[{3, "xyz", [3, 8, 4]}]]

  auto list_column = this->make_test_lists_column(2, {0, 2, 3}, this->leaf_data(), {1, 1});
  auto list_column_nested =
    this->make_test_lists_column(2, {0, 1, 2}, std::move(list_column), {1, 1});

  auto expected_data =
    this->make_test_lists_column(1, {0, 2}, this->concat({this->row0(), this->row1()}), {1});

  size_type index = 0;
  auto s          = get_element(list_column_nested->view(), index);
  auto typed_s    = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_data, typed_s->view());
}

TYPED_TEST(ListGetStructValueTest, NestedGetNonNullNonEmpty3)
{
  // 2-rows
  // [[{1, NULL, NULL}, NULL]]
  // [[{3, "xyz", [3, 8, 4]}]] <- get_element(1)

  auto list_column = this->make_test_lists_column(2, {0, 2, 3}, this->leaf_data(), {1, 1});
  auto list_column_nested =
    this->make_test_lists_column(2, {0, 1, 2}, std::move(list_column), {1, 1});

  auto expected_data = this->make_test_lists_column(1, {0, 1}, this->row2().release(), {1});

  size_type index = 1;
  auto s          = get_element(list_column_nested->view(), index);
  auto typed_s    = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  // Relax to equivalent. For `get_element`, the nested list column in struct
  // allocates `null_mask`.
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected_data, typed_s->view());
}

TYPED_TEST(ListGetStructValueTest, NestedGetNonNullEmpty)
{
  // 3-rows
  // [[{1, NULL, NULL}, NULL]]
  // []                        <- get_element(1)
  // [[{3, "xyz", [3, 8, 4]}]]

  auto list_column = this->make_test_lists_column(2, {0, 2, 3}, this->leaf_data(), {1, 1});
  auto list_column_nested =
    this->make_test_lists_column(3, {0, 1, 1, 2}, std::move(list_column), {1, 1, 1});

  auto expected_data =
    this->make_test_lists_column(0, {0}, this->zero_length_struct().release(), {1});

  size_type index = 1;
  auto s          = get_element(list_column_nested->view(), index);
  auto typed_s    = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  // Relax to equivalent. The sliced version still has the array for fields
  // allocated.
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected_data, typed_s->view());
}

TYPED_TEST(ListGetStructValueTest, NestedGetNull)
{
  // 3-rows
  // [[{1, NULL, NULL}, NULL]]
  // []
  // NULL                      <- get_element(2)

  using valid_t  = std::vector<valid_type>;
  using offset_t = cudf::test::fixed_width_column_wrapper<offset_type>;

  auto list_column = this->make_test_lists_column(2, {0, 2, 3}, this->leaf_data(), {1, 1});
  auto list_column_nested =
    this->make_test_lists_column(3, {0, 1, 1, 2}, std::move(list_column), {1, 1, 0});

  size_type index = 2;
  auto s          = get_element(list_column_nested->view(), index);
  auto typed_s    = static_cast<list_scalar const*>(s.get());

  auto nested = this->make_test_structs_column({}, {}, {}, valid_t{}.begin());
  auto expected_data =
    make_lists_column(0, offset_t{}.release(), nested.release(), 0, rmm::device_buffer{});

  EXPECT_FALSE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected_data, typed_s->view());
}

struct StructGetValueTest : public BaseFixture {
};
template <typename T>
struct StructGetValueTestTyped : public BaseFixture {
};

TYPED_TEST_SUITE(StructGetValueTestTyped, FixedWidthTypes);

TYPED_TEST(StructGetValueTestTyped, mixed_types_valid)
{
  using LCW = lists_column_wrapper<TypeParam, int32_t>;

  // col fields
  fixed_width_column_wrapper<TypeParam> f1{1, 2, 3};
  strings_column_wrapper f2{"aa", "bbb", "c"};
  dictionary_column_wrapper<TypeParam, uint32_t> f3{42, 42, 24};
  LCW f4{LCW{8, 8, 8}, LCW{9, 9}, LCW{10}};

  structs_column_wrapper col{f1, f2, f3, f4};

  size_type index = 2;
  auto s          = get_element(col, index);
  auto typed_s    = static_cast<struct_scalar const*>(s.get());

  // expect fields
  fixed_width_column_wrapper<TypeParam> ef1{3};
  strings_column_wrapper ef2{"c"};
  dictionary_column_wrapper<int32_t, TypeParam> ef3{24};
  LCW ef4{LCW{10}};

  table_view expect_data{{ef1, ef2, ef3, ef4}};

  EXPECT_TRUE(typed_s->is_valid());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expect_data, typed_s->view());
}

TYPED_TEST(StructGetValueTestTyped, mixed_types_valid_with_nulls)
{
  using LCW             = lists_column_wrapper<TypeParam, int32_t>;
  using validity_mask_t = std::vector<valid_type>;

  // col fields
  fixed_width_column_wrapper<TypeParam> f1({1, 2, 3}, {true, false, true});
  strings_column_wrapper f2({"aa", "bbb", "c"}, {false, false, true});
  dictionary_column_wrapper<TypeParam, uint32_t> f3({42, 42, 24},
                                                    validity_mask_t{true, true, true}.begin());
  LCW f4({LCW{8, 8, 8}, LCW{9, 9}, LCW{10}}, validity_mask_t{false, false, false}.begin());

  structs_column_wrapper col{f1, f2, f3, f4};

  size_type index = 1;
  auto s          = get_element(col, index);
  auto typed_s    = static_cast<struct_scalar const*>(s.get());

  // expect fields
  fixed_width_column_wrapper<TypeParam> ef1({-1}, {false});
  strings_column_wrapper ef2({""}, {false});

  dictionary_column_wrapper<TypeParam, uint32_t> x({42}, {true});
  dictionary_column_view dict_col(x);
  fixed_width_column_wrapper<TypeParam> new_key{24};
  auto ef3 = cudf::dictionary::add_keys(dict_col, new_key);

  LCW ef4({LCW{10}}, validity_mask_t{false}.begin());

  table_view expect_data{{ef1, ef2, *ef3, ef4}};

  EXPECT_TRUE(typed_s->is_valid());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expect_data, typed_s->view());
}

TYPED_TEST(StructGetValueTestTyped, mixed_types_invalid)
{
  using LCW             = lists_column_wrapper<TypeParam, int32_t>;
  using validity_mask_t = std::vector<valid_type>;

  // col fields
  fixed_width_column_wrapper<TypeParam> f1{1, 2, 3};
  strings_column_wrapper f2{"aa", "bbb", "c"};
  dictionary_column_wrapper<TypeParam, uint32_t> f3{42, 42, 24};
  LCW f4{LCW{8, 8, 8}, LCW{9, 9}, LCW{10}};

  structs_column_wrapper col({f1, f2, f3, f4}, validity_mask_t{false, true, true}.begin());

  size_type index = 0;
  auto s          = get_element(col, index);
  auto typed_s    = static_cast<struct_scalar const*>(s.get());

  EXPECT_FALSE(typed_s->is_valid());

  // expect to preserve types along column hierarchy.
  EXPECT_EQ(typed_s->view().column(0).type().id(), type_to_id<TypeParam>());
  EXPECT_EQ(typed_s->view().column(1).type().id(), type_id::STRING);
  EXPECT_EQ(typed_s->view().column(2).type().id(), type_id::DICTIONARY32);
  EXPECT_EQ(typed_s->view().column(2).child(1).type().id(), type_to_id<TypeParam>());
  EXPECT_EQ(typed_s->view().column(3).type().id(), type_id::LIST);
  EXPECT_EQ(typed_s->view().column(3).child(1).type().id(), type_to_id<TypeParam>());
}

TEST_F(StructGetValueTest, multi_level_nested)
{
  using LCW             = lists_column_wrapper<int32_t, int32_t>;
  using validity_mask_t = std::vector<valid_type>;

  // col fields
  LCW l3({LCW{1, 1, 1}, LCW{2, 2}, LCW{3}}, validity_mask_t{false, true, true}.begin());
  structs_column_wrapper l2{l3};
  auto l1 = make_lists_column(1,
                              fixed_width_column_wrapper<offset_type>{0, 3}.release(),
                              l2.release(),
                              0,
                              create_null_mask(1, mask_state::UNALLOCATED));
  std::vector<std::unique_ptr<column>> l0_fields;
  l0_fields.emplace_back(std::move(l1));
  structs_column_wrapper l0(std::move(l0_fields));

  size_type index = 0;
  auto s          = get_element(l0, index);
  auto typed_s    = static_cast<struct_scalar const*>(s.get());

  // Expect fields
  column_view cv = column_view(l0);
  table_view fields(std::vector<column_view>(cv.child_begin(), cv.child_end()));

  EXPECT_TRUE(typed_s->is_valid());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(fields, typed_s->view());
}

}  // namespace test
}  // namespace cudf
