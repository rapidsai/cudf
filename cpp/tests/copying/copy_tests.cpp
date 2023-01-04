/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/encode.hpp>
#include <cudf/scalar/scalar.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

template <typename T>
struct CopyTest : public cudf::test::BaseFixture {
};

TYPED_TEST_SUITE(CopyTest, cudf::test::FixedWidthTypesWithoutFixedPoint);

#define wrapper cudf::test::fixed_width_column_wrapper

TYPED_TEST(CopyTest, CopyIfElseTestShort)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<bool> mask_w{1, 0, 0, 0};

  wrapper<T, int32_t> lhs_w({5, 5, 5, 5}, {1, 1, 1, 1});
  wrapper<T, int32_t> rhs_w({6, 6, 6, 6}, {1, 1, 1, 1});
  wrapper<T, int32_t> expected_w({5, 6, 6, 6});

  auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected_w);
}

TYPED_TEST(CopyTest, CopyIfElseTestManyNulls)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<bool> mask_w{{1, 0, 0, 0, 0, 0, 1}, {1, 1, 1, 1, 1, 1, 0}};

  wrapper<T, int32_t> lhs_w({5, 5, 5, 5, 5, 5, 5}, {1, 1, 1, 1, 1, 1, 1});
  wrapper<T, int32_t> rhs_w({6, 6, 6, 6, 6, 6, 6}, {1, 0, 0, 0, 0, 0, 1});
  wrapper<T, int32_t> expected_w({5, 6, 6, 6, 6, 6, 6}, {1, 0, 0, 0, 0, 0, 1});

  auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected_w);
}

TYPED_TEST(CopyTest, CopyIfElseTestLong)
{
  using T = TypeParam;

  // make sure we span at least 2 warps
  int num_els = 64;

  bool mask[] = {true, false, true, false, true, true, true,  true,  true,  true,  true, true, true,
                 true, true,  true, true,  true, true, false, false, false, false, true, true, true,
                 true, true,  true, true,  true, true, false, false, false, false, true, true, true,
                 true, true,  true, true,  true, true, true,  true,  true,  true,  true, true, true,
                 true, true,  true, true,  true, true, true,  true,  true,  true,  true, true};
  cudf::test::fixed_width_column_wrapper<bool> mask_w(mask, mask + num_els);

  bool lhs_v[] = {true, true, true, true, false, false, true, true, true, true, true, true, true,
                  true, true, true, true, true,  true,  true, true, true, true, true, true, true,
                  true, true, true, true, true,  true,  true, true, true, true, true, true, true,
                  true, true, true, true, true,  true,  true, true, true, true, true, true, true,
                  true, true, true, true, true,  true,  true, true, true, true, true, true};
  wrapper<T, int32_t> lhs_w({5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                             5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                             5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
                            lhs_v);

  bool rhs_v[] = {true, true, true, true, true, true, false, false, true, true, true, true, true,
                  true, true, true, true, true, true, true,  true,  true, true, true, true, true,
                  true, true, true, true, true, true, true,  true,  true, true, true, true, true,
                  true, true, true, true, true, true, true,  true,  true, true, true, true, true,
                  true, true, true, true, true, true, true,  true,  true, true, true, true};
  wrapper<T, int32_t> rhs_w({6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                             6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                             6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6},
                            rhs_v);

  bool exp_v[] = {true, true, true, true, false, false, true, true, true, true, true, true, true,
                  true, true, true, true, true,  true,  true, true, true, true, true, true, true,
                  true, true, true, true, true,  true,  true, true, true, true, true, true, true,
                  true, true, true, true, true,  true,  true, true, true, true, true, true, true,
                  true, true, true, true, true,  true,  true, true, true, true, true, true};
  wrapper<T, int32_t> expected_w({5, 6, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6,
                                  6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5,
                                  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
                                 exp_v);

  auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected_w);
}

TYPED_TEST(CopyTest, CopyIfElseTestMultipleBlocks)
{
  using T = TypeParam;

  int num = 1000;  // larger than a single block
  std::vector<int32_t> h_lhs(num, 5);
  std::vector<int32_t> h_rhs(num, 6);
  std::vector<bool> h_mask(num, false);
  std::vector<bool> h_validity(num, true);
  h_validity[0] = 0;

  cudf::test::fixed_width_column_wrapper<T, int32_t> lhs_w(
    h_lhs.begin(), h_lhs.end(), h_validity.begin());
  cudf::test::fixed_width_column_wrapper<T, int32_t> rhs_w(
    h_rhs.begin(), h_rhs.end(), h_validity.begin());
  cudf::test::fixed_width_column_wrapper<bool> mask_w(h_mask.begin(), h_mask.end());

  auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), rhs_w);
}

TYPED_TEST(CopyTest, CopyIfElseTestEmptyInputs)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<bool> mask_w{};

  wrapper<T> lhs_w{};
  wrapper<T> rhs_w{};
  wrapper<T> expected_w{};

  auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected_w);
}

TYPED_TEST(CopyTest, CopyIfElseMixedInputValidity)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<bool> mask_w{1, 0, 1, 1};

  wrapper<T, int32_t> lhs_w({5, 5, 5, 5}, {1, 1, 1, 0});
  wrapper<T, int32_t> rhs_w({6, 6, 6, 6}, {1, 0, 1, 1});
  wrapper<T, int32_t> expected_w({5, 6, 5, 5}, {1, 0, 1, 0});

  auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected_w);
}

TYPED_TEST(CopyTest, CopyIfElseMixedInputValidity2)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<bool> mask_w{1, 0, 1, 1};

  wrapper<T, int32_t> lhs_w({5, 5, 5, 5}, {1, 1, 1, 0});
  wrapper<T, int32_t> rhs_w({6, 6, 6, 6});
  wrapper<T, int32_t> expected_w({5, 6, 5, 5}, {1, 1, 1, 0});

  auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected_w);
}

TYPED_TEST(CopyTest, CopyIfElseMixedInputValidity3)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<bool> mask_w{1, 0, 1, 1};

  wrapper<T, int32_t> lhs_w({5, 5, 5, 5});
  wrapper<T, int32_t> rhs_w({6, 6, 6, 6}, {1, 0, 1, 1});
  wrapper<T, int32_t> expected_w({5, 6, 5, 5}, {1, 0, 1, 1});

  auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected_w);
}

TYPED_TEST(CopyTest, CopyIfElseMixedInputValidity4)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<bool> mask_w{1, 0, 1, 1};

  wrapper<T, int32_t> lhs_w({5, 5, 5, 5});
  wrapper<T, int32_t> rhs_w({6, 6, 6, 6});
  wrapper<T, int32_t> expected_w({5, 6, 5, 5});

  auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected_w);
}

TYPED_TEST(CopyTest, CopyIfElseBadInputLength)
{
  using T = TypeParam;

  // mask length mismatch
  {
    cudf::test::fixed_width_column_wrapper<bool> mask_w{1, 1, 1};

    wrapper<T, int32_t> lhs_w({5, 5, 5, 5});
    wrapper<T, int32_t> rhs_w({6, 6, 6, 6});

    EXPECT_THROW(cudf::copy_if_else(lhs_w, rhs_w, mask_w), cudf::logic_error);
  }

  // column length mismatch
  {
    cudf::test::fixed_width_column_wrapper<bool> mask_w{1, 1, 1, 1};

    wrapper<T, int32_t> lhs_w({5, 5, 5});
    wrapper<T, int32_t> rhs_w({6, 6, 6, 6});

    EXPECT_THROW(cudf::copy_if_else(lhs_w, rhs_w, mask_w), cudf::logic_error);
  }
}

struct CopyEmptyNested : public cudf::test::BaseFixture {
};

TEST_F(CopyEmptyNested, CopyIfElseTestEmptyNestedColumns)
{
  // lists
  {
    cudf::test::lists_column_wrapper<cudf::string_view> col{{{"abc", "def"}, {"xyz"}}};
    auto lhs = cudf::empty_like(col);
    auto rhs = cudf::empty_like(col);
    cudf::test::fixed_width_column_wrapper<bool> mask{};

    auto expected = empty_like(col);

    auto out = cudf::copy_if_else(*lhs, *rhs, mask);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), *expected);
  }

  // structs
  {
    cudf::test::lists_column_wrapper<cudf::string_view> _col0{{{"abc", "def"}, {"xyz"}}};
    auto col0 = cudf::empty_like(_col0);
    cudf::test::fixed_width_column_wrapper<int> col1;

    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(col0));
    cols.push_back(col1.release());
    cudf::test::structs_column_wrapper struct_col(std::move(cols));
    auto lhs = cudf::empty_like(struct_col);
    auto rhs = cudf::empty_like(struct_col);

    cudf::test::fixed_width_column_wrapper<bool> mask{};

    auto expected = cudf::empty_like(struct_col);

    auto out = cudf::copy_if_else(*lhs, *rhs, mask);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), *expected);
  }
}

TEST_F(CopyEmptyNested, CopyIfElseTestEmptyNestedScalars)
{
  // lists
  {
    cudf::test::lists_column_wrapper<cudf::string_view> _col{{{"abc", "def"}, {"xyz"}}};
    std::unique_ptr<cudf::scalar> lhs = cudf::get_element(_col, 0);
    std::unique_ptr<cudf::scalar> rhs = cudf::get_element(_col, 0);

    cudf::test::fixed_width_column_wrapper<bool> mask{};

    auto expected = empty_like(_col);

    auto out = cudf::copy_if_else(*lhs, *rhs, mask);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), *expected);
  }

  // structs
  {
    cudf::test::lists_column_wrapper<cudf::string_view> col0{{{"abc", "def"}, {"xyz"}}};
    cudf::test::fixed_width_column_wrapper<int> col1{1};

    cudf::table_view tbl({col0, col1});
    cudf::struct_scalar lhs(tbl);
    cudf::struct_scalar rhs(tbl);

    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(col0.release());
    cols.push_back(col1.release());
    cudf::test::structs_column_wrapper struct_col(std::move(cols));

    cudf::test::fixed_width_column_wrapper<bool> mask{};

    auto expected = cudf::empty_like(struct_col);

    auto out = cudf::copy_if_else(lhs, rhs, mask);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), *expected);
  }
}

template <typename T>
struct CopyTestNumeric : public cudf::test::BaseFixture {
};
TYPED_TEST_SUITE(CopyTestNumeric, cudf::test::NumericTypes);

TYPED_TEST(CopyTestNumeric, CopyIfElseTestScalarColumn)
{
  using T = TypeParam;

  int num_els = 4;

  bool mask[] = {true, false, false, true};
  cudf::test::fixed_width_column_wrapper<bool> mask_w(mask, mask + num_els);

  cudf::numeric_scalar<T> lhs_w(5);

  const auto rhs = cudf::test::make_type_param_vector<T>({6, 6, 6, 6});
  bool rhs_v[]   = {true, false, true, true};
  wrapper<T> rhs_w(rhs.begin(), rhs.end(), rhs_v);

  const auto expected = cudf::test::make_type_param_vector<T>({5, 6, 6, 5});
  wrapper<T> expected_w(expected.begin(), expected.end(), rhs_v);

  auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected_w);
}

TYPED_TEST(CopyTestNumeric, CopyIfElseTestColumnScalar)
{
  using T = TypeParam;

  int num_els = 4;

  bool mask[]   = {true, false, false, true};
  bool mask_v[] = {true, true, true, false};
  cudf::test::fixed_width_column_wrapper<bool> mask_w(mask, mask + num_els, mask_v);

  const auto lhs = cudf::test::make_type_param_vector<T>({5, 5, 5, 5});
  bool lhs_v[]   = {false, true, true, true};
  wrapper<T> lhs_w(lhs.begin(), lhs.end(), lhs_v);

  cudf::numeric_scalar<T> rhs_w(6);

  const auto expected = cudf::test::make_type_param_vector<T>({5, 6, 6, 6});
  wrapper<T> expected_w(expected.begin(), expected.end(), lhs_v);

  auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected_w);
}

TYPED_TEST(CopyTestNumeric, CopyIfElseTestScalarScalar)
{
  using T = TypeParam;

  int num_els = 4;

  bool mask[] = {true, false, false, true};
  cudf::test::fixed_width_column_wrapper<bool> mask_w(mask, mask + num_els);

  cudf::numeric_scalar<T> lhs_w(5);
  cudf::numeric_scalar<T> rhs_w(6, false);

  const auto expected = cudf::test::make_type_param_vector<T>({5, 6, 6, 5});
  wrapper<T> expected_w(expected.begin(), expected.end(), mask);

  auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected_w);
}

template <typename T>
struct create_chrono_scalar {
  template <typename ChronoT = T, typename... Args>
  std::enable_if_t<std::is_same_v<typename cudf::is_timestamp_t<ChronoT>::type, std::true_type>,
                   cudf::timestamp_scalar<ChronoT>>
  operator()(Args&&... args) const
  {
    return cudf::timestamp_scalar<T>(std::forward<Args>(args)...);
  }

  template <typename ChronoT = T, typename... Args>
  std::enable_if_t<std::is_same_v<typename cudf::is_duration_t<ChronoT>::type, std::true_type>,
                   cudf::duration_scalar<ChronoT>>
  operator()(Args&&... args) const
  {
    return cudf::duration_scalar<T>(std::forward<Args>(args)...);
  }
};

template <typename T>
struct CopyTestChrono : public cudf::test::BaseFixture {
};
TYPED_TEST_SUITE(CopyTestChrono, cudf::test::ChronoTypes);

TYPED_TEST(CopyTestChrono, CopyIfElseTestScalarColumn)
{
  using T = TypeParam;

  int num_els = 4;

  bool mask[] = {true, false, false, true};
  cudf::test::fixed_width_column_wrapper<bool> mask_w(mask, mask + num_els);

  auto lhs_w = create_chrono_scalar<T>{}(cudf::test::make_type_param_scalar<T>(5), true);

  bool rhs_v[] = {true, false, true, true};
  wrapper<T, int32_t> rhs_w({6, 6, 6, 6}, rhs_v);

  wrapper<T, int32_t> expected_w({5, 6, 6, 5}, rhs_v);

  auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected_w);
}

TYPED_TEST(CopyTestChrono, CopyIfElseTestColumnScalar)
{
  using T = TypeParam;

  int num_els = 4;

  bool mask[] = {true, false, false, true};
  cudf::test::fixed_width_column_wrapper<bool> mask_w(mask, mask + num_els);

  bool lhs_v[] = {false, true, true, true};
  wrapper<T, int32_t> lhs_w({5, 5, 5, 5}, lhs_v);

  auto rhs_w = create_chrono_scalar<T>{}(cudf::test::make_type_param_scalar<T>(6), true);

  wrapper<T, int32_t> expected_w({5, 6, 6, 5}, lhs_v);

  auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected_w);
}

TYPED_TEST(CopyTestChrono, CopyIfElseTestScalarScalar)
{
  using T = TypeParam;

  int num_els = 4;

  bool mask[] = {true, false, false, true};
  cudf::test::fixed_width_column_wrapper<bool> mask_w(mask, mask + num_els);

  auto lhs_w = create_chrono_scalar<T>{}(cudf::test::make_type_param_scalar<T>(5), true);
  auto rhs_w = create_chrono_scalar<T>{}(cudf::test::make_type_param_scalar<T>(6), false);

  wrapper<T, int32_t> expected_w({5, 6, 6, 5}, mask);

  auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected_w);
}

struct CopyTestUntyped : public cudf::test::BaseFixture {
};

TEST_F(CopyTestUntyped, CopyIfElseTypeMismatch)
{
  cudf::test::fixed_width_column_wrapper<bool> mask_w{1, 1, 1, 1};

  wrapper<float> lhs_w{5, 5, 5, 5};
  wrapper<int32_t> rhs_w{6, 6, 6, 6};

  EXPECT_THROW(cudf::copy_if_else(lhs_w, rhs_w, mask_w), cudf::logic_error);
}

struct StringsCopyIfElseTest : public cudf::test::BaseFixture {
};

TEST_F(StringsCopyIfElseTest, CopyIfElse)
{
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });

  std::vector<const char*> h_strings1{"eee", "bb", "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings1(h_strings1.begin(), h_strings1.end(), valids);
  std::vector<const char*> h_strings2{"zz", "", "yyy", "w", "ééé", "ooo"};
  cudf::test::strings_column_wrapper strings2(h_strings2.begin(), h_strings2.end(), valids);

  bool mask[]   = {true, true, false, true, false, true};
  bool mask_v[] = {true, true, true, true, true, false};
  cudf::test::fixed_width_column_wrapper<bool> mask_w(mask, mask + 6, mask_v);

  auto results = cudf::copy_if_else(strings1, strings2, mask_w);

  std::vector<const char*> h_expected;
  for (cudf::size_type idx = 0; idx < static_cast<cudf::size_type>(h_strings1.size()); ++idx) {
    if (mask[idx] and mask_v[idx])
      h_expected.push_back(h_strings1[idx]);
    else
      h_expected.push_back(h_strings2[idx]);
  }
  cudf::test::strings_column_wrapper expected(h_expected.begin(), h_expected.end(), valids);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCopyIfElseTest, CopyIfElseScalarColumn)
{
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });

  std::vector<const char*> h_string1{"eee"};
  cudf::string_scalar strings1{h_string1[0]};
  std::vector<const char*> h_strings2{"zz", "", "yyy", "w", "ééé", "ooo"};
  cudf::test::strings_column_wrapper strings2(h_strings2.begin(), h_strings2.end(), valids);

  bool mask[]   = {true, false, true, false, true, false};
  bool mask_v[] = {true, true, true, true, true, false};
  cudf::test::fixed_width_column_wrapper<bool> mask_w(mask, mask + 6, mask_v);

  auto results = cudf::copy_if_else(strings1, strings2, mask_w);

  std::vector<const char*> h_expected;
  for (cudf::size_type idx = 0; idx < static_cast<cudf::size_type>(h_strings2.size()); ++idx) {
    if (mask[idx] and mask_v[idx]) {
      h_expected.push_back(h_string1[0]);
    } else {
      h_expected.push_back(h_strings2[idx]);
    }
  }
  cudf::test::strings_column_wrapper expected(h_expected.begin(), h_expected.end(), valids);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCopyIfElseTest, CopyIfElseColumnScalar)
{
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });

  std::vector<const char*> h_string1{"eee"};
  cudf::string_scalar strings1{h_string1[0]};
  std::vector<const char*> h_strings2{"zz", "", "yyy", "w", "ééé", "ooo"};
  cudf::test::strings_column_wrapper strings2(h_strings2.begin(), h_strings2.end(), valids);

  bool mask[] = {false, true, true, true, false, true};
  cudf::test::fixed_width_column_wrapper<bool> mask_w(mask, mask + 6);

  auto results = cudf::copy_if_else(strings2, strings1, mask_w);

  std::vector<const char*> h_expected;
  for (cudf::size_type idx = 0; idx < static_cast<cudf::size_type>(h_strings2.size()); ++idx) {
    if (mask[idx]) {
      h_expected.push_back(h_strings2[idx]);
    } else {
      h_expected.push_back(h_string1[0]);
    }
  }
  cudf::test::strings_column_wrapper expected(h_expected.begin(), h_expected.end(), valids);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCopyIfElseTest, CopyIfElseScalarScalar)
{
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });

  std::vector<const char*> h_string1{"eee"};
  cudf::string_scalar string1{h_string1[0]};
  std::vector<const char*> h_string2{"aaa"};
  cudf::string_scalar string2{h_string2[0], false};

  constexpr cudf::size_type mask_size = 6;
  bool mask[]                         = {true, false, true, false, true, false};
  cudf::test::fixed_width_column_wrapper<bool> mask_w(mask, mask + mask_size);

  auto results = cudf::copy_if_else(string1, string2, mask_w);

  std::vector<const char*> h_expected;
  for (bool idx : mask) {
    if (idx) {
      h_expected.push_back(h_string1[0]);
    } else {
      h_expected.push_back(h_string2[0]);
    }
  }
  cudf::test::strings_column_wrapper expected(h_expected.begin(), h_expected.end(), valids);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

template <typename T>
struct FixedPointTypes : public cudf::test::BaseFixture {
};

TYPED_TEST_SUITE(FixedPointTypes, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTypes, FixedPointSimple)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const mask     = cudf::test::fixed_width_column_wrapper<bool>{0, 1, 1, 1, 0, 0};
  auto const a        = fp_wrapper{{110, 220, 330, 440, 550, 660}, scale_type{-2}};
  auto const b        = fp_wrapper{{0, 0, 0, 0, 0, 0}, scale_type{-2}};
  auto const expected = fp_wrapper{{0, 220, 330, 440, 0, 0}, scale_type{-2}};
  auto const result   = cudf::copy_if_else(a, b, mask);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTypes, FixedPointLarge)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto a = thrust::make_counting_iterator(-1000);
  auto b = thrust::make_constant_iterator(0);
  auto m = cudf::detail::make_counting_transform_iterator(-1000, [](int i) { return i > 0; });
  auto e =
    cudf::detail::make_counting_transform_iterator(-1000, [](int i) { return std::max(0, i); });

  auto const mask     = cudf::test::fixed_width_column_wrapper<bool>(m, m + 2000);
  auto const A        = fp_wrapper{a, a + 2000, scale_type{-3}};
  auto const B        = fp_wrapper{b, b + 2000, scale_type{-3}};
  auto const expected = fp_wrapper{e, e + 2000, scale_type{-3}};
  auto const result   = cudf::copy_if_else(A, B, mask);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTypes, FixedPointScaleMismatch)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const mask = cudf::test::fixed_width_column_wrapper<bool>{0, 1, 1, 1, 0, 0};
  auto const a    = fp_wrapper{{110, 220, 330, 440, 550, 660}, scale_type{-2}};
  auto const b    = fp_wrapper{{0, 0, 0, 0, 0, 0}, scale_type{-1}};

  EXPECT_THROW(cudf::copy_if_else(a, b, mask), cudf::logic_error);
}

struct DictionaryCopyIfElseTest : public cudf::test::BaseFixture {
};

TEST_F(DictionaryCopyIfElseTest, ColumnColumn)
{
  auto valids = cudf::test::iterators::null_at(2);
  std::vector<const char*> h_strings1{"eee", "bb", "", "aa", "bb", "ééé"};
  cudf::test::dictionary_column_wrapper<std::string> input1(
    h_strings1.begin(), h_strings1.end(), valids);
  std::vector<const char*> h_strings2{"zz", "bb", "", "aa", "ééé", "ooo"};
  cudf::test::dictionary_column_wrapper<std::string> input2(
    h_strings2.begin(), h_strings2.end(), valids);

  bool mask[]   = {true, true, false, true, false, true};
  bool mask_v[] = {true, true, true, true, true, false};
  cudf::test::fixed_width_column_wrapper<bool> mask_w(mask, mask + 6, mask_v);

  auto results = cudf::copy_if_else(input1, input2, mask_w);
  auto decoded = cudf::dictionary::decode(cudf::dictionary_column_view(results->view()));

  std::vector<const char*> h_expected;
  for (cudf::size_type idx = 0; idx < static_cast<cudf::size_type>(h_strings1.size()); ++idx) {
    if (mask[idx] and mask_v[idx])
      h_expected.push_back(h_strings1[idx]);
    else
      h_expected.push_back(h_strings2[idx]);
  }
  cudf::test::strings_column_wrapper expected(h_expected.begin(), h_expected.end(), valids);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(decoded->view(), expected);
}

TEST_F(DictionaryCopyIfElseTest, ColumnScalar)
{
  std::string h_string{"eee"};
  cudf::string_scalar input1{h_string};
  std::vector<const char*> h_strings{"zz", "", "yyy", "w", "ééé", "ooo"};
  auto valids = cudf::test::iterators::null_at(1);
  cudf::test::dictionary_column_wrapper<std::string> input2(
    h_strings.begin(), h_strings.end(), valids);

  bool mask[] = {false, true, true, true, false, true};
  cudf::test::fixed_width_column_wrapper<bool> mask_w(mask, mask + 6);

  auto results = cudf::copy_if_else(input2, input1, mask_w);
  auto decoded = cudf::dictionary::decode(cudf::dictionary_column_view(results->view()));

  std::vector<const char*> h_expected1;
  std::vector<const char*> h_expected2;
  for (cudf::size_type idx = 0; idx < static_cast<cudf::size_type>(h_strings.size()); ++idx) {
    if (mask[idx]) {
      h_expected1.push_back(h_strings[idx]);
      h_expected2.push_back(h_string.c_str());
    } else {
      h_expected1.push_back(h_string.c_str());
      h_expected2.push_back(h_strings[idx]);
    }
  }

  cudf::test::strings_column_wrapper expected1(h_expected1.begin(), h_expected1.end(), valids);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(decoded->view(), expected1);

  results = cudf::copy_if_else(input1, input2, mask_w);
  decoded = cudf::dictionary::decode(cudf::dictionary_column_view(results->view()));

  cudf::test::strings_column_wrapper expected2(h_expected2.begin(), h_expected2.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(decoded->view(), expected2);
}

TEST_F(DictionaryCopyIfElseTest, TypeMismatch)
{
  cudf::test::dictionary_column_wrapper<int32_t> input1({1, 1, 1, 1});
  cudf::test::dictionary_column_wrapper<double> input2({1.0, 1.0, 1.0, 1.0});
  cudf::test::fixed_width_column_wrapper<bool> mask({1, 0, 0, 1});

  EXPECT_THROW(cudf::copy_if_else(input1, input2, mask), cudf::logic_error);

  cudf::string_scalar input3{"1"};
  EXPECT_THROW(cudf::copy_if_else(input1, input3, mask), cudf::logic_error);
  EXPECT_THROW(cudf::copy_if_else(input3, input2, mask), cudf::logic_error);
  EXPECT_THROW(cudf::copy_if_else(input2, input3, mask), cudf::logic_error);
}
