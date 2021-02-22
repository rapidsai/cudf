/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/detail/copy_if_else.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/scalar/scalar.hpp>

#include <rmm/cuda_stream_view.hpp>

template <typename T>
struct CopyTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(CopyTest, cudf::test::FixedWidthTypesWithoutFixedPoint);

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

struct copy_if_else_tiny_grid_functor {
  template <typename T, typename Filter, std::enable_if_t<cudf::is_fixed_width<T>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& lhs,
                                           cudf::column_view const& rhs,
                                           Filter filter,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
  {
    // output
    std::unique_ptr<cudf::column> out =
      cudf::allocate_like(lhs, lhs.size(), cudf::mask_allocation_policy::RETAIN, mr);

    // device views
    auto lhs_view = cudf::column_device_view::create(lhs);
    auto rhs_view = cudf::column_device_view::create(rhs);
    auto lhs_iter = cudf::detail::make_pair_iterator<T>(*lhs_view);
    auto rhs_iter = cudf::detail::make_pair_iterator<T>(*rhs_view);
    auto out_dv   = cudf::mutable_column_device_view::create(*out);

    // call the kernel with an artificially small grid
    cudf::detail::copy_if_else_kernel<32, T, decltype(lhs_iter), decltype(rhs_iter), Filter, false>
      <<<1, 32, 0, stream.value()>>>(lhs_iter, rhs_iter, filter, *out_dv, nullptr);

    return out;
  }

  template <typename T, typename Filter, std::enable_if_t<not cudf::is_fixed_width<T>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& lhs,
                                           cudf::column_view const& rhs,
                                           Filter filter,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
  {
    CUDF_FAIL("Unexpected test execution");
  }
};

std::unique_ptr<cudf::column> tiny_grid_launch(cudf::column_view const& lhs,
                                               cudf::column_view const& rhs,
                                               cudf::column_view const& boolean_mask)
{
  auto bool_mask_device_p                   = cudf::column_device_view::create(boolean_mask);
  cudf::column_device_view bool_mask_device = *bool_mask_device_p;
  auto filter                               = [bool_mask_device] __device__(cudf::size_type i) {
    return bool_mask_device.element<bool>(i);
  };
  return cudf::type_dispatcher(lhs.type(),
                               copy_if_else_tiny_grid_functor{},
                               lhs,
                               rhs,
                               filter,
                               rmm::cuda_stream_default,
                               rmm::mr::get_current_device_resource());
}

TYPED_TEST(CopyTest, CopyIfElseTestTinyGrid)
{
  using T = TypeParam;

  // make sure we span at least 2 warps
  int num_els = 64;

  bool mask[] = {1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  cudf::test::fixed_width_column_wrapper<bool> mask_w(mask, mask + num_els);

  wrapper<T, int32_t> lhs_w({5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                             5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                             5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5});

  wrapper<T, int32_t> rhs_w({6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                             6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                             6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6});

  wrapper<T, int32_t> expected_w({5, 6, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6,
                                  6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5,
                                  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5});

  auto out = tiny_grid_launch(lhs_w, rhs_w, mask_w);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected_w);
}

TYPED_TEST(CopyTest, CopyIfElseTestLong)
{
  using T = TypeParam;

  // make sure we span at least 2 warps
  int num_els = 64;

  bool mask[] = {1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  cudf::test::fixed_width_column_wrapper<bool> mask_w(mask, mask + num_els);

  bool lhs_v[] = {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  wrapper<T, int32_t> lhs_w({5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                             5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                             5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
                            lhs_v);

  bool rhs_v[] = {1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  wrapper<T, int32_t> rhs_w({6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                             6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                             6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6},
                            rhs_v);

  bool exp_v[] = {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  wrapper<T, int32_t> expected_w({5, 6, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6,
                                  6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5,
                                  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
                                 exp_v);

  auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected_w);
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

template <typename T>
struct CopyTestNumeric : public cudf::test::BaseFixture {
};
TYPED_TEST_CASE(CopyTestNumeric, cudf::test::NumericTypes);

TYPED_TEST(CopyTestNumeric, CopyIfElseTestScalarColumn)
{
  using T = TypeParam;

  int num_els = 4;

  bool mask[] = {1, 0, 0, 1};
  cudf::test::fixed_width_column_wrapper<bool> mask_w(mask, mask + num_els);

  cudf::numeric_scalar<T> lhs_w(5);

  const auto rhs = cudf::test::make_type_param_vector<T>({6, 6, 6, 6});
  bool rhs_v[]   = {1, 0, 1, 1};
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

  bool mask[]   = {1, 0, 0, 1};
  bool mask_v[] = {1, 1, 1, 0};
  cudf::test::fixed_width_column_wrapper<bool> mask_w(mask, mask + num_els, mask_v);

  const auto lhs = cudf::test::make_type_param_vector<T>({5, 5, 5, 5});
  bool lhs_v[]   = {0, 1, 1, 1};
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

  bool mask[] = {1, 0, 0, 1};
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
  typename std::enable_if_t<
    std::is_same<typename cudf::is_timestamp_t<ChronoT>::type, std::true_type>::value,
    cudf::timestamp_scalar<ChronoT>>
  operator()(Args&&... args) const
  {
    return cudf::timestamp_scalar<T>(std::forward<Args>(args)...);
  }

  template <typename ChronoT = T, typename... Args>
  typename std::enable_if_t<
    std::is_same<typename cudf::is_duration_t<ChronoT>::type, std::true_type>::value,
    cudf::duration_scalar<ChronoT>>
  operator()(Args&&... args) const
  {
    return cudf::duration_scalar<T>(std::forward<Args>(args)...);
  }
};

template <typename T>
struct CopyTestChrono : public cudf::test::BaseFixture {
};
TYPED_TEST_CASE(CopyTestChrono, cudf::test::ChronoTypes);

TYPED_TEST(CopyTestChrono, CopyIfElseTestScalarColumn)
{
  using T = TypeParam;

  int num_els = 4;

  bool mask[] = {1, 0, 0, 1};
  cudf::test::fixed_width_column_wrapper<bool> mask_w(mask, mask + num_els);

  auto lhs_w = create_chrono_scalar<T>{}(cudf::test::make_type_param_scalar<T>(5), true);

  bool rhs_v[] = {1, 0, 1, 1};
  wrapper<T, int32_t> rhs_w({6, 6, 6, 6}, rhs_v);

  wrapper<T, int32_t> expected_w({5, 6, 6, 5}, rhs_v);

  auto out = cudf::copy_if_else(lhs_w, rhs_w, mask_w);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected_w);
}

TYPED_TEST(CopyTestChrono, CopyIfElseTestColumnScalar)
{
  using T = TypeParam;

  int num_els = 4;

  bool mask[] = {1, 0, 0, 1};
  cudf::test::fixed_width_column_wrapper<bool> mask_w(mask, mask + num_els);

  bool lhs_v[] = {0, 1, 1, 1};
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

  bool mask[] = {1, 0, 0, 1};
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
  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  std::vector<const char*> h_strings1{"eee", "bb", "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings1(h_strings1.begin(), h_strings1.end(), valids);
  std::vector<const char*> h_strings2{"zz", "", "yyy", "w", "ééé", "ooo"};
  cudf::test::strings_column_wrapper strings2(h_strings2.begin(), h_strings2.end(), valids);

  bool mask[]   = {1, 1, 0, 1, 0, 1};
  bool mask_v[] = {1, 1, 1, 1, 1, 0};
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
  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  std::vector<const char*> h_string1{"eee"};
  cudf::string_scalar strings1{h_string1[0]};
  std::vector<const char*> h_strings2{"zz", "", "yyy", "w", "ééé", "ooo"};
  cudf::test::strings_column_wrapper strings2(h_strings2.begin(), h_strings2.end(), valids);

  bool mask[]   = {1, 0, 1, 0, 1, 0};
  bool mask_v[] = {1, 1, 1, 1, 1, 0};
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
  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  std::vector<const char*> h_string1{"eee"};
  cudf::string_scalar strings1{h_string1[0]};
  std::vector<const char*> h_strings2{"zz", "", "yyy", "w", "ééé", "ooo"};
  cudf::test::strings_column_wrapper strings2(h_strings2.begin(), h_strings2.end(), valids);

  bool mask[] = {0, 1, 1, 1, 0, 1};
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
  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  std::vector<const char*> h_string1{"eee"};
  cudf::string_scalar string1{h_string1[0]};
  std::vector<const char*> h_string2{"aaa"};
  cudf::string_scalar string2{h_string2[0], false};

  constexpr cudf::size_type mask_size = 6;
  bool mask[]                         = {1, 0, 1, 0, 1, 0};
  cudf::test::fixed_width_column_wrapper<bool> mask_w(mask, mask + mask_size);

  auto results = cudf::copy_if_else(string1, string2, mask_w);

  std::vector<const char*> h_expected;
  for (cudf::size_type idx = 0; idx < static_cast<cudf::size_type>(mask_size); ++idx) {
    if (mask[idx]) {
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

TYPED_TEST_CASE(FixedPointTypes, cudf::test::FixedPointTypes);

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
  using namespace cudf::test;
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
