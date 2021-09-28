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
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/copy_if_else.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

template <typename T>
struct CopyDetailTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(CopyDetailTest, cudf::test::FixedWidthTypesWithoutFixedPoint);

struct copy_if_else_tiny_grid_functor {
  template <typename T, typename Filter, CUDF_ENABLE_IF(cudf::is_rep_layout_compatible<T>())>
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
    auto lhs_iter = cudf::detail::make_optional_iterator<T>(*lhs_view, cudf::contains_nulls::NO{});
    auto rhs_iter = cudf::detail::make_optional_iterator<T>(*rhs_view, cudf::contains_nulls::NO{});
    auto out_dv   = cudf::mutable_column_device_view::create(*out);

    // call the kernel with an artificially small grid
    cudf::detail::copy_if_else_kernel<32, T, decltype(lhs_iter), decltype(rhs_iter), Filter, false>
      <<<1, 32, 0, stream.value()>>>(lhs_iter, rhs_iter, filter, *out_dv, nullptr);

    return out;
  }

  template <typename T, typename Filter, CUDF_ENABLE_IF(not cudf::is_rep_layout_compatible<T>())>
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

TYPED_TEST(CopyDetailTest, CopyIfElseTestTinyGrid)
{
  using T = TypeParam;

  // make sure we span at least 2 warps
  int num_els = 64;

  bool mask[] = {1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  cudf::test::fixed_width_column_wrapper<bool> mask_w(mask, mask + num_els);

  cudf::test::fixed_width_column_wrapper<T, int32_t> lhs_w(
    {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
     5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
     5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5});

  cudf::test::fixed_width_column_wrapper<T, int32_t> rhs_w(
    {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6});

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected_w(
    {5, 6, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6,
     6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5,
     5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5});

  auto out = tiny_grid_launch(lhs_w, rhs_w, mask_w);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected_w);
}
