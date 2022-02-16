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

#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <memory>
#include <type_traits>

// fixed_width, dict, string, list, struct
template <typename T, std::enable_if_t<cudf::is_fixed_width<T>()>* = nullptr>
std::unique_ptr<cudf::column> example_column()
{
  auto begin = thrust::make_counting_iterator(1);
  auto end   = thrust::make_counting_iterator(16);
  return cudf::test::fixed_width_column_wrapper<T>(begin, end).release();
}

template <typename T>
struct ColumnViewDeviceSpanTests : public cudf::test::BaseFixture {
};

using DeviceSpanTypes =
  cudf::test::Types<int32_t>;  // cudf::test::FixedWidthTypesWithoutFixedPoint;
TYPED_TEST_SUITE(ColumnViewDeviceSpanTests, DeviceSpanTypes);

TYPED_TEST(ColumnViewDeviceSpanTests, conversion_round_trip)
{
  auto col      = example_column<TypeParam>();
  auto col_view = cudf::column_view{*col};

  // Test implicit conversion, round trip
  cudf::device_span<const TypeParam> device_span_from_col_view = col_view;
  cudf::column_view col_view_from_device_span                  = device_span_from_col_view;
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col_view, col_view_from_device_span);
}
