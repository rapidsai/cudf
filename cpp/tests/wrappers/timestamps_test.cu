/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/legacy/binaryop.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/timestamp_utilities.cuh>
#include <tests/utilities/type_lists.hpp>
#include "cudf/column/column_view.hpp"
#include "cudf/types.hpp"
#include "tests/utilities/column_utilities.hpp"
#include "tests/utilities/column_wrapper.hpp"

#include <gmock/gmock.h>

template <typename T>
struct TimestampColumnTest : public cudf::test::BaseFixture {
  cudaStream_t stream() { return cudaStream_t(0); }
  cudf::size_type size() { return cudf::size_type(100); }
  cudf::data_type type() {
    return cudf::data_type{cudf::experimental::type_to_id<T>()};
  }
};

template <typename Timestamp>
struct compare_timestamp_elements_to_primitive_representation {
  cudf::column_device_view primitives;
  cudf::column_device_view timestamps;

  compare_timestamp_elements_to_primitive_representation(
      cudf::column_device_view& _primitives,
      cudf::column_device_view& _timestamps)
      : primitives(_primitives), timestamps(_timestamps) {}

  __host__ __device__ bool operator()(const int32_t element_index) {
    using Primitive = typename Timestamp::rep;
    auto primitive = primitives.element<Primitive>(element_index);
    auto timestamp = timestamps.element<Timestamp>(element_index);
    return primitive == timestamp.time_since_epoch().count();
  }
};

TYPED_TEST_CASE(TimestampColumnTest, cudf::test::TimestampTypes);

TYPED_TEST(TimestampColumnTest,
           TimestampDurationsMatchPrimitiveRepresentation) {
  using T = TypeParam;
  using Rep = typename T::rep;
  using namespace cudf::test;
  using namespace simt::std::chrono;

  auto start = milliseconds(-2500000000000);  // Sat, 11 Oct 1890 19:33:20 GMT
  auto stop_ = milliseconds(2500000000000);   // Mon, 22 Mar 2049 04:26:40 GMT
  auto timestamp_col = generate_timestamps<T>(
      this->size(), time_point_ms(start), time_point_ms(stop_));

  // rount-trip through the host to copy `timestamp_col` values
  // to a new fixed_width_column_wrapper `primitive_col`
  std::vector<Rep> timestamp_col_data;
  std::vector<cudf::bitmask_type> timestamp_col_mask;
  std::tie(timestamp_col_data, timestamp_col_mask) = to_host<Rep>(timestamp_col);

  auto primitive_col = fixed_width_column_wrapper<Rep>(
      timestamp_col_data.begin(), timestamp_col_data.end());

  thrust::device_vector<int32_t> indices(this->size());
  thrust::sequence(indices.begin(), indices.end());
  EXPECT_TRUE(
      thrust::all_of(indices.begin(), indices.end(),
                     compare_timestamp_elements_to_primitive_representation<T>{
                         *cudf::column_device_view::create(primitive_col),
                         *cudf::column_device_view::create(timestamp_col)}));
}

template <typename Timestamp>
struct compare_timestamp_elements {
  gdf_binary_operator comp;
  cudf::column_device_view lhs;
  cudf::column_device_view rhs;

  compare_timestamp_elements(gdf_binary_operator _comp,
                             cudf::column_device_view& _lhs,
                             cudf::column_device_view& _rhs)
      : comp(_comp), lhs(_lhs), rhs(_rhs) {}

  __host__ __device__ bool operator()(const int32_t element_index) {
    auto lhs_elt = lhs.element<Timestamp>(element_index);
    auto rhs_elt = rhs.element<Timestamp>(element_index);
    switch (comp) {
      case GDF_LESS:
        return lhs_elt < rhs_elt;
      case GDF_GREATER:
        return lhs_elt > rhs_elt;
      case GDF_LESS_EQUAL:
        return lhs_elt <= rhs_elt;
      case GDF_GREATER_EQUAL:
        return lhs_elt >= rhs_elt;
      default:
        return false;
    }
  }
};

TYPED_TEST(TimestampColumnTest, TimestampsCanBeComparedInDeviceCode) {
  using T = TypeParam;
  using namespace cudf::test;
  using namespace simt::std::chrono;

  auto start_lhs = milliseconds(-2500000000000);  // Sat, 11 Oct 1890 19:33:20 GMT
  auto start_rhs = milliseconds(-2400000000000);  // Tue, 12 Dec 1893 05:20:00 GMT
  auto stop_lhs_ = milliseconds(2500000000000);  // Mon, 22 Mar 2049 04:26:40 GMT
  auto stop_rhs_ = milliseconds(2600000000000);  // Wed, 22 May 2052 14:13:20 GMT

  auto timestamp_lhs_col = generate_timestamps<T>(
      this->size(), time_point_ms(start_lhs), time_point_ms(stop_lhs_));

  auto timestamp_rhs_col = generate_timestamps<T>(
      this->size(), time_point_ms(start_rhs), time_point_ms(stop_rhs_));

  thrust::device_vector<int32_t> indices(this->size());
  thrust::sequence(indices.begin(), indices.end());

  EXPECT_TRUE(thrust::all_of(
      indices.begin(), indices.end(),
      compare_timestamp_elements<TypeParam>{
          GDF_LESS,
          *cudf::column_device_view::create(timestamp_lhs_col),
          *cudf::column_device_view::create(timestamp_rhs_col)}));

  EXPECT_TRUE(thrust::all_of(
      indices.begin(), indices.end(),
      compare_timestamp_elements<TypeParam>{
          GDF_GREATER,
          *cudf::column_device_view::create(timestamp_rhs_col),
          *cudf::column_device_view::create(timestamp_lhs_col)}));

  EXPECT_TRUE(thrust::all_of(
      indices.begin(), indices.end(),
      compare_timestamp_elements<TypeParam>{
          GDF_LESS_EQUAL,
          *cudf::column_device_view::create(timestamp_lhs_col),
          *cudf::column_device_view::create(timestamp_lhs_col)}));

  EXPECT_TRUE(thrust::all_of(
      indices.begin(), indices.end(),
      compare_timestamp_elements<TypeParam>{
          GDF_GREATER_EQUAL,
          *cudf::column_device_view::create(timestamp_rhs_col),
          *cudf::column_device_view::create(timestamp_rhs_col)}));
}

TYPED_TEST(TimestampColumnTest, TimestampFactoryNullMaskAsParm) {
  rmm::device_buffer null_mask{
      create_null_mask(this->size(), cudf::mask_state::ALL_NULL)};
  auto column = cudf::make_timestamp_column(
      cudf::data_type{cudf::experimental::type_to_id<TypeParam>()},
      this->size(), null_mask, this->size(), this->stream(), this->mr());
  EXPECT_EQ(column->type(),
            cudf::data_type{cudf::experimental::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
  EXPECT_EQ(this->size(), column->null_count());
  EXPECT_TRUE(column->nullable());
  EXPECT_TRUE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(TimestampColumnTest, TimestampFactoryNullMaskAsEmptyParm) {
  rmm::device_buffer null_mask{};
  auto column = cudf::make_timestamp_column(
      cudf::data_type{cudf::experimental::type_to_id<TypeParam>()},
      this->size(), null_mask, 0, this->stream(), this->mr());
  EXPECT_EQ(column->type(),
            cudf::data_type{cudf::experimental::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
  EXPECT_EQ(0, column->null_count());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}
