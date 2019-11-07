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

#include <cudf/legacy/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>

#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/timestamp_utilities.cuh>

#include <simt/chrono>
#include <gmock/gmock.h>

template <typename T>
struct TimestampColumnTest : public cudf::test::BaseFixture {
  cudaStream_t stream() { return cudaStream_t(0); }
  cudf::size_type size() { return cudf::size_type(100); }
  cudf::data_type type() { return cudf::data_type{cudf::experimental::type_to_id<T>()}; }
};

template <typename Timestamp>
struct compare_timestamp_elements_to_primitive_representation {

  cudf::column_device_view primitives;
  cudf::column_device_view timestamps;

  compare_timestamp_elements_to_primitive_representation(
    cudf::column_device_view& _primitives,
    cudf::column_device_view& _timestamps
  ) : primitives(_primitives) , timestamps(_timestamps) {}

  __host__ __device__ bool operator()(const int32_t element_index) {
    using Primitive = typename Timestamp::rep;
    auto primitive = primitives.element<Primitive>(element_index);
    auto timestamp = timestamps.element<Timestamp>(element_index);
    return primitive == timestamp.time_since_epoch().count();
  }
};

TYPED_TEST_CASE(TimestampColumnTest, cudf::test::TimestampTypes);

TYPED_TEST(TimestampColumnTest, TimestampDurationsMatchPrimitiveRepresentation) {

  using namespace cudf::test;
  using namespace simt::std::chrono;
  using Rep = typename TypeParam::rep;
  using Period = typename TypeParam::period;

  auto start = milliseconds(-2500000000000); // Sat, 11 Oct 1890 19:33:20 GMT
  auto stop_ = milliseconds( 2500000000000); // Mon, 22 Mar 2049 04:26:40 GMT
  auto test_timestamps = generate_timestamps<Rep, Period>(this->size(),
                                                          time_point_ms(start),
                                                          time_point_ms(stop_));

  auto primitive_type = cudf::data_type{cudf::experimental::type_to_id<Rep>()};

  auto primitive_col = cudf::make_numeric_column(primitive_type, this->size(),
                                                 cudf::mask_state::ALL_VALID,
                                                 this->stream(), this->mr());

  auto timestamp_col = cudf::make_timestamp_column(this->type(), this->size(),
                                                   cudf::mask_state::ALL_VALID,
                                                   this->stream(), this->mr());

  cudf::mutable_column_view primitive_view = *primitive_col;
  cudf::mutable_column_view timestamp_view = *timestamp_col;

  auto primitive_dev_view = cudf::column_device_view::create(primitive_view);
  auto timestamp_dev_view = cudf::column_device_view::create(timestamp_view);

  CUDA_TRY(cudaMemcpy(timestamp_view.data<Rep>(),
    thrust::raw_pointer_cast(test_timestamps.data()),
    test_timestamps.size() * sizeof(Rep), cudaMemcpyDefault));

  CUDA_TRY(cudaMemcpy(primitive_view.data<Rep>(),
    thrust::raw_pointer_cast(test_timestamps.data()),
    test_timestamps.size() * sizeof(Rep), cudaMemcpyDefault));

  thrust::device_vector<int32_t> indices(this->size());
  thrust::sequence(indices.begin(), indices.end());
  EXPECT_TRUE(thrust::all_of(indices.begin(), indices.end(),
              compare_timestamp_elements_to_primitive_representation<TypeParam>{
                *primitive_dev_view, *timestamp_dev_view
              }));
}

template <typename Timestamp>
struct compare_timestamp_elements {

  gdf_binary_operator comp;
  cudf::column_device_view lhs;
  cudf::column_device_view rhs;

  compare_timestamp_elements(
    gdf_binary_operator _comp,
    cudf::column_device_view& _lhs,
    cudf::column_device_view& _rhs
  ) : comp(_comp), lhs(_lhs) , rhs(_rhs) {}

  __host__ __device__ bool operator()(const int32_t element_index) {
    auto lhs_elt = lhs.element<Timestamp>(element_index);
    auto rhs_elt = rhs.element<Timestamp>(element_index);
    switch (comp) {
      case GDF_LESS: return lhs_elt < rhs_elt;
      case GDF_GREATER: return lhs_elt > rhs_elt;
      case GDF_LESS_EQUAL: return lhs_elt <= rhs_elt;
      case GDF_GREATER_EQUAL: return lhs_elt >= rhs_elt;
      default: return false;
    }
  }
};

TYPED_TEST(TimestampColumnTest, TimestampsCanBeComparedInDeviceCode) {

  using namespace cudf::test;
  using namespace simt::std::chrono;
  using Rep = typename TypeParam::rep;
  using Period = typename TypeParam::period;

  auto start_lhs = milliseconds(-2500000000000); // Sat, 11 Oct 1890 19:33:20 GMT
  auto start_rhs = milliseconds(-2400000000000); // Tue, 12 Dec 1893 05:20:00 GMT
  auto stop_lhs_ = milliseconds( 2500000000000); // Mon, 22 Mar 2049 04:26:40 GMT
  auto stop_rhs_ = milliseconds( 2600000000000); // Wed, 22 May 2052 14:13:20 GMT

  auto test_timestamps_lhs = generate_timestamps<Rep, Period>(this->size(),
                                                              time_point_ms(start_lhs),
                                                              time_point_ms(stop_lhs_));

  auto test_timestamps_rhs = generate_timestamps<Rep, Period>(this->size(),
                                                              time_point_ms(start_rhs),
                                                              time_point_ms(stop_rhs_));

  auto timestamp_lhs_col = cudf::make_timestamp_column(this->type(), this->size(),
                                                       cudf::mask_state::ALL_VALID,
                                                       this->stream(), this->mr());

  auto timestamp_rhs_col = cudf::make_timestamp_column(this->type(), this->size(),
                                                       cudf::mask_state::ALL_VALID,
                                                       this->stream(), this->mr());

  cudf::mutable_column_view timestamp_lhs_view = *timestamp_lhs_col;
  cudf::mutable_column_view timestamp_rhs_view = *timestamp_rhs_col;

  auto timestamp_lhs_dev_view = cudf::column_device_view::create(timestamp_lhs_view);
  auto timestamp_rhs_dev_view = cudf::column_device_view::create(timestamp_rhs_view);

  CUDA_TRY(cudaMemcpy(timestamp_lhs_view.data<Rep>(),
    thrust::raw_pointer_cast(test_timestamps_lhs.data()),
    test_timestamps_lhs.size() * sizeof(Rep), cudaMemcpyDefault));

  CUDA_TRY(cudaMemcpy(timestamp_rhs_view.data<Rep>(),
    thrust::raw_pointer_cast(test_timestamps_rhs.data()),
    test_timestamps_rhs.size() * sizeof(Rep), cudaMemcpyDefault));

  thrust::device_vector<int32_t> indices(this->size());
  thrust::sequence(indices.begin(), indices.end());

  EXPECT_TRUE(thrust::all_of(indices.begin(), indices.end(),
              compare_timestamp_elements<TypeParam>{
                GDF_LESS,
                *timestamp_lhs_dev_view,
                *timestamp_rhs_dev_view
              }));

  EXPECT_TRUE(thrust::all_of(indices.begin(), indices.end(),
              compare_timestamp_elements<TypeParam>{
                GDF_GREATER,
                *timestamp_rhs_dev_view,
                *timestamp_lhs_dev_view
              }));

  EXPECT_TRUE(thrust::all_of(indices.begin(), indices.end(),
              compare_timestamp_elements<TypeParam>{
                GDF_LESS_EQUAL,
                *timestamp_lhs_dev_view,
                *timestamp_lhs_dev_view
              }));

  EXPECT_TRUE(thrust::all_of(indices.begin(), indices.end(),
              compare_timestamp_elements<TypeParam>{
                GDF_GREATER_EQUAL,
                *timestamp_rhs_dev_view,
                *timestamp_rhs_dev_view
              }));

}

TYPED_TEST(TimestampColumnTest, TimestampFactoryNullMaskAsParm) {
  rmm::device_buffer null_mask{
      create_null_mask(this->size(), cudf::mask_state::ALL_NULL)};
  auto column = cudf::make_timestamp_column(
      cudf::data_type{cudf::experimental::type_to_id<TypeParam>()}, this->size(),
      null_mask, this->size(), this->stream(), this->mr());
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
      cudf::data_type{cudf::experimental::type_to_id<TypeParam>()}, this->size(),
      null_mask, 0, this->stream(), this->mr());
  EXPECT_EQ(column->type(),
            cudf::data_type{cudf::experimental::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
  EXPECT_EQ(0, column->null_count());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}
