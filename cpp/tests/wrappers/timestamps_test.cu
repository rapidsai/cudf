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

#include <cudf/binaryop.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/timestamp_utilities.cuh>
#include <cudf_test/type_lists.hpp>

template <typename T>
struct ChronoColumnTest : public cudf::test::BaseFixture {
  cudaStream_t stream() { return cudaStream_t(0); }
  cudf::size_type size() { return cudf::size_type(100); }
  cudf::data_type type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

template <typename ChronoT>
struct compare_chrono_elements_to_primitive_representation {
  cudf::column_device_view primitives;
  cudf::column_device_view chronos;

  compare_chrono_elements_to_primitive_representation(cudf::column_device_view& _primitives,
                                                      cudf::column_device_view& _chronos)
    : primitives(_primitives), chronos(_chronos)
  {
  }

  template <typename T = ChronoT, typename std::enable_if_t<cudf::is_timestamp<T>()>* = nullptr>
  __host__ __device__ bool operator()(const int32_t element_index)
  {
    using Primitive = typename ChronoT::rep;
    auto primitive  = primitives.element<Primitive>(element_index);
    auto timestamp  = chronos.element<ChronoT>(element_index);
    return primitive == timestamp.time_since_epoch().count();
  }

  template <typename T = ChronoT, typename std::enable_if_t<cudf::is_duration<T>()>* = nullptr>
  __host__ __device__ bool operator()(const int32_t element_index)
  {
    using Primitive = typename ChronoT::rep;
    auto primitive  = primitives.element<Primitive>(element_index);
    auto dur        = chronos.element<ChronoT>(element_index);
    return primitive == dur.count();
  }
};

TYPED_TEST_CASE(ChronoColumnTest, cudf::test::ChronoTypes);

TYPED_TEST(ChronoColumnTest, ChronoDurationsMatchPrimitiveRepresentation)
{
  using T   = TypeParam;
  using Rep = typename T::rep;
  using namespace cudf::test;
  using namespace simt::std::chrono;

  auto start = milliseconds(-2500000000000);  // Sat, 11 Oct 1890 19:33:20 GMT
  auto stop_ = milliseconds(2500000000000);   // Mon, 22 Mar 2049 04:26:40 GMT
  auto chrono_col =
    generate_timestamps<T>(this->size(), time_point_ms(start), time_point_ms(stop_));

  // round-trip through the host to copy `chrono_col` values
  // to a new fixed_width_column_wrapper `primitive_col`
  // When C++17, use structured bindings
  thrust::host_vector<Rep> chrono_col_data;
  std::vector<cudf::bitmask_type> chrono_col_mask;
  std::tie(chrono_col_data, chrono_col_mask) = to_host<Rep>(chrono_col);

  auto primitive_col =
    fixed_width_column_wrapper<Rep>(chrono_col_data.begin(), chrono_col_data.end());

  thrust::device_vector<int32_t> indices(this->size());
  thrust::sequence(indices.begin(), indices.end());
  EXPECT_TRUE(thrust::all_of(indices.begin(),
                             indices.end(),
                             compare_chrono_elements_to_primitive_representation<T>{
                               *cudf::column_device_view::create(primitive_col),
                               *cudf::column_device_view::create(chrono_col)}));
}

template <typename ChronoT>
struct compare_chrono_elements {
  cudf::binary_operator comp;
  cudf::column_device_view lhs;
  cudf::column_device_view rhs;

  compare_chrono_elements(cudf::binary_operator _comp,
                          cudf::column_device_view& _lhs,
                          cudf::column_device_view& _rhs)
    : comp(_comp), lhs(_lhs), rhs(_rhs)
  {
  }

  __host__ __device__ bool operator()(const int32_t element_index)
  {
    auto lhs_elt = lhs.element<ChronoT>(element_index);
    auto rhs_elt = rhs.element<ChronoT>(element_index);
    switch (comp) {
      case cudf::binary_operator::LESS: return lhs_elt < rhs_elt;
      case cudf::binary_operator::GREATER: return lhs_elt > rhs_elt;
      case cudf::binary_operator::LESS_EQUAL: return lhs_elt <= rhs_elt;
      case cudf::binary_operator::GREATER_EQUAL: return lhs_elt >= rhs_elt;
      default: return false;
    }
  }
};

TYPED_TEST(ChronoColumnTest, ChronosCanBeComparedInDeviceCode)
{
  using T = TypeParam;
  using namespace cudf::test;
  using namespace simt::std::chrono;

  auto start_lhs = milliseconds(-2500000000000);  // Sat, 11 Oct 1890 19:33:20 GMT
  auto start_rhs = milliseconds(-2400000000000);  // Tue, 12 Dec 1893 05:20:00 GMT
  auto stop_lhs_ = milliseconds(2500000000000);   // Mon, 22 Mar 2049 04:26:40 GMT
  auto stop_rhs_ = milliseconds(2600000000000);   // Wed, 22 May 2052 14:13:20 GMT

  auto chrono_lhs_col =
    generate_timestamps<T>(this->size(), time_point_ms(start_lhs), time_point_ms(stop_lhs_));

  auto chrono_rhs_col =
    generate_timestamps<T>(this->size(), time_point_ms(start_rhs), time_point_ms(stop_rhs_));

  thrust::device_vector<int32_t> indices(this->size());
  thrust::sequence(indices.begin(), indices.end());

  EXPECT_TRUE(thrust::all_of(
    indices.begin(),
    indices.end(),
    compare_chrono_elements<TypeParam>{cudf::binary_operator::LESS,
                                       *cudf::column_device_view::create(chrono_lhs_col),
                                       *cudf::column_device_view::create(chrono_rhs_col)}));

  EXPECT_TRUE(thrust::all_of(
    indices.begin(),
    indices.end(),
    compare_chrono_elements<TypeParam>{cudf::binary_operator::GREATER,
                                       *cudf::column_device_view::create(chrono_rhs_col),
                                       *cudf::column_device_view::create(chrono_lhs_col)}));

  EXPECT_TRUE(thrust::all_of(
    indices.begin(),
    indices.end(),
    compare_chrono_elements<TypeParam>{cudf::binary_operator::LESS_EQUAL,
                                       *cudf::column_device_view::create(chrono_lhs_col),
                                       *cudf::column_device_view::create(chrono_lhs_col)}));

  EXPECT_TRUE(thrust::all_of(
    indices.begin(),
    indices.end(),
    compare_chrono_elements<TypeParam>{cudf::binary_operator::GREATER_EQUAL,
                                       *cudf::column_device_view::create(chrono_rhs_col),
                                       *cudf::column_device_view::create(chrono_rhs_col)}));
}

TYPED_TEST(ChronoColumnTest, ChronoFactoryNullMaskAsParm)
{
  rmm::device_buffer null_mask{create_null_mask(this->size(), cudf::mask_state::ALL_NULL)};
  auto column = make_fixed_width_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                        this->size(),
                                        std::move(null_mask),
                                        this->size(),
                                        this->stream(),
                                        this->mr());
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
  EXPECT_EQ(this->size(), column->null_count());
  EXPECT_TRUE(column->nullable());
  EXPECT_TRUE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

TYPED_TEST(ChronoColumnTest, ChronoFactoryNullMaskAsEmptyParm)
{
  rmm::device_buffer null_mask{};
  auto column = make_fixed_width_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                        this->size(),
                                        std::move(null_mask),
                                        0,
                                        this->stream(),
                                        this->mr());

  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
  EXPECT_EQ(0, column->null_count());
  EXPECT_FALSE(column->nullable());
  EXPECT_FALSE(column->has_nulls());
  EXPECT_EQ(0, column->num_children());
}

CUDF_TEST_PROGRAM_MAIN()
