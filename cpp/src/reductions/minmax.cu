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
// The translation unit for reduction `minmax`

#include <thrust/transform_reduce.h>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <type_traits>

namespace cudf {
namespace detail {

namespace {
/**
 * @brief stores the minimum and maximum
 * values that have been encountered so far
 *
 */
template <typename T>
struct minmax_pair {
  T min_val;
  T max_val;
  bool min_valid;
  bool max_valid;

  __host__ __device__ minmax_pair()
    : min_val(cudf::DeviceMin::identity<T>()),
      max_val(cudf::DeviceMax::identity<T>()),
      min_valid(false),
      max_valid(false){};
  __host__ __device__ minmax_pair(T val, bool valid_)
    : min_val(val), max_val(val), min_valid(valid_), max_valid(valid_){};
  __host__ __device__ minmax_pair(T min_val_, bool min_valid_, T max_val_, bool max_valid_)
    : min_val(min_val_), max_val(max_val_), min_valid(min_valid_), max_valid(max_valid_){};
};

/**
 * @brief functor that accepts two minmax_pairs and returns a
 * minmax_pair whose minimum and maximum values are the min() and max()
 * respectively of the minimums and maximums of the input pairs. Respects
 * validity.
 *
 */
template <typename T>
struct minmax_with_null_binary_op
  : public thrust::binary_function<minmax_pair<T>, minmax_pair<T>, minmax_pair<T>> {
  __host__ __device__ minmax_pair<T> operator()(const minmax_pair<T> &x,
                                                const minmax_pair<T> &y) const
  {
    T x_min = x.min_valid ? x.min_val : cudf::DeviceMin::identity<T>();
    T y_min = y.min_valid ? y.min_val : cudf::DeviceMin::identity<T>();
    T x_max = x.max_valid ? x.max_val : cudf::DeviceMax::identity<T>();
    T y_max = y.max_valid ? y.max_val : cudf::DeviceMax::identity<T>();

    // The only invalid situation is if we compare two invalid values.
    // Otherwise, we are certain to select a valid value due to the
    // identity functions above changing the comparison value.
    bool valid_min_result = x.min_valid || y.min_valid;
    bool valid_max_result = x.max_valid || y.max_valid;

    return minmax_pair<T>{
      thrust::min(x_min, y_min), valid_min_result, thrust::max(x_max, y_max), valid_max_result};
  }
};

/**
 * @brief functor that accepts two minmax_pairs and returns a
 * minmax_pair whose minimum and maximum values are the min() and max()
 * respectively of the minimums and maximums of the input pairs. Expects
 * no null values.
 *
 */
template <typename T>
struct minmax_no_null_binary_op
  : public thrust::binary_function<minmax_pair<T>, minmax_pair<T>, minmax_pair<T>> {
  __host__ __device__ minmax_pair<T> operator()(const minmax_pair<T> &x,
                                                const minmax_pair<T> &y) const
  {
    return minmax_pair<T>{
      thrust::min(x.min_val, y.min_val), true, thrust::max(x.max_val, y.max_val), true};
  }
};

/**
 * @brief functor that calls thrust::transform_reduce to produce a std::pair
 * of scalars that represent the minimum and maximum values of the input data
 * respectively. Note that dictionaries and non-relationally comparable objects
 * are not supported.
 *
 */
struct minmax_functor {
  template <typename T>
  // unable to support fixed point due to DeviceMin/DeviceMax not supporting fixed point
  std::enable_if_t<cudf::is_relationally_comparable<T, T>() and
                     not std::is_same<T, dictionary32>::value and not cudf::is_fixed_point<T>(),
                   std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>>>
  operator()(const cudf::column_view &col, rmm::mr::device_memory_resource *mr, cudaStream_t stream)
  {
    auto device_col = column_device_view::create(col, stream);

    // compute minimum and maximum values
    minmax_pair<T> result;
    if (col.nullable()) {
      result = thrust::transform_reduce(
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(col.size()),
        [d_col = *device_col] __device__(size_type index) -> minmax_pair<T> {
          return minmax_pair<T>(d_col.element<T>(index), d_col.is_valid(index));
        },
        minmax_pair<T>{},
        minmax_with_null_binary_op<T>{});
    } else {
      result = thrust::transform_reduce(
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(col.size()),
        [d_col = *device_col] __device__(size_type index) -> minmax_pair<T> {
          return minmax_pair<T>(d_col.element<T>(index), d_col.is_valid(index));
        },
        minmax_pair<T>{},
        minmax_no_null_binary_op<T>{});
    }

    std::unique_ptr<scalar> min =
      make_fixed_width_scalar<T>(result.min_val, result.min_valid, stream, mr);
    std::unique_ptr<scalar> max =
      make_fixed_width_scalar<T>(result.max_val, result.max_valid, stream, mr);
    return {std::move(min), std::move(max)};
  }

  template <typename T,
            std::enable_if_t<not cudf::is_relationally_comparable<T, T>() or
                             cudf::is_fixed_point<T>()> * = nullptr>
  std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> operator()(
    const cudf::column_view &col, rmm::mr::device_memory_resource *mr, cudaStream_t stream)
  {
    CUDF_FAIL("type not supported");
  }

  template <typename T, typename std::enable_if_t<std::is_same<T, dictionary32>::value> * = nullptr>
  std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> operator()(
    const cudf::column_view &col, rmm::mr::device_memory_resource *mr, cudaStream_t stream)
  {
    CUDF_FAIL("dictionary type not supported");
  }
};

}  // namespace

std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> minmax(
  const cudf::column_view &col,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0)
{
  return type_dispatcher(col.type(), minmax_functor{}, col, mr, stream);
}
}  // namespace detail

/**
 * @copydoc cudf::minmax
 */
std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> minmax(
  const cudf::column_view &col, rmm::mr::device_memory_resource *mr)
{
  return cudf::detail::minmax(col, mr);
}

}  // namespace cudf
