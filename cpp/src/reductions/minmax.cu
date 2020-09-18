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
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <type_traits>

namespace cudf {
namespace detail {

namespace {

/**
 * @brief Reduce the binary operation in device and return a device scalar.
 *
 * @tparam Op Binary operator functor
 * @tparam InputIterator Input iterator Type
 * @param d_in input iterator
 * @param num_items number of items to reduce
 * @param binary_op binary operator used to reduce
 * @param mr Device resource used for result allocation
 * @param stream CUDA stream to run kernels on.
 * @return rmm::device_scalar<OutputType>
 */
template <typename Op,
          typename InputIterator,
          typename OutputType = typename thrust::iterator_value<InputIterator>::type>
// typename std::enable_if_t<is_fixed_width<OutputType>()>* = nullptr>
rmm::device_scalar<OutputType> reduce_device(InputIterator d_in,
                                             cudf::size_type num_items,
                                             Op binary_op,
                                             rmm::mr::device_memory_resource *mr,
                                             cudaStream_t stream)
{
  OutputType identity{};
  rmm::device_scalar<OutputType> dev_result{identity, stream, mr};  // TODO remove mr

  // Allocate temporary storage
  rmm::device_buffer d_temp_storage;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(d_temp_storage.data(),
                            temp_storage_bytes,
                            d_in,
                            dev_result.data(),
                            num_items,
                            binary_op,
                            identity,
                            stream);
  d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};

  // Run reduction
  cub::DeviceReduce::Reduce(d_temp_storage.data(),
                            temp_storage_bytes,
                            d_in,
                            dev_result.data(),
                            num_items,
                            binary_op,
                            identity,
                            stream);
  return std::move(dev_result);
}

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
template <typename T, bool has_nulls = true>
struct minmax_binary_op
  : public thrust::binary_function<minmax_pair<T>, minmax_pair<T>, minmax_pair<T>> {
  __device__ minmax_pair<T> operator()(minmax_pair<T> const &lhs, minmax_pair<T> const &rhs) const
  {
    T const x_min = (lhs.min_valid || !has_nulls) ? lhs.min_val : cudf::DeviceMin::identity<T>();
    T const y_min = (rhs.min_valid || !has_nulls) ? rhs.min_val : cudf::DeviceMin::identity<T>();
    T const x_max = (lhs.max_valid || !has_nulls) ? lhs.max_val : cudf::DeviceMax::identity<T>();
    T const y_max = (rhs.max_valid || !has_nulls) ? rhs.max_val : cudf::DeviceMax::identity<T>();

    // The only invalid situation is if we compare two invalid values.
    // Otherwise, we are certain to select a valid value due to the
    // identity functions above changing the comparison value.
    bool const valid_min_result = !has_nulls || lhs.min_valid || rhs.min_valid;
    bool const valid_max_result = !has_nulls || lhs.max_valid || rhs.max_valid;

    return minmax_pair<T>{
      thrust::min(x_min, y_min), valid_min_result, thrust::max(x_max, y_max), valid_max_result};
  }
};

template <typename T>
struct minmax_iterfunctor {
  column_device_view d_col;
  __device__ minmax_pair<T> operator()(size_type index)
  {
    return minmax_pair<T>(d_col.element<T>(index), d_col.is_valid(index));
  };
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
  std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> operator()(
    const cudf::column_view &col, rmm::mr::device_memory_resource *mr, cudaStream_t stream)
  {
    auto device_col  = column_device_view::create(col, stream);
    using OutputType = minmax_pair<T>;

    // compute minimum and maximum values
    auto dev_result = [&]() -> rmm::device_scalar<OutputType> {
      auto op = minmax_iterfunctor<T>{*device_col};
      auto begin =
        thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0), op);
      // auto end = begin + col.size();

      if (col.nullable()) {
        auto binary_op = minmax_binary_op<T, true>{};
        return reduce_device(begin, col.size(), binary_op, mr, stream);
      } else {
        auto binary_op = minmax_binary_op<T, false>{};
        return reduce_device(begin, col.size(), binary_op, mr, stream);
      }
    }();

    using ScalarType = cudf::scalar_type_t<T>;
    auto min         = new ScalarType(T{}, false, stream, mr);
    auto max         = new ScalarType(T{}, false, stream, mr);

    device_single_thread(
      [result    = dev_result.data(),
       min_data  = min->data(),
       min_valid = min->validity_data(),
       max_data  = max->data(),
       max_valid = max->validity_data()] __device__() mutable {
        *min_data  = result->min_val;
        *min_valid = result->min_valid;
        *max_data  = result->max_val;
        *max_valid = result->max_valid;
      },
      stream);
    return {std::move(std::unique_ptr<scalar>(min)), std::move(std::unique_ptr<scalar>(max))};
  }
};

template <>
std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> minmax_functor::
operator()<cudf::dictionary32>(const cudf::column_view &col,
                               rmm::mr::device_memory_resource *mr,
                               cudaStream_t stream)
{
  CUDF_FAIL("dictionary type not supported");
}

template <>
std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> minmax_functor::
operator()<cudf::string_view>(const cudf::column_view &col,
                              rmm::mr::device_memory_resource *mr,
                              cudaStream_t stream)
{
  CUDF_FAIL("string type not supported");
}

template <>
std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> minmax_functor::
operator()<cudf::list_view>(const cudf::column_view &col,
                            rmm::mr::device_memory_resource *mr,
                            cudaStream_t stream)
{
  CUDF_FAIL("list type not supported");
}

template <>
std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> minmax_functor::
operator()<cudf::struct_view>(const cudf::column_view &col,
                              rmm::mr::device_memory_resource *mr,
                              cudaStream_t stream)
{
  CUDF_FAIL("struct type not supported");
}

// unable to support fixed point due to DeviceMin/DeviceMax not supporting fixed point
template <>
std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> minmax_functor::
operator()<numeric::decimal32>(const cudf::column_view &col,
                               rmm::mr::device_memory_resource *mr,
                               cudaStream_t stream)
{
  CUDF_FAIL("fixed-point type not supported");
}

template <>
std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> minmax_functor::
operator()<numeric::decimal64>(const cudf::column_view &col,
                               rmm::mr::device_memory_resource *mr,
                               cudaStream_t stream)
{
  CUDF_FAIL("fixed-point type not supported");
}

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
