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

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
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

  __host__ __device__ minmax_pair()
    : min_val(cudf::DeviceMin::identity<T>()), max_val(cudf::DeviceMax::identity<T>()){};
  __host__ __device__ minmax_pair(T val) : min_val(val), max_val(val){};
  __host__ __device__ minmax_pair(T min_val_, T max_val_) : min_val(min_val_), max_val(max_val_){};
};

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
template <typename T, typename Op, typename InputIterator, typename OutputType = minmax_pair<T>>
// typename thrust::iterator_value<InputIterator>::type>
// above produces compilation error where OutputType is somehow int
auto reduce_device(InputIterator d_in,
                   cudf::size_type num_items,
                   Op binary_op,
                   rmm::mr::device_memory_resource *mr,
                   cudaStream_t stream)  //-> typename std::enable_if_t<std::is_same<T, int>::value, rmm::device_scalar<OutputType>>
{
  OutputType identity{};
  rmm::device_scalar<OutputType> dev_result{identity, stream, mr};  // TODO remove mr

  // Allocate temporary storage
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(
    nullptr, temp_storage_bytes, d_in, dev_result.data(), num_items, binary_op, identity, stream);
  auto d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};

  // Run reduction
  cub::DeviceReduce::Reduce(d_temp_storage.data(),
                            temp_storage_bytes,
                            d_in,
                            dev_result.data(),
                            num_items,
                            binary_op,
                            identity,
                            stream);

  // this works as expected and prints values from the input
  /*
  thrust::for_each(
    rmm::exec_policy(stream)->on(stream), d_in, d_in + num_items, [] __device__(auto p) {
      printf("%d/%d\n", p.min_val, p.max_val);
    });
  */

  // dev_result is invalid and it appears no iterator access is happening via
  // cub::DeviceReduce::Reduce. The return value is identity.

  return dev_result;
}

/**
 * @brief functor that accepts two minmax_pairs and returns a
 * minmax_pair whose minimum and maximum values are the min() and max()
 * respectively of the minimums and maximums of the input pairs. Respects
 * validity.
 *
 */
template <typename T>
struct minmax_binary_op
  : public thrust::binary_function<minmax_pair<T>, minmax_pair<T>, minmax_pair<T>> {
  __device__ minmax_pair<T> operator()(minmax_pair<T> const &lhs, minmax_pair<T> const &rhs) const
  {
    return minmax_pair<T>{thrust::min(lhs.min_val, rhs.min_val),
                          thrust::max(lhs.max_val, rhs.max_val)};
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
  std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> operator()(
    const cudf::column_view &col, rmm::mr::device_memory_resource *mr, cudaStream_t stream)
  {
    auto device_col  = column_device_view::create(col, stream);
    using OutputType = minmax_pair<T>;

    using ScalarType = cudf::scalar_type_t<T>;
    auto min         = new ScalarType(T{}, false, stream, mr);
    auto max         = new ScalarType(T{}, false, stream, mr);

    auto null_count = col.null_count();

    if (null_count == col.size()) {
      // all nulls, no computation needed
      return {std::unique_ptr<scalar>(min), std::unique_ptr<scalar>(max)};
    }

    // compute minimum and maximum values
    auto dev_result = [&]() -> rmm::device_scalar<OutputType> {
      if (null_count > 0) {
        auto pair = make_pair_iterator<T, true>(*device_col);
        auto pair_to_minmax =
          thrust::make_transform_iterator(pair, [] __device__(auto i) -> minmax_pair<T> {
            return i.second ? minmax_pair<T>{i.first} : minmax_pair<T>{};
          });

        rmm::device_scalar<minmax_pair<T>> ret =
          reduce_device<T>(pair_to_minmax, col.size(), minmax_binary_op<T>{}, mr, stream);
        return ret;
      } else {
        auto col_to_minmax = thrust::make_transform_iterator(
          thrust::make_counting_iterator(size_type{0}),
          [dcol = *device_col] __device__(size_type i) -> minmax_pair<T> {
            return minmax_pair<T>{dcol.element<T>(i)};
          });

        rmm::device_scalar<minmax_pair<T>> ret =
          reduce_device<T>(col_to_minmax, col.size(), minmax_binary_op<T>{}, mr, stream);
        return ret;
      }
    }();

    device_single_thread(
      [result    = dev_result.data(),
       min_data  = min->data(),
       min_valid = min->validity_data(),
       max_data  = max->data(),
       max_valid = max->validity_data()] __device__() mutable {
        *min_data  = result->min_val;
        *min_valid = true;
        *max_data  = result->max_val;
        *max_valid = true;
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
