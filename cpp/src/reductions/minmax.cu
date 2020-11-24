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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <type_traits>

namespace cudf {
namespace detail {

namespace {

/**
 * @brief Basic element for the minmax reduce operation.
 *
 * Stores the minimum and maximum values that have been encountered so far
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
 * @brief Reduce for the minmax operation and return a device scalar.
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
template <typename T,
          typename Op,
          typename InputIterator,
          typename OutputType = typename thrust::iterator_value<InputIterator>::type>
rmm::device_scalar<OutputType> reduce_device(InputIterator d_in,
                                             cudf::size_type num_items,
                                             Op binary_op,
                                             rmm::cuda_stream_view stream)
{
  OutputType identity{};
  rmm::device_scalar<OutputType> result{identity, stream};

  // Allocate temporary storage
  size_t storage_bytes = 0;
  cub::DeviceReduce::Reduce(
    nullptr, storage_bytes, d_in, result.data(), num_items, binary_op, identity, stream.value());
  auto temp_storage = rmm::device_buffer{storage_bytes, stream};

  // Run reduction
  cub::DeviceReduce::Reduce(temp_storage.data(),
                            storage_bytes,
                            d_in,
                            result.data(),
                            num_items,
                            binary_op,
                            identity,
                            stream.value());

  return result;
}

/**
 * @brief Functor that accepts two minmax_pairs and returns a
 * minmax_pair whose minimum and maximum values are the min() and max()
 * respectively of the minimums and maximums of the input pairs.
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
 * @brief Creates a minmax_pair<T> from a T
 *
 */
template <typename T>
struct create_minmax {
  __device__ minmax_pair<T> operator()(T e) { return minmax_pair<T>{e}; }
};

/**
 * @brief Functor that takes a thrust::pair<T, bool> and produces a minmax_pair
 * that is <T, T> for minimum and maximum or <cudf::DeviceMin::identity<T>(),
 * cudf::DeviceMax::identity<T>()>
 *
 */
template <typename T>
struct create_minmax_with_nulls {
  __device__ minmax_pair<T> operator()(thrust::pair<T, bool> i)
  {
    return i.second ? minmax_pair<T>{i.first} : minmax_pair<T>{};
  }
};

/**
 * @brief Dispatch functor for minmax operation.
 *
 * This uses the reduce function to compute the min and max values
 * simultaneously for a column of data.
 *
 * @tparam T The input column's type
 */
struct minmax_functor {
  template <typename T>
  static constexpr bool is_supported()
  {
    return !(cudf::is_fixed_point<T>() || std::is_same<T, cudf::list_view>::value ||
             std::is_same<T, cudf::struct_view>::value);
  }

  template <typename T>
  auto reduce(column_view const &col, rmm::cuda_stream_view stream)
  {
    auto device_col = column_device_view::create(col, stream);
    // compute minimum and maximum values
    if (col.has_nulls()) {
      auto pair_to_minmax = thrust::make_transform_iterator(
        make_pair_iterator<T, true>(*device_col), create_minmax_with_nulls<T>{});
      return reduce_device<T>(pair_to_minmax, col.size(), minmax_binary_op<T>{}, stream);
    } else {
      auto col_to_minmax =
        thrust::make_transform_iterator(device_col->begin<T>(), create_minmax<T>{});
      return reduce_device<T>(col_to_minmax, col.size(), minmax_binary_op<T>{}, stream);
    }
  }

  /**
   * @brief Functor to copy a minmax_pair result to individual scalar instances.
   */
  template <typename T, typename ResultType = minmax_pair<T>>
  struct assign_min_max {
    __device__ void operator()()
    {
      *min_data = result->min_val;
      *max_data = result->max_val;
    }

    ResultType *result;
    T *min_data;
    T *max_data;
  };

  template <typename T,
            std::enable_if_t<is_supported<T>() and !std::is_same<T, cudf::string_view>::value and
                             !cudf::is_dictionary<T>()> * = nullptr>
  std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> operator()(
    cudf::column_view const &col, rmm::cuda_stream_view stream, rmm::mr::device_memory_resource *mr)
  {
    // compute minimum and maximum values
    auto dev_result = reduce<T>(col, stream);
    // create output scalars
    using ScalarType = cudf::scalar_type_t<T>;
    auto minimum     = new ScalarType(T{}, true, stream, mr);
    auto maximum     = new ScalarType(T{}, true, stream, mr);
    // copy dev_result to the output scalars
    device_single_thread(assign_min_max<T>{dev_result.data(), minimum->data(), maximum->data()},
                         stream.value());
    return {std::unique_ptr<scalar>(minimum), std::unique_ptr<scalar>(maximum)};
  }

  /**
   * @brief Specialization for strings column.
   */
  template <typename T, std::enable_if_t<std::is_same<T, cudf::string_view>::value> * = nullptr>
  std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> operator()(
    cudf::column_view const &col, rmm::cuda_stream_view stream, rmm::mr::device_memory_resource *mr)
  {
    // compute minimum and maximum values
    auto dev_result = reduce<cudf::string_view>(col, stream);
    // copy the minmax_pair to the host; does not copy the strings
    using OutputType = minmax_pair<cudf::string_view>;
    OutputType host_result;
    CUDA_TRY(cudaMemcpyAsync(
      &host_result, dev_result.data(), sizeof(OutputType), cudaMemcpyDeviceToHost, stream.value()));
    // strings are copied to create the scalars here
    return {std::make_unique<string_scalar>(host_result.min_val, true, stream, mr),
            std::make_unique<string_scalar>(host_result.max_val, true, stream, mr)};
  }

  /**
   * @brief Specialization for dictionary column.
   */
  template <typename T, std::enable_if_t<cudf::is_dictionary<T>()> * = nullptr>
  std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> operator()(
    cudf::column_view const &col, rmm::cuda_stream_view stream, rmm::mr::device_memory_resource *mr)
  {
    // compute minimum and maximum values
    auto dev_result = reduce<T>(col, stream);
    // copy the minmax_pair to the host to call get_element
    using OutputType = minmax_pair<T>;
    OutputType host_result;
    CUDA_TRY(cudaMemcpyAsync(
      &host_result, dev_result.data(), sizeof(OutputType), cudaMemcpyDeviceToHost, stream.value()));
    // get the keys for those indexes
    auto const keys = dictionary_column_view(col).keys();
    return {get_element(keys, static_cast<size_type>(host_result.min_val), stream, mr),
            get_element(keys, static_cast<size_type>(host_result.max_val), stream, mr)};
  }

  template <typename T, std::enable_if_t<!is_supported<T>()> * = nullptr>
  std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> operator()(
    cudf::column_view const &, rmm::cuda_stream_view, rmm::mr::device_memory_resource *)
  {
    CUDF_FAIL("type not supported for minmax() operation");
  }
};

}  // namespace

std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> minmax(
  cudf::column_view const &col, rmm::cuda_stream_view stream, rmm::mr::device_memory_resource *mr)
{
  if (col.null_count() == col.size()) {
    // this handles empty and all-null columns
    // return scalars with valid==false
    return {make_default_constructed_scalar(col.type()),
            make_default_constructed_scalar(col.type())};
  }

  return type_dispatcher(col.type(), minmax_functor{}, col, stream, mr);
}
}  // namespace detail

/**
 * @copydoc cudf::minmax
 */
std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> minmax(
  const column_view &col, rmm::mr::device_memory_resource *mr)
{
  return detail::minmax(col, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
