/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/reduction/detail/reduction_functions.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/utility>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform_reduce.h>

#include <type_traits>

namespace cudf {
namespace reduction {
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
    : min_val(cudf::DeviceMin::identity<T>()), max_val(cudf::DeviceMax::identity<T>()) {};
  __host__ __device__ minmax_pair(T val) : min_val(val), max_val(val) {};
  __host__ __device__ minmax_pair(T min_val_, T max_val_) : min_val(min_val_), max_val(max_val_) {};
};

/**
 * @brief Reduce for the minmax operation and return a device scalar.
 *
 * @tparam Op Binary operator functor
 * @tparam InputIterator Input iterator Type
 * @tparam OutputType Output scalar type
 * @param d_in input iterator
 * @param num_items number of items to reduce
 * @param binary_op binary operator used to reduce
 * @param stream CUDA stream to run kernels on.
 * @return cudf::detail::device_scalar<OutputType>
 */
template <typename Op,
          typename InputIterator,
          typename OutputType = cuda::std::iter_value_t<InputIterator>>
auto reduce_device(InputIterator d_in,
                   size_type num_items,
                   Op binary_op,
                   rmm::cuda_stream_view stream)
{
  OutputType identity{};
  cudf::detail::device_scalar<OutputType> result{identity, stream};

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
 */
template <typename T>
struct minmax_binary_op {
  __device__ minmax_pair<T> operator()(minmax_pair<T> const& lhs, minmax_pair<T> const& rhs) const
  {
    return minmax_pair<T>{cuda::std::min(lhs.min_val, rhs.min_val),
                          cuda::std::max(lhs.max_val, rhs.max_val)};
  }
};

/**
 * @brief Creates a minmax_pair<T> from a T
 */
template <typename T>
struct create_minmax {
  __device__ minmax_pair<T> operator()(T e) { return minmax_pair<T>{e}; }
};

/**
 * @brief Functor that takes a cuda::std::pair<T, bool> and produces a minmax_pair
 * that is <T, T> for minimum and maximum or <cudf::DeviceMin::identity<T>(),
 * cudf::DeviceMax::identity<T>()>
 */
template <typename T>
struct create_minmax_with_nulls {
  __device__ minmax_pair<T> operator()(cuda::std::pair<T, bool> i)
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
    return !(std::is_same_v<T, cudf::list_view> || std::is_same_v<T, cudf::struct_view>);
  }

  template <typename T>
  auto reduce(column_view const& col, rmm::cuda_stream_view stream)
  {
    auto device_col = column_device_view::create(col, stream);
    // compute minimum and maximum values
    if (col.has_nulls()) {
      auto pair_to_minmax = thrust::make_transform_iterator(
        make_pair_iterator<T, true>(*device_col), create_minmax_with_nulls<T>{});
      return reduce_device(pair_to_minmax, col.size(), minmax_binary_op<T>{}, stream);
    } else {
      auto col_to_minmax =
        thrust::make_transform_iterator(device_col->begin<T>(), create_minmax<T>{});
      return reduce_device(col_to_minmax, col.size(), minmax_binary_op<T>{}, stream);
    }
  }

  /**
   * @brief Functor to copy a minmax_pair result to individual scalar instances.
   *
   * @tparam T type of the data
   * @tparam ResultType result type to assign min, max to minmax_pair<T>
   */
  template <typename T, typename ResultType = minmax_pair<T>>
  struct assign_min_max {
    __device__ void operator()()
    {
      *min_data = result->min_val;
      *max_data = result->max_val;
    }

    ResultType* result;
    T* min_data;
    T* max_data;
  };

  template <typename T>
  std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> operator()(
    cudf::column_view const& col, rmm::cuda_stream_view stream, cudf::memory_resources resources)
    requires(is_supported<T>() and !std::is_same_v<T, cudf::string_view> and
             !cudf::is_dictionary<T>())
  {
    using storage_type = device_storage_type_t<T>;
    // compute minimum and maximum values
    auto dev_result = reduce<storage_type>(col, stream);
    // create output scalars
    using ScalarType = cudf::scalar_type_t<T>;
    auto minimum     = new ScalarType(T{}, true, stream, resources);
    auto maximum     = new ScalarType(T{}, true, stream, resources);
    // copy dev_result to the output scalars
    cudf::detail::device_single_thread(
      assign_min_max<storage_type>{dev_result.data(), minimum->data(), maximum->data()}, stream);
    return {std::unique_ptr<scalar>(minimum), std::unique_ptr<scalar>(maximum)};
  }

  /**
   * @brief Specialization for strings column.
   */
  template <typename T>
  std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> operator()(
    cudf::column_view const& col, rmm::cuda_stream_view stream, cudf::memory_resources resources)
    requires(std::is_same_v<T, cudf::string_view>)
  {
    // compute minimum and maximum values
    auto dev_result = reduce<cudf::string_view>(col, stream);
    // copy the minmax_pair to the host; does not copy the strings
    auto const host_result = dev_result.value(stream);
    // strings are copied to create the scalars here
    return {std::make_unique<string_scalar>(host_result.min_val, true, stream, resources),
            std::make_unique<string_scalar>(host_result.max_val, true, stream, resources)};
  }

  /**
   * @brief Specialization for dictionary column.
   */
  template <typename T>
  std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> operator()(
    cudf::column_view const& col, rmm::cuda_stream_view stream, cudf::memory_resources resources)
    requires(cudf::is_dictionary<T>())
  {
    // computes minimum and maximum on the dictionary indices as dictionary32 values
    auto d_indices     = reduce<T>(col, stream);
    auto const indices = d_indices.value(stream);
    // use these values to slice the keys column (add 1 for complete inclusion)
    auto keys = cudf::detail::slice(dictionary_column_view(col).keys(),
                                    {indices.min_val.value(), indices.max_val.value() + 1},
                                    stream)
                  .front();
    return type_dispatcher(keys.type(), minmax_functor{}, keys, stream, resources);
  }

  template <typename T>
  std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> operator()(
    cudf::column_view const&, rmm::cuda_stream_view, rmm::device_async_resource_ref)
    requires(!is_supported<T>())
  {
    CUDF_FAIL("type not supported for minmax() operation");
  }
};

}  // namespace

/**
 * @copydoc cudf::minmax
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> minmax(cudf::column_view const& col,
                                                                   rmm::cuda_stream_view stream,
                                                                   cudf::memory_resources resources)
{
  if (col.null_count() == col.size()) {
    // this handles empty and all-null columns
    // return scalars with valid==false
    return {make_default_constructed_scalar(col.type(), stream, resources),
            make_default_constructed_scalar(col.type(), stream, resources)};
  }

  return type_dispatcher(col.type(), minmax_functor{}, col, stream, resources);
}
}  // namespace detail
}  // namespace reduction

std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> minmax(column_view const& col,
                                                                   rmm::cuda_stream_view stream,
                                                                   cudf::memory_resources resources)
{
  CUDF_FUNC_RANGE();
  return reduction::detail::minmax(col, stream, resources);
}

}  // namespace cudf
