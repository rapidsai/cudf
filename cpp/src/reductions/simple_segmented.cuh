/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/detail/copy.hpp>
#include <cudf/detail/reduction.cuh>
#include <cudf/detail/unary.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace reduction {
namespace simple {

/**
 * @brief Segment reduction for 'sum', 'product', 'min', 'max', 'sum of squares'
 * which directly compute the reduction by a single step reduction call
 *
 * @tparam InputType  the input column data-type
 * @tparam ResultType   the output data-type
 * @tparam Op           the operator of cudf::reduction::op::

 * @param col Input column of data to reduce
 * @param offsets Indices to identify segment boundaries
 * @param stream Used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Output column in device memory
 */
template <typename InputType, typename ResultType, typename Op>
std::unique_ptr<column> simple_segmented_reduction(column_view const& col,
                                                   column_view const& offsets,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::mr::device_memory_resource* mr)
{
  // reduction by iterator
  auto dcol              = cudf::column_device_view::create(col, stream);
  auto simple_op         = Op{};
  size_type num_segments = offsets.size() - 1;

  auto result = [&] {
    if (col.has_nulls()) {
      auto f  = simple_op.template get_null_replacing_element_transformer<ResultType>();
      auto it = thrust::make_transform_iterator(dcol->pair_begin<InputType, true>(), f);
      return detail::segmented_reduce(
        it, offsets.begin<size_type>(), num_segments, simple_op, stream, mr);
    } else {
      auto f  = simple_op.template get_element_transformer<ResultType>();
      auto it = thrust::make_transform_iterator(dcol->begin<InputType>(), f);
      return detail::segmented_reduce(
        it, offsets.begin<size_type>(), num_segments, simple_op, stream, mr);
    }
  }();

  // TODO: null handling, gh9552
  //  result->set_valid_async(col.null_count() < col.size(), stream);
  return result;
}

/**
 * @brief Call reduce and return a column of type bool.
 *
 * This is used by operations `any()` and `all()`.
 *
 * @tparam Op The reduce operation to execute on the column.
 */
template <typename Op>
struct bool_result_element_dispatcher {
  template <typename ElementType,
            std::enable_if_t<std::is_arithmetic<ElementType>::value>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& col,
                                     column_view const& offsets,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return simple_segmented_reduction<ElementType, bool, Op>(col, offsets, stream, mr);
  }

  template <typename ElementType,
            typename... Args,
            std::enable_if_t<not std::is_arithmetic<ElementType>::value>* = nullptr>
  std::unique_ptr<column> operator()(Args&&...)
  {
    CUDF_FAIL("Reduction operator not supported for this type");
  }
};

/**
 * @brief Call reduce and return a column of type matching the input column.
 *
 * This is used by operations `min()` and `max()`.
 *
 * @tparam Op The reduce operation to execute on the column.
 */
template <typename Op>
struct same_column_type_dispatcher {
 private:
  template <typename ElementType>
  static constexpr bool is_supported()
  {
    return !(cudf::is_dictionary<ElementType>() || std::is_same_v<ElementType, cudf::list_view> ||
             std::is_same_v<ElementType, cudf::struct_view>);
  }

 public:
  template <typename ElementType,
            std::enable_if_t<is_supported<ElementType>() &&
                             not cudf::is_fixed_point<ElementType>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& col,
                                     column_view const& offsets,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return simple_segmented_reduction<ElementType, ElementType, Op>(col, offsets, stream, mr);
  }

  template <typename ElementType, std::enable_if_t<not is_supported<ElementType>()>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const&,
                                     rmm::cuda_stream_view,
                                     rmm::mr::device_memory_resource*)
  {
    CUDF_FAIL("Reduction operator not supported for this type");
  }
};

/**
 * @brief Call reduce and return a column of the type specified.
 *
 * This is used by operations sum(), product(), and sum_of_squares().
 * It only supports numeric types. If the output type is not the
 * same as the input type, an extra cast operation may incur.
 *
 * @tparam Op The reduce operation to execute on the column.
 */
template <typename Op>
struct column_type_dispatcher {
  /**
   * @brief Specialization for reducing floating-point column types to any output type.
   */
  template <typename ColumnType,
            typename std::enable_if_t<std::is_floating_point<ColumnType>::value>* = nullptr>
  std::unique_ptr<column> reduce_numeric(column_view const& col,
                                         column_view const& offsets,
                                         data_type const output_type,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
  {
    auto result = simple_segmented_reduction<ColumnType, double, Op>(col, offsets, stream, mr);
    if (output_type == result->type()) return result;
    return cudf::detail::cast(*result, output_type, stream, mr);
  }

  /**
   * @brief Specialization for reducing integer column types to any output type.
   */
  template <typename ColumnType,
            typename std::enable_if_t<std::is_integral<ColumnType>::value>* = nullptr>
  std::unique_ptr<column> reduce_numeric(column_view const& col,
                                         column_view const& offsets,
                                         data_type const output_type,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
  {
    auto result = simple_segmented_reduction<ColumnType, int64_t, Op>(col, offsets, stream, mr);
    if (output_type == result->type()) return result;
    return cudf::detail::cast(*result, output_type, stream, mr);
  }

  /**
   * @brief Called by the type-dispatcher to reduce the input column `col` using
   * the `Op` operation.
   *
   * @tparam ElementType The input column type or key type.
   * @param col Input column (must be numeric)
   * @param output_type Requested type of the scalar result
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned scalar's device memory
   */
  template <typename ColumnType,
            typename std::enable_if_t<cudf::is_numeric<ColumnType>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& col,
                                     column_view const& offsets,
                                     data_type const output_type,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    if (output_type.id() == cudf::type_to_id<ColumnType>())
      return simple_segmented_reduction<ColumnType, ColumnType, Op>(col, offsets, stream, mr);
    // reduce and map to output type
    return reduce_numeric<ColumnType>(col, offsets, output_type, stream, mr);
  }

  template <typename ColumnType,
            typename std::enable_if_t<not cudf::is_numeric<ColumnType>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const&,
                                     column_view const&,
                                     data_type const,
                                     rmm::cuda_stream_view,
                                     rmm::mr::device_memory_resource*)
  {
    CUDF_FAIL("Reduction operator not supported for this type");
  }
};

}  // namespace simple
}  // namespace reduction
}  // namespace cudf
