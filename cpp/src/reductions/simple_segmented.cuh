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
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/reduction.cuh>
#include <cudf/detail/unary.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

namespace cudf {
namespace reduction {
namespace simple {
namespace detail {

/**
 * @brief Segment reduction for 'sum', 'product', 'min', 'max', 'sum of squares'
 * which directly compute the reduction by a single step reduction call.
 *
 * @tparam InputType    the input column data-type
 * @tparam ResultType   the output data-type
 * @tparam Op           the operator of cudf::reduction::op::

 * @param col Input column of data to reduce.
 * @param offsets Indices to segment boundaries.
 * @param null_handling If `null_policy::INCLUDE`, all elements in a segment
 * must be valid for the reduced value to be valid. If `null_policy::EXCLUDE`,
 * the reduced value is valid if any element in the segment is valid.
 * @param stream Used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Output column in device memory
 */
template <typename InputType, typename ResultType, typename Op>
std::unique_ptr<column> simple_segmented_reduction(column_view const& col,
                                                   column_view const& offsets,
                                                   null_policy null_handling,
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
      return cudf::reduction::detail::segmented_reduce(
        it, offsets.begin<size_type>(), num_segments, simple_op, stream, mr);
    } else {
      auto f  = simple_op.template get_element_transformer<ResultType>();
      auto it = thrust::make_transform_iterator(dcol->begin<InputType>(), f);
      return cudf::reduction::detail::segmented_reduce(
        it, offsets.begin<size_type>(), num_segments, simple_op, stream, mr);
    }
  }();

  // Compute the output null mask
  auto const bitmask                 = col.null_mask();
  auto const first_bit_indices_begin = offsets.begin<size_type>();
  auto const first_bit_indices_end   = offsets.end<size_type>() - 1;
  auto const last_bit_indices_begin  = offsets.begin<size_type>() + 1;
  auto const [output_null_mask, output_null_count] =
    cudf::detail::segmented_null_mask_reduction(bitmask,
                                                first_bit_indices_begin,
                                                first_bit_indices_end,
                                                last_bit_indices_begin,
                                                null_handling,
                                                stream,
                                                mr);
  result->set_null_mask(output_null_mask, output_null_count, stream);

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
struct bool_result_column_dispatcher {
  template <typename ElementType,
            std::enable_if_t<std::is_arithmetic<ElementType>::value>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& col,
                                     column_view const& offsets,
                                     null_policy null_handling,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return simple_segmented_reduction<ElementType, bool, Op>(
      col, offsets, null_handling, stream, mr);
  }

  template <typename ElementType,
            std::enable_if_t<not std::is_arithmetic<ElementType>::value>* = nullptr>
  std::unique_ptr<column> operator()(column_view const&,
                                     column_view const&,
                                     null_policy,
                                     rmm::cuda_stream_view,
                                     rmm::mr::device_memory_resource*)
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
                                     null_policy null_handling,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return simple_segmented_reduction<ElementType, ElementType, Op>(
      col, offsets, null_handling, stream, mr);
  }

  template <typename ElementType,
            std::enable_if_t<not is_supported<ElementType>() or
                             cudf::is_fixed_point<ElementType>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const&,
                                     column_view const&,
                                     null_policy,
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
  template <typename ElementType,
            typename std::enable_if_t<std::is_floating_point<ElementType>::value>* = nullptr>
  std::unique_ptr<column> reduce_numeric(column_view const& col,
                                         column_view const& offsets,
                                         data_type const output_type,
                                         null_policy null_handling,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
  {
    auto result =
      simple_segmented_reduction<ElementType, double, Op>(col, offsets, null_handling, stream, mr);
    if (output_type == result->type()) { return result; }
    return cudf::detail::cast(*result, output_type, stream, mr);
  }

  /**
   * @brief Specialization for reducing integer column types to any output type.
   */
  template <typename ElementType,
            typename std::enable_if_t<std::is_integral<ElementType>::value>* = nullptr>
  std::unique_ptr<column> reduce_numeric(column_view const& col,
                                         column_view const& offsets,
                                         data_type const output_type,
                                         null_policy null_handling,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
  {
    auto result =
      simple_segmented_reduction<ElementType, int64_t, Op>(col, offsets, null_handling, stream, mr);
    if (output_type == result->type()) { return result; }
    return cudf::detail::cast(*result, output_type, stream, mr);
  }

  /**
   * @brief Called by the type-dispatcher to reduce the input column `col` using
   * the `Op` operation.
   *
   * @tparam ElementType The input column type or key type.
   * @param col Input column (must be numeric)
   * @param offsets Indices to segment boundaries
   * @param output_type Requested type of the scalar result
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned scalar's device memory
   */
  template <typename ElementType,
            typename std::enable_if_t<cudf::is_numeric<ElementType>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& col,
                                     column_view const& offsets,
                                     data_type const output_type,
                                     null_policy null_handling,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    if (output_type.id() == cudf::type_to_id<ElementType>()) {
      return simple_segmented_reduction<ElementType, ElementType, Op>(
        col, offsets, null_handling, stream, mr);
    }
    // reduce and map to output type
    return reduce_numeric<ElementType>(col, offsets, output_type, null_handling, stream, mr);
  }

  template <typename ElementType,
            typename std::enable_if_t<not cudf::is_numeric<ElementType>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const&,
                                     column_view const&,
                                     data_type const,
                                     null_policy,
                                     rmm::cuda_stream_view,
                                     rmm::mr::device_memory_resource*)
  {
    CUDF_FAIL("Reduction operator not supported for this type");
  }
};

}  // namespace detail
}  // namespace simple
}  // namespace reduction
}  // namespace cudf
