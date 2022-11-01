/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/reduction.cuh>
#include <cudf/detail/unary.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/element_argminmax.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <optional>
#include <type_traits>

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

 * @param col Input column of data to reduce
 * @param offsets Indices to segment boundaries
 * @param null_handling If `null_policy::INCLUDE`, all elements in a segment must be valid for the
 * reduced value to be valid. If `null_policy::EXCLUDE`, the reduced value is valid if any element
 * in the segment is valid.
 * @param init Optional initial value of the reduction
 * @param stream Used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Output column in device memory
 */
template <typename InputType, typename ResultType, typename Op>
std::unique_ptr<column> simple_segmented_reduction(
  column_view const& col,
  device_span<size_type const> offsets,
  null_policy null_handling,
  std::optional<std::reference_wrapper<scalar const>> init,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  // TODO: Rewrites this function to accept a pair of iterators for start/end indices
  // to enable `2N` type offset input.
  // reduction by iterator
  auto dcol               = cudf::column_device_view::create(col, stream);
  auto simple_op          = Op{};
  auto const num_segments = offsets.size() - 1;

  auto const binary_op = simple_op.get_binary_op();

  // Cast initial value
  ResultType initial_value = [&] {
    if (init.has_value() && init.value().get().is_valid()) {
      using ScalarType = cudf::scalar_type_t<InputType>;
      auto input_value = static_cast<const ScalarType*>(&init.value().get())->value(stream);
      return static_cast<ResultType>(input_value);
    } else {
      return simple_op.template get_identity<ResultType>();
    }
  }();

  auto const result_type =
    cudf::is_fixed_point(col.type()) ? col.type() : data_type{type_to_id<ResultType>()};
  auto result =
    make_fixed_width_column(result_type, num_segments, mask_state::UNALLOCATED, stream, mr);
  auto outit = result->mutable_view().template begin<ResultType>();

  // TODO: Explore rewriting null_replacing_element_transformer/element_transformer with nullate
  if (col.has_nulls()) {
    auto f  = simple_op.template get_null_replacing_element_transformer<ResultType>();
    auto it = thrust::make_transform_iterator(dcol->pair_begin<InputType, true>(), f);
    cudf::reduction::detail::segmented_reduce(
      it, offsets.begin(), offsets.end(), outit, binary_op, initial_value, stream);
  } else {
    auto f  = simple_op.template get_element_transformer<ResultType>();
    auto it = thrust::make_transform_iterator(dcol->begin<InputType>(), f);
    cudf::reduction::detail::segmented_reduce(
      it, offsets.begin(), offsets.end(), outit, binary_op, initial_value, stream);
  }

  // Compute the output null mask
  auto const bitmask                         = col.null_mask();
  auto const first_bit_indices_begin         = offsets.begin();
  auto const first_bit_indices_end           = offsets.end() - 1;
  auto const last_bit_indices_begin          = first_bit_indices_begin + 1;
  auto [output_null_mask, output_null_count] = cudf::detail::segmented_null_mask_reduction(
    bitmask,
    first_bit_indices_begin,
    first_bit_indices_end,
    last_bit_indices_begin,
    null_handling,
    init.has_value() ? std::optional(init.value().get().is_valid()) : std::nullopt,
    stream,
    mr);
  result->set_null_mask(std::move(output_null_mask), output_null_count);

  return result;
}

/**
 * @brief String segmented reduction for 'min', 'max'.
 *
 * This algorithm uses argmin/argmax as a custom comparator to build a gather
 * map, then builds the output.
 *
 * @tparam InputType    the input column data-type
 * @tparam Op           the operator of cudf::reduction::op::

 * @param col Input column of data to reduce
 * @param offsets Indices to segment boundaries
 * @param null_handling If `null_policy::INCLUDE`, all elements in a segment must be valid for the
 * reduced value to be valid. If `null_policy::EXCLUDE`, the reduced value is valid if any element
 * in the segment is valid.
 * @param stream Used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Output column in device memory
 */

template <typename InputType,
          typename Op,
          CUDF_ENABLE_IF(std::is_same_v<Op, cudf::reduction::op::min> ||
                         std::is_same_v<Op, cudf::reduction::op::max>)>
std::unique_ptr<column> string_segmented_reduction(column_view const& col,
                                                   device_span<size_type const> offsets,
                                                   null_policy null_handling,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::mr::device_memory_resource* mr)
{
  // Pass to simple_segmented_reduction, get indices to gather, perform gather here.
  auto device_col = cudf::column_device_view::create(col, stream);

  auto it                 = thrust::make_counting_iterator(0);
  auto const num_segments = static_cast<size_type>(offsets.size()) - 1;

  bool constexpr is_argmin = std::is_same_v<Op, cudf::reduction::op::min>;
  auto string_comparator =
    cudf::detail::element_argminmax_fn<InputType>{*device_col, col.has_nulls(), is_argmin};
  auto constexpr identity =
    is_argmin ? cudf::detail::ARGMIN_SENTINEL : cudf::detail::ARGMAX_SENTINEL;

  auto gather_map = make_fixed_width_column(
    data_type{type_to_id<size_type>()}, num_segments, mask_state::UNALLOCATED, stream, mr);

  auto gather_map_it = gather_map->mutable_view().begin<size_type>();

  cudf::reduction::detail::segmented_reduce(
    it, offsets.begin(), offsets.end(), gather_map_it, string_comparator, identity, stream);

  auto result = std::move(cudf::detail::gather(table_view{{col}},
                                               *gather_map,
                                               cudf::out_of_bounds_policy::NULLIFY,
                                               cudf::detail::negative_index_policy::NOT_ALLOWED,
                                               stream,
                                               mr)
                            ->release()[0]);
  auto [segmented_null_mask, segmented_null_count] =
    cudf::detail::segmented_null_mask_reduction(col.null_mask(),
                                                offsets.begin(),
                                                offsets.end() - 1,
                                                offsets.begin() + 1,
                                                null_handling,
                                                std::nullopt,
                                                stream,
                                                mr);

  // If the segmented null mask contains any null values, the segmented null mask
  // must be combined with the result null mask.
  if (segmented_null_count > 0) {
    if (result->null_count() == 0) {
      // The result has no nulls. Use the segmented null mask.
      result->set_null_mask(std::move(segmented_null_mask), segmented_null_count);
    } else {
      // Compute the logical AND of the segmented output null mask and the
      // result null mask to update the result null mask and null count.
      auto result_mview = result->mutable_view();
      std::vector masks{static_cast<bitmask_type const*>(result_mview.null_mask()),
                        static_cast<bitmask_type const*>(segmented_null_mask.data())};
      std::vector<size_type> begin_bits{0, 0};
      auto const valid_count = cudf::detail::inplace_bitmask_and(
        device_span<bitmask_type>(static_cast<bitmask_type*>(result_mview.null_mask()),
                                  num_bitmask_words(result->size())),
        masks,
        begin_bits,
        result->size(),
        stream);
      result->set_null_count(result->size() - valid_count);
    }
  }

  return result;
}

template <typename InputType,
          typename Op,
          CUDF_ENABLE_IF(!std::is_same_v<Op, cudf::reduction::op::min>() &&
                         !std::is_same_v<Op, cudf::reduction::op::max>())>
std::unique_ptr<column> string_segmented_reduction(column_view const& col,
                                                   device_span<size_type const> offsets,
                                                   null_policy null_handling,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FAIL("Segmented reduction on string column only supports min and max reduction.");
}

/**
 * @brief Fixed point segmented reduction for 'min', 'max'.
 *
 * @tparam InputType    the input column data-type
 * @tparam Op           the operator of cudf::reduction::op::

 * @param col Input column of data to reduce
 * @param offsets Indices to segment boundaries
 * @param null_handling If `null_policy::INCLUDE`, all elements in a segment must be valid for the
 * reduced value to be valid. If `null_policy::EXCLUDE`, the reduced value is valid if any element
 * in the segment is valid.
 * @param stream Used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Output column in device memory
 */

template <typename InputType,
          typename Op,
          CUDF_ENABLE_IF(std::is_same_v<Op, cudf::reduction::op::min> ||
                         std::is_same_v<Op, cudf::reduction::op::max>)>
std::unique_ptr<column> fixed_point_segmented_reduction(
  column_view const& col,
  device_span<size_type const> offsets,
  null_policy null_handling,
  std::optional<std::reference_wrapper<scalar const>> init,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  using RepType = device_storage_type_t<InputType>;
  return simple_segmented_reduction<RepType, RepType, Op>(
    col, offsets, null_handling, init, stream, mr);
}

template <typename InputType,
          typename Op,
          CUDF_ENABLE_IF(!std::is_same_v<Op, cudf::reduction::op::min>() &&
                         !std::is_same_v<Op, cudf::reduction::op::max>())>
std::unique_ptr<column> fixed_point_segmented_reduction(
  column_view const& col,
  device_span<size_type const> offsets,
  null_policy null_handling,
  std::optional<std::reference_wrapper<scalar const>>,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FAIL("Segmented reduction on fixed point column only supports min and max reduction.");
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
  template <typename ElementType, std::enable_if_t<cudf::is_numeric<ElementType>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& col,
                                     device_span<size_type const> offsets,
                                     null_policy null_handling,
                                     std::optional<std::reference_wrapper<scalar const>> init,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return simple_segmented_reduction<ElementType, bool, Op>(
      col, offsets, null_handling, init, stream, mr);
  }

  template <typename ElementType, std::enable_if_t<not cudf::is_numeric<ElementType>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const&,
                                     device_span<size_type const>,
                                     null_policy,
                                     std::optional<std::reference_wrapper<scalar const>>,
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
            CUDF_ENABLE_IF(is_supported<ElementType>() &&
                           !std::is_same_v<ElementType, string_view> &&
                           !cudf::is_fixed_point<ElementType>())>
  std::unique_ptr<column> operator()(column_view const& col,
                                     device_span<size_type const> offsets,
                                     null_policy null_handling,
                                     std::optional<std::reference_wrapper<scalar const>> init,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return simple_segmented_reduction<ElementType, ElementType, Op>(
      col, offsets, null_handling, init, stream, mr);
  }

  template <typename ElementType,
            CUDF_ENABLE_IF(is_supported<ElementType>() && std::is_same_v<ElementType, string_view>)>
  std::unique_ptr<column> operator()(column_view const& col,
                                     device_span<size_type const> offsets,
                                     null_policy null_handling,
                                     std::optional<std::reference_wrapper<scalar const>> init,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    if (init.has_value()) { CUDF_FAIL("Initial value not supported for strings"); }

    return string_segmented_reduction<ElementType, Op>(col, offsets, null_handling, stream, mr);
  }

  template <typename ElementType,
            CUDF_ENABLE_IF(is_supported<ElementType>() && cudf::is_fixed_point<ElementType>())>
  std::unique_ptr<column> operator()(column_view const& col,
                                     device_span<size_type const> offsets,
                                     null_policy null_handling,
                                     std::optional<std::reference_wrapper<scalar const>> init,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return fixed_point_segmented_reduction<ElementType, Op>(
      col, offsets, null_handling, init, stream, mr);
  }

  template <typename ElementType, CUDF_ENABLE_IF(!is_supported<ElementType>())>
  std::unique_ptr<column> operator()(column_view const&,
                                     device_span<size_type const>,
                                     null_policy,
                                     std::optional<std::reference_wrapper<scalar const>>,
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
 * same as the input type, an extra cast operation may occur.
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
                                         device_span<size_type const> offsets,
                                         data_type const output_type,
                                         null_policy null_handling,
                                         std::optional<std::reference_wrapper<scalar const>> init,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
  {
    // TODO: per gh-9988, we should change the compute precision to `output_type`.
    auto result = simple_segmented_reduction<ElementType, double, Op>(
      col, offsets, null_handling, init, stream, mr);
    if (output_type == result->type()) { return result; }
    return cudf::detail::cast(*result, output_type, stream, mr);
  }

  /**
   * @brief Specialization for reducing integer column types to any output type.
   */
  template <typename ElementType,
            typename std::enable_if_t<std::is_integral<ElementType>::value>* = nullptr>
  std::unique_ptr<column> reduce_numeric(column_view const& col,
                                         device_span<size_type const> offsets,
                                         data_type const output_type,
                                         null_policy null_handling,
                                         std::optional<std::reference_wrapper<scalar const>> init,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
  {
    // TODO: per gh-9988, we should change the compute precision to `output_type`.
    auto result = simple_segmented_reduction<ElementType, int64_t, Op>(
      col, offsets, null_handling, init, stream, mr);
    if (output_type == result->type()) { return result; }
    return cudf::detail::cast(*result, output_type, stream, mr);
  }

  /**
   * @brief Called by the type-dispatcher to reduce the input column `col` using
   * the `Op` operation.
   *
   * @tparam ElementType The input column type or key type
   * @param col Input column (must be numeric)
   * @param offsets Indices to segment boundaries
   * @param output_type Requested type of the scalar result
   * @param null_handling If `null_policy::INCLUDE`, all elements in a segment must be valid for the
   * reduced value to be valid. If `null_policy::EXCLUDE`, the reduced value is valid if any element
   * in the segment is valid.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned scalar's device memory
   */
  template <typename ElementType,
            typename std::enable_if_t<cudf::is_numeric<ElementType>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& col,
                                     device_span<size_type const> offsets,
                                     data_type const output_type,
                                     null_policy null_handling,
                                     std::optional<std::reference_wrapper<scalar const>> init,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    if (output_type.id() == cudf::type_to_id<ElementType>()) {
      return simple_segmented_reduction<ElementType, ElementType, Op>(
        col, offsets, null_handling, init, stream, mr);
    }
    // reduce and map to output type
    return reduce_numeric<ElementType>(col, offsets, output_type, null_handling, init, stream, mr);
  }

  template <typename ElementType,
            typename std::enable_if_t<not cudf::is_numeric<ElementType>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const&,
                                     device_span<size_type const>,
                                     data_type const,
                                     null_policy,
                                     std::optional<std::reference_wrapper<scalar const>>,
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
