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

#ifndef _GROUPBY_KERNEL_UTILS_H
#define _GROUPBY_KERNEL_UTILS_H

namespace cudf {
namespace groupby {

/**---------------------------------------------------------------------------*
 * @brief This functor is used by elementwise_aggregator to do in-place update
 * operations.
 * Base case for invalid SourceType and op combinations.
 *
 * For an invalid combination of SourceType and operator,
 *`target_type_t<SourceType, operator>` yields a `void` TargetType. This
 * specialization will be invoked for any invalid combination and cause a
 * runtime error.
 *
 * @note A struct is used instead of a function to allow for partial
 * specialization.
 *---------------------------------------------------------------------------**/
template <typename SourceType, operators op, bool values_have_nulls,
          typename Enable = void>
struct update_target_element {
  __device__ inline void operator()(gdf_column const& target,
                                    gdf_size_type target_index,
                                    gdf_column const& source,
                                    gdf_size_type source_index) {
    release_assert(false && "Invalid Source type and Aggregation combination.");
  }
};

/**---------------------------------------------------------------------------*
 * @brief Specialization for valid SourceType and op combinations.
 *
 * @tparam SourceType Type of the source element
 * @tparam op The operation to perform
 * @tparama values_have_nulls Indicates the potential for null values in the
 * source
 *---------------------------------------------------------------------------**/
template <typename SourceType, operators op, bool source_has_nulls>
struct update_target_element<
    SourceType, op, source_has_nulls,
    std::enable_if_t<not std::is_void<target_type_t<SourceType, op>>::value>> {
  /**---------------------------------------------------------------------------*
   * @brief Performs in-place update of a target element via a binary operation
   * with a source element.
   *
   * @note It is assumed the source element is not NULL, i.e., a NULL source
   * element should be detected before calling this function.
   *
   * @note If `source_has_nulls==true`, it is assumed that `target` is nullable
   *
   * If the target element is NULL, it is assumed that the target element was
   * initialized with the identity of the aggregation operation. The target is
   * updated with the result of the aggregation with the source element, and the
   * target column's bitmask is updated to indicate the target element is no
   * longer NULL.
   *
   * @param target Column containing target element
   * @param target_index Index of target element
   * @param source Column containing source element
   * @param source_index Index of source element
   *---------------------------------------------------------------------------**/
  __device__ inline void operator()(gdf_column const& target,
                                    gdf_size_type target_index,
                                    gdf_column const& source,
                                    gdf_size_type source_index) {
    using TargetType = target_type_t<SourceType, op>;
    
    assert(gdf_dtype_of<TargetType>() == target.dtype); 
    
    TargetType* const __restrict__ target_data{
        static_cast<TargetType*>(target.data)};
    SourceType const* const __restrict__ source_data{
        static_cast<SourceType const*>(source.data)};

    SourceType const source_element{source_data[source_index]};

    using FunctorType = corresponding_functor_t<op>;

    cudf::genericAtomicOperation(&target_data[target_index],
                                 static_cast<TargetType>(source_element),
                                 FunctorType{});

    bit_mask::bit_mask_t* const __restrict__ target_mask{
        reinterpret_cast<bit_mask::bit_mask_t*>(target.valid)};

    if (source_has_nulls) {
      if (not bit_mask::is_valid(target_mask, target_index)) {
        bit_mask::set_bit_safe(target_mask, target_index);
      }
    }
  }
};

/**---------------------------------------------------------------------------*
 * @brief Specialization for COUNT.
 *---------------------------------------------------------------------------**/
template <typename SourceType, bool source_has_nulls>
struct update_target_element<SourceType, COUNT, source_has_nulls,
                             std::enable_if_t<not std::is_void<
                                 target_type_t<SourceType, COUNT>>::value>> {
  /**---------------------------------------------------------------------------*
   * @brief Increments the target_element by 1.
   *
   * @note Assumes the target element is never NULL, and was intialized to 0.
   *
   * @param target Column containing target element
   * @param target_index Index of target element
   *---------------------------------------------------------------------------**/
  __device__ inline void operator()(gdf_column const& target,
                                    gdf_size_type target_index,
                                    gdf_column const&, gdf_size_type) {
    using TargetType = target_type_t<SourceType, COUNT>;
    assert(gdf_dtype_of<TargetType>() == target.dtype);

    TargetType* const __restrict__ target_data{
        static_cast<TargetType*>(target.data)};

    cudf::genericAtomicOperation(&target_data[target_index], TargetType{1},
                                 DeviceSum{});
  }
};

template <bool source_has_nulls>
struct elementwise_aggregator {
  template <typename SourceType>
  __device__ inline void operator()(gdf_column const& target,
                                    gdf_size_type target_index,
                                    gdf_column const& source,
                                    gdf_size_type source_index, operators op) {
    switch (op) {
      case MIN: {
        update_target_element<SourceType, MIN, source_has_nulls>{}(
            target, target_index, source, source_index);
        break;
      }
      case MAX: {
        update_target_element<SourceType, MAX, source_has_nulls>{}(
            target, target_index, source, source_index);
        break;
      }
      case SUM: {
        update_target_element<SourceType, SUM, source_has_nulls>{}(
            target, target_index, source, source_index);
        break;
      }
      case COUNT: {
        update_target_element<SourceType, COUNT, source_has_nulls>{}(
            target, target_index, source, source_index);
      }
      default:
        return;
    }
  }
};

/**---------------------------------------------------------------------------*
 * @brief Performs an in-place update by performing elementwise aggregation
 * operations between a target and source row.
 *
 * For `i` in `[0, num_columns)`, each element in the target row is updated as:
 *
 *```
 * target_row[i] = target_row[i] op[i] source_row[i]
 *```
 * @note If a source element is NULL, the aggregation operation for
 * that column is skipped.
 *
 * @note If a target element is NULL, it is assumed that the value of the NULL
 * element is the identity value of the aggregation operation being performed.
 * The aggregation operation is performed between the source element and the
 * identity value, and the target element's bit is set to indicate it is no
 * longer NULL.
 *
 * @note For COUNT, it is assumed the target element can *never* be NULL. As
 * such, it is expected the target element's bit is already set.
 *
 * @param target Table containing the target row
 * @param target_index Index of the target row
 * @param source Table cotaning the source row
 * @param source_index Index of the source row
 * @param ops Array of operators to perform between the elements of the
 * target and source rows
 *---------------------------------------------------------------------------**/
template <bool values_have_nulls = true>
__device__ inline void aggregate_row(device_table const& target,
                                     gdf_size_type target_index,
                                     device_table const& source,
                                     gdf_size_type source_index,
                                     operators* ops) {
  using namespace bit_mask;
  for (gdf_size_type i = 0; i < target.num_columns(); ++i) {
    bit_mask_t const* const __restrict__ source_mask{
        reinterpret_cast<bit_mask_t const*>(source.get_column(i)->valid)};

    if (values_have_nulls and nullptr != source_mask and
        not is_valid(source_mask, source_index)) {
      continue;
    }

    cudf::type_dispatcher(source.get_column(i)->dtype,
                          elementwise_aggregator<values_have_nulls>{},
                          *target.get_column(i), target_index,
                          *source.get_column(i), source_index, ops[i]);
  }
} 

}  // namespace groupby
}  // namespace cudf
#endif // _GROUPBY_KERNEL_UTILS_H
