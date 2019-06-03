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
#ifndef GROUPBY_KERNELS_CUH
#define GROUPBY_KERNELS_CUH

#include <groupby.hpp>
#include "type_info.hpp"

namespace cudf {
namespace groupby {
namespace hash {
/**---------------------------------------------------------------------------*
 * @brief Base case for invalid SourceType and op combinations.
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
 *---------------------------------------------------------------------------**/
template <typename SourceType, operators op, bool values_have_nulls>
struct update_target_element<
    SourceType, op, values_have_nulls,
    std::enable_if_t<not std::is_void<target_type_t<SourceType, op>>::value>> {
  /**---------------------------------------------------------------------------*
   * @brief Performs in-place update of a target element via a binary operation
   * with a source element.
   *
   * @note It is assumed the source element is not NULL, i.e., a NULL source
   * element should be detected before calling this function.
   *
   * @note It is assumed the target column is always nullable, i.e., has a valid
   * bitmask allocation.
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

cudf::genericAtomicOperation(
        &target_data[target_index], static_cast<TargetType>(source_element),
        FunctorType{});


    bit_mask::bit_mask_t* const __restrict__ target_mask{
        reinterpret_cast<bit_mask::bit_mask_t*>(target.valid)};

    if (values_have_nulls) {
      if (not bit_mask::is_valid(target_mask, target_index)) {
        bit_mask::set_bit_safe(target_mask, target_index);
      }
    }
  }
};

/**---------------------------------------------------------------------------*
 * @brief Specialization for COUNT.
 *---------------------------------------------------------------------------**/
template <typename SourceType, bool values_have_nulls>
struct update_target_element<SourceType, COUNT, values_have_nulls,
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

template <bool values_have_nulls>
struct elementwise_aggregator {
  template <typename SourceType>
  __device__ inline void operator()(gdf_column const& target,
                                    gdf_size_type target_index,
                                    gdf_column const& source,
                                    gdf_size_type source_index, operators op) {
    switch (op) {
      case MIN: {
        update_target_element<SourceType, MIN, values_have_nulls>{}(
            target, target_index, source, source_index);
        break;
      }
      case MAX: {
        update_target_element<SourceType, MAX, values_have_nulls>{}(
            target, target_index, source, source_index);
        break;
      }
      case SUM: {
        update_target_element<SourceType, SUM, values_have_nulls>{}(
            target, target_index, source, source_index);
        break;
      }
      case COUNT: {
        update_target_element<SourceType, COUNT, values_have_nulls>{}(
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
  thrust::for_each(
      thrust::seq, thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(target.num_columns()),
      [target, target_index, source, source_index, ops](gdf_size_type i) {
        bit_mask_t* const __restrict__ source_mask{
            reinterpret_cast<bit_mask_t*>(source.get_column(i)->valid)};

        if (values_have_nulls and nullptr != source_mask and
            not is_valid(source_mask, source_index)) {
          return;
        }

        cudf::type_dispatcher(source.get_column(i)->dtype,
                              elementwise_aggregator<values_have_nulls>{},
                              *target.get_column(i), target_index,
                              *source.get_column(i), source_index, ops[i]);
      });
}

template <bool nullable = true>
struct row_hasher {
  using result_type = hash_value_type;  // TODO Remove when aggregating
                                        // map::insert function is removed
  device_table table;
  row_hasher(device_table const& t) : table{t} {}

  __device__ auto operator()(gdf_size_type row_index) const {
    return hash_row<nullable>(table, row_index);
  }
};

template <bool skip_rows_with_nulls, bool values_have_nulls, typename Map>
__global__ void build_aggregation_map(
    Map* map, device_table input_keys, device_table input_values,
    device_table output_values, operators* ops,
    bit_mask::bit_mask_t const* const __restrict__ row_bitmask) {
  gdf_size_type i = threadIdx.x + blockIdx.x * blockDim.x;

  while (i < input_keys.num_rows()) {
    if (skip_rows_with_nulls and not bit_mask::is_valid(row_bitmask, i)) {
      i += blockDim.x * gridDim.x;
      continue;
    }

    auto result = map->insert(thrust::make_pair(i, i));

    aggregate_row<values_have_nulls>(output_values, result.first->second,
                                     input_values, i, ops);
    i += blockDim.x * gridDim.x;
  }
}

template <bool keys_have_nulls, bool values_have_nulls, typename Map>
__global__ void extract_groupby_result(Map* map, device_table const input_keys,
                                       device_table output_keys,
                                       device_table const sparse_output_values,
                                       device_table dense_output_values,
                                       gdf_size_type* output_write_index) {
  gdf_size_type i = threadIdx.x + blockIdx.x * blockDim.x;

  using pair_type = typename Map::value_type;

  pair_type const* const __restrict__ table_pairs{map->data()};

  while (i < map->capacity()) {
    gdf_size_type source_key_row_index;
    gdf_size_type source_value_row_index;

    // The way the aggregation map is built, these two indices will always be
    // equal, but lets be generic just in case that ever changes.
    thrust::tie(source_key_row_index, source_value_row_index) = table_pairs[i];

    if (source_key_row_index != map->get_unused_key()) {
      auto output_index = atomicAdd(output_write_index, 1);


      copy_row<keys_have_nulls>(output_keys, output_index, input_keys,
                                source_key_row_index);

      copy_row<values_have_nulls>(dense_output_values, output_index,
                                  sparse_output_values, source_value_row_index);
    }
    i += gridDim.x * blockDim.x;
  }
}

}  // namespace hash
}  // namespace groupby
}  // namespace cudf

#endif
