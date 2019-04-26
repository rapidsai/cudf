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

#include <cudf.h>
#include <bitmask/bit_mask.cuh>
#include <dataframe/device_table.cuh>
#include <groupby.hpp>
#include <hash/concurrent_unordered_map.cuh>
#include <types.hpp>
#include <utilities/device_atomics.cuh>
#include <utilities/release_assert.cuh>
#include <utilities/type_dispatcher.hpp>
#include "new_hash_groupby.hpp"

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/fill.h>
#include <type_traits>
#include <vector>

namespace cudf {
namespace detail {

namespace {

using namespace groupby;

/**---------------------------------------------------------------------------*
 * @brief Determines accumulator type based on input type and operation.
 *
 * @tparam InputType The type of the input to the aggregation operation
 * @tparam op The aggregation operation performed
 * @tparam dummy Dummy for SFINAE
 *---------------------------------------------------------------------------**/
template <typename SourceType, distributive_operators op, typename dummy = void>
struct target_type {
  using type = void;
};

// Computing MIN of SourceType, use SourceType accumulator
template <typename SourceType>
struct target_type<SourceType, distributive_operators::MIN> {
  using type = SourceType;
};

// Computing MAX of SourceType, use SourceType accumulator
template <typename SourceType>
struct target_type<SourceType, distributive_operators::MAX> {
  using type = SourceType;
};

// Always use int64_t accumulator for COUNT
template <typename SourceType>
struct target_type<SourceType, distributive_operators::COUNT> {
  using type = int64_t;
};

// Summing integers of any type, always use int64_t accumulator
template <typename SourceType>
struct target_type<
    SourceType, distributive_operators::SUM,
    typename std::enable_if_t<std::is_integral<SourceType>::value>> {
  using type = int64_t;
};

// Summing float/doubles, use same type accumulator
template <typename SourceType>
struct target_type<
    SourceType, distributive_operators::SUM,
    typename std::enable_if_t<std::is_floating_point<SourceType>::value>> {
  using type = SourceType;
};

/**---------------------------------------------------------------------------*
 * @brief Functor that uses the target_type trait to map the combination of a
 * dispatched SourceType and aggregation operation to required target gdf_dtype.
 *
 *---------------------------------------------------------------------------**/
struct type_mapper {
  template <typename SourceType>
  gdf_dtype operator()(distributive_operators op) {
    switch (op) {
      case distributive_operators::MIN:
        return gdf_dtype_of<typename target_type<
            SourceType, distributive_operators::MIN>::type>();
      case distributive_operators::MAX:
        return gdf_dtype_of<typename target_type<
            SourceType, distributive_operators::MAX>::type>();
      case distributive_operators::SUM:
        return gdf_dtype_of<typename target_type<
            SourceType, distributive_operators::SUM>::type>();
      case distributive_operators::COUNT:
        return gdf_dtype_of<typename target_type<
            SourceType, distributive_operators::COUNT>::type>();
      default:
        return GDF_invalid;
    }
  }
};

/**---------------------------------------------------------------------------*
 * @brief Deteremines target gdf_dtypes to use for combinations of source
 * gdf_dtypes and aggregation operations.
 *
 * Given vectors of source gdf_dtypes and corresponding aggregation operations
 * to be performed on that type, returns a vector the gdf_dtypes to use to store
 * the result of the aggregation operations.
 *
 * @param source_dtypes The source types
 * @param op The aggregation operations
 * @return Target gdf_dtypes to use for the target aggregation columns
 *---------------------------------------------------------------------------**/
std::vector<gdf_dtype> target_dtypes(
    std::vector<gdf_dtype> const& source_dtypes,
    std::vector<distributive_operators> const& operators) {
  std::vector<gdf_dtype> output_dtypes(source_dtypes.size());

  std::transform(
      source_dtypes.begin(), source_dtypes.end(), operators.begin(),
      output_dtypes.begin(),
      [](gdf_dtype source_dtype, distributive_operators op) {
        gdf_dtype t = cudf::type_dispatcher(source_dtype, type_mapper{}, op);
        CUDF_EXPECTS(
            t != GDF_invalid,
            "Invalid combination of input type and aggregation operation.");
        return t;
      });

  return output_dtypes;
}

template <distributive_operators op>
struct corresponding_functor {
  using type = void;
};

template <>
struct corresponding_functor<distributive_operators::MIN> {
  using type = DeviceMin;
};

template <>
struct corresponding_functor<distributive_operators::MAX> {
  using type = DeviceMax;
};

template <>
struct corresponding_functor<distributive_operators::SUM> {
  using type = DeviceSum;
};

template <>
struct corresponding_functor<distributive_operators::COUNT> {
  using type = DeviceSum;
};

/**---------------------------------------------------------------------------*
 * @brief Performs inplace update of a target element via a binary operation
 * with a source element.
 *
 * Atomically performs `target[target_index] = target[target_index] op
 * source[source_index]`
 *
 * @tparam TargetType Type of the target element
 * @tparam SourceType Type of the source element
 * @tparam Op Type of the binary operation to perform
 * @param target Column containing target element
 * @param target_index Index of the target element
 * @param source Column containing source element
 * @param source_index Index of the source element
 * @param op The aggregation operation to perform
 *---------------------------------------------------------------------------**/
// template <typename TargetType, typename SourceType, typename Op,
//          std::enable_if_t<not std::is_void<TargetType>::value, int>* =
//          nullptr>
//__device__ inline void binary_op(gdf_column const& target,
//                                 gdf_size_type target_index,
//                                 gdf_column const& source,
//                                 gdf_size_type source_index, Op&& op) {
//  assert(gdf_dtype_of<TargetType>() == target.dtype);
//
//  SourceType const& source_element{
//      static_cast<SourceType const*>(source.data)[source_index]};
//
//  cudf::genericAtomicOperation(
//      &(static_cast<TargetType*>(target.data)[target_index]),
//      static_cast<TargetType>(source_element), op);
//}
//
///**---------------------------------------------------------------------------*
// * @brief Increments a target value by one.
// *
// * @tparam TargetType Target element's type
// * @param target The column containing the target element
// * @param target_index Index of the target element
// *---------------------------------------------------------------------------**/
// template <typename TargetType>
//__device__ inline void count_op(gdf_column const& target,
//                                gdf_size_type target_index) {
//  static_assert(std::is_integral<TargetType>::value,
//                "TargetType of count operation must be integral.");
//  assert(gdf_dtype_of<TargetType>() == target.dtype);
//  cudf::genericAtomicOperation(
//      &(static_cast<TargetType*>(target.data)[target_index]), TargetType{1},
//      DeviceSum{});
//}

template <typename SourceType, distributive_operators op,
          std::enable_if_t<
              std::is_void<typename target_type<SourceType, op>::type>::value,
              int>* = nullptr>
__device__ inline void update_target(gdf_column const& target,
                                     gdf_size_type target_index,
                                     gdf_column const& source,
                                     gdf_size_type source_index) {
  release_assert(false && "Invalid Source type and Aggregation combination.");
}

template <
    typename SourceType, distributive_operators op,
    std::enable_if_t<
        not std::is_void<typename target_type<SourceType, op>::type>::value,
        int>* = nullptr>
__device__ inline void update_target(gdf_column const& target,
                                     gdf_size_type target_index,
                                     gdf_column const& source,
                                     gdf_size_type source_index) {

  using TargetType = typename target_type<SourceType, op>::type;
  assert(gdf_dtype_of<TargetType>() == target.dtype);

  TargetType* const __restrict__ target_data{
      static_cast<TargetType*>(target.data)};
  SourceType const* const __restrict__ source_data{
      static_cast<SourceType const*>(source.data)};
  SourceType const& source_element{source_data[source_index]};

  // Target element is NULL
  if (not gdf_is_valid(target.valid, target_index)) {
    TargetType const expected = target_data[target_index];

    TargetType const actual =
        atomicCAS(&target_data[target_index], expected,
                  static_cast<TargetType>(source_element));

    if (expected == actual) {
      bit_mask::set_bit_safe(
          reinterpret_cast<bit_mask::bit_mask_t*>(target.valid), target_index);
      return;
    }
  }

  using FunctorType = typename corresponding_functor<op>::type;

  cudf::genericAtomicOperation(&target_data[target_index],
                               static_cast<TargetType>(source_element),
                               FunctorType{});
}

struct elementwise_aggregator {
  template <typename SourceType>
  __device__ inline void operator()(gdf_column const& target,
                                    gdf_size_type target_index,
                                    gdf_column const& source,
                                    gdf_size_type source_index,
                                    distributive_operators op) {
    // TODO Can we avoid setting the target's valid bit for every binary
    // operation? Technically, it only needs to be set upon the first succesful
    // update of the target element.
    switch (op) {
      case distributive_operators::MIN: {
        update_target<SourceType, distributive_operators::MIN>(
            target, target_index, source, source_index);
        break;
      }
      case distributive_operators::MAX: {
        update_target<SourceType, distributive_operators::MAX>(
            target, target_index, source, source_index);
        break;
      }
      case distributive_operators::SUM: {
        update_target<SourceType, distributive_operators::SUM>(
            target, target_index, source, source_index);
        break;
      }
      case distributive_operators::COUNT: {
        // using TargetType =
        //    typename target_type<SourceType,
        //                         distributive_operators::COUNT>::type;
        // update_target<TargetType, SourceType>(target, target_index, source,
        //                                      source_index, DeviceCount{});
        // break;
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
 * @note It is assumed that the target element of a COUNT operation is *never*
 * NULL.
 *
 * @param target Table containing the target row
 * @param target_index Index of the target row
 * @param source Table cotaning the source row
 * @param source_index Index of the source row
 * @param ops Array of operators to perform between the elements of the
 * target and source rows
 *---------------------------------------------------------------------------**/
__device__ inline void aggregate_row(device_table const& target,
                                     gdf_size_type target_index,
                                     device_table const& source,
                                     gdf_size_type source_index,
                                     distributive_operators* ops) {
  thrust::for_each(
      thrust::seq, thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(target.num_columns()),
      [target, target_index, source, source_index, ops](gdf_size_type i) {
        if (gdf_is_valid(source.get_column(i)->valid, source_index)) {
          cudf::type_dispatcher(source.get_column(i)->dtype,
                                elementwise_aggregator{}, *target.get_column(i),
                                target_index, *source.get_column(i),
                                source_index, ops[i]);
        }
      });
}

}  // namespace

std::tuple<cudf::table, cudf::table> hash_groupby(
    cudf::table const& keys, cudf::table const& values,
    std::vector<cudf::groupby::distributive_operators> const& operators,
    groupby::Options options, cudaStream_t stream) {
  // The exact output size is unknown a priori, therefore, use the input size as
  // an upper bound
  gdf_size_type const output_size_estimate{keys.num_rows()};

  cudf::table output_keys{output_size_estimate, column_dtypes(keys), true,
                          stream};
  cudf::table output_values{output_size_estimate,
                            target_dtypes(column_dtypes(values), operators),
                            true, stream};

  using map_type = concurrent_unordered_map<
      gdf_size_type, gdf_size_type, std::numeric_limits<gdf_size_type>::max(),
      default_hash<gdf_size_type>, equal_to<gdf_size_type>,
      legacy_allocator<thrust::pair<gdf_size_type, gdf_size_type>>>;

  std::unique_ptr<map_type> map =
      std::make_unique<map_type>(compute_hash_table_size(keys.num_rows()), 0);

  rmm::device_vector<groupby::distributive_operators> d_operators(operators);

  CHECK_STREAM(stream);

  return std::make_tuple(output_keys, output_values);
}

}  // namespace detail
}  // namespace cudf
