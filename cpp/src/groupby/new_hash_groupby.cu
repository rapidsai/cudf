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

struct identity_initializer {
  template <typename T>
  T get_identity(distributive_operators op) {
    switch (op) {
      case distributive_operators::SUM:
        return cudf::DeviceSum::identity<T>();
      case distributive_operators::MIN:
        return cudf::DeviceMin::identity<T>();
      case distributive_operators::MAX:
        return cudf::DeviceMax::identity<T>();
      case distributive_operators::COUNT:
        return cudf::DeviceSum::identity<T>();
      default:
        CUDF_FAIL("Invalid aggregation operation.");
    }
  }

  template <typename T>
  void operator()(gdf_column const& col, distributive_operators op,
                  cudaStream_t stream = 0) {
    T* typed_data = static_cast<T*>(col.data);
    thrust::fill(rmm::exec_policy(stream)->on(stream), typed_data,
                 typed_data + col.size, get_identity<T>(op));

    // For COUNT operator, initialize column's bitmask to be all valid
    if ((nullptr != col.valid) and (distributive_operators::COUNT == op)) {
      CUDA_TRY(cudaMemsetAsync(
          col.valid, 0xff,
          sizeof(gdf_valid_type) * gdf_valid_allocation_size(col.size),
          stream));
    }
  }
};

/**---------------------------------------------------------------------------*
 * @brief Initializes each column in a table with a corresponding identity value
 * of an aggregation operation.
 *
 * The `i`th column will be initialized with the identity value of the `i`th
 * aggregation operation.
 *
 * @note The validity bitmask for the column corresponding to a COUNT operator
 * will be initialized to all valid.
 *
 * @param table The table of columns to initialize.
 * @param operators The aggregation operations whose identity values will be
 *used to initialize the columns.
 *---------------------------------------------------------------------------**/
void initialize_with_identity(
    cudf::table const& table,
    std::vector<distributive_operators> const& operators,
    cudaStream_t stream = 0) {
  // TODO: Initialize all the columns in a single kernel instead of invoking one
  // kernel per column
  for (gdf_size_type i = 0; i < table.num_columns(); ++i) {
    gdf_column const* col = table.get_column(i);
    cudf::type_dispatcher(col->dtype, identity_initializer{}, *col,
                          operators[i]);
  }
}

/**---------------------------------------------------------------------------*
 * @brief Determines accumulator type based on input type and operation.
 *
 * @tparam InputType The type of the input to the aggregation operation
 * @tparam op The aggregation operation performed
 * @tparam dummy Dummy for SFINAE
 *---------------------------------------------------------------------------**/
template <typename SourceType, distributive_operators op, typename dummy = void>
struct result_type {
  using type = void;
};

// Computing MIN of SourceType, use SourceType accumulator
template <typename SourceType>
struct result_type<SourceType, distributive_operators::MIN> {
  using type = SourceType;
};

// Computing MAX of SourceType, use SourceType accumulator
template <typename SourceType>
struct result_type<SourceType, distributive_operators::MAX> {
  using type = SourceType;
};

// Always use int64_t accumulator for COUNT
template <typename SourceType>
struct result_type<SourceType, distributive_operators::COUNT> {
  using type = int64_t;
};

// Summing integers of any type, always use int64_t accumulator
template <typename SourceType>
struct result_type<
    SourceType, distributive_operators::SUM,
    typename std::enable_if_t<std::is_integral<SourceType>::value>> {
  using type = int64_t;
};

// Summing float/doubles, use same type accumulator
template <typename SourceType>
struct result_type<
    SourceType, distributive_operators::SUM,
    typename std::enable_if_t<std::is_floating_point<SourceType>::value>> {
  using type = SourceType;
};

/**---------------------------------------------------------------------------*
 * @brief Error case for invalid combinations of SourceType and operator.
 *
 * For an invalid combination of SourceType and operator,
 * `result_type<SourceType, operator>::type` yields a `void` TargetType. This
 * specialization will be invoked for any invalid combination and cause a
 * runtime error.
 *---------------------------------------------------------------------------**/
template <typename TargetType, typename SourceType, typename Op,
          std::enable_if_t<std::is_void<TargetType>::value, int>* = nullptr>
__device__ inline void binary_op(gdf_column const& target,
                                 gdf_size_type target_index,
                                 gdf_column const& source,
                                 gdf_size_type source_index, Op&& op) {
  release_assert(false && "Invalid Source type and Aggregation combination.");
}

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
template <typename TargetType, typename SourceType, typename Op,
          std::enable_if_t<not std::is_void<TargetType>::value, int>* = nullptr>
__device__ inline void binary_op(gdf_column const& target,
                                 gdf_size_type target_index,
                                 gdf_column const& source,
                                 gdf_size_type source_index, Op&& op) {
  assert(gdf_dtype_of<TargetType>() == target.dtype);

  SourceType const& source_element{
      static_cast<SourceType const*>(source.data)[source_index]};

  cudf::genericAtomicOperation(
      &(static_cast<TargetType*>(target.data)[target_index]),
      static_cast<TargetType>(source_element), op);
}

/**---------------------------------------------------------------------------*
 * @brief Increments a target value by one.
 *
 * @tparam TargetType Target element's type
 * @param target The column containing the target element
 * @param target_index Index of the target element
 *---------------------------------------------------------------------------**/
template <typename TargetType>
__device__ inline void count_op(gdf_column const& target,
                                gdf_size_type target_index) {
  static_assert(std::is_integral<TargetType>::value,
                "TargetType of count operation must be integral.");
  assert(gdf_dtype_of<TargetType>() == target.dtype);
  cudf::genericAtomicOperation(
      &(static_cast<TargetType*>(target.data)[target_index]), TargetType{1},
      DeviceSum{});
}

/**---------------------------------------------------------------------------*
 * @brief Sets the specified bit in a column's validity bitmask.
 *
 * @note Setting a bit invokes an atomic operation. Therefore, to avoid
 * unnecessary/expensive atomic operations, the bit is only set if it is not
 * already set.
 *
 * @param target Column whose bitmask will be set
 * @param target_index Index of the bit to set
 *---------------------------------------------------------------------------**/
__device__ inline void set_valid_bit(gdf_column const& target,
                                     gdf_size_type target_index) {
  if (not gdf_is_valid(target.valid, target_index)) {
    bit_mask::set_bit_safe(
        reinterpret_cast<bit_mask::bit_mask_t*>(target.valid), target_index);
  }
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
        using TargetType =
            typename result_type<SourceType, distributive_operators::MIN>::type;
        binary_op<TargetType, SourceType>(target, target_index, source,
                                          source_index, DeviceMin{});
        set_valid_bit(target, target_index);
        break;
      }
      case distributive_operators::MAX: {
        using TargetType =
            typename result_type<SourceType, distributive_operators::MAX>::type;
        binary_op<TargetType, SourceType>(target, target_index, source,
                                          source_index, DeviceMax{});
        set_valid_bit(target, target_index);
        break;
      }
      case distributive_operators::SUM: {
        using TargetType =
            typename result_type<SourceType, distributive_operators::SUM>::type;
        binary_op<TargetType, SourceType>(target, target_index, source,
                                          source_index, DeviceSum{});
        set_valid_bit(target, target_index);
        break;
      }
      case distributive_operators::COUNT: {
        using TargetType =
            typename result_type<SourceType,
                                 distributive_operators::COUNT>::type;
        count_op<TargetType>(target, target_index);
        break;
      }
      default:
        return;
    }
  }
};  // namespace

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

struct type_mapper {
  template <typename InputT>
  gdf_dtype operator()(distributive_operators op) {
    switch (op) {
      case distributive_operators::MIN:
        return gdf_dtype_of<
            typename result_type<InputT, distributive_operators::MIN>::type>();
      case distributive_operators::MAX:
        return gdf_dtype_of<
            typename result_type<InputT, distributive_operators::MAX>::type>();
      case distributive_operators::SUM:
        return gdf_dtype_of<
            typename result_type<InputT, distributive_operators::SUM>::type>();
      case distributive_operators::COUNT:
        return gdf_dtype_of<typename result_type<
            InputT, distributive_operators::COUNT>::type>();
      default:
        return GDF_invalid;
    }
  }
};

/**---------------------------------------------------------------------------*
 * @brief Returns the output gdf_dtype to use for a combination of input
 * gdf_dtype and aggregation operation.
 *
 * @param input_type The type of the input aggregation column
 * @param op The aggregation operation
 * @return gdf_dtype Type to use for output aggregation column
 *---------------------------------------------------------------------------**/
gdf_dtype output_dtype(gdf_dtype input_type, distributive_operators op) {
  return cudf::type_dispatcher(input_type, type_mapper{}, op);
}
}  // namespace

std::tuple<cudf::table, cudf::table> hash_groupby(
    cudf::table const& keys, cudf::table const& values,
    std::vector<cudf::groupby::distributive_operators> const& operators,
    groupby::Options options, cudaStream_t stream) {
  // The exact output size is unknown a priori, therefore, use the input size as
  // an upper bound
  gdf_size_type const output_size{keys.num_rows()};

  // Allocate output keys
  std::vector<gdf_dtype> key_dtypes(keys.num_columns());
  std::transform(keys.begin(), keys.end(), key_dtypes.begin(),
                 [](gdf_column const* col) { return col->dtype; });
  cudf::table output_keys{output_size, key_dtypes, true, stream};

  // Allocate/initialize output values
  // TODO: Move to function.
  std::vector<gdf_dtype> output_dtypes(values.num_columns());
  std::transform(
      values.begin(), values.end(), operators.begin(), output_dtypes.begin(),
      [](gdf_column const* input_col, groupby::distributive_operators op) {
        gdf_dtype t = output_dtype(input_col->dtype, op);
        CUDF_EXPECTS(
            t != GDF_invalid,
            "Invalid combination of input type and aggregation operation.");
        return t;
      });
  cudf::table output_values{output_size, output_dtypes, true, stream};
  initialize_with_identity(output_values, operators, stream);

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
