/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "jit/cache.hpp"
#include "jit/parser.hpp"
#include "jit/util.hpp"
#include "lead_lag_nested.cuh"
#include "nth_element.cuh"
#include "reductions/nested_type_minmax_util.cuh"
#include "rolling.hpp"
#include "rolling_collect_list.cuh"
#include "rolling_jit.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/lists/detail/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/climits>
#include <cuda/std/limits>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>

#include <jit_preprocessed_files/rolling/jit/kernel.cu.jit.hpp>

#include <memory>

namespace cudf {

namespace detail {

/// Helper function to materialize preceding/following offsets.
template <typename Calculator>
std::unique_ptr<column> expand_to_column(Calculator const& calc,
                                         size_type const& num_rows,
                                         rmm::cuda_stream_view stream)
{
  auto window_column = cudf::make_numeric_column(
    cudf::data_type{type_to_id<size_type>()}, num_rows, cudf::mask_state::UNALLOCATED, stream);

  auto begin = cudf::detail::make_counting_transform_iterator(0, calc);

  thrust::copy_n(
    rmm::exec_policy(stream), begin, num_rows, window_column->mutable_view().data<size_type>());

  return window_column;
}

/**
 * @brief Operator for applying a generic (non-specialized) rolling aggregation on a single window.
 */
template <typename InputType, aggregation::Kind op>
struct DeviceRolling {
  size_type min_periods;

  // what operations do we support
  template <typename T = InputType, aggregation::Kind O = op>
  static constexpr bool is_supported()
  {
    return cudf::detail::is_valid_aggregation<T, O>() && has_corresponding_operator<O>() &&
           // MIN/MAX only supports fixed width types
           (((O == aggregation::MIN || O == aggregation::MAX) && cudf::is_fixed_width<T>()) ||
            (O == aggregation::SUM) || (O == aggregation::MEAN));
  }

  // operations we do support
  template <typename T = InputType, aggregation::Kind O = op>
  explicit DeviceRolling(size_type _min_periods, std::enable_if_t<is_supported<T, O>()>* = nullptr)
    : min_periods(_min_periods)
  {
  }

  // operations we don't support
  template <typename T = InputType, aggregation::Kind O = op>
  explicit DeviceRolling(size_type _min_periods, std::enable_if_t<!is_supported<T, O>()>* = nullptr)
    : min_periods(_min_periods)
  {
    CUDF_FAIL("Invalid aggregation/type pair");
  }

  // perform the windowing operation
  template <typename OutputType, bool has_nulls>
  bool __device__ operator()(column_device_view const& input,
                             column_device_view const&,
                             mutable_column_device_view& output,
                             size_type start_index,
                             size_type end_index,
                             size_type current_index) const
  {
    using AggOp = typename corresponding_operator<op>::type;
    AggOp agg_op;

    cudf::size_type count = 0;
    OutputType val        = AggOp::template identity<OutputType>();

    for (size_type j = start_index; j < end_index; j++) {
      if (!has_nulls || input.is_valid(j)) {
        OutputType element = input.element<device_storage_type_t<InputType>>(j);
        val                = agg_op(element, val);
        count++;
      }
    }

    bool output_is_valid = (count >= min_periods);

    // store the output value, one per thread
    cudf::detail::rolling_store_output_functor<OutputType, op == aggregation::MEAN>{}(
      output.element<OutputType>(current_index), val, count);

    return output_is_valid;
  }
};

/**
 * @brief The base struct used for checking if the combination of input type and aggregation op is
 * supported.
 */
template <typename InputType, aggregation::Kind op>
struct DeviceRollingArgMinMaxBase {
  size_type min_periods;
  explicit DeviceRollingArgMinMaxBase(size_type _min_periods) : min_periods(_min_periods) {}

  static constexpr bool is_supported()
  {
    // Right now only support ARGMIN/ARGMAX of strings and structs.
    auto const type_supported =
      std::is_same_v<InputType, cudf::string_view> || std::is_same_v<InputType, cudf::struct_view>;
    auto const op_supported = op == aggregation::Kind::ARGMIN || op == aggregation::Kind::ARGMAX;

    return type_supported && op_supported;
  }
};

/**
 * @brief Operator for applying an ARGMAX/ARGMIN rolling aggregation on a single window for string.
 */
template <aggregation::Kind op>
struct DeviceRollingArgMinMaxString : DeviceRollingArgMinMaxBase<cudf::string_view, op> {
  explicit DeviceRollingArgMinMaxString(size_type _min_periods)
    : DeviceRollingArgMinMaxBase<cudf::string_view, op>(_min_periods)
  {
  }
  using DeviceRollingArgMinMaxBase<cudf::string_view, op>::min_periods;

  template <typename OutputType, bool has_nulls>
  bool __device__ operator()(column_device_view const& input,
                             column_device_view const&,
                             mutable_column_device_view& output,
                             size_type start_index,
                             size_type end_index,
                             size_type current_index)
  {
    auto constexpr default_output = (op == aggregation::ARGMIN) ? ARGMIN_SENTINEL : ARGMAX_SENTINEL;

    using InputType = cudf::string_view;
    using AggOp     = typename corresponding_operator<op>::type;
    AggOp agg_op;

    cudf::size_type count = 0;
    InputType val         = AggOp::template identity<InputType>();
    OutputType val_index  = default_output;

    for (size_type j = start_index; j < end_index; j++) {
      if (!has_nulls || input.is_valid(j)) {
        InputType element = input.element<InputType>(j);
        val               = agg_op(element, val);
        if (val == element) { val_index = j; }
        count++;
      }
    }

    bool output_is_valid = (count >= min_periods);
    // Use the sentinel value (i.e., -1) for the output will help identify null elements while
    // gathering for Min and Max.
    output.element<OutputType>(current_index) = output_is_valid ? val_index : default_output;

    // The gather mask shouldn't contain null values, so
    // always return zero
    return true;
  }
};

/**
 * @brief Operator for applying an ARGMAX/ARGMIN rolling aggregation on a single window for struct.
 */
template <aggregation::Kind op, typename Comparator>
struct DeviceRollingArgMinMaxStruct : DeviceRollingArgMinMaxBase<cudf::struct_view, op> {
  DeviceRollingArgMinMaxStruct(size_type _min_periods, Comparator const& _comp)
    : DeviceRollingArgMinMaxBase<cudf::struct_view, op>(_min_periods), comp(_comp)
  {
  }
  using DeviceRollingArgMinMaxBase<cudf::struct_view, op>::min_periods;
  Comparator comp;

  template <typename OutputType, bool has_nulls>
  bool __device__ operator()(column_device_view const& input,
                             column_device_view const&,
                             mutable_column_device_view& output,
                             size_type start_index,
                             size_type end_index,
                             size_type current_index)
  {
    auto constexpr default_output = (op == aggregation::ARGMIN) ? ARGMIN_SENTINEL : ARGMAX_SENTINEL;

    auto const valid_count =
      has_nulls ? thrust::count_if(thrust::seq,
                                   thrust::make_counting_iterator(start_index),
                                   thrust::make_counting_iterator(end_index),
                                   [&input](size_type idx) { return input.is_valid_nocheck(idx); })
                : end_index - start_index;

    // Use the sentinel value (i.e., -1) for the output will help identify null elements while
    // gathering for Min and Max.
    output.element<OutputType>(current_index) =
      (valid_count >= min_periods) ? thrust::reduce(thrust::seq,
                                                    thrust::make_counting_iterator(start_index),
                                                    thrust::make_counting_iterator(end_index),
                                                    size_type{start_index},
                                                    comp)
                                   : default_output;

    // The gather mask shouldn't contain null values, so always return true.
    return true;
  }
};

/**
 * @brief Operator for applying a COUNT_VALID rolling aggregation on a single window.
 */
template <typename InputType>
struct DeviceRollingCountValid {
  size_type min_periods;

  // what operations do we support
  template <typename T = InputType, aggregation::Kind O = aggregation::COUNT_VALID>
  static constexpr bool is_supported()
  {
    return true;
  }

  DeviceRollingCountValid(size_type _min_periods) : min_periods(_min_periods) {}

  template <typename OutputType, bool has_nulls>
  bool __device__ operator()(column_device_view const& input,
                             column_device_view const&,
                             mutable_column_device_view& output,
                             size_type start_index,
                             size_type end_index,
                             size_type current_index)
  {
    bool output_is_valid = ((end_index - start_index) >= min_periods);

    if (output_is_valid) {
      cudf::size_type count = 0;

      if (!has_nulls) {
        count = end_index - start_index;
      } else {
        count = thrust::count_if(thrust::seq,
                                 thrust::make_counting_iterator(start_index),
                                 thrust::make_counting_iterator(end_index),
                                 [&input](auto i) { return input.is_valid_nocheck(i); });
      }
      output.element<OutputType>(current_index) = count;
    }

    return output_is_valid;
  }
};

/**
 * @brief Operator for applying a COUNT_ALL rolling aggregation on a single window.
 */
template <typename InputType>
struct DeviceRollingCountAll {
  size_type min_periods;

  // what operations do we support
  template <typename T = InputType, aggregation::Kind O = aggregation::COUNT_ALL>
  static constexpr bool is_supported()
  {
    return true;
  }

  DeviceRollingCountAll(size_type _min_periods) : min_periods(_min_periods) {}

  template <typename OutputType, bool has_nulls>
  bool __device__ operator()(column_device_view const&,
                             column_device_view const&,
                             mutable_column_device_view& output,
                             size_type start_index,
                             size_type end_index,
                             size_type current_index)
  {
    cudf::size_type count = end_index - start_index;

    bool output_is_valid                      = count >= min_periods;
    output.element<OutputType>(current_index) = count;

    return output_is_valid;
  }
};

/**
 * @brief Operator for applying a VAR rolling aggregation on a single window.
 */
template <typename InputType>
struct DeviceRollingVariance {
  size_type const min_periods;
  size_type const ddof;

  // what operations do we support
  template <typename T = InputType, aggregation::Kind O = aggregation::VARIANCE>
  static constexpr bool is_supported()
  {
    return is_fixed_width<InputType>() and not is_chrono<InputType>();
  }

  DeviceRollingVariance(size_type _min_periods, size_type _ddof)
    : min_periods(_min_periods), ddof{_ddof}
  {
  }

  template <typename OutputType, bool has_nulls>
  bool __device__ operator()(column_device_view const& input,
                             column_device_view const&,
                             mutable_column_device_view& output,
                             size_type start_index,
                             size_type end_index,
                             size_type current_index) const
  {
    using DeviceInputType = device_storage_type_t<InputType>;

    // valid counts in the window
    cudf::size_type const count =
      has_nulls ? thrust::count_if(thrust::seq,
                                   thrust::make_counting_iterator(start_index),
                                   thrust::make_counting_iterator(end_index),
                                   [&input](auto i) { return input.is_valid_nocheck(i); })
                : end_index - start_index;

    // Result will be null if any of the following conditions are met:
    // - All inputs are null
    // - Number of valid inputs is less than `min_periods`
    bool output_is_valid = count > 0 and (count >= min_periods);

    if (output_is_valid) {
      if (count >= ddof) {
        // Welford algorithm
        // See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        OutputType m{0}, m2{0};
        size_type running_count{0};

        for (size_type i = start_index; i < end_index; i++) {
          if (has_nulls and input.is_null_nocheck(i)) { continue; }

          OutputType const x = static_cast<OutputType>(input.element<DeviceInputType>(i));

          running_count++;
          OutputType const tmp1 = x - m;
          m += tmp1 / running_count;
          OutputType const tmp2 = x - m;
          m2 += tmp1 * tmp2;
        }
        if constexpr (is_fixed_point<InputType>()) {
          // For fixed_point types, the previous computed value used unscaled rep-value,
          // the final result should be multiplied by the square of decimal `scale`.
          OutputType scaleby = exp10(static_cast<double>(input.type().scale()));
          scaleby *= scaleby;
          output.element<OutputType>(current_index) = m2 / (count - ddof) * scaleby;
        } else {
          output.element<OutputType>(current_index) = m2 / (count - ddof);
        }
      } else {
        output.element<OutputType>(current_index) =
          cuda::std::numeric_limits<OutputType>::signaling_NaN();
      }
    }

    return output_is_valid;
  }
};

/**
 * @brief Operator for applying a ROW_NUMBER rolling aggregation on a single window.
 */
template <typename InputType>
struct DeviceRollingRowNumber {
  size_type min_periods;

  // what operations do we support
  template <typename T = InputType, aggregation::Kind O = aggregation::ROW_NUMBER>
  static constexpr bool is_supported()
  {
    return true;
  }

  DeviceRollingRowNumber(size_type _min_periods) : min_periods(_min_periods) {}

  template <typename OutputType, bool has_nulls>
  bool __device__ operator()(column_device_view const&,
                             column_device_view const&,
                             mutable_column_device_view& output,
                             size_type start_index,
                             size_type end_index,
                             size_type current_index)
  {
    bool output_is_valid                      = end_index - start_index >= min_periods;
    output.element<OutputType>(current_index) = current_index - start_index + 1;

    return output_is_valid;
  }
};

struct agg_specific_empty_output {
  template <typename InputType, aggregation::Kind op>
  std::unique_ptr<column> operator()(column_view const& input, rolling_aggregation const&) const
  {
    using target_type = cudf::detail::target_type_t<InputType, op>;

    if constexpr (std::is_same_v<cudf::detail::target_type_t<InputType, op>, void>) {
      CUDF_FAIL("Unsupported combination of column-type and aggregation.");
    }

    if constexpr (cudf::is_fixed_width<target_type>()) {
      return cudf::make_empty_column(type_to_id<target_type>());
    }

    if constexpr (op == aggregation::COLLECT_LIST) {
      return cudf::make_lists_column(
        0, make_empty_column(type_to_id<size_type>()), empty_like(input), 0, {});
    }

    return empty_like(input);
  }
};

static std::unique_ptr<column> empty_output_for_rolling_aggregation(column_view const& input,
                                                                    rolling_aggregation const& agg)
{
  // TODO:
  //  Ideally, for UDF aggregations, the returned column would match
  //  the agg's return type. It currently returns empty_like(input), because:
  //    1. This preserves prior behavior for empty input columns.
  //    2. There is insufficient information to construct nested return columns.
  //       `cudf::make_udf_aggregation()` expresses the return type as a `data_type`
  //        which cannot express recursively nested types (e.g. `STRUCT<LIST<INT32>>`.)
  //    3. In any case, UDFs that return nested types are not currently supported.
  //  Constructing a more accurate return type for UDFs will be taken up
  //  at a later date.
  return agg.kind == aggregation::CUDA || agg.kind == aggregation::PTX
           ? empty_like(input)
           : cudf::detail::dispatch_type_and_aggregation(
               input.type(), agg.kind, agg_specific_empty_output{}, input, agg);
}

/**
 * @brief Operator for applying a LEAD rolling aggregation on a single window.
 */
template <typename InputType>
struct DeviceRollingLead {
  size_type row_offset;

  // what operations do we support
  template <typename T = InputType, aggregation::Kind O = aggregation::LEAD>
  static constexpr bool is_supported()
  {
    return cudf::is_fixed_width<T>();
  }

  template <typename T = InputType, std::enable_if_t<is_supported<T>()>* = nullptr>
  DeviceRollingLead(size_type _row_offset) : row_offset(_row_offset)
  {
  }

  template <typename T = InputType, std::enable_if_t<!is_supported<T>()>* = nullptr>
  DeviceRollingLead(size_type _row_offset) : row_offset(_row_offset)
  {
    CUDF_FAIL("Invalid aggregation/type pair");
  }

  template <typename OutputType, bool has_nulls>
  bool __device__ operator()(column_device_view const& input,
                             column_device_view const& default_outputs,
                             mutable_column_device_view& output,
                             size_type,
                             size_type end_index,
                             size_type current_index)
  {
    // Offsets have already been normalized.

    // Check if row is invalid.
    if (row_offset > (end_index - current_index - 1)) {
      // Invalid row marked. Use default value, if available.
      if (default_outputs.size() == 0 || default_outputs.is_null(current_index)) { return false; }

      output.element<OutputType>(current_index) =
        default_outputs.element<OutputType>(current_index);
      return true;
    }

    // Not an invalid row.
    auto index   = current_index + row_offset;
    auto is_null = input.is_null(index);
    if (!is_null) {
      output.element<OutputType>(current_index) =
        input.element<device_storage_type_t<InputType>>(index);
    }
    return !is_null;
  }
};

/**
 * @brief Operator for applying a LAG rolling aggregation on a single window.
 */
template <typename InputType>
struct DeviceRollingLag {
  size_type row_offset;

  // what operations do we support
  template <typename T = InputType, aggregation::Kind O = aggregation::LAG>
  static constexpr bool is_supported()
  {
    return cudf::is_fixed_width<T>();
  }

  template <typename T = InputType, std::enable_if_t<is_supported<T>()>* = nullptr>
  DeviceRollingLag(size_type _row_offset) : row_offset(_row_offset)
  {
  }

  template <typename T = InputType, std::enable_if_t<!is_supported<T>()>* = nullptr>
  DeviceRollingLag(size_type _row_offset) : row_offset(_row_offset)
  {
    CUDF_FAIL("Invalid aggregation/type pair");
  }

  template <typename OutputType, bool has_nulls>
  bool __device__ operator()(column_device_view const& input,
                             column_device_view const& default_outputs,
                             mutable_column_device_view& output,
                             size_type start_index,
                             size_type,
                             size_type current_index)
  {
    // Offsets have already been normalized.

    // Check if row is invalid.
    if (row_offset > (current_index - start_index)) {
      // Invalid row marked. Use default value, if available.
      if (default_outputs.size() == 0 || default_outputs.is_null(current_index)) { return false; }

      output.element<OutputType>(current_index) =
        default_outputs.element<OutputType>(current_index);
      return true;
    }

    // Not an invalid row.
    auto index   = current_index - row_offset;
    auto is_null = input.is_null(index);
    if (!is_null) {
      output.element<OutputType>(current_index) =
        input.element<device_storage_type_t<InputType>>(index);
    }
    return !is_null;
  }
};

/**
 * @brief Maps an `InputType and `aggregation::Kind` value to its corresponding
 * rolling window operator.
 *
 * @tparam InputType The input type to map to its corresponding operator
 * @tparam k The `aggregation::Kind` value to map to its corresponding operator
 */
template <typename InputType, aggregation::Kind k>
struct corresponding_rolling_operator {
  using type = DeviceRolling<InputType, k>;
};

template <typename InputType>
struct corresponding_rolling_operator<InputType, aggregation::ARGMIN> {
  using type = DeviceRollingArgMinMaxBase<InputType, aggregation::ARGMIN>;
};

template <typename InputType>
struct corresponding_rolling_operator<InputType, aggregation::ARGMAX> {
  using type = DeviceRollingArgMinMaxBase<InputType, aggregation::ARGMAX>;
};

template <typename InputType>
struct corresponding_rolling_operator<InputType, aggregation::COUNT_VALID> {
  using type = DeviceRollingCountValid<InputType>;
};

template <typename InputType>
struct corresponding_rolling_operator<InputType, aggregation::COUNT_ALL> {
  using type = DeviceRollingCountAll<InputType>;
};

template <typename InputType>
struct corresponding_rolling_operator<InputType, aggregation::ROW_NUMBER> {
  using type = DeviceRollingRowNumber<InputType>;
};

template <typename InputType>
struct corresponding_rolling_operator<InputType, aggregation::Kind::VARIANCE> {
  using type = DeviceRollingVariance<InputType>;
};

template <typename InputType>
struct corresponding_rolling_operator<InputType, aggregation::Kind::LEAD> {
  using type = DeviceRollingLead<InputType>;
};

template <typename InputType>
struct corresponding_rolling_operator<InputType, aggregation::Kind::LAG> {
  using type = DeviceRollingLag<InputType>;
};

/**
 * @brief Functor for creating a device rolling operator based on input type and aggregation type.
 */
template <typename InputType, aggregation::Kind k, typename = void>
struct create_rolling_operator {
  auto operator()(size_type min_periods, rolling_aggregation const&)
  {
    return typename corresponding_rolling_operator<InputType, k>::type(min_periods);
  }
};

template <typename InputType>
struct create_rolling_operator<InputType, aggregation::Kind::VARIANCE> {
  auto operator()(size_type min_periods, rolling_aggregation const& agg)
  {
    return DeviceRollingVariance<InputType>{
      min_periods, dynamic_cast<cudf::detail::var_aggregation const&>(agg)._ddof};
  }
};

template <typename InputType>
struct create_rolling_operator<InputType, aggregation::Kind::LEAD> {
  auto operator()(size_type, rolling_aggregation const& agg)
  {
    return DeviceRollingLead<InputType>{
      dynamic_cast<cudf::detail::lead_lag_aggregation const&>(agg).row_offset};
  }
};

template <typename InputType>
struct create_rolling_operator<InputType, aggregation::Kind::LAG> {
  auto operator()(size_type, rolling_aggregation const& agg)
  {
    return DeviceRollingLag<InputType>{
      dynamic_cast<cudf::detail::lead_lag_aggregation const&>(agg).row_offset};
  }
};

template <typename InputType, aggregation::Kind k>
struct create_rolling_operator<
  InputType,
  k,
  typename std::enable_if_t<std::is_same_v<InputType, cudf::string_view> &&
                            (k == aggregation::Kind::ARGMIN || k == aggregation::Kind::ARGMAX)>> {
  auto operator()(size_type min_periods, rolling_aggregation const&)
  {
    return DeviceRollingArgMinMaxString<k>{min_periods};
  }
};

template <typename InputType, aggregation::Kind k>
struct create_rolling_operator<
  InputType,
  k,
  typename std::enable_if_t<std::is_same_v<InputType, cudf::struct_view> &&
                            (k == aggregation::Kind::ARGMIN || k == aggregation::Kind::ARGMAX)>> {
  template <typename Comparator>
  auto operator()(size_type min_periods, Comparator const& comp)
  {
    return DeviceRollingArgMinMaxStruct<k, Comparator>{min_periods, comp};
  }
};

/**
 * @brief Rolling window specific implementation of simple_aggregations_collector.
 *
 * The purpose of this class is to preprocess incoming aggregation/type pairs and
 * potentially transform them into other aggregation/type pairs. Typically when this
 * happens, the equivalent aggregation/type implementation of finalize() will perform
 * some postprocessing step.
 *
 * An example of this would be applying a MIN aggregation to strings. This cannot be done
 * directly in the rolling operation, so instead the following happens:
 *
 * - the rolling_aggregation_preprocessor transforms the incoming MIN/string pair to
 *   an ARGMIN/int pair.
 * - The ARGMIN/int has the rolling operation applied to it, generating a list of indices
 *   that can then be used as a gather map.
 * - The rolling_aggregation_postprocessor then takes this gather map and performs a final
 *   gather() on the input string data to generate the final output.
 *
 * Another example is COLLECT_LIST. COLLECT_LIST is odd in that it doesn't go through the
 * normal gpu rolling kernel at all. It has a completely custom implementation. So the
 * following happens:
 *
 * - the rolling_aggregation_preprocessor transforms the COLLECT_LIST aggregation into nothing,
 *   since no actual rolling window operation will be performed.
 * - the rolling_aggregation_postprocessor calls the specialized rolling_collect_list()
 *   function to generate the final output.
 *
 */
class rolling_aggregation_preprocessor final : public cudf::detail::simple_aggregations_collector {
 public:
  using cudf::detail::simple_aggregations_collector::visit;

  // NOTE : all other aggregations are passed through unchanged via the default
  // visit() function in the simple_aggregations_collector.

  // MIN aggregations with strings are processed in 2 passes. The first pass performs
  // the rolling operation on a ARGMIN aggregation to generate indices instead of values.
  // Then a second pass uses those indices to gather the final strings.  This step
  // translates the MIN -> ARGMIN aggregation
  std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                  cudf::detail::min_aggregation const&) override
  {
    std::vector<std::unique_ptr<aggregation>> aggs;
    aggs.push_back(col_type.id() == type_id::STRING || col_type.id() == type_id::STRUCT
                     ? make_argmin_aggregation()
                     : make_min_aggregation());
    return aggs;
  }

  // MAX aggregations with strings are processed in 2 passes. The first pass performs
  // the rolling operation on a ARGMAX aggregation to generate indices instead of values.
  // Then a second pass uses those indices to gather the final strings.  This step
  // translates the MAX -> ARGMAX aggregation
  std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                  cudf::detail::max_aggregation const&) override
  {
    std::vector<std::unique_ptr<aggregation>> aggs;
    aggs.push_back(col_type.id() == type_id::STRING || col_type.id() == type_id::STRUCT
                     ? make_argmax_aggregation()
                     : make_max_aggregation());
    return aggs;
  }

  // COLLECT_LIST aggregations do not perform a rolling operation at all. They get processed
  // entirely in the finalize() step.
  std::vector<std::unique_ptr<aggregation>> visit(
    data_type, cudf::detail::collect_list_aggregation const&) override
  {
    return {};
  }

  // COLLECT_SET aggregations do not perform a rolling operation at all. They get processed
  // entirely in the finalize() step.
  std::vector<std::unique_ptr<aggregation>> visit(
    data_type, cudf::detail::collect_set_aggregation const&) override
  {
    return {};
  }

  // STD aggregations depends on VARIANCE aggregation. Each element is applied
  // with square-root in the finalize() step.
  std::vector<std::unique_ptr<aggregation>> visit(data_type,
                                                  cudf::detail::std_aggregation const& agg) override
  {
    std::vector<std::unique_ptr<aggregation>> aggs;
    aggs.push_back(make_variance_aggregation(agg._ddof));
    return aggs;
  }

  // LEAD and LAG have custom behaviors for non fixed-width types.
  std::vector<std::unique_ptr<aggregation>> visit(
    data_type col_type, cudf::detail::lead_lag_aggregation const& agg) override
  {
    // no rolling operation for non-fixed width.  just a postprocess step at the end
    if (!cudf::is_fixed_width(col_type)) { return {}; }
    // otherwise, pass through
    std::vector<std::unique_ptr<aggregation>> aggs;
    aggs.push_back(agg.clone());
    return aggs;
  }

  // NTH_ELEMENT aggregations are computed in finalize(). Skip preprocessing.
  std::vector<std::unique_ptr<aggregation>> visit(
    data_type, cudf::detail::nth_element_aggregation const&) override
  {
    return {};
  }
};

/**
 * @brief Rolling window specific implementation of aggregation_finalizer.
 *
 * The purpose of this class is to postprocess rolling window data depending on the
 * aggregation/type pair. See the description of rolling_aggregation_preprocessor for
 * a detailed description.
 *
 */
template <typename PrecedingWindowIterator, typename FollowingWindowIterator>
class rolling_aggregation_postprocessor final : public cudf::detail::aggregation_finalizer {
 public:
  using cudf::detail::aggregation_finalizer::visit;

  rolling_aggregation_postprocessor(column_view const& _input,
                                    column_view const& _default_outputs,
                                    data_type _result_type,
                                    PrecedingWindowIterator _preceding_window_begin,
                                    FollowingWindowIterator _following_window_begin,
                                    int _min_periods,
                                    std::unique_ptr<column>&& _intermediate,
                                    rmm::cuda_stream_view _stream,
                                    rmm::device_async_resource_ref _mr)
    :

      input(_input),
      default_outputs(_default_outputs),
      result_type(_result_type),
      preceding_window_begin(_preceding_window_begin),
      following_window_begin(_following_window_begin),
      min_periods(_min_periods),
      intermediate(std::move(_intermediate)),
      result(nullptr),
      stream(_stream),
      mr(_mr)
  {
  }

  // all non-specialized aggregation types simply pass the intermediate result through.
  void visit(aggregation const&) override { result = std::move(intermediate); }

  // perform a final gather on the generated ARGMIN data
  void visit(cudf::detail::min_aggregation const&) override
  {
    if (result_type.id() == type_id::STRING || result_type.id() == type_id::STRUCT) {
      // The rows that represent null elements will have negative values in gather map,
      // and that's why nullify_out_of_bounds/ignore_out_of_bounds is true.
      auto output_table = detail::gather(table_view{{input}},
                                         intermediate->view(),
                                         cudf::out_of_bounds_policy::NULLIFY,
                                         detail::negative_index_policy::NOT_ALLOWED,
                                         stream,
                                         mr);
      result            = std::make_unique<cudf::column>(std::move(output_table->get_column(0)));
    } else {
      result = std::move(intermediate);
    }
  }

  // perform a final gather on the generated ARGMAX data
  void visit(cudf::detail::max_aggregation const&) override
  {
    if (result_type.id() == type_id::STRING || result_type.id() == type_id::STRUCT) {
      // The rows that represent null elements will have negative values in gather map,
      // and that's why nullify_out_of_bounds/ignore_out_of_bounds is true.
      auto output_table = detail::gather(table_view{{input}},
                                         intermediate->view(),
                                         cudf::out_of_bounds_policy::NULLIFY,
                                         detail::negative_index_policy::NOT_ALLOWED,
                                         stream,
                                         mr);
      result            = std::make_unique<cudf::column>(std::move(output_table->get_column(0)));
    } else {
      result = std::move(intermediate);
    }
  }

  // perform the actual COLLECT_LIST operation entirely.
  void visit(cudf::detail::collect_list_aggregation const& agg) override
  {
    result = rolling_collect_list(input,
                                  default_outputs,
                                  preceding_window_begin,
                                  following_window_begin,
                                  min_periods,
                                  agg._null_handling,
                                  stream,
                                  mr);
  }

  // perform the actual COLLECT_SET operation entirely.
  void visit(cudf::detail::collect_set_aggregation const& agg) override
  {
    auto const collected_list = rolling_collect_list(input,
                                                     default_outputs,
                                                     preceding_window_begin,
                                                     following_window_begin,
                                                     min_periods,
                                                     agg._null_handling,
                                                     stream,
                                                     cudf::get_current_device_resource_ref());

    result = lists::detail::distinct(
      lists_column_view{collected_list->view()}, agg._nulls_equal, agg._nans_equal, stream, mr);
  }

  // perform the element-wise square root operation on result of VARIANCE
  void visit(cudf::detail::std_aggregation const&) override
  {
    result = detail::unary_operation(intermediate->view(), unary_operator::SQRT, stream, mr);
  }

  std::unique_ptr<column> get_result()
  {
    CUDF_EXPECTS(result != nullptr,
                 "Calling result on rolling aggregation postprocessor that has not been visited in "
                 "rolling_window");
    return std::move(result);
  }

  // LEAD and LAG have custom behaviors for non fixed-width types.
  void visit(cudf::detail::lead_lag_aggregation const& agg) override
  {
    // if this is non-fixed width, run the custom lead-lag code
    if (!cudf::is_fixed_width(result_type)) {
      result =
        cudf::detail::compute_lead_lag_for_nested<PrecedingWindowIterator, FollowingWindowIterator>(
          agg.kind,
          input,
          default_outputs,
          preceding_window_begin,
          following_window_begin,
          agg.row_offset,
          stream,
          mr);
    }
    // otherwise just pass through the intermediate
    else {
      result = std::move(intermediate);
    }
  }

  // Nth_ELEMENT aggregation.
  void visit(cudf::detail::nth_element_aggregation const& agg) override
  {
    result =
      agg._null_handling == null_policy::EXCLUDE
        ? rolling::nth_element<null_policy::EXCLUDE>(
            agg._n, input, preceding_window_begin, following_window_begin, min_periods, stream, mr)
        : rolling::nth_element<null_policy::INCLUDE>(
            agg._n, input, preceding_window_begin, following_window_begin, min_periods, stream, mr);
  }

 private:
  column_view input;
  column_view default_outputs;
  data_type result_type;
  PrecedingWindowIterator preceding_window_begin;
  FollowingWindowIterator following_window_begin;
  int min_periods;
  std::unique_ptr<column> intermediate;
  std::unique_ptr<column> result;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;
};

/**
 * @brief Computes the rolling window function
 *
 * @tparam OutputType Datatype of `output`
 * @tparam block_size CUDA block size for the kernel
 * @tparam has_nulls true if the input column has nulls
 * @tparam DeviceRollingOperator An operator that performs a single windowing operation
 * @tparam PrecedingWindowIterator iterator type (inferred)
 * @tparam FollowingWindowIterator iterator type (inferred)
 * @param[in] input Input column device view
 * @param[in] default_outputs A column of per-row default values to be returned instead
 *            of nulls for certain aggregation types.
 * @param[out] output Output column device view
 * @param[out] output_valid_count Output count of valid values
 * @param[in] device_operator The operator used to perform a single window operation
 * @param[in] preceding_window_begin Rolling window size iterator, accumulates from
 *            in_col[i-preceding_window] to in_col[i] inclusive
 * @param[in] following_window_begin Rolling window size iterator in the forward
 *            direction, accumulates from in_col[i] to in_col[i+following_window] inclusive
 */
template <typename OutputType,
          int block_size,
          bool has_nulls,
          typename DeviceRollingOperator,
          typename PrecedingWindowIterator,
          typename FollowingWindowIterator>
__launch_bounds__(block_size) CUDF_KERNEL
  void gpu_rolling(column_device_view input,
                   column_device_view default_outputs,
                   mutable_column_device_view output,
                   size_type* __restrict__ output_valid_count,
                   DeviceRollingOperator device_operator,
                   PrecedingWindowIterator preceding_window_begin,
                   FollowingWindowIterator following_window_begin)
{
  thread_index_type i            = blockIdx.x * block_size + threadIdx.x;
  thread_index_type const stride = block_size * gridDim.x;

  size_type warp_valid_count{0};

  auto active_threads = __ballot_sync(0xffff'ffffu, i < input.size());
  while (i < input.size()) {
    // to prevent overflow issues when computing bounds use int64_t
    int64_t const preceding_window = preceding_window_begin[i];
    int64_t const following_window = following_window_begin[i];

    // compute bounds
    auto const start = static_cast<size_type>(
      min(static_cast<int64_t>(input.size()), max(int64_t{0}, i - preceding_window + 1)));
    auto const end = static_cast<size_type>(
      min(static_cast<int64_t>(input.size()), max(int64_t{0}, i + following_window + 1)));
    auto const start_index = min(start, end);
    auto const end_index   = max(start, end);

    // aggregate
    // TODO: We should explore using shared memory to avoid redundant loads.
    //       This might require separating the kernel into a special version
    //       for dynamic and static sizes.

    volatile bool output_is_valid = false;
    output_is_valid               = device_operator.template operator()<OutputType, has_nulls>(
      input, default_outputs, output, start_index, end_index, i);

    // set the mask
    cudf::bitmask_type const result_mask{__ballot_sync(active_threads, output_is_valid)};

    // only one thread writes the mask
    if (0 == threadIdx.x % cudf::detail::warp_size) {
      output.set_mask_word(cudf::word_index(i), result_mask);
      warp_valid_count += __popc(result_mask);
    }

    // process next element
    i += stride;
    active_threads = __ballot_sync(active_threads, i < input.size());
  }

  // sum the valid counts across the whole block
  size_type block_valid_count =
    cudf::detail::single_lane_block_sum_reduce<block_size, 0>(warp_valid_count);

  if (threadIdx.x == 0) { atomicAdd(output_valid_count, block_valid_count); }
}

/**
 * @brief Type/aggregation dispatched functor for launching the gpu rolling window
 *        kernel.
 */
template <typename InputType>
struct rolling_window_launcher {
  template <aggregation::Kind op,
            typename PrecedingWindowIterator,
            typename FollowingWindowIterator>
  std::enable_if_t<corresponding_rolling_operator<InputType, op>::type::is_supported(),
                   std::unique_ptr<column>>
  operator()(column_view const& input,
             column_view const& default_outputs,
             PrecedingWindowIterator preceding_window_begin,
             FollowingWindowIterator following_window_begin,
             int min_periods,
             [[maybe_unused]] rolling_aggregation const& agg,
             rmm::cuda_stream_view stream,
             rmm::device_async_resource_ref mr)
  {
    auto const do_rolling = [&](auto const& device_op) {
      auto output = make_fixed_width_column(
        target_type(input.type(), op), input.size(), mask_state::UNINITIALIZED, stream, mr);

      auto const d_inp_ptr         = column_device_view::create(input, stream);
      auto const d_default_out_ptr = column_device_view::create(default_outputs, stream);
      auto const d_out_ptr = mutable_column_device_view::create(output->mutable_view(), stream);
      auto d_valid_count   = cudf::detail::device_scalar<size_type>{0, stream};

      auto constexpr block_size = 256;
      auto const grid           = cudf::detail::grid_1d(input.size(), block_size);
      using OutType             = device_storage_type_t<target_type_t<InputType, op>>;

      if (input.has_nulls()) {
        gpu_rolling<OutType, block_size, true>
          <<<grid.num_blocks, block_size, 0, stream.value()>>>(*d_inp_ptr,
                                                               *d_default_out_ptr,
                                                               *d_out_ptr,
                                                               d_valid_count.data(),
                                                               device_op,
                                                               preceding_window_begin,
                                                               following_window_begin);
      } else {
        gpu_rolling<OutType, block_size, false>
          <<<grid.num_blocks, block_size, 0, stream.value()>>>(*d_inp_ptr,
                                                               *d_default_out_ptr,
                                                               *d_out_ptr,
                                                               d_valid_count.data(),
                                                               device_op,
                                                               preceding_window_begin,
                                                               following_window_begin);
      }

      auto const valid_count = d_valid_count.value(stream);
      output->set_null_count(output->size() - valid_count);

      return output;
    };  // end do_rolling

    auto constexpr is_arg_minmax =
      op == aggregation::Kind::ARGMIN || op == aggregation::Kind::ARGMAX;

    if constexpr (is_arg_minmax && std::is_same_v<InputType, cudf::struct_view>) {
      // Using comp_generator to create a LESS operator for finding ARGMIN/ARGMAX of structs.
      auto const comp_generator =
        cudf::reduction::detail::comparison_binop_generator::create<op>(input, stream);
      auto const device_op =
        create_rolling_operator<InputType, op>{}(min_periods, comp_generator.binop());
      return do_rolling(device_op);
    } else {  // all the remaining rolling operations
      auto const device_op = create_rolling_operator<InputType, op>{}(min_periods, agg);
      return do_rolling(device_op);
    }
  }

  template <aggregation::Kind op,
            typename PrecedingWindowIterator,
            typename FollowingWindowIterator>
  std::enable_if_t<!corresponding_rolling_operator<InputType, op>::type::is_supported(),
                   std::unique_ptr<column>>
  operator()(column_view const&,
             column_view const&,
             PrecedingWindowIterator,
             FollowingWindowIterator,
             int,
             rolling_aggregation const&,
             rmm::cuda_stream_view,
             rmm::device_async_resource_ref)
  {
    CUDF_FAIL("Invalid aggregation type/pair");
  }
};

/**
 * @brief Functor for performing the high level rolling logic.
 *
 * This does 3 basic things:
 *
 * - It calls the preprocess step on incoming aggregation/type pairs
 * - It calls the aggregation-dispatched gpu-rolling operation
 * - It calls the final postprocess step
 */
struct dispatch_rolling {
  template <typename InputType, typename PrecedingWindowIterator, typename FollowingWindowIterator>
  std::unique_ptr<column> operator()(column_view const& input,
                                     column_view const& default_outputs,
                                     PrecedingWindowIterator preceding_window_begin,
                                     FollowingWindowIterator following_window_begin,
                                     size_type min_periods,
                                     rolling_aggregation const& agg,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  {
    // do any preprocessing of aggregations (eg, MIN -> ARGMIN, COLLECT_LIST -> nothing)
    rolling_aggregation_preprocessor preprocessor;
    auto preprocessed_aggs = agg.get_simple_aggregations(input.type(), preprocessor);
    CUDF_EXPECTS(preprocessed_aggs.size() <= 1,
                 "Encountered a non-trivial rolling aggregation result");

    // perform the rolling window if we produced an aggregation to use
    auto intermediate = preprocessed_aggs.size() > 0
                          ? aggregation_dispatcher(
                              dynamic_cast<rolling_aggregation const&>(*preprocessed_aggs[0]).kind,
                              rolling_window_launcher<InputType>{},
                              input,
                              default_outputs,
                              preceding_window_begin,
                              following_window_begin,
                              min_periods,
                              dynamic_cast<rolling_aggregation const&>(*preprocessed_aggs[0]),
                              stream,
                              mr)
                          : nullptr;

    // finalize.
    auto const result_type = target_type(input.type(), agg.kind);
    rolling_aggregation_postprocessor postprocessor(input,
                                                    default_outputs,
                                                    result_type,
                                                    preceding_window_begin,
                                                    following_window_begin,
                                                    min_periods,
                                                    std::move(intermediate),
                                                    stream,
                                                    mr);
    agg.finalize(postprocessor);
    return postprocessor.get_result();
  }
};

// Applies a user-defined rolling window function to the values in a column.
template <typename PrecedingWindowIterator, typename FollowingWindowIterator>
std::unique_ptr<column> rolling_window_udf(column_view const& input,
                                           PrecedingWindowIterator preceding_window,
                                           std::string const& preceding_window_str,
                                           FollowingWindowIterator following_window,
                                           std::string const& following_window_str,
                                           size_type min_periods,
                                           rolling_aggregation const& agg,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  static_assert(warp_size == cudf::detail::size_in_bits<cudf::bitmask_type>(),
                "bitmask_type size does not match CUDA warp size");

  if (input.has_nulls()) {
    CUDF_FAIL("Currently the UDF version of rolling window does NOT support inputs with nulls.");
  }

  min_periods = std::max(min_periods, 0);

  auto& udf_agg = dynamic_cast<udf_aggregation const&>(agg);

  std::string hash = "prog_rolling." + std::to_string(std::hash<std::string>{}(udf_agg._source));

  std::string cuda_source;
  switch (udf_agg.kind) {
    case aggregation::Kind::PTX:
      cuda_source += cudf::jit::parse_single_function_ptx(udf_agg._source,
                                                          udf_agg._function_name,
                                                          cudf::type_to_name(udf_agg._output_type),
                                                          {0, 5});  // args 0 and 5 are pointers.
      break;
    case aggregation::Kind::CUDA:
      cuda_source += cudf::jit::parse_single_function_cuda(udf_agg._source, udf_agg._function_name);
      break;
    default: CUDF_FAIL("Unsupported UDF type.");
  }

  std::unique_ptr<column> output = make_numeric_column(
    udf_agg._output_type, input.size(), cudf::mask_state::UNINITIALIZED, stream, mr);

  auto output_view = output->mutable_view();
  cudf::detail::device_scalar<size_type> device_valid_count{0, stream};

  std::string kernel_name =
    jitify2::reflection::Template("cudf::rolling::jit::gpu_rolling_new")  //
      .instantiate(cudf::type_to_name(input.type()),  // list of template arguments
                   cudf::type_to_name(output->type()),
                   udf_agg._operator_name,
                   preceding_window_str.c_str(),
                   following_window_str.c_str());

  cudf::jit::get_program_cache(*rolling_jit_kernel_cu_jit)
    .get_kernel(
      kernel_name, {}, {{"rolling/jit/operation-udf.hpp", cuda_source}}, {"-arch=sm_."})  //
    ->configure_1d_max_occupancy(0, 0, 0, stream.value())                                 //
    ->launch(input.size(),
             cudf::jit::get_data_ptr(input),
             input.null_mask(),
             cudf::jit::get_data_ptr(output_view),
             output_view.null_mask(),
             device_valid_count.data(),
             preceding_window,
             following_window,
             min_periods);

  output->set_null_count(output->size() - device_valid_count.value(stream));

  // check the stream for debugging
  CUDF_CHECK_CUDA(stream.value());

  return output;
}

/**
 * @copydoc cudf::rolling_window(column_view const& input,
 *                               PrecedingWindowIterator preceding_window_begin,
 *                               FollowingWindowIterator following_window_begin,
 *                               size_type min_periods,
 *                               rolling_aggregation const& agg,
 *                               rmm::device_async_resource_ref mr)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
template <typename PrecedingWindowIterator, typename FollowingWindowIterator>
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& default_outputs,
                                       PrecedingWindowIterator preceding_window_begin,
                                       FollowingWindowIterator following_window_begin,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  static_assert(warp_size == cudf::detail::size_in_bits<cudf::bitmask_type>(),
                "bitmask_type size does not match CUDA warp size");

  if (input.is_empty()) { return cudf::detail::empty_output_for_rolling_aggregation(input, agg); }

  if (cudf::is_dictionary(input.type())) {
    CUDF_EXPECTS(agg.kind == aggregation::COUNT_ALL || agg.kind == aggregation::COUNT_VALID ||
                   agg.kind == aggregation::ROW_NUMBER || agg.kind == aggregation::MIN ||
                   agg.kind == aggregation::MAX || agg.kind == aggregation::LEAD ||
                   agg.kind == aggregation::LAG,
                 "Invalid aggregation for dictionary column");
  }

  if (agg.kind != aggregation::LEAD && agg.kind != aggregation::LAG) {
    CUDF_EXPECTS(default_outputs.is_empty(),
                 "Only LEAD/LAG window functions support default values.");
  }

  min_periods = std::max(min_periods, 0);

  auto input_col = cudf::is_dictionary(input.type())
                     ? dictionary_column_view(input).get_indices_annotated()
                     : input;

  auto output = cudf::type_dispatcher(input_col.type(),
                                      dispatch_rolling{},
                                      input_col,
                                      default_outputs,
                                      preceding_window_begin,
                                      following_window_begin,
                                      min_periods,
                                      agg,
                                      stream,
                                      mr);

  if (!cudf::is_dictionary(input.type())) return output;

  // dictionary column post processing
  if (agg.kind == aggregation::COUNT_ALL || agg.kind == aggregation::COUNT_VALID ||
      agg.kind == aggregation::ROW_NUMBER) {
    return output;
  }

  // output is new dictionary indices (including nulls)
  auto keys = std::make_unique<column>(dictionary_column_view(input).keys(), stream, mr);
  auto const indices_type = output->type();        // capture these
  auto const output_size  = output->size();        // before calling
  auto const null_count   = output->null_count();  // release()
  auto contents           = output->release();
  // create indices column from output column data
  auto indices = std::make_unique<column>(indices_type,
                                          output_size,
                                          std::move(*(contents.data.release())),
                                          rmm::device_buffer{0, stream, mr},
                                          0);
  // create dictionary from keys and indices
  return make_dictionary_column(
    std::move(keys), std::move(indices), std::move(*(contents.null_mask.release())), null_count);
}

}  // namespace detail

}  // namespace cudf
