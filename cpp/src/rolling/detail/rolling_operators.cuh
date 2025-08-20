/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "rolling.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/limits>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>

#include <memory>

namespace cudf {

namespace detail {

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

    if (output_is_valid) {
      // store the output value, one per thread, but only if the
      // output is valid. min_periods is required to be >= 1, and so
      // here, count must be nonzero. We need to avoid storing if
      // count is zero since this could cause UB in some aggregations,
      // which may cause the compiler to deduce nonsense about the loop
      // that increments count.
      cudf::detail::rolling_store_output_functor<OutputType, op == aggregation::MEAN>{}(
        output.element<OutputType>(current_index), val, count);
    }

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

  template <typename T = InputType>
  DeviceRollingLead(size_type _row_offset)
    requires(is_supported<T>())
    : row_offset(_row_offset)
  {
  }

  template <typename T = InputType>
  DeviceRollingLead(size_type _row_offset)
    requires(!is_supported<T>())
    : row_offset(_row_offset)
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

  template <typename T = InputType>
  DeviceRollingLag(size_type _row_offset)
    requires(is_supported<T>())
    : row_offset(_row_offset)
  {
  }

  template <typename T = InputType>
  DeviceRollingLag(size_type _row_offset)
    requires(!is_supported<T>())
    : row_offset(_row_offset)
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

}  // namespace detail

}  // namespace cudf
