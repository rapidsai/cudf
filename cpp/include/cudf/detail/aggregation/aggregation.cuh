/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf/aggregation.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/traits.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>

namespace cudf {
namespace detail {
/**
 * @brief Maps an `aggregation::Kind` value to it's corresponding binary
 * operator.
 *
 * @note Not all values of `aggregation::Kind` have a valid corresponding binary
 * operator. For these values `E`,
 * `std::is_same_v<corresponding_operator<E>::type, void>`.
 *
 * @tparam k The `aggregation::Kind` value to map to its corresponding operator
 */
template <aggregation::Kind k>
struct corresponding_operator {
  using type = void;
};

template <>
struct corresponding_operator<aggregation::MIN> {
  using type = DeviceMin;
};
template <>
struct corresponding_operator<aggregation::MAX> {
  using type = DeviceMax;
};
template <>
struct corresponding_operator<aggregation::ARGMIN> {
  using type = DeviceMin;
};
template <>
struct corresponding_operator<aggregation::ARGMAX> {
  using type = DeviceMax;
};
template <>
struct corresponding_operator<aggregation::ANY> {
  using type = DeviceMax;
};
template <>
struct corresponding_operator<aggregation::ALL> {
  using type = DeviceMin;
};
template <>
struct corresponding_operator<aggregation::SUM> {
  using type = DeviceSum;
};
template <>
struct corresponding_operator<aggregation::PRODUCT> {
  using type = DeviceProduct;
};
template <>
struct corresponding_operator<aggregation::SUM_OF_SQUARES> {
  using type = DeviceSum;
};
template <>
struct corresponding_operator<aggregation::STD> {
  using type = DeviceSum;
};
template <>
struct corresponding_operator<aggregation::VARIANCE> {
  using type = DeviceSum;
};
template <>
struct corresponding_operator<aggregation::MEAN> {
  using type = DeviceSum;
};
template <>
struct corresponding_operator<aggregation::COUNT_VALID> {
  using type = DeviceCount;
};
template <>
struct corresponding_operator<aggregation::COUNT_ALL> {
  using type = DeviceCount;
};

template <aggregation::Kind k>
using corresponding_operator_t = typename corresponding_operator<k>::type;

template <aggregation::Kind k>
constexpr bool has_corresponding_operator()
{
  return !std::is_same_v<typename corresponding_operator<k>::type, void>;
}

template <typename Source,
          aggregation::Kind k,
          bool target_has_nulls,
          bool source_has_nulls,
          typename Enable = void>
struct update_target_element {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    CUDF_UNREACHABLE("Invalid source type and aggregation combination.");
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element<
  Source,
  aggregation::MIN,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<is_fixed_width<Source>() && cudf::has_atomic_support<Source>() &&
                   !is_fixed_point<Source>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target = target_type_t<Source, aggregation::MIN>;
    cudf::detail::atomic_min(&target.element<Target>(target_index),
                             static_cast<Target>(source.element<Source>(source_index)));

    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element<
  Source,
  aggregation::MIN,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<is_fixed_point<Source>() &&
                   cudf::has_atomic_support<device_storage_type_t<Source>>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target       = target_type_t<Source, aggregation::MIN>;
    using DeviceTarget = device_storage_type_t<Target>;
    using DeviceSource = device_storage_type_t<Source>;

    cudf::detail::atomic_min(&target.element<DeviceTarget>(target_index),
                             static_cast<DeviceTarget>(source.element<DeviceSource>(source_index)));

    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element<
  Source,
  aggregation::MAX,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<is_fixed_width<Source>() && cudf::has_atomic_support<Source>() &&
                   !is_fixed_point<Source>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target = target_type_t<Source, aggregation::MAX>;
    cudf::detail::atomic_max(&target.element<Target>(target_index),
                             static_cast<Target>(source.element<Source>(source_index)));

    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element<
  Source,
  aggregation::MAX,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<is_fixed_point<Source>() &&
                   cudf::has_atomic_support<device_storage_type_t<Source>>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target       = target_type_t<Source, aggregation::MAX>;
    using DeviceTarget = device_storage_type_t<Target>;
    using DeviceSource = device_storage_type_t<Source>;

    cudf::detail::atomic_max(&target.element<DeviceTarget>(target_index),
                             static_cast<DeviceTarget>(source.element<DeviceSource>(source_index)));

    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element<
  Source,
  aggregation::SUM,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<cudf::is_fixed_width<Source>() && cudf::has_atomic_support<Source>() &&
                   !cudf::is_fixed_point<Source>() && !cudf::is_timestamp<Source>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target = target_type_t<Source, aggregation::SUM>;
    cudf::detail::atomic_add(&target.element<Target>(target_index),
                             static_cast<Target>(source.element<Source>(source_index)));

    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element<
  Source,
  aggregation::SUM,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<is_fixed_point<Source>() &&
                   cudf::has_atomic_support<device_storage_type_t<Source>>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target       = target_type_t<Source, aggregation::SUM>;
    using DeviceTarget = device_storage_type_t<Target>;
    using DeviceSource = device_storage_type_t<Source>;

    cudf::detail::atomic_add(&target.element<DeviceTarget>(target_index),
                             static_cast<DeviceTarget>(source.element<DeviceSource>(source_index)));

    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

/**
 * @brief Function object to update a single element in a target column using
 * the dictionary key addressed by the specific index.
 *
 * SFINAE is used to prevent recursion for dictionary type. Dictionary keys cannot be a
 * dictionary.
 *
 */
template <bool target_has_nulls = true>
struct update_target_from_dictionary {
  template <typename Source,
            aggregation::Kind k,
            std::enable_if_t<!is_dictionary<Source>()>* = nullptr>
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    update_target_element<Source, k, target_has_nulls, false>{}(
      target, target_index, source, source_index);
  }
  template <typename Source,
            aggregation::Kind k,
            std::enable_if_t<is_dictionary<Source>()>* = nullptr>
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
  }
};

/**
 * @brief Specialization function for dictionary type and aggregations.
 *
 * The `source` column is a dictionary type. This functor de-references the
 * dictionary's keys child column and maps the input source index through
 * the dictionary's indices child column to pass to the `update_target_element`
 * in the above `update_target_from_dictionary` using the type-dispatcher to
 * resolve the keys column type.
 *
 * `update_target_element( target, target_index, source.keys(), source.indices()[source_index] )`
 *
 * @tparam target_has_nulls Indicates presence of null elements in `target`
 * @tparam source_has_nulls Indicates presence of null elements in `source`.
 */
template <aggregation::Kind k, bool target_has_nulls, bool source_has_nulls>
struct update_target_element<
  dictionary32,
  k,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<not(k == aggregation::ARGMIN or k == aggregation::ARGMAX or
                       k == aggregation::COUNT_VALID or k == aggregation::COUNT_ALL)>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    dispatch_type_and_aggregation(
      source.child(cudf::dictionary_column_view::keys_column_index).type(),
      k,
      update_target_from_dictionary<target_has_nulls>{},
      target,
      target_index,
      source.child(cudf::dictionary_column_view::keys_column_index),
      static_cast<cudf::size_type>(source.element<dictionary32>(source_index)));
  }
};

template <typename T>
constexpr bool is_product_supported()
{
  return is_numeric<T>();
}

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element<Source,
                             aggregation::SUM_OF_SQUARES,
                             target_has_nulls,
                             source_has_nulls,
                             std::enable_if_t<is_product_supported<Source>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target = target_type_t<Source, aggregation::SUM_OF_SQUARES>;
    auto value   = static_cast<Target>(source.element<Source>(source_index));
    cudf::detail::atomic_add(&target.element<Target>(target_index), value * value);
    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element<Source,
                             aggregation::PRODUCT,
                             target_has_nulls,
                             source_has_nulls,
                             std::enable_if_t<is_product_supported<Source>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target = target_type_t<Source, aggregation::PRODUCT>;
    cudf::detail::atomic_mul(&target.element<Target>(target_index),
                             static_cast<Target>(source.element<Source>(source_index)));
    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element<
  Source,
  aggregation::COUNT_VALID,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<is_valid_aggregation<Source, aggregation::COUNT_VALID>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target = target_type_t<Source, aggregation::COUNT_VALID>;
    cudf::detail::atomic_add(&target.element<Target>(target_index), Target{1});

    // It is assumed the output for COUNT_VALID is initialized to be all valid
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element<
  Source,
  aggregation::COUNT_ALL,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<is_valid_aggregation<Source, aggregation::COUNT_ALL>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    using Target = target_type_t<Source, aggregation::COUNT_ALL>;
    cudf::detail::atomic_add(&target.element<Target>(target_index), Target{1});

    // It is assumed the output for COUNT_ALL is initialized to be all valid
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element<
  Source,
  aggregation::ARGMAX,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<is_valid_aggregation<Source, aggregation::ARGMAX>() and
                   cudf::is_relationally_comparable<Source, Source>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target = target_type_t<Source, aggregation::ARGMAX>;
    auto old     = cudf::detail::atomic_cas(
      &target.element<Target>(target_index), ARGMAX_SENTINEL, source_index);
    if (old != ARGMAX_SENTINEL) {
      while (source.element<Source>(source_index) > source.element<Source>(old)) {
        old = cudf::detail::atomic_cas(&target.element<Target>(target_index), old, source_index);
      }
    }

    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element<
  Source,
  aggregation::ARGMIN,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<is_valid_aggregation<Source, aggregation::ARGMIN>() and
                   cudf::is_relationally_comparable<Source, Source>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target = target_type_t<Source, aggregation::ARGMIN>;
    auto old     = cudf::detail::atomic_cas(
      &target.element<Target>(target_index), ARGMIN_SENTINEL, source_index);
    if (old != ARGMIN_SENTINEL) {
      while (source.element<Source>(source_index) < source.element<Source>(old)) {
        old = cudf::detail::atomic_cas(&target.element<Target>(target_index), old, source_index);
      }
    }

    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

/**
 * @brief Function object to update a single element in a target column by
 * performing an aggregation operation with a single element from a source
 * column.
 *
 * @tparam target_has_nulls Indicates presence of null elements in `target`
 * @tparam source_has_nulls Indicates presence of null elements in `source`.
 */
template <bool target_has_nulls = true, bool source_has_nulls = true>
struct elementwise_aggregator {
  template <typename Source, aggregation::Kind k>
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    update_target_element<Source, k, target_has_nulls, source_has_nulls>{}(
      target, target_index, source, source_index);
  }
};

/**
 * @brief Updates a row in `target` by performing elementwise aggregation
 * operations with a row in `source`.
 *
 * For the row in `target` specified by `target_index`, each element at `i` is
 * updated by:
 * ```c++
 * target_row[i] = aggs[i](target_row[i], source_row[i])
 * ```
 *
 * This function only supports aggregations that can be done in a "single pass",
 * i.e., given an initial value `R`, the aggregation `op` can be computed on a series
 * of elements `e[i] for i in [0,n)` by computing `R = op(e[i],R)` for any order
 * of the values of `i`.
 *
 * The initial value and validity of `R` depends on the aggregation:
 * SUM: 0 and NULL
 * MIN: Max value of type and NULL
 * MAX: Min value of type and NULL
 * COUNT_VALID: 0 and VALID
 * COUNT_ALL:   0 and VALID
 * ARGMAX: `ARGMAX_SENTINEL` and NULL
 * ARGMIN: `ARGMIN_SENTINEL` and NULL
 *
 * It is required that the elements of `target` be initialized with the corresponding
 * initial values and validity specified above.
 *
 * Handling of null elements in both `source` and `target` depends on the aggregation:
 * SUM, MIN, MAX, ARGMIN, ARGMAX:
 *  - `source`: Skipped
 *  - `target`: Updated from null to valid upon first successful aggregation
 * COUNT_VALID, COUNT_ALL:
 *  - `source`: Skipped
 *  - `target`: Cannot be null
 *
 * @param target Table containing the row to update
 * @param target_index Index of the row to update in `target`
 * @param source Table containing the row used to update the row in `target`.
 * The invariant `source.num_columns() >= target.num_columns()` must hold.
 * @param source_index Index of the row to use in `source`
 * @param aggs Array of aggregations to perform between elements of the `target`
 * and `source` rows. Must contain at least `target.num_columns()` valid
 * `aggregation::Kind` values.
 */
template <bool target_has_nulls = true, bool source_has_nulls = true>
__device__ inline void aggregate_row(mutable_table_device_view target,
                                     size_type target_index,
                                     table_device_view source,
                                     size_type source_index,
                                     aggregation::Kind const* aggs)
{
  for (auto i = 0; i < target.num_columns(); ++i) {
    dispatch_type_and_aggregation(source.column(i).type(),
                                  aggs[i],
                                  elementwise_aggregator<target_has_nulls, source_has_nulls>{},
                                  target.column(i),
                                  target_index,
                                  source.column(i),
                                  source_index);
  }
}

/**
 * @brief Dispatched functor to initialize a column with the identity of an
 * aggregation operation.
 *
 * Given a type `T` and `aggregation kind k`, determines and sets the value of
 * each element of the passed column to the appropriate initial value for the
 * aggregation.
 *
 * The initial values set as per aggregation are:
 * SUM: 0
 * COUNT_VALID: 0 and VALID
 * COUNT_ALL:   0 and VALID
 * MIN: Max value of type `T`
 * MAX: Min value of type `T`
 * ARGMAX: `ARGMAX_SENTINEL`
 * ARGMIN: `ARGMIN_SENTINEL`
 *
 * Only works on columns of fixed-width types.
 */
struct identity_initializer {
 private:
  template <typename T, aggregation::Kind k>
  static constexpr bool is_supported()
  {
    return cudf::is_fixed_width<T>() and
           (k == aggregation::SUM or k == aggregation::MIN or k == aggregation::MAX or
            k == aggregation::COUNT_VALID or k == aggregation::COUNT_ALL or
            k == aggregation::ARGMAX or k == aggregation::ARGMIN or
            k == aggregation::SUM_OF_SQUARES or k == aggregation::STD or
            k == aggregation::VARIANCE or
            (k == aggregation::PRODUCT and is_product_supported<T>()));
  }

  template <typename T, aggregation::Kind k>
  std::enable_if_t<not std::is_same_v<corresponding_operator_t<k>, void>, T>
  identity_from_operator()
  {
    using DeviceType = device_storage_type_t<T>;
    return corresponding_operator_t<k>::template identity<DeviceType>();
  }

  template <typename T, aggregation::Kind k>
  std::enable_if_t<std::is_same_v<corresponding_operator_t<k>, void>, T> identity_from_operator()
  {
    CUDF_FAIL("Unable to get identity/sentinel from device operator");
  }

  template <typename T, aggregation::Kind k>
  T get_identity()
  {
    if (k == aggregation::ARGMAX || k == aggregation::ARGMIN) {
      if constexpr (cudf::is_timestamp<T>())
        return k == aggregation::ARGMAX ? T{typename T::duration(ARGMAX_SENTINEL)}
                                        : T{typename T::duration(ARGMIN_SENTINEL)};
      else {
        using DeviceType = device_storage_type_t<T>;
        return k == aggregation::ARGMAX ? static_cast<DeviceType>(ARGMAX_SENTINEL)
                                        : static_cast<DeviceType>(ARGMIN_SENTINEL);
      }
    }
    return identity_from_operator<T, k>();
  }

 public:
  template <typename T, aggregation::Kind k>
  std::enable_if_t<is_supported<T, k>(), void> operator()(mutable_column_view const& col,
                                                          rmm::cuda_stream_view stream)
  {
    using DeviceType = device_storage_type_t<T>;
    thrust::fill(rmm::exec_policy(stream),
                 col.begin<DeviceType>(),
                 col.end<DeviceType>(),
                 get_identity<DeviceType, k>());
  }

  template <typename T, aggregation::Kind k>
  std::enable_if_t<not is_supported<T, k>(), void> operator()(mutable_column_view const& col,
                                                              rmm::cuda_stream_view stream)
  {
    CUDF_FAIL("Unsupported aggregation for initializing values");
  }
};

/**
 * @brief Initializes each column in a table with a corresponding identity value
 * of an aggregation operation.
 *
 * The `i`th column will be initialized with the identity value of the `i`th
 * aggregation operation in `aggs`.
 *
 * @throw cudf::logic_error if column type and corresponding agg are incompatible
 * @throw cudf::logic_error if column type is not fixed-width
 *
 * @param table The table of columns to initialize.
 * @param aggs A vector of aggregation operations corresponding to the table
 * columns. The aggregations determine the identity value for each column.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
void initialize_with_identity(mutable_table_view& table,
                              std::vector<aggregation::Kind> const& aggs,
                              rmm::cuda_stream_view stream);

}  // namespace detail
}  // namespace cudf
