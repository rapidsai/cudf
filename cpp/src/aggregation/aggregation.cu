/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>

namespace cudf {
namespace detail {
namespace {

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
    return (cudf::is_fixed_width<T>() and
            (k == aggregation::SUM or k == aggregation::MIN or k == aggregation::MAX or
             k == aggregation::COUNT_VALID or k == aggregation::COUNT_ALL or
             k == aggregation::ARGMAX or k == aggregation::ARGMIN or
             k == aggregation::SUM_OF_SQUARES or k == aggregation::M2 or k == aggregation::STD or
             k == aggregation::VARIANCE or
             (k == aggregation::PRODUCT and is_product_supported<T>()))) or
           (k == aggregation::SUM_WITH_OVERFLOW and std::is_same_v<T, cudf::struct_view>);
  }

  template <typename T, aggregation::Kind k>
  T identity_from_operator()
    requires(not std::is_same_v<corresponding_operator_t<k>, void>)
  {
    using DeviceType = device_storage_type_t<T>;
    return corresponding_operator_t<k>::template identity<DeviceType>();
  }

  template <typename T, aggregation::Kind k>
  T identity_from_operator()
    requires(std::is_same_v<corresponding_operator_t<k>, void>)
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
  void operator()(mutable_column_view const& col, rmm::cuda_stream_view stream)
    requires(is_supported<T, k>())
  {
    if constexpr (k == aggregation::SUM_WITH_OVERFLOW) {
      // SUM_WITH_OVERFLOW uses a struct with sum and overflow children
      auto sum_col      = col.child(0);
      auto overflow_col = col.child(1);

      // Initialize sum column using standard SUM aggregation dispatch
      dispatch_type_and_aggregation(
        sum_col.type(), aggregation::SUM, identity_initializer{}, sum_col, stream);
      thrust::uninitialized_fill_n(
        rmm::exec_policy_nosync(stream), overflow_col.begin<bool>(), col.size(), false);
    } else if constexpr (std::is_same_v<T, cudf::struct_view>) {
      // This should only happen for SUM_WITH_OVERFLOW, but handle it just in case
      CUDF_FAIL("Struct columns are only supported for SUM_WITH_OVERFLOW aggregation");
    } else {
      using DeviceType = device_storage_type_t<T>;
      thrust::fill(rmm::exec_policy_nosync(stream),
                   col.begin<DeviceType>(),
                   col.end<DeviceType>(),
                   get_identity<DeviceType, k>());
    }
  }

  template <typename T, aggregation::Kind k>
  void operator()(mutable_column_view const& col, rmm::cuda_stream_view stream)
    requires(not is_supported<T, k>())
  {
    CUDF_FAIL("Unsupported aggregation for initializing values");
  }
};
}  // namespace

void initialize_with_identity(mutable_table_view const& table,
                              host_span<cudf::aggregation::Kind const> aggs,
                              rmm::cuda_stream_view stream)
{
  // TODO: Initialize all the columns in a single kernel instead of invoking one
  // kernel per column
  for (size_type i = 0; i < table.num_columns(); ++i) {
    auto col = table.column(i);
    dispatch_type_and_aggregation(col.type(), aggs[i], identity_initializer{}, col, stream);
  }
}

}  // namespace detail
}  // namespace cudf
