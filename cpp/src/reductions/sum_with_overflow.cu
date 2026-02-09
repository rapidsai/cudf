/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/reduction/detail/reduction_functions.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/limits>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

namespace cudf::reduction::detail {

// Simple pair to hold sum and overflow flag
struct sum_overflow_result {
  int64_t sum;
  bool overflow;

  CUDF_HOST_DEVICE sum_overflow_result() : sum(0), overflow(false) {}
  CUDF_HOST_DEVICE sum_overflow_result(int64_t s, bool o) : sum(s), overflow(o) {}
};

// Binary operator for combining sum_overflow_result values
struct overflow_sum_op {
  __device__ sum_overflow_result operator()(sum_overflow_result const& lhs,
                                            sum_overflow_result const& rhs) const
  {
    // If either operand already has overflow, result has overflow
    if (lhs.overflow || rhs.overflow) {
      // Still compute the sum for consistency, but mark as overflow
      // This addition may wrap but we've already detected overflow
      return sum_overflow_result{lhs.sum + rhs.sum, true};
    }

    // Check for overflow BEFORE performing the addition to avoid UB
    bool overflow_detected = false;

    // Check for positive overflow: would the addition exceed INT64_MAX?
    if (rhs.sum > 0 && lhs.sum > cuda::std::numeric_limits<int64_t>::max() - rhs.sum) {
      overflow_detected = true;
    }
    // Check for negative overflow: would the addition go below INT64_MIN?
    else if (rhs.sum < 0 && lhs.sum < cuda::std::numeric_limits<int64_t>::min() - rhs.sum) {
      overflow_detected = true;
    }

    // Perform the addition (safe if no overflow detected)
    int64_t const result_sum = lhs.sum + rhs.sum;

    return sum_overflow_result{result_sum, overflow_detected};
  }
};

// Transform function to convert int64_t values to sum_overflow_result
struct to_sum_overflow {
  __device__ sum_overflow_result operator()(int64_t value) const
  {
    return sum_overflow_result{value, false};
  }
};

// Transform functor for null-aware conversion using index
struct null_aware_to_sum_overflow {
  cudf::column_device_view const* dcol_ptr;

  CUDF_HOST_DEVICE null_aware_to_sum_overflow(cudf::column_device_view const* dcol) : dcol_ptr(dcol)
  {
  }

  __device__ sum_overflow_result operator()(cudf::size_type idx) const
  {
    return dcol_ptr->is_valid(idx) ? sum_overflow_result{dcol_ptr->element<int64_t>(idx), false}
                                   : sum_overflow_result{0, false};
  }
};

std::unique_ptr<cudf::scalar> sum_with_overflow(
  column_view const& col,
  cudf::data_type const output_dtype,
  std::optional<std::reference_wrapper<scalar const>> init,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // SUM_WITH_OVERFLOW only supports int64_t input
  CUDF_EXPECTS(col.type().id() == cudf::type_id::INT64,
               "SUM_WITH_OVERFLOW only supports int64_t input types",
               std::invalid_argument);

  // Handle empty column
  if (col.size() == 0 || col.size() == col.null_count()) {
    // Create struct with {null sum, false overflow}
    auto sum_scalar =
      cudf::make_default_constructed_scalar(cudf::data_type{cudf::type_id::INT64}, stream, mr);
    sum_scalar->set_valid_async(false, stream);
    auto overflow_scalar = cudf::make_fixed_width_scalar<bool>(false, stream, mr);

    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(cudf::make_column_from_scalar(*sum_scalar, 1, stream, mr));
    children.push_back(cudf::make_column_from_scalar(*overflow_scalar, 1, stream, mr));

    // Use host_span of column_views instead of table_view to avoid double wrapping
    std::vector<cudf::column_view> child_views;
    child_views.push_back(children[0]->view());
    child_views.push_back(children[1]->view());

    return cudf::make_struct_scalar(
      cudf::host_span<cudf::column_view const>{child_views}, stream, mr);
  }

  // Create device view
  auto dcol = cudf::column_device_view::create(col, stream);

  // Set up initial value
  sum_overflow_result initial_value{0, false};
  if (init.has_value() && init.value().get().is_valid(stream)) {
    auto const& init_scalar = static_cast<cudf::numeric_scalar<int64_t> const&>(init.value().get());
    initial_value.sum       = init_scalar.value(stream);
  }

  // Perform the reduction using thrust::transform_reduce
  auto counting_iter = thrust::make_counting_iterator<cudf::size_type>(0);
  auto dcol_ptr      = dcol.get();
  sum_overflow_result result;

  if (col.has_nulls()) {
    // Use null-aware transform functor
    result = thrust::transform_reduce(rmm::exec_policy_nosync(stream),
                                      counting_iter,
                                      counting_iter + col.size(),
                                      null_aware_to_sum_overflow{dcol_ptr},
                                      initial_value,
                                      overflow_sum_op{});
  } else {
    // Use direct iterator for non-null case
    auto input_iter = dcol->begin<int64_t>();
    result          = thrust::transform_reduce(rmm::exec_policy_nosync(stream),
                                      input_iter,
                                      input_iter + col.size(),
                                      to_sum_overflow{},
                                      initial_value,
                                      overflow_sum_op{});
  }

  // Create result struct scalar with {sum: int64_t, overflow: bool}
  auto sum_scalar      = cudf::make_fixed_width_scalar<int64_t>(result.sum, stream, mr);
  auto overflow_scalar = cudf::make_fixed_width_scalar<bool>(result.overflow, stream, mr);

  // Create struct scalar using cudf::make_struct_scalar with host_span of column_views
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(cudf::make_column_from_scalar(*sum_scalar, 1, stream, mr));
  children.push_back(cudf::make_column_from_scalar(*overflow_scalar, 1, stream, mr));

  // Use host_span of column_views instead of table_view to avoid double wrapping
  std::vector<cudf::column_view> child_views;
  child_views.push_back(children[0]->view());
  child_views.push_back(children[1]->view());

  return cudf::make_struct_scalar(
    cudf::host_span<cudf::column_view const>{child_views}, stream, mr);
}

}  // namespace cudf::reduction::detail
