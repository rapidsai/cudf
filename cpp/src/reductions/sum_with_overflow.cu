/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/reduction/detail/reduction_functions.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/numeric>
#include <thrust/transform_reduce.h>

namespace cudf::reduction::detail {

namespace {

// `wraps` is the net number of times the running sum has stepped outside [MIN, MAX].
// A final `wraps == 0` means the true sum fits in DeviceType, i.e. no overflow.
template <typename DeviceType>
struct sum_overflow_result {
  DeviceType sum;
  cudf::size_type wraps;

  CUDF_HOST_DEVICE sum_overflow_result() : sum{0}, wraps{0} {}
  CUDF_HOST_DEVICE sum_overflow_result(DeviceType s, cudf::size_type w) : sum{s}, wraps{w} {}
};

template <typename DeviceType>
struct overflow_sum_op {
  __device__ sum_overflow_result<DeviceType> operator()(
    sum_overflow_result<DeviceType> const& lhs, sum_overflow_result<DeviceType> const& rhs) const
  {
    auto const r     = cuda::add_overflow<DeviceType>(lhs.sum, rhs.sum);
    auto const carry = r.overflow ? (rhs.sum > DeviceType{0} ? 1 : -1) : 0;
    return sum_overflow_result<DeviceType>{r.value, lhs.wraps + rhs.wraps + carry};
  }
};

template <typename DeviceType>
struct to_sum_overflow {
  __device__ sum_overflow_result<DeviceType> operator()(DeviceType value) const
  {
    return sum_overflow_result<DeviceType>{value, 0};
  }
};

template <typename DeviceType>
struct null_aware_to_sum_overflow {
  cudf::column_device_view dcol;

  CUDF_HOST_DEVICE null_aware_to_sum_overflow(cudf::column_device_view const& d) : dcol{d} {}

  __device__ sum_overflow_result<DeviceType> operator()(cudf::size_type idx) const
  {
    return dcol.is_valid(idx) ? sum_overflow_result<DeviceType>{dcol.element<DeviceType>(idx), 0}
                              : sum_overflow_result<DeviceType>{DeviceType{0}, 0};
  }
};

template <typename Source>
std::unique_ptr<cudf::scalar> make_sum_overflow_struct_scalar(
  device_storage_type_t<Source> sum_value,
  bool overflow_value,
  bool sum_is_valid,
  cudf::data_type const& source_type,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const temp_mr = cudf::get_current_device_resource_ref();

  std::unique_ptr<cudf::scalar> sum_scalar;
  if constexpr (cudf::is_fixed_point<Source>()) {
    sum_scalar = cudf::make_fixed_point_scalar<Source>(
      sum_value, numeric::scale_type{source_type.scale()}, stream, temp_mr);
  } else {
    sum_scalar =
      cudf::make_fixed_width_scalar<Source>(static_cast<Source>(sum_value), stream, temp_mr);
  }
  sum_scalar->set_valid_async(sum_is_valid, stream);
  auto overflow_scalar = cudf::make_fixed_width_scalar<bool>(overflow_value, stream, temp_mr);

  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(cudf::make_column_from_scalar(*sum_scalar, 1, stream, temp_mr));
  children.push_back(cudf::make_column_from_scalar(*overflow_scalar, 1, stream, temp_mr));

  std::vector<cudf::column_view> child_views{children[0]->view(), children[1]->view()};
  return cudf::make_struct_scalar(
    cudf::host_span<cudf::column_view const>{child_views}, stream, mr);
}

template <typename Source>
std::unique_ptr<cudf::scalar> sum_with_overflow_impl(
  column_view const& col,
  std::optional<std::reference_wrapper<scalar const>> init,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  using DeviceType = device_storage_type_t<Source>;

  if (init.has_value() && !init.value().get().is_valid(stream)) {
    return make_sum_overflow_struct_scalar<Source>(
      DeviceType{0}, false, false, col.type(), stream, mr);
  }

  if (col.size() == 0 || col.size() == col.null_count()) {
    return make_sum_overflow_struct_scalar<Source>(
      DeviceType{0}, false, false, col.type(), stream, mr);
  }

  auto dcol = cudf::column_device_view::create(col, stream);

  sum_overflow_result<DeviceType> initial_value{DeviceType{0}, 0};
  if (init.has_value()) {
    auto const& init_scalar = static_cast<cudf::scalar_type_t<Source> const&>(init.value().get());
    initial_value.sum       = static_cast<DeviceType>(init_scalar.value(stream));
  }

  auto counting_iter = cuda::counting_iterator<cudf::size_type>{0};
  sum_overflow_result<DeviceType> result;

  if (col.has_nulls()) {
    result = thrust::transform_reduce(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      counting_iter,
      counting_iter + col.size(),
      null_aware_to_sum_overflow<DeviceType>{*dcol},
      initial_value,
      overflow_sum_op<DeviceType>{});
  } else {
    auto input_iter = dcol->begin<DeviceType>();
    result          = thrust::transform_reduce(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      input_iter,
      input_iter + col.size(),
      to_sum_overflow<DeviceType>{},
      initial_value,
      overflow_sum_op<DeviceType>{});
  }

  // On overflow the sum value is unspecified; the boolean flag is the source of truth.
  return make_sum_overflow_struct_scalar<Source>(
    result.sum, result.wraps != 0, true, col.type(), stream, mr);
}

struct sum_with_overflow_dispatcher {
  template <typename Source>
    requires((cudf::is_integral_not_bool<Source>() && cudf::is_signed<Source>()) ||
             cudf::is_fixed_point<Source>())
  std::unique_ptr<cudf::scalar> operator()(column_view const& col,
                                           std::optional<std::reference_wrapper<scalar const>> init,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const
  {
    return sum_with_overflow_impl<Source>(col, init, stream, mr);
  }

  template <typename Source>
    requires(!((cudf::is_integral_not_bool<Source>() && cudf::is_signed<Source>()) ||
               cudf::is_fixed_point<Source>()))
  std::unique_ptr<cudf::scalar> operator()(column_view const&,
                                           std::optional<std::reference_wrapper<scalar const>>,
                                           rmm::cuda_stream_view,
                                           rmm::device_async_resource_ref) const
  {
    CUDF_FAIL("SUM_WITH_OVERFLOW reduction supports only signed integer and decimal types.",
              std::invalid_argument);
  }
};

}  // namespace

std::unique_ptr<cudf::scalar> sum_with_overflow(
  column_view const& col,
  cudf::data_type const output_dtype,
  std::optional<std::reference_wrapper<scalar const>> init,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(output_dtype.id() == type_id::STRUCT,
               "SUM_WITH_OVERFLOW output dtype must be STRUCT.",
               std::invalid_argument);
  return cudf::type_dispatcher(col.type(), sum_with_overflow_dispatcher{}, col, init, stream, mr);
}

}  // namespace cudf::reduction::detail
