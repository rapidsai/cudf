/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy_if_else.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/replace.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

namespace cudf {
namespace detail {
namespace {

struct replace_nans_functor {
  template <typename T, typename Replacement>
  std::enable_if_t<std::is_floating_point_v<T>, std::unique_ptr<column>> operator()(
    column_view const& input,
    Replacement const& replacement,
    bool replacement_nullable,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    CUDF_EXPECTS(input.type() == replacement.type(),
                 "Input and replacement must be of the same type");

    if (input.is_empty()) { return cudf::make_empty_column(input.type()); }

    auto input_device_view = column_device_view::create(input, stream);
    size_type size         = input.size();

    auto predicate = [dinput = *input_device_view] __device__(auto i) {
      return dinput.is_null(i) or !std::isnan(dinput.element<T>(i));
    };

    auto input_iterator =
      make_optional_iterator<T>(*input_device_view, nullate::DYNAMIC{input.has_nulls()});
    auto replacement_iterator =
      make_optional_iterator<T>(replacement, nullate::DYNAMIC{replacement_nullable});
    return copy_if_else(input.has_nulls() or replacement_nullable,
                        input_iterator,
                        input_iterator + size,
                        replacement_iterator,
                        predicate,
                        input.type(),
                        stream,
                        mr);
  }

  template <typename T, typename... Args>
  std::enable_if_t<!std::is_floating_point_v<T>, std::unique_ptr<column>> operator()(Args&&...)
  {
    CUDF_FAIL("NAN is not supported in a Non-floating point type column");
  }
};

}  // namespace

std::unique_ptr<column> replace_nans(column_view const& input,
                                     column_view const& replacement,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(input.size() == replacement.size(),
               "Input and replacement must be of the same size");

  return type_dispatcher(input.type(),
                         replace_nans_functor{},
                         input,
                         *column_device_view::create(replacement, stream),
                         replacement.nullable(),
                         stream,
                         mr);
}

std::unique_ptr<column> replace_nans(column_view const& input,
                                     scalar const& replacement,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  return type_dispatcher(
    input.type(), replace_nans_functor{}, input, replacement, true, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> replace_nans(column_view const& input,
                                     column_view const& replacement,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_nans(input, replacement, cudf::get_default_stream(), mr);
}

std::unique_ptr<column> replace_nans(column_view const& input,
                                     scalar const& replacement,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_nans(input, replacement, cudf::get_default_stream(), mr);
}

}  // namespace cudf

namespace {  // anonymous

template <typename T>
struct normalize_nans_and_zeros_lambda {
  cudf::column_device_view in;
  T __device__ operator()(cudf::size_type i)
  {
    auto e = in.element<T>(i);
    if (isnan(e)) { return std::numeric_limits<T>::quiet_NaN(); }
    if (T{0.0} == e) { return T{0.0}; }
    return e;
  }
};

/**
 * @brief Functor called by the `type_dispatcher` in order to invoke and instantiate
 *        `normalize_nans_and_zeros` with the appropriate data types.
 */
struct normalize_nans_and_zeros_kernel_forwarder {
  // floats and doubles. what we really care about.
  template <typename T, std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
  void operator()(cudf::column_device_view in,
                  cudf::mutable_column_device_view out,
                  rmm::cuda_stream_view stream)
  {
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(in.size()),
                      out.head<T>(),
                      normalize_nans_and_zeros_lambda<T>{in});
  }

  // if we get in here for anything but a float or double, that's a problem.
  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point_v<T>, void> operator()(Args&&...)
  {
    CUDF_FAIL("Unexpected non floating-point type.");
  }
};

}  // end anonymous namespace

namespace cudf {
namespace detail {
void normalize_nans_and_zeros(mutable_column_view in_out, rmm::cuda_stream_view stream)
{
  if (in_out.is_empty()) { return; }
  CUDF_EXPECTS(
    in_out.type() == data_type(type_id::FLOAT32) || in_out.type() == data_type(type_id::FLOAT64),
    "Expects float or double input");

  // wrapping the in_out data in a column_view so we can call the same lower level code.
  // that we use for the non in-place version.
  column_view input = in_out;

  // to device. unique_ptr which gets automatically cleaned up when we leave
  auto device_in = column_device_view::create(input, stream);

  // from device. unique_ptr which gets automatically cleaned up when we leave.
  auto device_out = mutable_column_device_view::create(in_out, stream);

  // invoke the actual kernel.
  cudf::type_dispatcher(
    input.type(), normalize_nans_and_zeros_kernel_forwarder{}, *device_in, *device_out, stream);
}

std::unique_ptr<column> normalize_nans_and_zeros(column_view const& input,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::mr::device_memory_resource* mr)
{
  // output. copies the input
  auto out = std::make_unique<column>(input, stream, mr);

  // from device. unique_ptr which gets automatically cleaned up when we leave.
  auto out_view = out->mutable_view();
  normalize_nans_and_zeros(out_view, stream);
  out->set_null_count(input.null_count());

  return out;
}

}  // namespace detail

/**
 * @brief Makes all Nans and zeroes positive.
 *
 * Converts floating point values from @p in_out using the following rules:
 *        Convert  -NaN  -> NaN
 *        Convert  -0.0  -> 0.0
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
std::unique_ptr<column> normalize_nans_and_zeros(column_view const& input,
                                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::normalize_nans_and_zeros(input, cudf::get_default_stream(), mr);
}

/**
 * @brief Makes all Nans and zeroes positive.
 *
 * Converts floating point values from @p in_out using the following rules:
 *        Convert  -NaN  -> NaN
 *        Convert  -0.0  -> 0.0
 *
 * @throws cudf::logic_error if column does not have floating point data type.
 * @param[in, out] in_out mutable_column_view representing input data. data is processed in-place
 */
void normalize_nans_and_zeros(mutable_column_view& in_out)
{
  CUDF_FUNC_RANGE();
  detail::normalize_nans_and_zeros(in_out, cudf::get_default_stream());
}

}  // namespace cudf
