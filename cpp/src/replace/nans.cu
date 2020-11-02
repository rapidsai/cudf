/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/transform_scan.h>

namespace cudf {
namespace detail {
namespace {

struct replace_nans_functor {
  template <typename T, typename Replacement>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<column>> operator()(
    column_view const& input,
    Replacement const& replacement,
    bool replacement_nullable,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
  {
    CUDF_EXPECTS(input.type() == replacement.type(),
                 "Input and replacement must be of the same type");

    if (input.is_empty()) { return cudf::make_empty_column(input.type()); }

    auto input_device_view = column_device_view::create(input);
    size_type size         = input.size();

    auto predicate = [dinput = *input_device_view] __device__(auto i) {
      return dinput.is_null(i) or !std::isnan(dinput.element<T>(i));
    };

    if (input.has_nulls()) {
      auto input_pair_iterator = make_pair_iterator<T, true>(*input_device_view);
      if (replacement_nullable) {
        auto replacement_pair_iterator = make_pair_iterator<T, true>(replacement);
        return copy_if_else(true,
                            input_pair_iterator,
                            input_pair_iterator + size,
                            replacement_pair_iterator,
                            predicate,
                            stream,
                            mr);
      } else {
        auto replacement_pair_iterator = make_pair_iterator<T, false>(replacement);
        return copy_if_else(true,
                            input_pair_iterator,
                            input_pair_iterator + size,
                            replacement_pair_iterator,
                            predicate,
                            stream,
                            mr);
      }
    } else {
      auto input_pair_iterator = make_pair_iterator<T, false>(*input_device_view);
      if (replacement_nullable) {
        auto replacement_pair_iterator = make_pair_iterator<T, true>(replacement);
        return copy_if_else(true,
                            input_pair_iterator,
                            input_pair_iterator + size,
                            replacement_pair_iterator,
                            predicate,
                            stream,
                            mr);
      } else {
        auto replacement_pair_iterator = make_pair_iterator<T, false>(replacement);
        return copy_if_else(false,
                            input_pair_iterator,
                            input_pair_iterator + size,
                            replacement_pair_iterator,
                            predicate,
                            stream,
                            mr);
      }
    }
  }

  template <typename T, typename... Args>
  std::enable_if_t<!std::is_floating_point<T>::value, std::unique_ptr<column>> operator()(
    Args&&... args)
  {
    CUDF_FAIL("NAN is not supported in a Non-floating point type column");
  }
};

}  // namespace
std::unique_ptr<column> replace_nans(column_view const& input,
                                     column_view const& replacement,
                                     cudaStream_t stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(input.size() == replacement.size(),
               "Input and replacement must be of the same size");

  return type_dispatcher(input.type(),
                         replace_nans_functor{},
                         input,
                         *column_device_view::create(replacement),
                         replacement.nullable(),
                         mr,
                         stream);
}

std::unique_ptr<column> replace_nans(column_view const& input,
                                     scalar const& replacement,
                                     cudaStream_t stream,
                                     rmm::mr::device_memory_resource* mr)
{
  return type_dispatcher(
    input.type(), replace_nans_functor{}, input, replacement, true, mr, stream);
}

}  // namespace detail

std::unique_ptr<column> replace_nans(column_view const& input,
                                     column_view const& replacement,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_nans(input, replacement, 0, mr);
}

std::unique_ptr<column> replace_nans(column_view const& input,
                                     scalar const& replacement,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_nans(input, replacement, 0, mr);
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
  template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
  void operator()(cudf::column_device_view in,
                  cudf::mutable_column_device_view out,
                  cudaStream_t stream)
  {
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(in.size()),
                      out.head<T>(),
                      normalize_nans_and_zeros_lambda<T>{in});
  }

  // if we get in here for anything but a float or double, that's a problem.
  template <typename T, std::enable_if_t<not std::is_floating_point<T>::value>* = nullptr>
  void operator()(cudf::column_device_view in,
                  cudf::mutable_column_device_view out,
                  cudaStream_t stream)
  {
    CUDF_FAIL("Unexpected non floating-point type.");
  }
};

}  // end anonymous namespace

namespace cudf {
namespace detail {
void normalize_nans_and_zeros(mutable_column_view in_out, cudaStream_t stream = 0)
{
  if (in_out.is_empty()) { return; }
  CUDF_EXPECTS(
    in_out.type() == data_type(type_id::FLOAT32) || in_out.type() == data_type(type_id::FLOAT64),
    "Expects float or double input");

  // wrapping the in_out data in a column_view so we can call the same lower level code.
  // that we use for the non in-place version.
  column_view input = in_out;

  // to device. unique_ptr which gets automatically cleaned up when we leave
  auto device_in = column_device_view::create(input);

  // from device. unique_ptr which gets automatically cleaned up when we leave.
  auto device_out = mutable_column_device_view::create(in_out);

  // invoke the actual kernel.
  cudf::type_dispatcher(
    input.type(), normalize_nans_and_zeros_kernel_forwarder{}, *device_in, *device_out, stream);
}

}  // namespace detail

/**
 * @brief Makes all NaNs and zeroes positive.
 *
 * Converts floating point values from @p input using the following rules:
 *        Convert  -NaN  -> NaN
 *        Convert  -0.0  -> 0.0
 *
 * @throws cudf::logic_error if column does not have floating point data type.
 * @param[in] column_view representing input data
 * @param[in] device_memory_resource allocator for allocating output data
 *
 * @returns new column with the modified data
 */
std::unique_ptr<column> normalize_nans_and_zeros(column_view const& input,
                                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  // output. copies the input
  std::unique_ptr<column> out = std::make_unique<column>(input, (cudaStream_t)0, mr);
  // from device. unique_ptr which gets automatically cleaned up when we leave.
  auto out_view = out->mutable_view();

  detail::normalize_nans_and_zeros(out_view, 0);

  return out;
}

/**
 * @brief Makes all Nans and zeroes positive.
 *
 * Converts floating point values from @p in_out using the following rules:
 *        Convert  -NaN  -> NaN
 *        Convert  -0.0  -> 0.0
 *
 * @throws cudf::logic_error if column does not have floating point data type.
 * @param[in, out] mutable_column_view representing input data. data is processed in-place
 */
void normalize_nans_and_zeros(mutable_column_view& in_out)
{
  CUDF_FUNC_RANGE();
  detail::normalize_nans_and_zeros(in_out, 0);
}

}  // namespace cudf
