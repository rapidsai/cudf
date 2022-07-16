/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/detail/reduction_functions.hpp>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <reductions/simple.cuh>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

namespace cudf {
namespace reduction {
namespace detail {
namespace {

/**
 * @brief Compute reduction all() for dictionary columns.
 *
 * This compiles 10x faster than using thrust::reduce or the
 * cudf::simple::reduction::detail::reduce utility.
 * Both of these use the CUB DeviceReduce which aggressively inlines
 * the input iterator logic.
 */
struct all_fn {
  template <typename Iterator>
  struct all_true_fn {
    __device__ void operator()(size_type idx)
    {
      if (*d_result && (iter[idx] != *d_result)) atomicAnd(d_result, false);
    }
    Iterator iter;
    bool* d_result;
  };

  template <typename T, std::enable_if_t<std::is_arithmetic_v<T>>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& input,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    auto const d_dict = cudf::column_device_view::create(input, stream);
    auto const iter   = [&] {
      auto null_iter =
        cudf::reduction::op::min{}.template get_null_replacing_element_transformer<bool>();
      auto pair_iter =
        cudf::dictionary::detail::make_dictionary_pair_iterator<T>(*d_dict, input.has_nulls());
      return thrust::make_transform_iterator(pair_iter, null_iter);
    }();
    auto result = std::make_unique<numeric_scalar<bool>>(true, true, stream, mr);
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       input.size(),
                       all_true_fn<decltype(iter)>{iter, result->data()});
    return result;
  }
  template <typename T, std::enable_if_t<!std::is_arithmetic_v<T>>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const&,
                                     rmm::cuda_stream_view,
                                     rmm::mr::device_memory_resource*)
  {
    CUDF_FAIL("Unexpected key type for dictionary in reduction all()");
  }
};

}  // namespace
}  // namespace detail

std::unique_ptr<cudf::scalar> all(column_view const& col,
                                  cudf::data_type const output_dtype,
                                  std::optional<std::reference_wrapper<scalar const>> init,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(output_dtype == cudf::data_type(cudf::type_id::BOOL8),
               "all() operation can be applied with output type `BOOL8` only");

  if (cudf::is_dictionary(col.type())) {
    return cudf::type_dispatcher(
      dictionary_column_view(col).keys().type(), detail::all_fn{}, col, stream, mr);
  }
  // dispatch for non-dictionary types
  return cudf::type_dispatcher(
    col.type(),
    simple::detail::bool_result_element_dispatcher<cudf::reduction::op::min>{},
    col,
    init,
    stream,
    mr);
}

}  // namespace reduction
}  // namespace cudf
