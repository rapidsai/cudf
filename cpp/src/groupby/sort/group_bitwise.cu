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

#include <cudf/aggregation.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/std/functional>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>

namespace cudf::groupby::detail {

namespace {

struct bitwise_group_reduction_functor {
  template <typename T, CUDF_ENABLE_IF(std::is_integral_v<T>)>
  std::unique_ptr<column> operator()(bitwise_op bit_op,
                                     column_view const& values,
                                     device_span<size_type const> group_labels,
                                     size_type num_groups,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const

  {
    auto result =
      make_fixed_width_column(values.type(), num_groups, mask_state::UNALLOCATED, stream, mr);
    if (values.is_empty()) { return result; }

    auto const do_reduction = [&](auto const& inp_iter, auto const& out_iter, auto const& binop) {
      thrust::reduce_by_key(rmm::exec_policy_nosync(stream),
                            group_labels.data(),
                            group_labels.data() + group_labels.size(),
                            inp_iter,
                            thrust::make_discard_iterator(),
                            out_iter,
                            cuda::std::equal_to{},
                            binop);
    };

    auto const d_values_ptr       = column_device_view::create(values, stream);
    auto const result_begin       = result->mutable_view().begin<T>();
    auto const compute_bitwise_op = [&](auto const init, auto const bitop_fn) {
      auto const inp_values = cudf::detail::make_counting_transform_iterator(
        0, cudf::detail::null_replaced_value_accessor{*d_values_ptr, init, values.has_nulls()});
      do_reduction(inp_values, result_begin, bitop_fn);
    };

    if (bit_op == bitwise_op::AND) {
      using OpType = DeviceBitAnd;
      compute_bitwise_op(OpType::identity<T>(), OpType{});
    } else if (bit_op == bitwise_op::OR) {
      using OpType = DeviceBitOr;
      compute_bitwise_op(OpType::identity<T>(), OpType{});
    } else {  // if (bit_op == bitwise_op::XOR)
      using OpType = DeviceBitXor;
      compute_bitwise_op(OpType::identity<T>(), OpType{});
    }

    if (values.has_nulls()) {
      rmm::device_uvector<bool> validity(num_groups, stream);
      do_reduction(cudf::detail::make_validity_iterator(*d_values_ptr),
                   validity.begin(),
                   cuda::std::logical_or{});

      auto [null_mask, null_count] =
        cudf::detail::valid_if(validity.begin(), validity.end(), cuda::std::identity{}, stream, mr);
      if (null_count > 0) { result->set_null_mask(std::move(null_mask), null_count); }
    }
    return result;
  }

  template <typename T, typename... Args, CUDF_ENABLE_IF(!std::is_integral<T>())>
  std::unique_ptr<column> operator()(Args...) const
  {
    CUDF_FAIL("Bitwise operations are only supported for integral types.");
  }
};

}  // namespace

std::unique_ptr<column> group_bitwise(bitwise_op bit_op,
                                      column_view const& values,
                                      device_span<size_type const> group_labels,
                                      size_type num_groups,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  return cudf::type_dispatcher(values.type(),
                               bitwise_group_reduction_functor{},
                               bit_op,
                               values,
                               group_labels,
                               num_groups,
                               stream,
                               mr);
}

}  // namespace cudf::groupby::detail
