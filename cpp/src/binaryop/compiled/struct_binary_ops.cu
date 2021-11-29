/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/detail/iterator.cuh>
#include <cudf/table/row_operators.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace binops::compiled::detail {
template <typename Comparator>
void struct_compare(mutable_column_view& out,
                    Comparator compare,
                    bool is_lhs_scalar,
                    bool is_rhs_scalar,
                    bool flip_output,
                    rmm::cuda_stream_view stream)
{
  auto d_out = column_device_view::create(out, stream);
  auto optional_iter =
    cudf::detail::make_optional_iterator<bool>(*d_out, contains_nulls::DYNAMIC{}, out.has_nulls());
  thrust::tabulate(
    rmm::exec_policy(stream),
    out.begin<bool>(),
    out.end<bool>(),
    [optional_iter, is_lhs_scalar, is_rhs_scalar, flip_output, compare] __device__(size_type i) {
      auto lhs = is_lhs_scalar ? 0 : i;
      auto rhs = is_rhs_scalar ? 0 : i;
      return optional_iter[i].has_value() and
             (flip_output ? not compare(lhs, rhs) : compare(lhs, rhs));
    });
}
}  //  namespace binops::compiled::detail

#define INSTANTIATE_STRUCT_COMPARE(comp_op)                        \
  template void binops::compiled::detail::struct_compare<comp_op>( \
    mutable_column_view&, comp_op, bool, bool, bool, rmm::cuda_stream_view);

INSTANTIATE_STRUCT_COMPARE(row_equality_comparator<true>);
INSTANTIATE_STRUCT_COMPARE(row_equality_comparator<false>);
INSTANTIATE_STRUCT_COMPARE(row_lexicographic_comparator<true>);
INSTANTIATE_STRUCT_COMPARE(row_lexicographic_comparator<false>);
}  //  namespace cudf
