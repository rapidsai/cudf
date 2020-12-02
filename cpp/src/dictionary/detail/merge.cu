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
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/detail/merge.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace dictionary {
namespace detail {

std::unique_ptr<column> merge(dictionary_column_view const& lcol,
                              dictionary_column_view const& rcol,
                              cudf::detail::index_vector const& row_order,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  auto const lcol_iter = cudf::detail::indexalator_factory::make_input_iterator(lcol.indices());
  auto const rcol_iter = cudf::detail::indexalator_factory::make_input_iterator(rcol.indices());

  // create output indices column
  auto const merged_size  = lcol.size() + rcol.size();
  auto const indices_type = get_indices_type_for_size(merged_size);
  auto indices_column =
    make_fixed_width_column(indices_type, merged_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto output_iter =
    cudf::detail::indexalator_factory::make_output_iterator(indices_column->mutable_view());

  // merge the input indices columns into the output column
  thrust::transform(rmm::exec_policy(stream)->on(stream.value()),
                    row_order.begin(),
                    row_order.end(),
                    output_iter,
                    [lcol_iter, rcol_iter] __device__(auto const& index_pair) {
                      auto index = thrust::get<1>(index_pair);
                      return (thrust::get<0>(index_pair) == cudf::detail::side::LEFT
                                ? lcol_iter[index]
                                : rcol_iter[index]);
                    });

  // build dictionary; the validity mask is updated by the caller
  return make_dictionary_column(
    std::make_unique<column>(lcol.keys(), stream, mr),
    std::move(indices_column),
    rmm::device_buffer{
      lcol.has_nulls() || rcol.has_nulls() ? static_cast<size_t>(merged_size) : 0, stream, mr},
    lcol.null_count() + rcol.null_count());
}

}  // namespace detail
}  // namespace dictionary
}  // namespace cudf
