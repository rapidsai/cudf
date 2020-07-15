/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/strings/copying.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <thrust/sequence.h>

namespace cudf {
namespace strings {
namespace detail {
// new strings column from subset of this strings instance
std::unique_ptr<cudf::column> slice(strings_column_view const& strings,
                                    size_type start,
                                    size_type end,
                                    size_type step,
                                    cudaStream_t stream,
                                    rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = strings.size();
  if (strings_count == 0) return make_empty_strings_column(mr, stream);
  if (step == 0) step = 1;
  CUDF_EXPECTS(step > 0, "Parameter step must be positive integer.");
  if (end < 0 || end > strings_count) end = strings_count;
  CUDF_EXPECTS(((start >= 0) && (start < end)), "Invalid start parameter value.");
  strings_count = cudf::util::round_up_safe<size_type>((end - start), step);
  //
  auto execpol = rmm::exec_policy(stream);
  // build indices
  rmm::device_vector<size_type> indices(strings_count);
  thrust::sequence(execpol->on(stream), indices.begin(), indices.end(), start, step);
  // create a column_view as a wrapper of these indices
  column_view indices_view(
    data_type{type_id::INT32}, strings_count, indices.data().get(), nullptr, 0);
  // build a new strings column from the indices
  auto sliced_table = cudf::detail::gather(table_view{{strings.parent()}},
                                           indices_view,
                                           cudf::detail::out_of_bounds_policy::NULLIFY,
                                           cudf::detail::negative_index_policy::NOT_ALLOWED,
                                           mr,
                                           stream)
                        ->release();
  std::unique_ptr<column> output_column(std::move(sliced_table.front()));
  if (output_column->null_count() == 0)
    output_column->set_null_mask(rmm::device_buffer{0, stream, mr}, 0);
  return output_column;
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
