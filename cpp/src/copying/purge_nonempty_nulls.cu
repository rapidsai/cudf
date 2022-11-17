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
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/utilities/default_stream.hpp>

#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>

namespace cudf {
namespace detail {

using cudf::type_id;

namespace {

/// Check if nonempty-null checks can be skipped for a given type.
bool type_may_have_nonempty_nulls(cudf::type_id const& type)
{
  return type == type_id::STRING || type == type_id::LIST || type == type_id::STRUCT;
}

/// Check if the (STRING/LIST) column has any null rows with non-zero length.
bool has_nonempty_null_rows(cudf::column_view const& input, rmm::cuda_stream_view stream)
{
  if (not input.has_nulls()) { return false; }  // No nulls => no dirty rows.

  // Cross-reference nullmask and offsets.
  auto const type         = input.type().id();
  auto const offsets      = (type == type_id::STRING) ? (strings_column_view{input}).offsets()
                                                      : (lists_column_view{input}).offsets();
  auto const d_input      = cudf::column_device_view::create(input, stream);
  auto const is_dirty_row = [d_input = *d_input, offsets = offsets.begin<size_type>()] __device__(
                              size_type const& row_idx) {
    return d_input.is_null_nocheck(row_idx) && (offsets[row_idx] != offsets[row_idx + 1]);
  };

  auto const row_begin = thrust::counting_iterator<cudf::size_type>(0);
  auto const row_end   = row_begin + input.size();
  return thrust::count_if(rmm::exec_policy(stream), row_begin, row_end, is_dirty_row) > 0;
}

}  // namespace

/**
 * @copydoc cudf::detail::has_nonempty_nulls
 */
bool has_nonempty_nulls(cudf::column_view const& input, rmm::cuda_stream_view stream)
{
  auto const type = input.type().id();

  if (not type_may_have_nonempty_nulls(type)) { return false; }

  // For types with variable-length rows, check if any rows are "dirty".
  // A dirty row is a null row with non-zero length.
  if ((type == type_id::STRING || type == type_id::LIST) && has_nonempty_null_rows(input, stream)) {
    return true;
  }

  // For complex types, check if child columns need purging.
  if ((type == type_id::STRUCT || type == type_id::LIST) &&
      std::any_of(input.child_begin(), input.child_end(), [stream](auto const& child) {
        return cudf::detail::has_nonempty_nulls(child, stream);
      })) {
    return true;
  }

  return false;
}

std::unique_ptr<column> purge_nonempty_nulls(column_view const& input,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  // If not compound types (LIST/STRING/STRUCT/DICTIONARY) then just copy the input into output.
  if (!cudf::is_compound(input.type())) { return std::make_unique<column>(input, stream, mr); }

  // Implement via identity gather.
  auto gathered_table = cudf::detail::gather(table_view{{input}},
                                             thrust::make_counting_iterator(0),
                                             thrust::make_counting_iterator(input.size()),
                                             out_of_bounds_policy::DONT_CHECK,
                                             stream,
                                             mr);
  return std::move(gathered_table->release().front());
}

}  // namespace detail

/**
 * @copydoc cudf::may_have_nonempty_nulls
 */
bool may_have_nonempty_nulls(column_view const& input)
{
  auto const type = input.type().id();

  if (not detail::type_may_have_nonempty_nulls(type)) { return false; }

  if ((type == type_id::STRING || type == type_id::LIST) && input.has_nulls()) { return true; }

  if ((type == type_id::STRUCT || type == type_id::LIST) &&
      std::any_of(input.child_begin(), input.child_end(), may_have_nonempty_nulls)) {
    return true;
  }

  return false;
}

/**
 * @copydoc cudf::has_nonempty_nulls
 */
bool has_nonempty_nulls(column_view const& input)
{
  return detail::has_nonempty_nulls(input, cudf::get_default_stream());
}

/**
 * @copydoc cudf::purge_nonempty_nulls(column_view const&, rmm::mr::device_memory_resource*)
 */
std::unique_ptr<cudf::column> purge_nonempty_nulls(column_view const& input,
                                                   rmm::mr::device_memory_resource* mr)
{
  return detail::purge_nonempty_nulls(input, cudf::get_default_stream(), mr);
}

}  // namespace cudf
