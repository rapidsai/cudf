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
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <thrust/iterator/counting_iterator.h>

namespace cudf {
namespace detail {

using cudf::type_id;

namespace {

/// Check if sanitize checks can be skipped for a given type.
bool cannot_need_sanitize(cudf::type_id const& type)
{
  return type != type_id::STRING && type != type_id::LIST && type != type_id::STRUCT;
}

/// Check if the (STRING/LIST) column has any null rows with non-zero length.
bool has_dirty_rows(cudf::column_view const& input, rmm::cuda_stream_view stream)
{
  if (not input.nullable()) { return false; }  // No nulls => no dirty rows.

  // Cross-reference nullmask and offsets.
  auto const type         = input.type().id();
  auto const offsets      = (type == type_id::STRING) ? (strings_column_view{input}).offsets()
                                                      : (lists_column_view{input}).offsets();
  auto const d_input      = cudf::column_device_view::create(input);
  auto const is_dirty_row = [d_input = *d_input, offsets = offsets.begin<size_type>()] __device__(
                              size_type const& row_idx) {
    return d_input.is_null_nocheck(row_idx) && (offsets[row_idx] != offsets[row_idx + 1]);
  };

  auto const row_begin = thrust::make_counting_iterator<cudf::size_type>(0);
  auto const row_end   = row_begin + input.size();
  return thrust::any_of(rmm::exec_policy(stream), row_begin, row_end, is_dirty_row);
}

bool has_dirty_children(cudf::column_view const& input, rmm::cuda_stream_view stream)
{
  return std::any_of(input.child_begin(), input.child_end(), [stream](auto const& child) {
    return cudf::detail::needs_sanitize(child, stream);
  });
}

}  // namespace

/**
 * @copydoc cudf::detail::needs_sanitize
 */
bool needs_sanitize(cudf::column_view const& input, rmm::cuda_stream_view stream)
{
  auto const type = input.type().id();

  if (cannot_need_sanitize(type)) { return false; }

  // For types with variable-length rows, check if any rows are "dirty".
  // A dirty row is a null row with non-zero length.
  if ((type == type_id::STRING || type == type_id::LIST) && has_dirty_rows(input, stream)) {
    return true;
  }

  // For complex types, check if child columns need sanitization.
  if ((type == type_id::STRUCT || type == type_id::LIST) && has_dirty_children(input, stream)) {
    return true;
  }

  return false;
}

}  // namespace detail

/**
 * @copydoc cudf::needs_sanitize
 */
bool needs_sanitize(column_view const& input) { return detail::needs_sanitize(input); }

}  // namespace cudf
