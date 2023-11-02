/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
#include <cudf/detail/gather.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace dictionary {
namespace detail {
/**
 * @brief Decode a column from a dictionary.
 */
std::unique_ptr<column> decode(dictionary_column_view const& source,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  if (source.is_empty()) return make_empty_column(type_id::EMPTY);

  column_view indices{source.indices().type(),
                      source.size(),
                      source.indices().head(),
                      nullptr,  // no nulls for gather indices
                      0,
                      source.offset()};
  // use gather to create the output column -- use ignore_out_of_bounds=true
  auto table_column = cudf::detail::gather(table_view{{source.keys()}},
                                           indices,
                                           cudf::out_of_bounds_policy::NULLIFY,
                                           cudf::detail::negative_index_policy::NOT_ALLOWED,
                                           stream,
                                           mr)
                        ->release();
  auto output_column = std::unique_ptr<column>(std::move(table_column.front()));

  // apply any nulls to the output column
  output_column->set_null_mask(cudf::detail::copy_bitmask(source.parent(), stream, mr),
                               source.null_count());

  return output_column;
}

}  // namespace detail

std::unique_ptr<column> decode(dictionary_column_view const& source,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::decode(source, stream, mr);
}

}  // namespace dictionary
}  // namespace cudf
