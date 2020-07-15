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
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

namespace cudf {
namespace dictionary {
namespace detail {
/**
 * @brief Decode a column from a dictionary.
 */
std::unique_ptr<column> decode(dictionary_column_view const& source,
                               rmm::mr::device_memory_resource* mr,
                               cudaStream_t stream)
{
  if (source.size() == 0) return make_empty_column(data_type{type_id::EMPTY});

  column_view indices{cudf::data_type{cudf::type_id::INT32},
                      source.size(),
                      source.indices().head<int32_t>(),
                      nullptr,
                      0,
                      source.offset()};  // no nulls for gather indices
  // use gather to create the output column -- use ignore_out_of_bounds=true
  auto table_column = cudf::detail::gather(table_view{{source.keys()}},
                                           indices,
                                           cudf::detail::out_of_bounds_policy::IGNORE,
                                           cudf::detail::negative_index_policy::NOT_ALLOWED,
                                           mr,
                                           stream)
                        ->release();
  auto output_column = std::unique_ptr<column>(std::move(table_column.front()));

  // apply any nulls to the output column
  output_column->set_null_mask(copy_bitmask(source.parent(), stream, mr), source.null_count());

  return output_column;
}

}  // namespace detail

std::unique_ptr<column> decode(dictionary_column_view const& source,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::decode(source, mr);
}

}  // namespace dictionary
}  // namespace cudf
