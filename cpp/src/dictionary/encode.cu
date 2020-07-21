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
#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

namespace cudf {
namespace dictionary {
namespace detail {
/**
 * @brief Create a new dictionary column from a column_view.
 *
 */
std::unique_ptr<column> encode(column_view const& input_column,
                               data_type indices_type,
                               rmm::mr::device_memory_resource* mr,
                               cudaStream_t stream)
{
  CUDF_EXPECTS(indices_type.id() == type_id::INT32, "only type_id::INT32 type for indices");
  CUDF_EXPECTS(input_column.type().id() != type_id::DICTIONARY32,
               "cannot encode a dictionary from a dictionary");

  // this returns a column with no null entries
  // - it appears to ignore the null entries in the input and tries to place the value regardless
  auto codified       = cudf::detail::encode(input_column, mr, stream);
  auto keys_column    = std::move(codified.first);
  auto indices_column = std::move(codified.second);

  // we should probably copy/cast to type_id::INT32 type if different
  CUDF_EXPECTS(indices_column->type() == indices_type, "expecting type_id::INT32 indices type");

  // create column with keys_column and indices_column
  return make_dictionary_column(std::move(keys_column),
                                std::move(indices_column),
                                copy_bitmask(input_column, stream, mr),
                                input_column.null_count());
}

}  // namespace detail

// external API

std::unique_ptr<column> encode(column_view const& input_column,
                               data_type indices_type,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::encode(input_column, indices_type, mr);
}

}  // namespace dictionary
}  // namespace cudf
