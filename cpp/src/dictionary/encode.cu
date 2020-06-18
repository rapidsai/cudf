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
#include <cudf/detail/search.hpp>
#include <cudf/detail/stream_compaction.hpp>
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
  CUDF_EXPECTS(indices_type.id() == INT32, "only INT32 type for indices");
  CUDF_EXPECTS(input_column.type().id() != DICTIONARY32,
               "cannot encode a dictionary from a dictionary");

  // side effects of this function were are now dependent on:
  // - resulting column elements are sorted ascending
  // - nulls are sorted to the beginning
  auto table_keys = cudf::detail::drop_duplicates(table_view{{input_column}},
                                                  std::vector<size_type>{0},
                                                  duplicate_keep_option::KEEP_FIRST,
                                                  null_equality::EQUAL,
                                                  mr,
                                                  stream)
                      ->release();  // true == nulls are equal
  std::unique_ptr<column> keys_column(std::move(table_keys.front()));

  if (input_column.has_nulls()) {
    // the single null entry should be at the beginning -- side effect from drop_duplicates
    // copy the column without the null entry
    keys_column = std::make_unique<column>(
      slice(keys_column->view(), std::vector<size_type>{1, keys_column->size()}).front(),
      stream,
      mr);
    keys_column->set_null_mask(rmm::device_buffer{0, stream, mr}, 0);  // remove the null-mask
  }

  // this returns a column with no null entries
  // - it appears to ignore the null entries in the input and tries to place the value regardless
  auto indices_column = cudf::detail::lower_bound(table_view{{keys_column->view()}},
                                                  table_view{{input_column}},
                                                  std::vector<order>{order::ASCENDING},
                                                  std::vector<null_order>{null_order::AFTER},
                                                  mr,
                                                  stream);
  // we should probably copy/cast to INT32 type if different
  CUDF_EXPECTS(indices_column->type() == indices_type, "expecting INT32 indices type");

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
