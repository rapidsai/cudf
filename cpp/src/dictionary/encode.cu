/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace dictionary {
namespace detail {
/**
 * @copydoc cudf::dictionary::encode
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> encode(column_view const& input_column,
                               data_type indices_type,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(is_signed(indices_type) && is_index_type(indices_type),
               "indices must be type signed integer");
  CUDF_EXPECTS(input_column.type().id() != type_id::DICTIONARY32,
               "cannot encode a dictionary from a dictionary");

  auto codified       = cudf::detail::encode(cudf::table_view({input_column}), stream, mr);
  auto keys_table     = std::move(codified.first);
  auto indices_column = std::move(codified.second);
  auto keys_column    = std::move(keys_table->release().front());

  if (keys_column->has_nulls()) {
    keys_column = std::make_unique<column>(
      cudf::detail::slice(
        keys_column->view(), std::vector<size_type>{0, keys_column->size() - 1}, stream)
        .front(),
      stream,
      mr);
    keys_column->set_null_mask(rmm::device_buffer{0, stream, mr}, 0);  // remove the null-mask
  }

  // create column with keys_column and indices_column
  return make_dictionary_column(std::move(keys_column),
                                std::move(indices_column),
                                cudf::detail::copy_bitmask(input_column, stream, mr),
                                input_column.null_count());
}

/**
 * @copydoc cudf::dictionary::detail::get_indices_type_for_size
 */
data_type get_indices_type_for_size(size_type keys_size)
{
  if (keys_size <= std::numeric_limits<int8_t>::max()) return data_type{type_id::INT8};
  if (keys_size <= std::numeric_limits<int16_t>::max()) return data_type{type_id::INT16};
  return data_type{type_id::INT32};
}

}  // namespace detail

// external API

std::unique_ptr<column> encode(column_view const& input_column,
                               data_type indices_type,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::encode(input_column, indices_type, stream, mr);
}

}  // namespace dictionary
}  // namespace cudf
