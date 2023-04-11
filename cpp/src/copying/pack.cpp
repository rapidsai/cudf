/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cudf/detail/copy.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

namespace {

/**
 * @brief The data that is stored as anonymous bytes in the `packed_columns` metadata
 * field.
 *
 * The metadata field of the `packed_columns` struct is simply an array of these.
 * This struct is exposed here because it is needed by both contiguous_split, pack
 * and unpack.
 */
struct serialized_column {
  serialized_column(data_type _type,
                    size_type _size,
                    size_type _null_count,
                    int64_t _data_offset,
                    int64_t _null_mask_offset,
                    size_type _num_children)
    : type(_type),
      size(_size),
      null_count(_null_count),
      data_offset(_data_offset),
      null_mask_offset(_null_mask_offset),
      num_children(_num_children),
      pad(0)
  {
  }

  data_type type;
  size_type size;
  size_type null_count;
  int64_t data_offset;       // offset into contiguous data buffer, or -1 if column data is null
  int64_t null_mask_offset;  // offset into contiguous data buffer, or -1 if column data is null
  size_type num_children;
  // Explicitly pad to avoid uninitialized padding bits, allowing `serialized_column` to be bit-wise
  // comparable
  int pad;
};

/**
 * @brief Deserialize a single column into a column_view
 *
 * Deserializes a single column (it's children are assumed to be already deserialized)
 * non-recursively.
 *
 * @param serial_column Serialized column information
 * @param children Children for the column
 * @param base_ptr Base pointer for the entire contiguous buffer from which all columns
 * were serialized into
 * @return Fully formed column_view
 */
column_view deserialize_column(serialized_column serial_column,
                               std::vector<column_view> const& children,
                               uint8_t const* base_ptr)
{
  auto const data_ptr =
    serial_column.data_offset != -1 ? base_ptr + serial_column.data_offset : nullptr;
  auto const null_mask_ptr =
    serial_column.null_mask_offset != -1
      ? reinterpret_cast<bitmask_type const*>(base_ptr + serial_column.null_mask_offset)
      : nullptr;

  return column_view(serial_column.type,
                     serial_column.size,
                     data_ptr,
                     null_mask_ptr,
                     serial_column.null_count,
                     0,
                     children);
}

/**
 * @brief Build and add metadata for a column and all of it's children, recursively
 *
 *
 * @param metadata Output vector of serialized_column metadata
 * @param col Column to build metadata for
 * @param base_ptr Base pointer for the entire contiguous buffer from which all columns
 * were serialized into
 * @param data_size Size of the incoming buffer
 */
void build_column_metadata(std::vector<serialized_column>& metadata,
                           column_view const& col,
                           uint8_t const* base_ptr,
                           size_t data_size)
{
  uint8_t const* data_ptr = col.size() == 0 || !col.head<uint8_t>() ? nullptr : col.head<uint8_t>();
  if (data_ptr != nullptr) {
    CUDF_EXPECTS(data_ptr >= base_ptr && data_ptr < base_ptr + data_size,
                 "Encountered column data outside the range of input buffer");
  }
  int64_t const data_offset = data_ptr ? data_ptr - base_ptr : -1;

  uint8_t const* null_mask_ptr = col.size() == 0 || !col.nullable()
                                   ? nullptr
                                   : reinterpret_cast<uint8_t const*>(col.null_mask());
  if (null_mask_ptr != nullptr) {
    CUDF_EXPECTS(null_mask_ptr >= base_ptr && null_mask_ptr < base_ptr + data_size,
                 "Encountered column null mask outside the range of input buffer");
  }
  int64_t const null_mask_offset = null_mask_ptr ? null_mask_ptr - base_ptr : -1;

  // add metadata
  metadata.emplace_back(
    col.type(), col.size(), col.null_count(), data_offset, null_mask_offset, col.num_children());

  std::for_each(
    col.child_begin(), col.child_end(), [&metadata, &base_ptr, &data_size](column_view const& col) {
      build_column_metadata(metadata, col, base_ptr, data_size);
    });
}

}  // anonymous namespace

/**
 * @copydoc cudf::detail::pack
 */
packed_columns pack(cudf::table_view const& input,
                    rmm::cuda_stream_view stream,
                    rmm::mr::device_memory_resource* mr)
{
  // do a contiguous_split with no splits to get the memory for the table
  // arranged as we want it
  auto contig_split_result = cudf::detail::contiguous_split(input, {}, stream, mr);
  return contig_split_result.empty() ? packed_columns{} : std::move(contig_split_result[0].data);
}

template <typename ColumnIter>
packed_columns::metadata pack_metadata(ColumnIter begin,
                                       ColumnIter end,
                                       uint8_t const* contiguous_buffer,
                                       size_t buffer_size)
{
  std::vector<serialized_column> metadata;

  // first metadata entry is a stub indicating how many total (top level) columns
  // there are
  metadata.emplace_back(
    data_type{type_id::EMPTY}, static_cast<size_type>(std::distance(begin, end)), 0, -1, -1, 0);

  std::for_each(begin, end, [&metadata, &contiguous_buffer, &buffer_size](column_view const& col) {
    build_column_metadata(metadata, col, contiguous_buffer, buffer_size);
  });

  // convert to anonymous bytes
  std::vector<uint8_t> metadata_bytes;
  auto const metadata_begin = reinterpret_cast<uint8_t const*>(metadata.data());
  std::copy(metadata_begin,
            metadata_begin + (metadata.size() * sizeof(serialized_column)),
            std::back_inserter(metadata_bytes));

  return packed_columns::metadata{std::move(metadata_bytes)};
}

/**
 * @copydoc cudf::detail::unpack
 */
table_view unpack(uint8_t const* metadata, uint8_t const* gpu_data)
{
  // gpu data can be null if everything is empty but the metadata must always be valid
  CUDF_EXPECTS(metadata != nullptr, "Encountered invalid packed column input");
  auto serialized_columns = reinterpret_cast<serialized_column const*>(metadata);
  uint8_t const* base_ptr = gpu_data;
  // first entry is a stub where size == the total # of top level columns (see pack_metadata above)
  auto const num_columns = serialized_columns[0].size;
  size_t current_index   = 1;

  std::function<std::vector<column_view>(size_type)> get_columns;
  get_columns = [&serialized_columns, &current_index, base_ptr, &get_columns](size_t num_columns) {
    std::vector<column_view> cols;
    for (size_t i = 0; i < num_columns; i++) {
      auto serial_column = serialized_columns[current_index];
      current_index++;

      std::vector<column_view> children = get_columns(serial_column.num_children);

      cols.emplace_back(deserialize_column(serial_column, children, base_ptr));
    }

    return cols;
  };

  return table_view{get_columns(num_columns)};
}

}  // namespace detail

/**
 * @copydoc cudf::pack
 */
packed_columns pack(cudf::table_view const& input, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::pack(input, cudf::get_default_stream(), mr);
}

/**
 * @copydoc cudf::pack_metadata
 */
packed_columns::metadata pack_metadata(table_view const& table,
                                       uint8_t const* contiguous_buffer,
                                       size_t buffer_size)
{
  CUDF_FUNC_RANGE();
  return table.is_empty()
           ? packed_columns::metadata{}
           : detail::pack_metadata(table.begin(), table.end(), contiguous_buffer, buffer_size);
}

/**
 * @copydoc cudf::unpack
 */
table_view unpack(packed_columns const& input)
{
  CUDF_FUNC_RANGE();
  return input.metadata_->size() == 0
           ? table_view{}
           : detail::unpack(input.metadata_->data(),
                            reinterpret_cast<uint8_t const*>(input.gpu_data->data()));
}

/**
 * @copydoc cudf::unpack(uint8_t const*, uint8_t const* )
 */
table_view unpack(uint8_t const* metadata, uint8_t const* gpu_data)
{
  CUDF_FUNC_RANGE();
  return detail::unpack(metadata, gpu_data);
}

}  // namespace cudf
