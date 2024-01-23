/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "reader_impl.hpp"
#include "reader_impl_chunking.hpp"
#include "reader_impl_helpers.hpp"

namespace cudf::io::orc::detail {

reader::impl::impl(std::vector<std::unique_ptr<datasource>>&& sources,
                   orc_reader_options const& options,
                   rmm::cuda_stream_view stream,
                   rmm::mr::device_memory_resource* mr)
  : _stream(stream),
    _mr(mr),
    _timestamp_type{options.get_timestamp_type()},
    _use_index{options.is_enabled_use_index()},
    _use_np_dtypes{options.is_enabled_use_np_dtypes()},
    _decimal128_columns{options.get_decimal128_columns()},
    _col_meta{std::make_unique<reader_column_meta>()},
    _sources(std::move(sources)),
    _metadata{_sources, stream},
    _selected_columns{_metadata.select_columns(options.get_columns())}
{
}

table_with_metadata reader::impl::read(uint64_t skip_rows,
                                       std::optional<size_type> const& num_rows_opt,
                                       std::vector<std::vector<size_type>> const& stripes)
{
  prepare_data(skip_rows, num_rows_opt, stripes);
  return read_chunk_internal();
}

table_metadata reader::impl::make_output_metadata()
{
  if (_output_metadata) { return table_metadata{*_output_metadata}; }

  // Copy user data to the output metadata.
  table_metadata out_metadata;
  out_metadata.per_file_user_data.reserve(_metadata.per_file_metadata.size());
  std::transform(_metadata.per_file_metadata.cbegin(),
                 _metadata.per_file_metadata.cend(),
                 std::back_inserter(out_metadata.per_file_user_data),
                 [](auto const& meta) {
                   std::unordered_map<std::string, std::string> kv_map;
                   std::transform(meta.ff.metadata.cbegin(),
                                  meta.ff.metadata.cend(),
                                  std::inserter(kv_map, kv_map.end()),
                                  [](auto const& kv) {
                                    return std::pair{kv.name, kv.value};
                                  });
                   return kv_map;
                 });
  out_metadata.user_data = {out_metadata.per_file_user_data[0].begin(),
                            out_metadata.per_file_user_data[0].end()};

  // Save the output table metadata into `_output_metadata` for reuse next time.
  _output_metadata = std::make_unique<table_metadata>(out_metadata);

  return out_metadata;
}

table_with_metadata reader::impl::read_chunk_internal()
{
  // There is no columns in the table.
  if (_selected_columns.num_levels() == 0) { return {std::make_unique<table>(), table_metadata{}}; }

  std::vector<std::unique_ptr<column>> out_columns;
  auto out_metadata = make_output_metadata();

  // If no rows or stripes to read, return empty columns
  if (_file_itm_data->rows_to_read == 0 || _file_itm_data->selected_stripes.empty()) {
    std::transform(_selected_columns.levels[0].begin(),
                   _selected_columns.levels[0].end(),
                   std::back_inserter(out_columns),
                   [&](auto const col_meta) {
                     out_metadata.schema_info.emplace_back("");
                     return create_empty_column(col_meta.id,
                                                _metadata,
                                                _decimal128_columns,
                                                _use_np_dtypes,
                                                _timestamp_type,
                                                out_metadata.schema_info.back(),
                                                _stream);
                   });
    return {std::make_unique<table>(std::move(out_columns)), std::move(out_metadata)};
  }

  // Create columns from buffer with respective schema information.
  std::transform(
    _selected_columns.levels[0].begin(),
    _selected_columns.levels[0].end(),
    std::back_inserter(out_columns),
    [&](auto const& orc_col_meta) {
      out_metadata.schema_info.emplace_back("");
      auto col_buffer = assemble_buffer(
        orc_col_meta.id, 0, *_col_meta, _metadata, _selected_columns, _out_buffers, _stream, _mr);
      return make_column(col_buffer, &out_metadata.schema_info.back(), std::nullopt, _stream);
    });

  return {std::make_unique<table>(std::move(out_columns)), std::move(out_metadata)};
}

// Forward to implementation
reader::reader(std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
               orc_reader_options const& options,
               rmm::cuda_stream_view stream,
               rmm::mr::device_memory_resource* mr)
  : _impl{std::make_unique<impl>(std::move(sources), options, stream, mr)}
{
}

// Destructor within this translation unit
reader::~reader() = default;

// Forward to implementation
table_with_metadata reader::read(orc_reader_options const& options)
{
  return _impl->read(options.get_skip_rows(), options.get_num_rows(), options.get_stripes());
}

}  // namespace cudf::io::orc::detail
