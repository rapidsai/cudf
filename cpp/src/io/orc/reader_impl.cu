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

// #define PRINT_DEBUG

// TODO: remove
#include <cudf_test/debug_utilities.hpp>

#include <cudf/concatenate.hpp>
//
//
//
#include "io/comp/gpuinflate.hpp"
#include "io/comp/nvcomp_adapter.hpp"
#include "io/orc/reader_impl.hpp"
#include "io/orc/reader_impl_chunking.hpp"
#include "io/orc/reader_impl_helpers.hpp"
#include "io/utilities/config_utils.hpp"

#include <cudf/detail/copy.hpp>
#include <cudf/detail/timezone.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <algorithm>
#include <iterator>

namespace cudf::io::orc::detail {

void reader::impl::prepare_data(int64_t skip_rows,
                                std::optional<int64_t> const& num_rows_opt,
                                std::vector<std::vector<size_type>> const& stripes,
                                read_mode mode)
{
  // Selected columns at different levels of nesting are stored in different elements
  // of `selected_columns`; thus, size == 1 means no nested columns
  CUDF_EXPECTS(skip_rows == 0 or _selected_columns.num_levels() == 1,
               "skip_rows is not supported by nested columns");

  // There are no columns in the table.
  if (_selected_columns.num_levels() == 0) { return; }

  std::cout << "call global, skip = " << skip_rows << std::endl;

  global_preprocess(skip_rows, num_rows_opt, stripes, mode);

  if (!_chunk_read_data.more_table_chunk_to_output()) {
    if (!_chunk_read_data.more_stripe_to_decode() && _chunk_read_data.more_stripe_to_load()) {
      printf("load more data\n\n");
      load_data();
    }

    if (_chunk_read_data.more_stripe_to_decode()) {
      printf("decode more data\n\n");
      decompress_and_decode();
    }
  }

  printf("done load and decode data\n\n");

  // decompress_and_decode();
  // while (_chunk_read_data.more_stripe_to_decode()) {
  //   decompress_and_decode();
  //   _file_itm_data.out_buffers.push_back(std::move(_out_buffers));
  // }
}

table_with_metadata reader::impl::make_output_chunk()
{
  {
    _stream.synchronize();
    auto peak_mem = mem_stats_logger.peak_memory_usage();
    std::cout << "start to make out, peak_memory_usage: " << peak_mem << "("
              << (peak_mem * 1.0) / (1024.0 * 1024.0) << " MB)" << std::endl;
  }

  // There is no columns in the table.
  if (_selected_columns.num_levels() == 0) { return {std::make_unique<table>(), table_metadata{}}; }

  // If no rows or stripes to read, return empty columns
  if (!_chunk_read_data.more_table_chunk_to_output()) {
    printf("has no next\n");
    std::vector<std::unique_ptr<column>> out_columns;
    auto out_metadata = get_meta_with_user_data();
    std::transform(_selected_columns.levels[0].begin(),
                   _selected_columns.levels[0].end(),
                   std::back_inserter(out_columns),
                   [&](auto const& col_meta) {
                     out_metadata.schema_info.emplace_back("");
                     return create_empty_column(col_meta.id,
                                                _metadata,
                                                _config.decimal128_columns,
                                                _config.use_np_dtypes,
                                                _config.timestamp_type,
                                                out_metadata.schema_info.back(),
                                                _stream);
                   });
    return {std::make_unique<table>(std::move(out_columns)), std::move(out_metadata)};
  }

#if 1
  auto out_table = [&] {
    if (_chunk_read_data.output_table_chunks.size() == 1) {
      _chunk_read_data.curr_output_table_chunk++;
      printf("one chunk, no more table---------------------------------\n");
      return std::move(_chunk_read_data.decoded_table);
    }

    {
      _stream.synchronize();
      auto peak_mem = mem_stats_logger.peak_memory_usage();
      std::cout << "prepare to make out, peak_memory_usage: " << peak_mem << "("
                << (peak_mem * 1.0) / (1024.0 * 1024.0) << " MB)" << std::endl;
    }

    auto const out_chunk =
      _chunk_read_data.output_table_chunks[_chunk_read_data.curr_output_table_chunk++];
    auto const out_tview =
      cudf::detail::slice(_chunk_read_data.decoded_table->view(),
                          {static_cast<size_type>(out_chunk.start_idx),
                           static_cast<size_type>(out_chunk.start_idx + out_chunk.count)},
                          _stream)[0];
    {
      _stream.synchronize();
      auto peak_mem = mem_stats_logger.peak_memory_usage();
      std::cout << "done make out, peak_memory_usage: " << peak_mem << "("
                << (peak_mem * 1.0) / (1024.0 * 1024.0) << " MB)" << std::endl;
    }

    // If this is the last slice, we also delete the decoded_table to free up memory.
    if (!_chunk_read_data.more_table_chunk_to_output()) {
      _chunk_read_data.decoded_table.reset(nullptr);
    }

    return std::make_unique<table>(out_tview, _stream, _mr);
  }();

#endif

  if (!_chunk_read_data.has_next()) {
    static int count{0};
    count++;
    _stream.synchronize();
    auto peak_mem = mem_stats_logger.peak_memory_usage();
    std::cout << "complete, " << count << ", peak_memory_usage: " << peak_mem
              << " , MB = " << (peak_mem * 1.0) / (1024.0 * 1024.0) << std::endl;
  } else {
    _stream.synchronize();
    auto peak_mem = mem_stats_logger.peak_memory_usage();
    std::cout << "done, partial, peak_memory_usage: " << peak_mem
              << " , MB = " << (peak_mem * 1.0) / (1024.0 * 1024.0) << std::endl;
  }

  return {std::move(out_table), _out_metadata};
}

table_metadata reader::impl::get_meta_with_user_data()
{
  if (_meta_with_user_data) { return table_metadata{*_meta_with_user_data}; }

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

  // Save the output table metadata into `_meta_with_user_data` for reuse next time.
  _meta_with_user_data = std::make_unique<table_metadata>(out_metadata);

  return out_metadata;
}

reader::impl::impl(std::vector<std::unique_ptr<datasource>>&& sources,
                   orc_reader_options const& options,
                   rmm::cuda_stream_view stream,
                   rmm::mr::device_memory_resource* mr)
  : reader::impl::impl(0UL, 0UL, std::move(sources), options, stream, mr)
{
}

reader::impl::impl(std::size_t output_size_limit,
                   std::size_t data_read_limit,
                   std::vector<std::unique_ptr<datasource>>&& sources,
                   orc_reader_options const& options,
                   rmm::cuda_stream_view stream,
                   rmm::mr::device_memory_resource* mr)
  : reader::impl::impl(output_size_limit,
                       data_read_limit,
                       DEFAULT_OUTPUT_ROW_GRANULARITY,
                       std::move(sources),
                       options,
                       stream,
                       mr)
{
}

reader::impl::impl(std::size_t output_size_limit,
                   std::size_t data_read_limit,
                   size_type output_row_granularity,
                   std::vector<std::unique_ptr<datasource>>&& sources,
                   orc_reader_options const& options,
                   rmm::cuda_stream_view stream,
                   rmm::mr::device_memory_resource* mr)
  : _stream(stream),
    _mr(mr),
    mem_stats_logger(mr),
    _config{options.get_timestamp_type(),
            options.is_enabled_use_index(),
            options.is_enabled_use_np_dtypes(),
            options.get_decimal128_columns(),
            options.get_skip_rows(),
            options.get_num_rows(),
            options.get_stripes()},
    _col_meta{std::make_unique<reader_column_meta>()},
    _sources(std::move(sources)),
    _metadata{_sources, stream},
    _selected_columns{_metadata.select_columns(options.get_columns())},
    _chunk_read_data{
      output_size_limit,
      data_read_limit,
      output_row_granularity > 0 ? output_row_granularity : DEFAULT_OUTPUT_ROW_GRANULARITY}
{
  printf("construct reader , limit = %d, %d, gradunarity %d \n",

         (int)output_size_limit,
         (int)data_read_limit,
         (int)output_row_granularity

  );
}

table_with_metadata reader::impl::read(int64_t skip_rows,
                                       std::optional<int64_t> const& num_rows_opt,
                                       std::vector<std::vector<size_type>> const& stripes)
{
  prepare_data(skip_rows, num_rows_opt, stripes, read_mode::READ_ALL);
  return make_output_chunk();
}

bool reader::impl::has_next()
{
  printf("==================query has next \n");
  prepare_data(
    _config.skip_rows, _config.num_read_rows, _config.selected_stripes, read_mode::CHUNKED_READ);

  printf("has next: %d\n", (int)_chunk_read_data.has_next());
  return _chunk_read_data.has_next();
}

table_with_metadata reader::impl::read_chunk()
{
  printf("==================call read chunk\n");
  {
    _stream.synchronize();
    auto peak_mem = mem_stats_logger.peak_memory_usage();
    std::cout << "\n\n\n------------start read chunk, peak_memory_usage: " << peak_mem << "("
              << (peak_mem * 1.0) / (1024.0 * 1024.0) << " MB)" << std::endl;
  }

  {
    static int count{0};
    ++count;

#if 0
    if (count == 3) {
      _file_itm_data.lvl_stripe_data.clear();
      {
        _stream.synchronize();
        auto peak_mem = mem_stats_logger.peak_memory_usage();
        std::cout << "clear all, peak_memory_usage: " << peak_mem << "("
                  << (peak_mem * 1.0) / (1024.0 * 1024.0) << " MB)" << std::endl;
      }
      exit(0);
    }
#endif
  }

  prepare_data(
    _config.skip_rows, _config.num_read_rows, _config.selected_stripes, read_mode::CHUNKED_READ);

  {
    _stream.synchronize();
    auto peak_mem = mem_stats_logger.peak_memory_usage();
    std::cout << "done prepare data, peak_memory_usage: " << peak_mem << "("
              << (peak_mem * 1.0) / (1024.0 * 1024.0) << " MB)" << std::endl;
  }

  return make_output_chunk();
}

}  // namespace cudf::io::orc::detail
