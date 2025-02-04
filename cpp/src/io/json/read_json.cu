/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include "io/comp/io_uncomp.hpp"
#include "io/json/nested_json.hpp"
#include "io/utilities/getenv_or.hpp"
#include "read_json.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/json.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scatter.h>

#include <BS_thread_pool.hpp>
#include <BS_thread_pool_utils.hpp>

#include <numeric>

namespace cudf::io::json::detail {

namespace {

namespace pools {

BS::thread_pool& tpool()
{
  static std::size_t pool_size =
    getenv_or("LIBCUDF_HOST_COMPRESSION_NUM_THREADS", std::thread::hardware_concurrency());
  static BS::thread_pool _tpool(pool_size);
  return _tpool;
}

}  // namespace pools

class compressed_host_buffer_source final : public datasource {
 public:
  explicit compressed_host_buffer_source(std::unique_ptr<datasource> const& src,
                                         compression_type comptype)
    : _comptype{comptype}, _dbuf_ptr{src->host_read(0, src->size())}
  {
    auto ch_buffer = host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(_dbuf_ptr->data()),
                                              _dbuf_ptr->size());
    if (_comptype == compression_type::GZIP || _comptype == compression_type::ZIP ||
        _comptype == compression_type::SNAPPY) {
      _decompressed_ch_buffer_size = cudf::io::detail::get_uncompressed_size(_comptype, ch_buffer);
    } else {
      _decompressed_buffer         = cudf::io::detail::decompress(_comptype, ch_buffer);
      _decompressed_ch_buffer_size = _decompressed_buffer.size();
    }
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    auto ch_buffer = host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(_dbuf_ptr->data()),
                                              _dbuf_ptr->size());
    if (_decompressed_buffer.empty()) {
      auto decompressed_hbuf = cudf::io::detail::decompress(_comptype, ch_buffer);
      auto const count       = std::min(size, decompressed_hbuf.size() - offset);
      bool partial_read      = offset + count < decompressed_hbuf.size();
      if (!partial_read) {
        std::memcpy(dst, decompressed_hbuf.data() + offset, count);
        return count;
      }
      _decompressed_buffer = std::move(decompressed_hbuf);
    }
    auto const count = std::min(size, _decompressed_buffer.size() - offset);
    std::memcpy(dst, _decompressed_buffer.data() + offset, count);
    return count;
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    auto ch_buffer = host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(_dbuf_ptr->data()),
                                              _dbuf_ptr->size());
    if (_decompressed_buffer.empty()) {
      auto decompressed_hbuf = cudf::io::detail::decompress(_comptype, ch_buffer);
      auto const count       = std::min(size, decompressed_hbuf.size() - offset);
      bool partial_read      = offset + count < decompressed_hbuf.size();
      if (!partial_read)
        return std::make_unique<owning_buffer<std::vector<uint8_t>>>(
          std::move(decompressed_hbuf), decompressed_hbuf.data() + offset, count);
      _decompressed_buffer = std::move(decompressed_hbuf);
    }
    auto const count = std::min(size, _decompressed_buffer.size() - offset);
    return std::make_unique<non_owning_buffer>(_decompressed_buffer.data() + offset, count);
  }

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view stream) override
  {
    auto& thread_pool = pools::tpool();
    return thread_pool.submit_task([this, offset, size, dst, stream] {
      auto hbuf = host_read(offset, size);
      CUDF_CUDA_TRY(
        cudaMemcpyAsync(dst, hbuf->data(), hbuf->size(), cudaMemcpyHostToDevice, stream.value()));
      stream.synchronize();
      return hbuf->size();
    });
  }

  [[nodiscard]] bool supports_device_read() const override { return true; }

  [[nodiscard]] size_t size() const override { return _decompressed_ch_buffer_size; }

 private:
  std::unique_ptr<datasource::buffer> _dbuf_ptr;
  compression_type _comptype;
  size_t _decompressed_ch_buffer_size;
  std::vector<std::uint8_t> _decompressed_buffer;
};

// Return total size of sources enclosing the passed range
std::size_t sources_size(host_span<std::unique_ptr<datasource>> const sources,
                         std::size_t range_offset,
                         std::size_t range_size)
{
  return std::accumulate(sources.begin(), sources.end(), 0ul, [=](std::size_t sum, auto& source) {
    auto const size = source->size();
    // TODO take care of 0, 0, or *, 0 case.
    return sum +
           (range_size == 0 or range_offset + range_size > size ? size - range_offset : range_size);
  });
}

// Return estimated size of subchunk using a heuristic involving the byte range size and the minimum
// subchunk size
std::size_t estimate_size_per_subchunk(std::size_t chunk_size)
{
  auto geometric_mean = [](double a, double b) { return std::sqrt(a * b); };
  // NOTE: heuristic for choosing subchunk size: geometric mean of minimum subchunk size (set to
  // 10kb) and the byte range size
  return geometric_mean(std::ceil(static_cast<double>(chunk_size) / num_subchunks),
                        min_subchunk_size);
}

/**
 * @brief Return the batch size for the JSON reader.
 *
 * The datasources passed to the JSON reader are read iteratively in batches demarcated by byte
 * range offsets. The tokenizer requires the JSON buffer read in each batch to be of size at most
 * INT_MAX bytes.
 * Since the byte range corresponding to a given batch can cause the last JSON line
 * in the batch to be incomplete, the batch size returned by this function allows for an additional
 * `max_subchunks_prealloced` subchunks to be allocated beyond the byte range offsets. Since the
 * size of the subchunk depends on the size of the byte range, the batch size is variable and cannot
 * be directly controlled by the user. As a workaround, the environment variable
 * LIBCUDF_JSON_BATCH_SIZE can be used to set a fixed batch size at runtime.
 *
 * @return size in bytes
 */
std::size_t get_batch_size(std::size_t chunk_size)
{
  auto const size_per_subchunk = estimate_size_per_subchunk(chunk_size);
  auto const batch_limit       = static_cast<std::size_t>(std::numeric_limits<int32_t>::max()) -
                           (max_subchunks_prealloced * size_per_subchunk);
  return std::min(batch_limit, getenv_or<std::size_t>("LIBCUDF_JSON_BATCH_SIZE", batch_limit));
}

/**
 * @brief Extract the first delimiter character position in the string
 *
 * @param d_data Device span in which to search for delimiter character
 * @param delimiter Delimiter character to search for
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return Position of first delimiter character in device array
 */
size_type find_first_delimiter(device_span<char const> d_data,
                               char const delimiter,
                               rmm::cuda_stream_view stream)
{
  auto const first_delimiter_position =
    thrust::find(rmm::exec_policy(stream), d_data.begin(), d_data.end(), delimiter);
  return first_delimiter_position != d_data.end()
           ? static_cast<size_type>(thrust::distance(d_data.begin(), first_delimiter_position))
           : -1;
}

/**
 * @brief Get the byte range between record starts and ends starting from the given range.
 *
 * if get_byte_range_offset == 0, then we can skip the first delimiter search
 * if get_byte_range_offset != 0, then we need to search for the first delimiter in given range.
 * if not found, skip this chunk, if found, then search for first delimiter in next range until we
 * find a delimiter. Use this as actual range for parsing.
 *
 * @param sources Data sources to read from
 * @param reader_opts JSON reader options with range offset and range size
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @returns Data source owning buffer enclosing the bytes read
 */
datasource::owning_buffer<rmm::device_buffer> get_record_range_raw_input(
  host_span<std::unique_ptr<datasource>> sources,
  json_reader_options const& reader_opts,
  rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  std::size_t const total_source_size = sources_size(sources, 0, 0);
  auto constexpr num_delimiter_chars  = 1;
  auto const delimiter                = reader_opts.get_delimiter();
  auto const num_extra_delimiters     = num_delimiter_chars * sources.size();
  std::size_t const chunk_offset      = reader_opts.get_byte_range_offset();
  std::size_t chunk_size              = reader_opts.get_byte_range_size();

  CUDF_EXPECTS(total_source_size ? chunk_offset < total_source_size : !chunk_offset,
               "Invalid offsetting",
               std::invalid_argument);
  auto should_load_till_last_source = !chunk_size || chunk_size >= total_source_size - chunk_offset;
  chunk_size = should_load_till_last_source ? total_source_size - chunk_offset : chunk_size;

  int num_subchunks_prealloced        = should_load_till_last_source ? 0 : max_subchunks_prealloced;
  std::size_t const size_per_subchunk = estimate_size_per_subchunk(chunk_size);

  std::size_t buffer_size =
    std::min(total_source_size, chunk_size + num_subchunks_prealloced * size_per_subchunk) +
    num_extra_delimiters;
  rmm::device_buffer buffer(buffer_size, stream);
  device_span<char> bufspan(reinterpret_cast<char*>(buffer.data()), buffer.size());

  // Offset within buffer indicating first read position
  std::int64_t buffer_offset = 0;
  auto readbufspan =
    ingest_raw_input(bufspan, sources, chunk_offset, chunk_size, delimiter, stream);

  auto const shift_for_nonzero_offset = std::min<std::int64_t>(chunk_offset, 1);
  auto const first_delim_pos =
    chunk_offset == 0 ? 0 : find_first_delimiter(readbufspan, delimiter, stream);
  if (first_delim_pos == -1) {
    // return empty owning datasource buffer
    auto empty_buf = rmm::device_buffer(0, stream);
    return datasource::owning_buffer<rmm::device_buffer>(std::move(empty_buf));
  } else if (!should_load_till_last_source) {
    // Find next delimiter
    std::int64_t next_delim_pos     = -1;
    std::size_t next_subchunk_start = chunk_offset + chunk_size;
    while (next_delim_pos < buffer_offset) {
      for (int subchunk = 0;
           subchunk < num_subchunks_prealloced && next_delim_pos < buffer_offset &&
           next_subchunk_start < total_source_size;
           subchunk++) {
        buffer_offset += readbufspan.size();
        readbufspan    = ingest_raw_input(bufspan.last(buffer_size - buffer_offset),
                                       sources,
                                       next_subchunk_start,
                                       size_per_subchunk,
                                       delimiter,
                                       stream);
        next_delim_pos = find_first_delimiter(readbufspan, delimiter, stream) + buffer_offset;
        next_subchunk_start += size_per_subchunk;
      }
      if (next_delim_pos < buffer_offset) {
        if (next_subchunk_start >= total_source_size) {
          // If we have reached the end of source list but the source does not terminate with a
          // delimiter character
          next_delim_pos = buffer_offset + readbufspan.size();
        } else {
          // Our buffer_size estimate is insufficient to read until the end of the line! We need to
          // allocate more memory and try again!
          num_subchunks_prealloced *= 2;
          buffer_size = std::min(total_source_size,
                                 buffer_size + num_subchunks_prealloced * size_per_subchunk) +
                        num_extra_delimiters;
          buffer.resize(buffer_size, stream);
          bufspan = device_span<char>(reinterpret_cast<char*>(buffer.data()), buffer.size());
        }
      }
    }

    auto const batch_limit = static_cast<size_t>(std::numeric_limits<int32_t>::max());
    CUDF_EXPECTS(static_cast<size_t>(next_delim_pos - first_delim_pos - shift_for_nonzero_offset) <
                   batch_limit,
                 "The size of the JSON buffer returned by every batch cannot exceed INT_MAX bytes");
    return datasource::owning_buffer<rmm::device_buffer>(
      std::move(buffer),
      reinterpret_cast<uint8_t*>(buffer.data()) + first_delim_pos + shift_for_nonzero_offset,
      next_delim_pos - first_delim_pos - shift_for_nonzero_offset);
  }

  // Add delimiter to end of buffer - possibly adding an empty line to the input buffer - iff we are
  // reading till the end of the last source i.e. should_load_till_last_source is true Note that the
  // table generated from the JSONL input remains unchanged since empty lines are ignored by the
  // parser.
  size_t num_chars = readbufspan.size() - first_delim_pos - shift_for_nonzero_offset;
  if (num_chars) {
    auto last_char = delimiter;
    cudf::detail::cuda_memcpy_async<char>(
      device_span<char>(reinterpret_cast<char*>(buffer.data()), buffer.size())
        .subspan(readbufspan.size(), 1),
      host_span<char const>(&last_char, 1, false),
      stream);
    num_chars++;
  }

  return datasource::owning_buffer<rmm::device_buffer>(
    std::move(buffer),
    reinterpret_cast<uint8_t*>(buffer.data()) + first_delim_pos + shift_for_nonzero_offset,
    num_chars);
}

// Helper function to read the current batch using byte range offsets and size
// passed
table_with_metadata read_batch(host_span<std::unique_ptr<datasource>> sources,
                               json_reader_options const& reader_opts,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  datasource::owning_buffer<rmm::device_buffer> bufview =
    get_record_range_raw_input(sources, reader_opts, stream);

  // If input JSON buffer has single quotes and option to normalize single quotes is enabled,
  // invoke pre-processing FST
  if (reader_opts.is_enabled_normalize_single_quotes()) {
    normalize_single_quotes(
      bufview, reader_opts.get_delimiter(), stream, cudf::get_current_device_resource_ref());
  }

  auto buffer =
    cudf::device_span<char const>(reinterpret_cast<char const*>(bufview.data()), bufview.size());
  stream.synchronize();
  return device_parse_nested_json(buffer, reader_opts, stream, mr);
}

table_with_metadata read_json_impl(host_span<std::unique_ptr<datasource>> sources,
                                   json_reader_options const& reader_opts,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  /*
   * The batched JSON reader enforces that the size of each batch is at most INT_MAX
   * bytes (~2.14GB). Batches are defined to be byte range chunks - characterized by
   * chunk offset and chunk size - that may span across multiple source files.
   * Note that the batched reader does not work for compressed inputs or for regular
   * JSON inputs.
   */
  std::size_t const total_source_size = sources_size(sources, 0, 0);

  // Batching is enabled only for JSONL inputs, not regular JSON files
  CUDF_EXPECTS(
    reader_opts.is_enabled_lines() || total_source_size < std::numeric_limits<int32_t>::max(),
    "Parsing Regular JSON inputs of size greater than INT_MAX bytes is not supported");

  std::size_t chunk_offset     = reader_opts.get_byte_range_offset();
  std::size_t chunk_size       = reader_opts.get_byte_range_size();
  chunk_size                   = !chunk_size ? total_source_size - chunk_offset
                                             : std::min(chunk_size, total_source_size - chunk_offset);
  std::size_t const batch_size = get_batch_size(chunk_size);

  /*
   * Identify the position (zero-indexed) of starting source file from which to begin
   * batching based on byte range offset. If the offset is larger than the sum of all
   * source sizes, then start_source is total number of source files i.e. no file is
   * read
   */

  // Prefix sum of source file sizes
  std::size_t pref_source_size = 0;
  // Starting source file from which to being batching evaluated using byte range offset
  std::size_t const start_source = [chunk_offset, &sources, &pref_source_size]() {
    for (std::size_t src_idx = 0; src_idx < sources.size(); ++src_idx) {
      if (pref_source_size + sources[src_idx]->size() > chunk_offset) { return src_idx; }
      pref_source_size += sources[src_idx]->size();
    }
    return sources.size();
  }();
  /*
   * Construct batches of byte ranges spanning source files, with the starting position of batches
   * indicated by `batch_offsets`. `pref_bytes_size` gives the bytes position from which the current
   * batch begins, and `end_bytes_size` gives the terminal bytes position after which reading
   * stops.
   */
  std::size_t pref_bytes_size = chunk_offset;
  std::size_t end_bytes_size  = chunk_offset + chunk_size;
  std::vector<std::size_t> batch_offsets{pref_bytes_size};
  for (std::size_t i = start_source; i < sources.size() && pref_bytes_size < end_bytes_size;) {
    pref_source_size += sources[i]->size();
    // If the current source file can subsume multiple batches, we split the file until the
    // boundary of the last batch exceeds the end of the file (indexed by `pref_source_size`)
    while (pref_bytes_size < end_bytes_size &&
           pref_source_size >= std::min(pref_bytes_size + batch_size, end_bytes_size)) {
      auto next_batch_size = std::min(batch_size, end_bytes_size - pref_bytes_size);
      batch_offsets.push_back(batch_offsets.back() + next_batch_size);
      pref_bytes_size += next_batch_size;
    }
    i++;
  }
  /*
   * If there is a single batch, then we can directly return the table without the
   * unnecessary concatenate. The size of batch_offsets is 1 if all sources are empty,
   * or if end_bytes_size is larger than total_source_size.
   */
  if (batch_offsets.size() <= 2) return read_batch(sources, reader_opts, stream, mr);

  std::vector<cudf::io::table_with_metadata> partial_tables;
  json_reader_options batched_reader_opts{reader_opts};

  // recursive lambda to construct schema_element. Here, we assume that the table from the
  // first batch contains all the columns in the concatenated table, and that the partial tables
  // from all following batches contain the same set of columns
  std::function<schema_element(cudf::host_span<column_view const> cols,
                               cudf::host_span<column_name_info const> names,
                               schema_element & schema)>
    construct_schema;
  schema_element schema{data_type{cudf::type_id::STRUCT}};
  construct_schema = [&construct_schema](cudf::host_span<column_view const> children,
                                         cudf::host_span<column_name_info const> children_props,
                                         schema_element& schema) -> schema_element {
    CUDF_EXPECTS(
      children.size() == children_props.size(),
      "Mismatch in the number of children columns and children column properties received");

    if (schema.type == data_type{cudf::type_id::LIST}) {
      schema.column_order = {"element"};
      CUDF_EXPECTS(children.size() == 2, "List should have two children");
      auto element_idx = children_props[0].name == "element" ? 0 : 1;
      schema_element child_schema{children[element_idx].type()};
      std::vector<column_view> grandchildren_cols;
      std::transform(children[element_idx].child_begin(),
                     children[element_idx].child_end(),
                     std::back_inserter(grandchildren_cols),
                     [](auto& gc) { return gc; });
      schema.child_types["element"] =
        construct_schema(grandchildren_cols, children_props[element_idx].children, child_schema);
    } else {
      std::vector<std::string> col_order;
      std::transform(children_props.begin(),
                     children_props.end(),
                     std::back_inserter(col_order),
                     [](auto& c_prop) { return c_prop.name; });
      schema.column_order = std::move(col_order);
      for (auto i = 0ul; i < children.size(); i++) {
        schema_element child_schema{children[i].type()};
        std::vector<column_view> grandchildren_cols;
        std::transform(children[i].child_begin(),
                       children[i].child_end(),
                       std::back_inserter(grandchildren_cols),
                       [](auto& gc) { return gc; });
        schema.child_types[children_props[i].name] =
          construct_schema(grandchildren_cols, children_props[i].children, child_schema);
      }
    }

    return schema;
  };
  batched_reader_opts.set_byte_range_offset(batch_offsets[0]);
  batched_reader_opts.set_byte_range_size(batch_offsets[1] - batch_offsets[0]);
  partial_tables.emplace_back(
    read_batch(sources, batched_reader_opts, stream, cudf::get_current_device_resource_ref()));

  auto& tbl = partial_tables.back().tbl;
  std::vector<column_view> children;
  for (size_type j = 0; j < tbl->num_columns(); j++) {
    children.emplace_back(tbl->get_column(j));
  }
  batched_reader_opts.set_dtypes(
    construct_schema(children, partial_tables.back().metadata.schema_info, schema));
  batched_reader_opts.enable_prune_columns(true);

  // Dispatch individual batches to read_batch and push the resulting table into
  // partial_tables array. Note that the reader options need to be updated for each
  // batch to adjust byte range offset and byte range size.
  for (std::size_t batch_offset_pos = 1; batch_offset_pos < batch_offsets.size() - 1;
       batch_offset_pos++) {
    batched_reader_opts.set_byte_range_offset(batch_offsets[batch_offset_pos]);
    batched_reader_opts.set_byte_range_size(batch_offsets[batch_offset_pos + 1] -
                                            batch_offsets[batch_offset_pos]);
    auto partial_table =
      read_batch(sources, batched_reader_opts, stream, cudf::get_current_device_resource_ref());
    if (partial_table.tbl->num_columns() == 0 && partial_table.tbl->num_rows() == 0) {
      CUDF_EXPECTS(batch_offset_pos == batch_offsets.size() - 2,
                   "Only the partial table generated by the last batch can be empty");
      break;
    }
    partial_tables.emplace_back(std::move(partial_table));
  }

  auto expects_schema_equality =
    std::all_of(partial_tables.begin() + 1,
                partial_tables.end(),
                [&gt = partial_tables[0].metadata.schema_info](auto& ptbl) {
                  return ptbl.metadata.schema_info == gt;
                });
  CUDF_EXPECTS(expects_schema_equality,
               "Mismatch in JSON schema across batches in multi-source multi-batch reading");

  auto partial_table_views = std::vector<cudf::table_view>(partial_tables.size());
  std::transform(partial_tables.begin(),
                 partial_tables.end(),
                 partial_table_views.begin(),
                 [](auto const& table) { return table.tbl->view(); });
  return table_with_metadata{cudf::concatenate(partial_table_views, stream, mr),
                             {partial_tables[0].metadata.schema_info}};
}

}  // anonymous namespace

device_span<char> ingest_raw_input(device_span<char> buffer,
                                   host_span<std::unique_ptr<datasource>> sources,
                                   std::size_t range_offset,
                                   std::size_t range_size,
                                   char delimiter,
                                   rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  // We append a line delimiter between two files to make sure the last line of file i and the first
  // line of file i+1 don't end up on the same JSON line, if file i does not already end with a line
  // delimiter.
  auto constexpr num_delimiter_chars = 1;
  std::vector<std::future<size_t>> thread_tasks;

  auto delimiter_map = cudf::detail::make_empty_host_vector<std::size_t>(sources.size(), stream);
  std::vector<std::size_t> prefsum_source_sizes(sources.size());
  std::vector<std::unique_ptr<datasource::buffer>> h_buffers;
  std::size_t bytes_read = 0;
  std::transform_inclusive_scan(sources.begin(),
                                sources.end(),
                                prefsum_source_sizes.begin(),
                                std::plus<std::size_t>{},
                                [](std::unique_ptr<datasource> const& s) { return s->size(); });
  auto upper =
    std::upper_bound(prefsum_source_sizes.begin(), prefsum_source_sizes.end(), range_offset);
  std::size_t start_source = std::distance(prefsum_source_sizes.begin(), upper);

  auto const total_bytes_to_read = std::min(range_size, prefsum_source_sizes.back() - range_offset);
  range_offset -= start_source ? prefsum_source_sizes[start_source - 1] : 0;

  size_t const num_streams =
    std::min<std::size_t>({sources.size() - start_source + 1,
                           cudf::detail::global_cuda_stream_pool().get_stream_pool_size(),
                           pools::tpool().get_thread_count()});
  auto stream_pool = cudf::detail::fork_streams(stream, num_streams);
  for (std::size_t i = start_source, cur_stream = 0;
       i < sources.size() && bytes_read < total_bytes_to_read;
       i++) {
    if (sources[i]->is_empty()) continue;
    auto data_size = std::min(sources[i]->size() - range_offset, total_bytes_to_read - bytes_read);
    auto destination = reinterpret_cast<uint8_t*>(buffer.data()) + bytes_read +
                       (num_delimiter_chars * delimiter_map.size());
    if (sources[i]->supports_device_read()) {
      thread_tasks.emplace_back(sources[i]->device_read_async(
        range_offset, data_size, destination, stream_pool[cur_stream++ % stream_pool.size()]));
      bytes_read += data_size;
    } else {
      h_buffers.emplace_back(sources[i]->host_read(range_offset, data_size));
      auto const& h_buffer = h_buffers.back();
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        destination, h_buffer->data(), h_buffer->size(), cudaMemcpyHostToDevice, stream.value()));
      bytes_read += h_buffer->size();
    }
    range_offset = 0;
    delimiter_map.push_back(bytes_read + (num_delimiter_chars * delimiter_map.size()));
  }
  // Removing delimiter inserted after last non-empty source is read
  if (!delimiter_map.empty()) { delimiter_map.pop_back(); }

  // If this is a multi-file source, we scatter the JSON line delimiters between files
  if (sources.size() > 1 && !delimiter_map.empty()) {
    static_assert(num_delimiter_chars == 1,
                  "Currently only single-character delimiters are supported");
    auto const delimiter_source = thrust::make_constant_iterator(delimiter);
    auto const d_delimiter_map  = cudf::detail::make_device_uvector_async(
      delimiter_map, stream, cudf::get_current_device_resource_ref());
    thrust::scatter(rmm::exec_policy_nosync(stream),
                    delimiter_source,
                    delimiter_source + d_delimiter_map.size(),
                    d_delimiter_map.data(),
                    buffer.data());
  }
  stream.synchronize();

  if (thread_tasks.size()) {
    auto const bytes_read = std::accumulate(
      thread_tasks.begin(), thread_tasks.end(), std::size_t{0}, [](std::size_t sum, auto& task) {
        return sum + task.get();
      });
    CUDF_EXPECTS(bytes_read == total_bytes_to_read, "something's fishy");
  }

  return buffer.first(bytes_read + (delimiter_map.size() * num_delimiter_chars));
}

table_with_metadata read_json(host_span<std::unique_ptr<datasource>> sources,
                              json_reader_options const& reader_opts,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (reader_opts.get_byte_range_offset() != 0 or reader_opts.get_byte_range_size() != 0) {
    CUDF_EXPECTS(reader_opts.is_enabled_lines(),
                 "Specifying a byte range is supported only for JSON Lines");
  }

  if (sources.size() > 1) {
    CUDF_EXPECTS(reader_opts.is_enabled_lines(),
                 "Multiple inputs are supported only for JSON Lines format");
  }

  if (reader_opts.get_compression() == compression_type::NONE)
    return read_json_impl(sources, reader_opts, stream, mr);

  std::vector<std::unique_ptr<datasource>> compressed_sources;
  std::vector<std::future<std::unique_ptr<compressed_host_buffer_source>>> thread_tasks;
  auto& thread_pool = pools::tpool();
  for (auto& src : sources) {
    thread_tasks.emplace_back(thread_pool.submit_task([&reader_opts, &src] {
      return std::make_unique<compressed_host_buffer_source>(src, reader_opts.get_compression());
    }));
  }
  std::transform(thread_tasks.begin(),
                 thread_tasks.end(),
                 std::back_inserter(compressed_sources),
                 [](auto& task) { return task.get(); });
  // in read_json_impl, we need the compressed source size to actually be the
  // uncompressed source size for correct batching
  return read_json_impl(compressed_sources, reader_opts, stream, mr);
}

}  // namespace cudf::io::json::detail
