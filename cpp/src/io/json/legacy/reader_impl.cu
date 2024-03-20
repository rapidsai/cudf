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

#include "hash/concurrent_unordered_map.cuh"
#include "io/comp/io_uncomp.hpp"
#include "io/utilities/column_buffer.hpp"
#include "io/utilities/parsing_utils.cuh"
#include "json_gpu.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/utilities/visitor_overload.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/json.hpp>
#include <cudf/io/json.hpp>
#include <cudf/sorting.hpp>
#include <cudf/strings/detail/replace.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/optional>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

using cudf::host_span;

namespace cudf::io::json::detail::legacy {

using col_map_ptr_type = std::unique_ptr<col_map_type, std::function<void(col_map_type*)>>;

/**
 * @brief Aggregate the table containing keys info by their hash values.
 *
 * @param[in] info Table with columns containing key offsets, lengths and hashes, respectively
 *
 * @return Table with data aggregated by key hash values
 */
std::unique_ptr<table> aggregate_keys_info(std::unique_ptr<table> info)
{
  auto const info_view = info->view();
  std::vector<groupby::aggregation_request> requests;
  requests.emplace_back(groupby::aggregation_request{info_view.column(0)});
  requests.back().aggregations.emplace_back(make_min_aggregation<groupby_aggregation>());
  requests.back().aggregations.emplace_back(make_nth_element_aggregation<groupby_aggregation>(0));

  requests.emplace_back(groupby::aggregation_request{info_view.column(1)});
  requests.back().aggregations.emplace_back(make_min_aggregation<groupby_aggregation>());
  requests.back().aggregations.emplace_back(make_nth_element_aggregation<groupby_aggregation>(0));

  // Aggregate by hash values
  groupby::groupby gb_obj(
    table_view({info_view.column(2)}), null_policy::EXCLUDE, sorted::NO, {}, {});

  auto result = gb_obj.aggregate(requests);  // TODO: no stream parameter?

  std::vector<std::unique_ptr<column>> out_columns;
  out_columns.emplace_back(std::move(result.second[0].results[0]));  // offsets
  out_columns.emplace_back(std::move(result.second[1].results[0]));  // lengths
  out_columns.emplace_back(std::move(result.first->release()[0]));   // hashes
  return std::make_unique<table>(std::move(out_columns));
}

/**
 * @brief Initializes the (key hash -> column index) hash map.
 */
col_map_ptr_type create_col_names_hash_map(column_view column_name_hashes,
                                           rmm::cuda_stream_view stream)
{
  auto key_col_map       = col_map_type::create(column_name_hashes.size(), stream);
  auto const column_data = column_name_hashes.data<uint32_t>();
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     column_name_hashes.size(),
                     [map = *key_col_map, column_data] __device__(size_type idx) mutable {
                       map.insert(thrust::make_pair(column_data[idx], idx));
                     });
  return key_col_map;
}

/**
 * @brief Create a table whose columns contain the information on JSON objects' keys.
 *
 * The columns contain name offsets in the file, name lengths and name hashes, respectively.
 *
 * @param[in] options Parsing options (e.g. delimiter and quotation character)
 * @param[in] data Input JSON device data
 * @param[in] row_offsets Device array of row start locations in the input buffer
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 *
 * @return std::unique_ptr<table> cudf table with three columns (offsets, lengths, hashes)
 */
std::unique_ptr<table> create_json_keys_info_table(parse_options_view const& parse_opts,
                                                   device_span<char const> const data,
                                                   device_span<uint64_t const> const row_offsets,
                                                   rmm::cuda_stream_view stream)
{
  // Count keys
  rmm::device_scalar<unsigned long long int> key_counter(0, stream);
  collect_keys_info(parse_opts, data, row_offsets, key_counter.data(), {}, stream);

  // Allocate columns to store hash value, length, and offset of each JSON object key in the input
  auto const num_keys = key_counter.value(stream);
  std::vector<std::unique_ptr<column>> info_columns;
  info_columns.emplace_back(
    make_numeric_column(data_type(type_id::UINT64), num_keys, mask_state::UNALLOCATED, stream));
  info_columns.emplace_back(
    make_numeric_column(data_type(type_id::UINT16), num_keys, mask_state::UNALLOCATED, stream));
  info_columns.emplace_back(
    make_numeric_column(data_type(type_id::UINT32), num_keys, mask_state::UNALLOCATED, stream));
  // Create a table out of these columns to pass them around more easily
  auto info_table           = std::make_unique<table>(std::move(info_columns));
  auto const info_table_mdv = mutable_table_device_view::create(info_table->mutable_view(), stream);

  // Reset the key counter - now used for indexing
  key_counter.set_value_to_zero_async(stream);
  // Fill the allocated columns
  collect_keys_info(parse_opts, data, row_offsets, key_counter.data(), {*info_table_mdv}, stream);
  return info_table;
}

/**
 * @brief Extract the keys from the JSON file the name offsets/lengths.
 */
std::vector<std::string> create_key_strings(char const* h_data,
                                            table_view sorted_info,
                                            rmm::cuda_stream_view stream)
{
  auto const num_cols = sorted_info.num_rows();
  std::vector<uint64_t> h_offsets(num_cols);
  CUDF_CUDA_TRY(cudaMemcpyAsync(h_offsets.data(),
                                sorted_info.column(0).data<uint64_t>(),
                                sizeof(uint64_t) * num_cols,
                                cudaMemcpyDefault,
                                stream.value()));

  std::vector<uint16_t> h_lens(num_cols);
  CUDF_CUDA_TRY(cudaMemcpyAsync(h_lens.data(),
                                sorted_info.column(1).data<uint16_t>(),
                                sizeof(uint16_t) * num_cols,
                                cudaMemcpyDefault,
                                stream.value()));

  std::vector<std::string> names(num_cols);
  std::transform(h_offsets.cbegin(),
                 h_offsets.cend(),
                 h_lens.cbegin(),
                 names.begin(),
                 [&](auto offset, auto len) { return std::string(h_data + offset, len); });
  return names;
}

auto sort_keys_info_by_offset(std::unique_ptr<table> info)
{
  auto const agg_offset_col_view = info->get_column(0).view();
  return sort_by_key(info->view(), table_view({agg_offset_col_view}));
}

/**
 * @brief Extract JSON object keys from a JSON file.
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 *
 * @return Names of JSON object keys in the file
 */
std::pair<std::vector<std::string>, col_map_ptr_type> get_json_object_keys_hashes(
  parse_options_view const& parse_opts,
  host_span<char const> h_data,
  device_span<uint64_t const> rec_starts,
  device_span<char const> d_data,
  rmm::cuda_stream_view stream)
{
  auto info = create_json_keys_info_table(parse_opts, d_data, rec_starts, stream);

  auto aggregated_info = aggregate_keys_info(std::move(info));
  auto sorted_info     = sort_keys_info_by_offset(std::move(aggregated_info));

  return {create_key_strings(h_data.data(), sorted_info->view(), stream),
          create_col_names_hash_map(sorted_info->get_column(2).view(), stream)};
}

std::vector<uint8_t> ingest_raw_input(host_span<std::unique_ptr<datasource>> sources,
                                      compression_type compression,
                                      size_t range_offset,
                                      size_t range_size,
                                      size_t range_size_padded)
{
  CUDF_FUNC_RANGE();
  // Iterate through the user defined sources and read the contents into the local buffer
  size_t total_source_size = 0;
  for (auto const& source : sources) {
    total_source_size += source->size();
  }
  total_source_size = total_source_size - (range_offset * sources.size());

  auto buffer = std::vector<uint8_t>(total_source_size);

  size_t bytes_read = 0;
  for (auto const& source : sources) {
    if (!source->is_empty()) {
      auto data_size   = (range_size_padded != 0) ? range_size_padded : source->size();
      auto destination = buffer.data() + bytes_read;
      bytes_read += source->host_read(range_offset, data_size, destination);
    }
  }

  if (compression == compression_type::NONE) {
    return buffer;
  } else {
    return decompress(compression, buffer);
  }
}

bool should_load_whole_source(json_reader_options const& reader_opts)
{
  return reader_opts.get_byte_range_offset() == 0 and  //
         reader_opts.get_byte_range_size() == 0;
}

rmm::device_uvector<uint64_t> find_record_starts(json_reader_options const& reader_opts,
                                                 host_span<char const> h_data,
                                                 device_span<char const> d_data,
                                                 rmm::cuda_stream_view stream)
{
  std::vector<char> chars_to_count{'\n'};
  // Currently, ignoring lineterminations within quotes is handled by recording the records of both,
  // and then filtering out the records that is a quotechar or a linetermination within a quotechar
  // pair.
  // If not starting at an offset, add an extra row to account for the first row in the file
  cudf::size_type prefilter_count = ((reader_opts.get_byte_range_offset() == 0) ? 1 : 0);
  if (should_load_whole_source(reader_opts)) {
    prefilter_count += count_all_from_set(d_data, chars_to_count, stream);
  } else {
    prefilter_count += count_all_from_set(h_data, chars_to_count, stream);
  }

  rmm::device_uvector<uint64_t> rec_starts(prefilter_count, stream);

  auto* find_result_ptr = rec_starts.data();
  // Manually adding an extra row to account for the first row in the file
  if (reader_opts.get_byte_range_offset() == 0) {
    find_result_ptr++;
    CUDF_CUDA_TRY(cudaMemsetAsync(rec_starts.data(), 0ull, sizeof(uint64_t), stream.value()));
  }

  std::vector<char> chars_to_find{'\n'};
  // Passing offset = 1 to return positions AFTER the found character
  if (should_load_whole_source(reader_opts)) {
    find_all_from_set(d_data, chars_to_find, 1, find_result_ptr, stream);
  } else {
    find_all_from_set(h_data, chars_to_find, 1, find_result_ptr, stream);
  }

  // Previous call stores the record positions as encountered by all threads
  // Sort the record positions as subsequent processing may require filtering
  // certain rows or other processing on specific records
  thrust::sort(rmm::exec_policy(stream), rec_starts.begin(), rec_starts.end());

  auto filtered_count = prefilter_count;

  // Exclude the ending newline as it does not precede a record start
  if (h_data.back() == '\n') { filtered_count--; }
  rec_starts.resize(filtered_count, stream);

  return rec_starts;
}

/**
 * @brief Uploads the relevant segment of the input json data onto the GPU.
 *
 * Sets the d_data_ data member.
 * Only rows that need to be parsed are copied, based on the byte range
 * Also updates the array of record starts to match the device data offset.
 */
rmm::device_uvector<char> upload_data_to_device(json_reader_options const& reader_opts,
                                                host_span<char const> h_data,
                                                rmm::device_uvector<uint64_t>& rec_starts,
                                                rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  size_t end_offset = h_data.size();

  // Trim lines that are outside range
  auto h_rec_starts = cudf::detail::make_std_vector_sync(rec_starts, stream);

  if (reader_opts.get_byte_range_size() != 0) {
    auto it = h_rec_starts.end() - 1;
    while (it >= h_rec_starts.begin() && *it > reader_opts.get_byte_range_size()) {
      end_offset = *it;
      --it;
    }
    h_rec_starts.erase(it + 1, h_rec_starts.end());
  }

  // Resize to exclude rows outside of the range
  // Adjust row start positions to account for the data subcopy
  size_t start_offset = h_rec_starts.front();
  rec_starts.resize(h_rec_starts.size(), stream);
  thrust::transform(rmm::exec_policy(stream),
                    rec_starts.begin(),
                    rec_starts.end(),
                    thrust::make_constant_iterator(start_offset),
                    rec_starts.begin(),
                    thrust::minus<uint64_t>());

  size_t const bytes_to_upload = end_offset - start_offset;
  CUDF_EXPECTS(bytes_to_upload <= h_data.size(),
               "Error finding the record within the specified byte range.\n");

  // Upload the raw data that is within the rows of interest
  return cudf::detail::make_device_uvector_async(
    h_data.subspan(start_offset, bytes_to_upload), stream, rmm::mr::get_current_device_resource());
}

std::pair<std::vector<std::string>, col_map_ptr_type> get_column_names_and_map(
  parse_options_view const& parse_opts,
  host_span<char const> h_data,
  device_span<uint64_t const> rec_starts,
  device_span<char const> d_data,
  rmm::cuda_stream_view stream)
{
  // If file only contains one row, use the file size for the row size
  uint64_t first_row_len = d_data.size();
  if (rec_starts.size() > 1) {
    // Set first_row_len to the offset of the second row, if it exists
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      &first_row_len, rec_starts.data() + 1, sizeof(uint64_t), cudaMemcpyDefault, stream.value()));
  }
  std::vector<char> first_row(first_row_len);
  CUDF_CUDA_TRY(cudaMemcpyAsync(first_row.data(),
                                d_data.data(),
                                first_row_len * sizeof(char),
                                cudaMemcpyDefault,
                                stream.value()));
  stream.synchronize();

  // Determine the row format between:
  //   JSON array - [val1, val2, ...] and
  //   JSON object - {"col1":val1, "col2":val2, ...}
  // based on the top level opening bracket
  auto const first_square_bracket = std::find(first_row.begin(), first_row.end(), '[');
  auto const first_curly_bracket  = std::find(first_row.begin(), first_row.end(), '{');
  CUDF_EXPECTS(first_curly_bracket != first_row.end() || first_square_bracket != first_row.end(),
               "Input data is not a valid JSON file.");
  // If the first opening bracket is '{', assume object format
  if (first_curly_bracket < first_square_bracket) {
    // use keys as column names if input rows are objects
    return get_json_object_keys_hashes(parse_opts, h_data, rec_starts, d_data, stream);
  } else {
    int cols_found    = 0;
    bool quotation    = false;
    auto column_names = std::vector<std::string>();
    for (size_t pos = 0; pos < first_row.size(); ++pos) {
      // Flip the quotation flag if current character is a quotechar
      if (first_row[pos] == parse_opts.quotechar) {
        quotation = !quotation;
      }
      // Check if end of a column/row
      else if (pos == first_row.size() - 1 ||
               (!quotation && first_row[pos] == parse_opts.delimiter)) {
        column_names.emplace_back(std::to_string(cols_found++));
      }
    }
    return {column_names, col_map_type::create(0, stream)};
  }
}

std::vector<data_type> get_data_types(json_reader_options const& reader_opts,
                                      parse_options_view const& parse_opts,
                                      std::vector<std::string> const& column_names,
                                      col_map_type* column_map,
                                      device_span<uint64_t const> rec_starts,
                                      device_span<char const> data,
                                      rmm::cuda_stream_view stream)
{
  bool has_to_infer_column_types =
    std::visit([](auto const& dtypes) { return dtypes.empty(); }, reader_opts.get_dtypes());

  if (!has_to_infer_column_types) {
    return std::visit(
      cudf::detail::visitor_overload{
        [&](std::vector<data_type> const& dtypes) {
          CUDF_EXPECTS(dtypes.size() == column_names.size(), "Must specify types for all columns");
          return dtypes;
        },
        [&](std::map<std::string, data_type> const& dtypes) {
          std::vector<data_type> sorted_dtypes;
          std::transform(std::cbegin(column_names),
                         std::cend(column_names),
                         std::back_inserter(sorted_dtypes),
                         [&](auto const& column_name) {
                           auto const it = dtypes.find(column_name);
                           CUDF_EXPECTS(it != dtypes.end(), "Must specify types for all columns");
                           return it->second;
                         });
          return sorted_dtypes;
        },
        [&](std::map<std::string, schema_element> const& dtypes) {
          std::vector<data_type> sorted_dtypes;
          std::transform(std::cbegin(column_names),
                         std::cend(column_names),
                         std::back_inserter(sorted_dtypes),
                         [&](auto const& column_name) {
                           auto const it = dtypes.find(column_name);
                           CUDF_EXPECTS(it != dtypes.end(), "Must specify types for all columns");
                           return it->second.type;
                         });
          return sorted_dtypes;
        }},
      reader_opts.get_dtypes());
  } else {
    CUDF_EXPECTS(not rec_starts.empty(), "No data available for data type inference.\n");
    auto const num_columns       = column_names.size();
    auto const do_set_null_count = column_map->capacity() > 0;

    auto const h_column_infos = detect_data_types(
      parse_opts, data, rec_starts, do_set_null_count, num_columns, column_map, stream);

    auto get_type_id = [&](auto const& cinfo) {
      auto int_count_total =
        cinfo.big_int_count + cinfo.negative_small_int_count + cinfo.positive_small_int_count;
      if (cinfo.null_count == static_cast<int>(rec_starts.size())) {
        // Entire column is NULL; allocate the smallest amount of memory
        return type_id::INT8;
      } else if (cinfo.string_count > 0) {
        return type_id::STRING;
      } else if (cinfo.datetime_count > 0) {
        return type_id::TIMESTAMP_MILLISECONDS;
      } else if (cinfo.float_count > 0) {
        return type_id::FLOAT64;
      } else if (cinfo.big_int_count == 0 && int_count_total != 0) {
        return type_id::INT64;
      } else if (cinfo.big_int_count != 0 && cinfo.negative_small_int_count != 0) {
        return type_id::STRING;
      } else if (cinfo.big_int_count != 0) {
        return type_id::UINT64;
      } else if (cinfo.bool_count > 0) {
        return type_id::BOOL8;
      } else {
        CUDF_FAIL("Data type detection failed.\n");
      }
    };

    std::vector<data_type> dtypes;

    std::transform(std::cbegin(h_column_infos),
                   std::cend(h_column_infos),
                   std::back_inserter(dtypes),
                   [&](auto const& cinfo) { return data_type{get_type_id(cinfo)}; });

    return dtypes;
  }
}

table_with_metadata convert_data_to_table(parse_options_view const& parse_opts,
                                          std::vector<data_type> const& dtypes,
                                          std::vector<std::string>&& column_names,
                                          col_map_type* column_map,
                                          device_span<uint64_t const> rec_starts,
                                          device_span<char const> data,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  auto const num_columns = dtypes.size();
  auto const num_records = rec_starts.size();

  // alloc output buffers.
  std::vector<cudf::io::detail::column_buffer> out_buffers;
  for (size_t col = 0; col < num_columns; ++col) {
    out_buffers.emplace_back(dtypes[col], num_records, true, stream, mr);
  }

  thrust::host_vector<data_type> h_dtypes(num_columns);
  thrust::host_vector<void*> h_data(num_columns);
  thrust::host_vector<bitmask_type*> h_valid(num_columns);

  for (size_t i = 0; i < num_columns; ++i) {
    h_dtypes[i] = dtypes[i];
    h_data[i]   = out_buffers[i].data();
    h_valid[i]  = out_buffers[i].null_mask();
  }

  auto d_dtypes = cudf::detail::make_device_uvector_async<data_type>(
    h_dtypes, stream, rmm::mr::get_current_device_resource());
  auto d_data = cudf::detail::make_device_uvector_async<void*>(
    h_data, stream, rmm::mr::get_current_device_resource());
  auto d_valid = cudf::detail::make_device_uvector_async<cudf::bitmask_type*>(
    h_valid, stream, rmm::mr::get_current_device_resource());
  auto d_valid_counts = cudf::detail::make_zeroed_device_uvector_async<cudf::size_type>(
    num_columns, stream, rmm::mr::get_current_device_resource());

  convert_json_to_columns(
    parse_opts, data, rec_starts, d_dtypes, column_map, d_data, d_valid, d_valid_counts, stream);

  stream.synchronize();

  // postprocess columns
  auto target_chars   = std::vector<char>{'\\', '"', '\\', '\\', '\\', 't', '\\', 'r', '\\', 'b'};
  auto target_offsets = std::vector<size_type>{0, 2, 4, 6, 8, 10};

  auto repl_chars   = std::vector<char>{'"', '\\', '\t', '\r', '\b'};
  auto repl_offsets = std::vector<size_type>{0, 1, 2, 3, 4, 5};

  auto target =
    make_strings_column(static_cast<size_type>(target_offsets.size() - 1),
                        std::make_unique<cudf::column>(
                          cudf::detail::make_device_uvector_async(
                            target_offsets, stream, rmm::mr::get_current_device_resource()),
                          rmm::device_buffer{},
                          0),
                        cudf::detail::make_device_uvector_async(
                          target_chars, stream, rmm::mr::get_current_device_resource())
                          .release(),
                        0,
                        {});
  auto repl = make_strings_column(
    static_cast<size_type>(repl_offsets.size() - 1),
    std::make_unique<cudf::column>(cudf::detail::make_device_uvector_async(
                                     repl_offsets, stream, rmm::mr::get_current_device_resource()),
                                   rmm::device_buffer{},
                                   0),
    cudf::detail::make_device_uvector_async(
      repl_chars, stream, rmm::mr::get_current_device_resource())
      .release(),
    0,
    {});

  auto const h_valid_counts = cudf::detail::make_std_vector_sync(d_valid_counts, stream);
  std::vector<std::unique_ptr<column>> out_columns;
  for (size_t i = 0; i < num_columns; ++i) {
    out_buffers[i].null_count() = num_records - h_valid_counts[i];

    auto out_column = make_column(out_buffers[i], nullptr, std::nullopt, stream);
    if (out_column->type().id() == type_id::STRING) {
      // Need to remove escape character in case of '\"' and '\\'
      out_columns.emplace_back(cudf::strings::detail::replace(
        out_column->view(), target->view(), repl->view(), stream, mr));
    } else {
      out_columns.emplace_back(std::move(out_column));
    }
    if (out_columns.back()->null_count() == 0) {
      out_columns.back()->set_null_mask(rmm::device_buffer{0, stream, mr}, 0);
    }
  }

  std::vector<column_name_info> column_infos;
  column_infos.reserve(column_names.size());
  std::transform(std::make_move_iterator(column_names.begin()),
                 std::make_move_iterator(column_names.end()),
                 std::back_inserter(column_infos),
                 [](auto const& col_name) { return column_name_info{col_name}; });

  // This is to ensure the stream-ordered make_stream_column calls above complete before
  // the temporary std::vectors are destroyed on exit from this function.
  stream.synchronize();

  CUDF_EXPECTS(!out_columns.empty(), "No columns created from json input");

  return table_with_metadata{std::make_unique<table>(std::move(out_columns)), {column_infos}};
}

/**
 * @brief Read an entire set or a subset of data from the source
 *
 * @param[in] options reader options with Number of bytes offset from the start,
 * Bytes to read; use `0` for all remaining data
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 *
 * @return Table and its metadata
 */
table_with_metadata read_json(host_span<std::unique_ptr<datasource>> sources,
                              json_reader_options const& reader_opts,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(not sources.empty(), "No sources were defined");
  CUDF_EXPECTS(sources.size() == 1 or reader_opts.get_compression() == compression_type::NONE,
               "Multiple compressed inputs are not supported");
  CUDF_EXPECTS(reader_opts.is_enabled_lines(), "Only JSON Lines format is currently supported.\n");

  auto parse_opts = parse_options{',', '\n', '\"', '.'};

  parse_opts.trie_true  = cudf::detail::create_serialized_trie({"true"}, stream);
  parse_opts.trie_false = cudf::detail::create_serialized_trie({"false"}, stream);
  parse_opts.trie_na    = cudf::detail::create_serialized_trie({"", "null"}, stream);

  parse_opts.dayfirst = reader_opts.is_enabled_dayfirst();

  auto range_offset      = reader_opts.get_byte_range_offset();
  auto range_size        = reader_opts.get_byte_range_size();
  auto range_size_padded = reader_opts.get_byte_range_size_with_padding();

  auto const h_raw_data = ingest_raw_input(
    sources, reader_opts.get_compression(), range_offset, range_size, range_size_padded);
  host_span<char const> h_data{reinterpret_cast<char const*>(h_raw_data.data()), h_raw_data.size()};

  CUDF_EXPECTS(not h_data.empty(), "Ingest failed: uncompressed input data has zero size.\n");

  auto d_data = rmm::device_uvector<char>(0, stream);

  if (should_load_whole_source(reader_opts)) {
    d_data = cudf::detail::make_device_uvector_async(
      h_data, stream, rmm::mr::get_current_device_resource());
  }

  auto rec_starts = find_record_starts(reader_opts, h_data, d_data, stream);

  CUDF_EXPECTS(rec_starts.size() > 0, "Error enumerating records.\n");

  if (not should_load_whole_source(reader_opts)) {
    d_data = upload_data_to_device(reader_opts, h_data, rec_starts, stream);
  }

  CUDF_EXPECTS(not d_data.is_empty(), "Error uploading input data to the GPU.\n");

  auto column_names_and_map =
    get_column_names_and_map(parse_opts.view(), h_data, rec_starts, d_data, stream);

  auto column_names = std::get<0>(column_names_and_map);
  auto column_map   = std::move(std::get<1>(column_names_and_map));

  CUDF_EXPECTS(not column_names.empty(), "Error determining column names.\n");

  auto dtypes = get_data_types(
    reader_opts, parse_opts.view(), column_names, column_map.get(), rec_starts, d_data, stream);

  CUDF_EXPECTS(not dtypes.empty(), "Error in data type detection.\n");

  return convert_data_to_table(parse_opts.view(),
                               dtypes,
                               std::move(column_names),
                               column_map.get(),
                               rec_starts,
                               d_data,
                               stream,
                               mr);
}

}  // namespace cudf::io::json::detail::legacy
