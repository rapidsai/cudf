/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_multifile_composer.hpp"

#include "hybrid_scan_multifile_common.hpp"

#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

using cudf::io::parquet::experimental::use_data_page_mask;

namespace {

std::vector<std::vector<cudf::io::text::byte_range_info>> column_chunks_byte_ranges_per_source(
  std::pair<std::vector<cudf::io::text::byte_range_info>, std::vector<cudf::size_type>> const&
    byte_ranges_and_source_map,
  std::size_t num_sources)
{
  auto const& [byte_ranges, source_map] = byte_ranges_and_source_map;
  CUDF_EXPECTS(byte_ranges.size() == source_map.size(), "Invalid source map size");

  auto byte_ranges_per_source =
    std::vector<std::vector<cudf::io::text::byte_range_info>>(num_sources);
  std::for_each(byte_ranges.begin(),
                byte_ranges.end(),
                [&, range_index = std::size_t{0}](auto const& range) mutable {
                  auto const source_index = source_map[range_index++];
                  CUDF_EXPECTS(source_index >= 0 and static_cast<std::size_t>(source_index) <
                                                       byte_ranges_per_source.size(),
                               "Invalid byte range source index");
                  byte_ranges_per_source[source_index].push_back(range);
                });
  return byte_ranges_per_source;
}

struct column_chunk_data {
  std::vector<rmm::device_buffer> buffers;
  std::vector<std::vector<cudf::device_span<uint8_t const>>> per_source_spans;
  std::vector<cudf::device_span<uint8_t const>> flat_spans;
};

column_chunk_data fetch_column_chunks(
  multifile_inputs const& inputs,
  std::pair<std::vector<cudf::io::text::byte_range_info>, std::vector<cudf::size_type>> const&
    byte_ranges_and_source_map,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto byte_ranges_per_source =
    column_chunks_byte_ranges_per_source(byte_ranges_and_source_map, inputs.datasources.size());
  auto [buffers, per_source_spans, tasks] = cudf::io::parquet::fetch_byte_ranges_to_device_async(
    inputs.datasource_refs,
    cudf::host_span<std::vector<cudf::io::text::byte_range_info> const>{byte_ranges_per_source},
    stream,
    mr);
  tasks.get();

  auto flat_spans = std::vector<cudf::device_span<uint8_t const>>{};
  for (auto const& source_spans : per_source_spans) {
    flat_spans.insert(flat_spans.end(), source_spans.begin(), source_spans.end());
  }

  return {std::move(buffers), std::move(per_source_spans), std::move(flat_spans)};
}

}  // namespace

std::tuple<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>> hybrid_scan_multifile(
  cudf::io::source_info const& source_info,
  cudf::ast::operation const& filter_expression,
  std::optional<std::vector<std::string>> const& payload_column_names,
  bool case_sensitive_names,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto options = cudf::io::parquet_reader_options::builder()
                   .filter(filter_expression)
                   .case_sensitive_names(case_sensitive_names)
                   .build();
  if (payload_column_names.has_value()) { options.set_column_names(payload_column_names.value()); }

  auto inputs = multifile_inputs(source_info);
  auto reader =
    cudf::io::parquet::experimental::hybrid_scan_multifile{inputs.footer_byte_spans, options};
  setup_page_indexes(reader, inputs);

  auto const input_row_groups = reader.all_row_groups(options);
  auto const row_groups = reader.filter_row_groups_with_stats(input_row_groups, options, stream);
  auto row_mask = reader.build_row_mask_with_page_index_stats(row_groups, options, stream, mr);

  auto filter_column_chunks = fetch_column_chunks(
    inputs, reader.filter_column_chunks_byte_ranges(row_groups, options), stream, mr);
  auto row_mask_view = row_mask->mutable_view();
  auto filter_result = reader.materialize_filter_columns(row_groups,
                                                         filter_column_chunks.flat_spans,
                                                         row_mask_view,
                                                         use_data_page_mask::YES,
                                                         options,
                                                         stream,
                                                         mr);

  auto payload_column_chunks = fetch_column_chunks(
    inputs, reader.payload_column_chunks_byte_ranges(row_groups, options), stream, mr);
  auto payload_result = reader.materialize_payload_columns(row_groups,
                                                           payload_column_chunks.flat_spans,
                                                           row_mask_view,
                                                           use_data_page_mask::YES,
                                                           options,
                                                           stream,
                                                           mr);

  return std::tuple{std::move(filter_result.tbl), std::move(payload_result.tbl)};
}

std::unique_ptr<cudf::table> hybrid_scan_multifile_single_step(
  cudf::io::source_info const& source_info,
  cudf::ast::operation const& filter_expression,
  std::optional<std::vector<std::string>> const& column_names,
  bool case_sensitive_names,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto options = cudf::io::parquet_reader_options::builder()
                   .filter(filter_expression)
                   .case_sensitive_names(case_sensitive_names)
                   .build();
  if (column_names.has_value()) { options.set_column_names(column_names.value()); }

  auto inputs = multifile_inputs(source_info);
  auto reader =
    cudf::io::parquet::experimental::hybrid_scan_multifile{inputs.footer_byte_spans, options};
  setup_page_indexes(reader, inputs);

  auto const input_row_groups = reader.all_row_groups(options);
  auto const row_groups = reader.filter_row_groups_with_stats(input_row_groups, options, stream);

  auto all_column_chunks = fetch_column_chunks(
    inputs, reader.all_column_chunks_byte_ranges(row_groups, options), stream, mr);
  return reader
    .materialize_all_columns(row_groups, all_column_chunks.flat_spans, options, stream, mr)
    .tbl;
}
