/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

/**
 * @brief Struct to hold multifile datasources and footer buffers along with their byte spans
 */
struct multifile_inputs {
  /**
   * @brief Construct datasources, datasource refs, and footer byte spans from source info
   */
  explicit multifile_inputs(cudf::io::source_info const& source_info)
    : datasources{cudf::io::make_datasources(source_info)}
  {
    datasource_refs.reserve(datasources.size());
    footer_buffers.reserve(datasources.size());
    footer_byte_spans.reserve(datasources.size());

    for (auto const& datasource : datasources) {
      datasource_refs.emplace_back(*datasource);
      footer_buffers.emplace_back(cudf::io::parquet::fetch_footer_to_host(datasource_refs.back()));
      footer_byte_spans.emplace_back(*footer_buffers.back());
    }
  }

  std::vector<std::unique_ptr<cudf::io::datasource>> datasources;
  std::vector<std::reference_wrapper<cudf::io::datasource>> datasource_refs;
  std::vector<std::unique_ptr<cudf::io::datasource::buffer>> footer_buffers;
  std::vector<cudf::host_span<uint8_t const>> footer_byte_spans;
};

/**
 * @brief Construct source info from host buffers
 */
template <typename Buffers>
cudf::io::source_info build_source_info(Buffers const& file_buffers)
{
  std::vector<cudf::host_span<char const>> spans;
  spans.reserve(file_buffers.size());
  for (auto const& buf : file_buffers) {
    spans.emplace_back(buf.data(), buf.size());
  }
  return cudf::io::source_info(cudf::host_span<cudf::host_span<char const>>{spans});
}

/**
 * @brief Fetch and set up page indexes for all sources in a multifile reader
 */
inline void setup_page_indexes(cudf::io::parquet::experimental::hybrid_scan_multifile const& reader,
                               multifile_inputs const& inputs)
{
  auto const page_index_byte_ranges = reader.page_index_byte_ranges();
  std::vector<cudf::host_span<uint8_t const>> page_index_byte_spans;
  page_index_byte_spans.reserve(page_index_byte_ranges.size());

  auto const page_index_buffers = cudf::io::parquet::fetch_page_indexes_to_host(
    cudf::host_span<std::reference_wrapper<cudf::io::datasource> const>{inputs.datasource_refs},
    cudf::host_span<cudf::io::parquet::byte_range_info const>{page_index_byte_ranges});
  std::transform(page_index_buffers.begin(),
                 page_index_buffers.end(),
                 std::back_inserter(page_index_byte_spans),
                 [](auto const& buffer) { return cudf::host_span<uint8_t const>{*buffer}; });

  reader.setup_page_indexes(
    cudf::host_span<cudf::host_span<uint8_t const> const>{page_index_byte_spans});
}

/**
 * @brief Group flattened byte ranges by their corresponding source indices
 *
 * @param byte_ranges_and_source_map Pair of flattened byte ranges and their parallel source map
 * @param num_sources Number of sources
 * @return Vector of byte ranges, one inner vector per source
 */
inline std::vector<std::vector<cudf::io::text::byte_range_info>> group_byte_ranges_by_source(
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
