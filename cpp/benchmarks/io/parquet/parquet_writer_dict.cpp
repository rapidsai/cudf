/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <nvbench/nvbench.cuh>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace {
// Dict-encoding-focused Parquet writer benchmark. Built to isolate the two
// encoder pathologies reported in https://github.com/rapidsai/cudf/issues/13995:
//
//   1. `dict_id` assignment order. cuDF's `collect_map_entries_kernel` walks
//      the hash map in slot order, so the `dict_id` assigned to a value is
//      uncorrelated with how early it appears in the column. Early pages
//      therefore end up referencing high-indexed entries, defeating any
//      per-page bit-width savings.
//   2. Chunk-wide RLE bit width. `build_chunk_dictionaries` derives one
//      `dict_rle_bits = ceil(log2(num_dict_entries))` for the entire chunk,
//      so a page touching only a handful of entries still bit-packs at the
//      chunk-wide width.
//
// Workload shape (deliberately the simplest construction that exposes both
// problems on a single row group / single chunk):
//
//   * 1 INT64 column, 1 row group = 1 chunk, 10 pages per chunk.
//     INT64 is the type reported in the issue; fixed width keeps the
//     per-row byte count deterministic so the file-size delta is driven
//     entirely by how the encoder packs the dict-index stream.
//   * cardinality = 64,000 (ceil(log2) = 16 bits chunk-wide).
//   * Pages 0..`hot_pages-1` (the "common" pages) draw uniformly from a
//     small "frequent set" of `frequent_set_size` values shared across
//     all common pages.
//   * Pages `hot_pages..pages_per_chunk-1` (the "rare" pages) draw
//     uniformly from the remaining `cardinality - frequent_set_size`
//     values. These never appear in any common page, so only the rare
//     pages force the chunk-wide bit width up.
//
// Under an ideal encoder (first-appearance ordering + per-page bit width)
// the common pages need only `ceil(log2(frequent_set_size))` bits per
// value while the rare pages need `ceil(log2(cardinality))` bits. With
// the constants below (frequent_set_size = 64), that is 6 bits vs. 16
// bits, saving ~10 bits/value on every common-page row. The current
// cuDF encoder produces 16 bits on every page, so the headline
// `encoded_file_size` metric cleanly resolves the optimization target.

// Distribution shape — see file header. These are intentionally fixed so that
// the encoded-file-size number is directly comparable across phases. Adjust
// together with the phase baseline if the shape changes.
constexpr cudf::size_type num_cols          = 1;
constexpr cudf::size_type cardinality       = 64'000;
constexpr cudf::size_type pages_per_chunk   = 10;
constexpr cudf::size_type hot_pages         = pages_per_chunk - 2;
constexpr cudf::size_type frequent_set_size = 64;
constexpr std::uint32_t dict_rng_seed       = 0xC0DEFACE;

// Build the row-to-dictionary-index mapping on host for a single column.
// Index space (disjoint, exactly covers [0, cardinality)):
//   * frequent set: [0, frequent_set_size)
//   * rare set:     [frequent_set_size, cardinality)
template <typename T>
std::vector<T> build_numeric_column(cudf::size_type num_rows, cudf::size_type page_size_rows)
{
  CUDF_EXPECTS(num_rows == pages_per_chunk * page_size_rows,
               "num_rows must equal pages_per_chunk * page_size_rows");
  CUDF_EXPECTS(frequent_set_size < cardinality,
               "cardinality must leave room for a nonempty rare set");

  std::mt19937 rng{dict_rng_seed};
  std::uniform_int_distribution<cudf::size_type> freq_dist(0, frequent_set_size - 1);
  std::uniform_int_distribution<cudf::size_type> rare_dist(frequent_set_size, cardinality - 1);
  auto const threshold = hot_pages * page_size_rows;

  cudf::size_type row_idx = 0;
  std::vector<T> values(num_rows);
  std::generate_n(values.begin(), num_rows, [&]() {
    return row_idx++ < threshold ? static_cast<T>(freq_dist(rng)) : static_cast<T>(rare_dist(rng));
  });

  return values;
}

[[nodiscard]] std::unique_ptr<cudf::table> build_table(cudf::size_type num_rows,
                                                       cudf::size_type page_size_rows)
{
  auto const values = build_numeric_column<int64_t>(num_rows, page_size_rows);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(num_cols);
  cols.emplace_back(
    cudf::test::fixed_width_column_wrapper<int64_t>(values.begin(), values.end()).release());
  return std::make_unique<cudf::table>(std::move(cols));
}

}  // namespace

void BM_parq_write_dict_encoding(nvbench::state& state)
{
  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const page_size_rows = num_rows / pages_per_chunk;

  auto const tbl  = build_table(num_rows, page_size_rows);
  auto const view = tbl->view();

  std::size_t encoded_file_size = 0;

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch&, auto& timer) {
               cuio_source_sink_pair source_sink(io_type::FILEPATH);

               // Lift the byte caps well above the row caps so page boundaries
               // are driven purely by `max_page_size_rows` (100K rows = 800KB
               // for INT64, which exceeds the default 512KB page byte cap).
               // This guarantees exactly `pages_per_chunk` pages per chunk,
               // matching the host-side data layout.
               constexpr std::size_t page_bytes_cap = std::size_t{64} << 20;

               timer.start();
               auto const write_opts =
                 cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
                   .compression(cudf::io::compression_type::NONE)
                   .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
                   .row_group_size_rows(num_rows)
                   .max_page_size_rows(page_size_rows)
                   .max_page_size_bytes(page_bytes_cap)
                   .build();
               cudf::io::write_parquet(write_opts);
               timer.stop();

               encoded_file_size = source_sink.size();
             });

  state.add_element_count(static_cast<double>(view.num_rows()), "rows");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(encoded_file_size, "encoded_file_size", "encoded_file_size");

  // TODO(phase 2): once per-page `dict_rle_bits` is plumbed into the writer
  // path and exposed at the reader, replace this whole-file proxy with an
  // exact min/max/mean across emitted pages. Until then we report aggregate
  // bits-per-value, which tracks overall compression ratio but not the
  // per-page distribution we ultimately want to shrink.
  auto const total_values = static_cast<std::size_t>(view.num_rows()) * num_cols;
  if (total_values > 0) {
    auto const bits_per_value_avg =
      static_cast<double>(encoded_file_size * 8) / static_cast<double>(total_values);
    state.add_element_count(bits_per_value_avg, "bits_per_value_avg");
  }
}

NVBENCH_BENCH(BM_parq_write_dict_encoding)
  .set_name("parquet_write_dict_encoding")
  .set_min_samples(3)
  .add_int64_axis("num_rows", {1'000'000});
