/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// EXPERIMENTAL/DNM: case-study benchmark for issue #17313. Exposes the writer's
// `compression_threshold` knob to NVBench so reviewers can sweep
// `compressed_size < uncompressed_size * threshold` end-to-end.
//
// Scope: read-side performance only. The write step is not timed; it only exists to materialize
// the file. The metric of interest is read-throughput / GPU read-time after the writer's
// compress-or-not decision flips, plus the `encoded_file_size` and per-page
// (num_pages, num_compressed_pages) counts that landed on host memory.
//
// Two bench registrations live in this TU:
//   - `parquet_read_compression_threshold_strings`         : V1 page headers, full encoding axis.
//   - `parquet_read_compression_threshold_strings_v2dict`  : V2 page headers, DICTIONARY only,
//                                                            pruned codec axis.
//
// Per-cell strategy:
//   - `(cardinality, run_length, avg_string_length)` is registered last on the NVBench axis list,
//     so NVBench iterates those three outermost. Any consecutive cells with the same triple share
//     the random table via a single-entry cache (`g_cached_*`), avoiding ~10s of
//     `create_random_table` work for every (codec, encoding, threshold) inner cell.
//   - `max_page_size_bytes(256 KiB)` is set on the writer so 512 MiB of data lands as ~2000 pages
//     per file. That gives the threshold decision enough statistical weight per cell.
//   - After write we re-open the file (HOST_BUFFER datasource), parse the parquet footer + page
//     index via `CompactProtocolReader`, and walk every PageHeader to count
//     `compressed_page_size < uncompressed_page_size`. The two counts are emitted as NVBench
//     element-count summaries; the timed read path is unchanged.
//
// This file is exploratory; it stays narrow (single string column, HOST_BUFFER only) and is not
// intended to land alongside production code. See plan:
// `~/.cursor/plans/compressed-page_broad_sweep_*.plan.md`.

#include "reader_common.hpp"

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

// Public cuDF IO headers for the bench body and the page-counting helper.
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

// Internal cuDF parquet utility: thrift compact-protocol reader. The bench TU is in-tree and DNM,
// and other in-tree code (e.g. `src/io/parquet/experimental/hybrid_scan_helpers.cpp`) reaches
// into this same `parquet::detail` namespace. Acceptable for an exploratory bench.
#include "io/parquet/compact_protocol_reader.hpp"

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <cstdlib>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

// Cell-skip threshold matched to the plan: at cardinality * avg_string_length > 1 GiB the unique
// string pool starts to dominate host memory and the data generator OOMs on a laptop.
constexpr std::size_t oom_skip_bytes = 1ull << 30;

// At cardinality >= 1'000'000 the writer's dictionary policy (ADAPTIVE) effectively bails out of
// dictionary encoding because the dictionary itself stops fitting. Re-running the same shape under
// the DICTIONARY label would double-count those cells against PLAIN, so we drop them.
constexpr cudf::size_type dict_cardinality_skip = 1'000'000;

// Force the writer to land ~2000 pages per file (at 512 MiB / 256 KiB ~= 2048 pages). Gives the
// per-page compression decision enough statistical weight for the case study.
constexpr std::size_t bench_max_page_size_bytes = 256ull * 1024ull;

bool oom_prone(cudf::size_type cardinality, cudf::size_type avg_string_length)
{
  return static_cast<std::size_t>(cardinality) * static_cast<std::size_t>(avg_string_length) >
         oom_skip_bytes;
}

cudf::io::column_encoding parse_encoding(std::string const& name)
{
  if (name == "DICTIONARY") { return cudf::io::column_encoding::DICTIONARY; }
  if (name == "PLAIN") { return cudf::io::column_encoding::PLAIN; }
  if (name == "DELTA_LENGTH_BYTE_ARRAY") {
    return cudf::io::column_encoding::DELTA_LENGTH_BYTE_ARRAY;
  }
  throw std::invalid_argument("Unsupported encoding axis value: " + name);
}

// ----------------------------------------------------------------------------------------------
// Single-entry table cache keyed on (cardinality, run_length, avg_string_length).
//
// NVBench evaluates axis combinations in the reverse order they were registered: the LAST axis
// added iterates innermost. We therefore register (io_type, compression_type, encoding,
// threshold) FIRST and (cardinality, run_length, avg_string_length, data_size) LAST, so the
// expensive triple iterates outermost. Within a (card, RL, ASL) group, all (codec * encoding *
// threshold) cells share the same random table.
// ----------------------------------------------------------------------------------------------

struct cache_key {
  cudf::size_type cardinality{};
  cudf::size_type run_length{};
  cudf::size_type avg_string_length{};

  bool operator==(cache_key const& other) const noexcept
  {
    return cardinality == other.cardinality && run_length == other.run_length &&
           avg_string_length == other.avg_string_length;
  }
};

cache_key g_cached_key{};
std::unique_ptr<cudf::table> g_cached_table{};

cudf::table_view get_or_build_table(cache_key const& key, std::size_t data_size_bytes)
{
  if (g_cached_table && g_cached_key == key) { return g_cached_table->view(); }

  // Mirrors parquet_reader_strings.cpp: NORMAL distribution centered on avg_string_length
  // with ~3-sigma half-width.
  auto const d_type     = get_type_or_group(static_cast<int32_t>(data_type::STRING));
  auto const half_width = key.avg_string_length >> 3;
  auto const length_min = key.avg_string_length - half_width;
  auto const length_max = key.avg_string_length + half_width;
  data_profile profile =
    data_profile_builder()
      .cardinality(key.cardinality)
      .avg_run_length(key.run_length)
      .distribution(data_type::STRING, distribution_id::NORMAL, length_min, length_max);

  g_cached_table = create_random_table(d_type, table_size_bytes{data_size_bytes}, profile);
  g_cached_key   = key;
  return g_cached_table->view();
}

// ----------------------------------------------------------------------------------------------
// Per-page count helper.
//
// Re-opens the just-written file as a HOST_BUFFER datasource, fetches the footer + page index,
// parses each ColumnChunk's OffsetIndex, then walks every PageLocation to read the PageHeader
// thrift struct at the page's file offset. A page is "compressed" iff its compressed size is
// strictly less than its uncompressed size (the same predicate the writer kernel uses).
// ----------------------------------------------------------------------------------------------

struct page_counts {
  std::size_t total{0};
  std::size_t compressed{0};
};

page_counts count_pages_in_written_file(cuio_source_sink_pair& source_sink)
{
  using cudf::io::parquet::detail::CompactProtocolReader;

  auto const src_info = source_sink.make_source_info();

  // Materialize a datasource. The bench TU is HOST_BUFFER-only, but the branch below covers the
  // FILEPATH case for future flexibility.
  std::unique_ptr<cudf::io::datasource> datasrc;
  if (src_info.type() == cudf::io::io_type::HOST_BUFFER) {
    datasrc = cudf::io::datasource::create(src_info.host_buffers()[0]);
  } else if (src_info.type() == cudf::io::io_type::FILEPATH) {
    datasrc = cudf::io::datasource::create(src_info.filepaths()[0]);
  } else {
    return {};  // Unsupported bench source -- skip page counting.
  }

  // Footer -> FileMetaData.
  auto const footer_buf = cudf::io::parquet::fetch_footer_to_host(*datasrc);
  CompactProtocolReader cp(footer_buf->data(), footer_buf->size());
  cudf::io::parquet::FileMetaData md;
  cp.read(&md);

  // Cache the full file bytes once so we can pluck PageHeader bytes by file offset cheaply.
  auto const file_size  = source_sink.size();
  auto const file_bytes = datasrc->host_read(0, file_size);
  auto const* base      = file_bytes->data();

  // Page index -> per-column-chunk OffsetIndex. We follow the recipe from
  // src/io/parquet/reader_impl_helpers.cpp lines 353-396: column_index and offset_index live
  // back-to-back in [min_offset, max_offset) at the end of the file just before the footer.
  if (md.row_groups.empty() || md.row_groups.front().columns.empty()) { return {}; }
  int64_t const min_offset = md.row_groups.front().columns.front().column_index_offset;
  auto const& last_col     = md.row_groups.back().columns.back();
  int64_t const max_offset = last_col.offset_index_offset + last_col.offset_index_length;
  if (max_offset <= min_offset) {
    return {};  // No page index. Should not happen; cuDF writes the page index by default.
  }
  auto const page_idx_buf =
    datasrc->host_read(static_cast<std::size_t>(min_offset),
                       static_cast<std::size_t>(max_offset - min_offset));
  auto const* idx_base = page_idx_buf->data();

  page_counts counts{};
  for (auto& rg : md.row_groups) {
    for (auto& col : rg.columns) {
      if (col.offset_index_length <= 0 || col.offset_index_offset <= 0) { continue; }
      // Materialize this column's OffsetIndex.
      cp.init(idx_base + (col.offset_index_offset - min_offset),
              static_cast<std::size_t>(col.offset_index_length));
      cudf::io::parquet::OffsetIndex offset_index;
      cp.read(&offset_index);

      for (auto const& pl : offset_index.page_locations) {
        if (pl.offset < 0 ||
            static_cast<std::size_t>(pl.offset) + static_cast<std::size_t>(pl.compressed_page_size) >
              file_size) {
          continue;  // Defensive: malformed page index; skip.
        }
        cp.init(base + pl.offset, static_cast<std::size_t>(pl.compressed_page_size));
        cudf::io::parquet::PageHeader hdr;
        cp.read(&hdr);
        counts.total += 1;
        if (hdr.compressed_page_size < hdr.uncompressed_page_size) { counts.compressed += 1; }
      }
    }
  }
  return counts;
}

// ----------------------------------------------------------------------------------------------
// Shared per-cell body. `write_v2_headers` selects V1 vs V2 page headers; everything else comes
// from the NVBench axis values.
// ----------------------------------------------------------------------------------------------

void run_compression_threshold_cell(nvbench::state& state, bool write_v2_headers)
{
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length  = static_cast<cudf::size_type>(state.get_int64("run_length"));
  auto const avg_string_length =
    static_cast<cudf::size_type>(state.get_int64("avg_string_length"));
  auto const data_size      = static_cast<size_t>(state.get_int64("data_size"));
  auto const source_type    = retrieve_io_type_enum(state.get_string("io_type"));
  auto const compression    = retrieve_compression_type_enum(state.get_string("compression_type"));
  auto const threshold      = state.get_float64("threshold");
  auto const encoding_name  = state.get_string("encoding");
  auto const encoding       = parse_encoding(encoding_name);

  if (oom_prone(cardinality, avg_string_length)) {
    state.skip("Skipping OOM-prone configuration: cardinality * avg_string_length > 1 GiB");
    return;
  }

  // Dictionary at huge cardinality degenerates to PLAIN inside the writer. Drop those cells so the
  // DICTIONARY facet stays meaningful and we don't double-count against PLAIN.
  if (encoding == cudf::io::column_encoding::DICTIONARY && cardinality >= dict_cardinality_skip) {
    state.skip(
      "Skipping degenerate DICTIONARY at large cardinality (writer policy falls back to PLAIN)");
    return;
  }

  // Reduce the large-string materialization threshold so we can still write a 512 MiB table when
  // avg_string_length is multi-KiB; matches parquet_reader_strings.cpp's behavior.
  setenv("LIBCUDF_LARGE_STRINGS_THRESHOLD", "1", 1);

  cuio_source_sink_pair source_sink(source_type);

  // Build (or reuse) the random table for this (cardinality, run_length, avg_string_length).
  cache_key const key{cardinality, run_length, avg_string_length};
  auto const view = get_or_build_table(key, data_size);

  cudf::io::table_input_metadata expected_metadata(view);
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    expected_metadata.column_metadata[i].set_encoding(encoding);
  }

  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
      .metadata(expected_metadata)
      .compression(compression)
      .compression_threshold(threshold)
      .write_v2_headers(write_v2_headers)
      .max_page_size_bytes(bench_max_page_size_bytes)
      // STATISTICS_COLUMN emits the per-column-chunk page index (ColumnIndex + OffsetIndex)
      // that the page-counting helper relies on. Default STATISTICS_ROWGROUP would leave
      // num_pages = 0.
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN);
  cudf::io::write_parquet(write_opts);
  auto const num_rows_written = view.num_rows();

  // Per-page compressed-page count, before the timed read. Done here so the count is recorded
  // even if the timed read itself is the slowest part of the cell.
  page_counts const pc = count_pages_in_written_file(source_sink);
  state.add_element_count(static_cast<double>(pc.total), "num_pages");
  state.add_element_count(static_cast<double>(pc.compressed), "num_compressed_pages");

  parquet_read_common(num_rows_written, /*num_cols_to_read=*/1, source_sink, state);

  unsetenv("LIBCUDF_LARGE_STRINGS_THRESHOLD");
}

}  // namespace

void BM_parquet_read_compression_threshold_strings(nvbench::state& state)
{
  run_compression_threshold_cell(state, /*write_v2_headers=*/false);
}

// Axis order: NVBench iterates the LAST axis innermost. So the (cardinality, run_length,
// avg_string_length, data_size) tuple is registered last, which means it iterates outermost ->
// the table cache stays warm across the (codec * encoding * threshold) inner loop.
NVBENCH_BENCH(BM_parquet_read_compression_threshold_strings)
  .set_name("parquet_read_compression_threshold_strings")
  .add_string_axis("io_type", {"HOST_BUFFER"})
  .add_string_axis("compression_type", {"NONE", "SNAPPY", "ZSTD", "LZ4"})
  .add_string_axis("encoding", {"DICTIONARY", "PLAIN", "DELTA_LENGTH_BYTE_ARRAY"})
  .add_float64_axis("threshold",
                    {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                     0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95})
  .add_int64_axis("cardinality", {100, 10'000, 1'000'000})
  .add_int64_axis("run_length", {1, 8, 32})
  .add_int64_axis("avg_string_length", {32, 4096, 65536})
  .set_min_samples(4)
  .add_int64_axis("data_size", {512 << 20});

// V2-DICTIONARY supplemental sweep: tells apart V1 PLAIN_DICTIONARY from V2 RLE_DICTIONARY.
// Pruned axes (codec {NONE, SNAPPY} only, run_length=1) keep the wall time to a small fraction of
// the V1 main sweep while preserving the V1/V2 contrast.
void BM_parquet_read_compression_threshold_strings_v2dict(nvbench::state& state)
{
  run_compression_threshold_cell(state, /*write_v2_headers=*/true);
}

NVBENCH_BENCH(BM_parquet_read_compression_threshold_strings_v2dict)
  .set_name("parquet_read_compression_threshold_strings_v2dict")
  .add_string_axis("io_type", {"HOST_BUFFER"})
  .add_string_axis("compression_type", {"NONE", "SNAPPY"})
  .add_string_axis("encoding", {"DICTIONARY"})
  .add_float64_axis("threshold",
                    {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                     0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95})
  .add_int64_axis("cardinality", {100, 10'000, 1'000'000})
  .add_int64_axis("run_length", {1})
  .add_int64_axis("avg_string_length", {32, 4096, 65536})
  .set_min_samples(2)
  .add_int64_axis("data_size", {512 << 20});
