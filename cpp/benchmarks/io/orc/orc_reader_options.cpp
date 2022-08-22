/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/fixture/rmm_pool_raii.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/io/orc.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

constexpr int64_t data_size = 512 << 20;

enum class uses_index : bool { YES, NO };

enum class uses_numpy_dtype : bool { YES, NO };

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  uses_index,
  [](auto value) {
    switch (value) {
      case uses_index::YES: return "YES";
      case uses_index::NO: return "NO";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  uses_numpy_dtype,
  [](auto value) {
    switch (value) {
      case uses_numpy_dtype::YES: return "YES";
      case uses_numpy_dtype::NO: return "NO";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  column_selection,
  [](auto value) {
    switch (value) {
      case column_selection::ALL: return "ALL";
      case column_selection::ALTERNATE: return "ALTERNATE";
      case column_selection::FIRST_HALF: return "FIRST_HALF";
      case column_selection::SECOND_HALF: return "SECOND_HALF";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  row_selection,
  [](auto value) {
    switch (value) {
      case row_selection::ALL: return "ALL";
      case row_selection::NROWS: return "NROWS";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  cudf::type_id,
  [](auto value) {
    switch (value) {
      case cudf::type_id::EMPTY: return "EMPTY";
      case cudf::type_id::TIMESTAMP_NANOSECONDS: return "TIMESTAMP_NANOSECONDS";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

std::vector<std::string> get_col_names(cudf::io::source_info const& source)
{
  cudf::io::orc_reader_options const read_options =
    cudf::io::orc_reader_options::builder(source).num_rows(1);
  return cudf::io::read_orc(read_options).metadata.column_names;
}

template <column_selection ColSelection,
          row_selection RowSelection,
          uses_index UsesIndex,
          uses_numpy_dtype UsesNumpyDType,
          cudf::type_id Timestamp>
void BM_orc_read_varying_options(nvbench::state& state,
                                 nvbench::type_list<nvbench::enum_type<ColSelection>,
                                                    nvbench::enum_type<RowSelection>,
                                                    nvbench::enum_type<UsesIndex>,
                                                    nvbench::enum_type<UsesNumpyDType>,
                                                    nvbench::enum_type<Timestamp>>)
{
  cudf::rmm_pool_raii rmm_pool;

  auto constexpr num_chunks = 1;

  auto const use_index     = UsesIndex == uses_index::YES;
  auto const use_np_dtypes = UsesNumpyDType == uses_numpy_dtype::YES;
  auto const ts_type       = cudf::data_type{Timestamp};

  // skip_rows is not supported on nested types
  auto const data_types =
    dtypes_for_column_selection(get_type_or_group({int32_t(type_group_id::INTEGRAL_SIGNED),
                                                   int32_t(type_group_id::FLOATING_POINT),
                                                   int32_t(type_group_id::FIXED_POINT),
                                                   int32_t(type_group_id::TIMESTAMP),
                                                   int32_t(cudf::type_id::STRING)}),
                                ColSelection);
  auto const tbl  = create_random_table(data_types, table_size_bytes{data_size});
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(io_type::HOST_BUFFER);
  cudf::io::orc_writer_options options =
    cudf::io::orc_writer_options::builder(source_sink.make_sink_info(), view);
  cudf::io::write_orc(options);

  auto const cols_to_read =
    select_column_names(get_col_names(source_sink.make_source_info()), ColSelection);
  cudf::io::orc_reader_options read_options =
    cudf::io::orc_reader_options::builder(source_sink.make_source_info())
      .columns(cols_to_read)
      .use_index(use_index)
      .use_np_dtypes(use_np_dtypes)
      .timestamp_type(ts_type);

  auto const num_stripes              = data_size / (64 << 20);
  cudf::size_type const chunk_row_cnt = view.num_rows() / num_chunks;

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::default_stream_value.value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      try_drop_l3_cache();

      timer.start();
      cudf::size_type rows_read = 0;
      for (int32_t chunk = 0; chunk < num_chunks; ++chunk) {
        auto const is_last_chunk = chunk == (num_chunks - 1);
        switch (RowSelection) {
          case row_selection::ALL: break;
          case row_selection::STRIPES: {
            auto stripes_to_read = segments_in_chunk(num_stripes, num_chunks, chunk);
            if (is_last_chunk) {
              // Need to assume that an additional "overflow" stripe is present
              stripes_to_read.push_back(num_stripes);
            }
            read_options.set_stripes({stripes_to_read});
          } break;
          case row_selection::NROWS:
            read_options.set_skip_rows(chunk * chunk_row_cnt);
            read_options.set_num_rows(chunk_row_cnt);
            if (is_last_chunk) read_options.set_num_rows(-1);
            break;
          default: CUDF_FAIL("Unsupported row selection method");
        }

        rows_read += cudf::io::read_orc(read_options).tbl->num_rows();
      }

      CUDF_EXPECTS(rows_read == view.num_rows(), "Benchmark did not read the entire table");
      timer.stop();
    });

  auto const elapsed_time   = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  auto const data_processed = data_size * cols_to_read.size() / view.num_columns();
  state.add_element_count(static_cast<double>(data_processed) / elapsed_time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

using col_selections = nvbench::enum_type_list<column_selection::ALL,
                                               column_selection::ALTERNATE,
                                               column_selection::FIRST_HALF,
                                               column_selection::SECOND_HALF>;

NVBENCH_BENCH_TYPES(BM_orc_read_varying_options,
                    NVBENCH_TYPE_AXES(col_selections,
                                      nvbench::enum_type_list<row_selection::ALL>,
                                      nvbench::enum_type_list<uses_index::YES>,
                                      nvbench::enum_type_list<uses_numpy_dtype::YES>,
                                      nvbench::enum_type_list<cudf::type_id::EMPTY>))
  .set_name("orc_read_column_selection")
  .set_type_axes_names(
    {"column_selection", "row_selection", "uses_index", "uses_numpy_dtype", "timestamp_type"});

NVBENCH_BENCH_TYPES(
  BM_orc_read_varying_options,
  NVBENCH_TYPE_AXES(
    nvbench::enum_type_list<column_selection::ALL>,
    nvbench::enum_type_list<row_selection::NROWS>,
    nvbench::enum_type_list<uses_index::YES, uses_index::NO>,
    nvbench::enum_type_list<uses_numpy_dtype::YES, uses_numpy_dtype::NO>,
    nvbench::enum_type_list<cudf::type_id::EMPTY, cudf::type_id::TIMESTAMP_NANOSECONDS>))
  .set_name("orc_read_misc_options")
  .set_type_axes_names(
    {"column_selection", "row_selection", "uses_index", "uses_numpy_dtype", "timestamp_type"});
