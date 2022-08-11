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
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <nvbench/nvbench.cuh>

#include <cudf/column/column.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/table/table.hpp>

#include <thrust/iterator/transform_iterator.h>

// to enable, run cmake with -DBUILD_BENCHMARKS=ON

constexpr int64_t data_size = 512 << 20;

namespace cudf_io = cudf::io;

void nvbench_orc_write(nvbench::state& state)
{
  cudf::size_type num_cols = state.get_int64("num_columns");

  auto tbl =
    create_random_table(cycle_dtypes(get_type_or_group({int32_t(type_group_id::INTEGRAL_SIGNED),
                                                        int32_t(type_group_id::FLOATING_POINT),
                                                        int32_t(type_group_id::FIXED_POINT),
                                                        int32_t(type_group_id::TIMESTAMP),
                                                        int32_t(cudf::type_id::STRING),
                                                        int32_t(cudf::type_id::STRUCT),
                                                        int32_t(cudf::type_id::LIST)}),
                                     num_cols),
                        table_size_bytes{data_size});
  cudf::table_view view = tbl->view();

  auto mem_stats_logger = cudf::memory_stats_logger();

  state.add_global_memory_reads<int64_t>(data_size);
  state.add_element_count(view.num_columns() * view.num_rows());

  size_t encoded_file_size = 0;

  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               cuio_source_sink_pair source_sink(io_type::VOID);
               timer.start();

               cudf_io::orc_writer_options opts =
                 cudf_io::orc_writer_options::builder(source_sink.make_sink_info(), view);
               cudf_io::write_orc(opts);

               timer.stop();
               encoded_file_size = source_sink.size();
             });

  state.add_buffer_size(mem_stats_logger.peak_memory_usage(), "pmu", "Peak Memory Usage");
  state.add_buffer_size(encoded_file_size, "efs", "Encoded File Size");
  state.add_buffer_size(view.num_rows(), "trc", "Total Rows");
}

void nvbench_orc_chunked_write(nvbench::state& state)
{
  cudf::size_type num_cols   = state.get_int64("num_columns");
  cudf::size_type num_tables = state.get_int64("num_chunks");

  std::vector<std::unique_ptr<cudf::table>> tables;
  for (cudf::size_type idx = 0; idx < num_tables; idx++) {
    tables.push_back(
      create_random_table(cycle_dtypes(get_type_or_group({int32_t(type_group_id::INTEGRAL_SIGNED),
                                                          int32_t(type_group_id::FLOATING_POINT),
                                                          int32_t(type_group_id::FIXED_POINT),
                                                          int32_t(type_group_id::TIMESTAMP),
                                                          int32_t(cudf::type_id::STRING),
                                                          int32_t(cudf::type_id::STRUCT),
                                                          int32_t(cudf::type_id::LIST)}),
                                       num_cols),
                          table_size_bytes{size_t(data_size / num_tables)}));
  }

  auto mem_stats_logger = cudf::memory_stats_logger();

  auto size_iter = thrust::make_transform_iterator(
    tables.begin(), [](auto const& i) { return i->num_columns() * i->num_rows(); });
  auto row_count_iter =
    thrust::make_transform_iterator(tables.begin(), [](auto const& i) { return i->num_rows(); });
  auto total_elements = std::accumulate(size_iter, size_iter + num_tables, 0);
  auto total_rows     = std::accumulate(row_count_iter, row_count_iter + num_tables, 0);

  state.add_global_memory_reads<int64_t>(data_size);
  state.add_element_count(total_elements);

  size_t encoded_file_size = 0;

  state.exec(
    nvbench::exec_tag::timer | nvbench::exec_tag::sync, [&](nvbench::launch& launch, auto& timer) {
      cuio_source_sink_pair source_sink(io_type::VOID);
      timer.start();

      cudf_io::chunked_orc_writer_options opts =
        cudf_io::chunked_orc_writer_options::builder(source_sink.make_sink_info());
      cudf_io::orc_chunked_writer writer(opts);
      std::for_each(tables.begin(),
                    tables.end(),
                    [&writer](std::unique_ptr<cudf::table> const& tbl) { writer.write(*tbl); });
      writer.close();

      timer.stop();
      encoded_file_size = source_sink.size();
    });

  state.add_buffer_size(mem_stats_logger.peak_memory_usage(), "pmu", "Peak Memory Usage");
  state.add_buffer_size(encoded_file_size, "efs", "Encoded File Size");
  state.add_buffer_size(total_rows, "trc", "Total Rows");
}

NVBENCH_BENCH(nvbench_orc_write)
  .set_name("orc_write")
  .set_min_samples(4)
  .add_int64_axis("num_columns", {8, 64});

NVBENCH_BENCH(nvbench_orc_chunked_write)
  .set_name("orc_chunked_write")
  .set_min_samples(4)
  .add_int64_axis("num_columns", {8, 64})
  .add_int64_axis("num_chunks", {8, 64});
