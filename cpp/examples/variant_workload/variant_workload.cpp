/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// A multi-path VARIANT extraction workload that exercises
// `cudf::io::parquet::extract_variant_field` over a representative set of
// nested paths.
//
// The example synthesizes three VARIANT columns (A/B/C) entirely in device
// memory, then extracts 57 distinct JSONPath-like paths from each row and
// prints aggregate throughput.  The path set mixes top-level fields,
// array-and-object interleaving, and a deep fan-out into a shared sub-tree
// (see `variant_workload_fixture.hpp` for the full path table).  No Parquet
// I/O is involved; the workload measures only the extract_variant_field
// kernels plus their cast-to-STRING step.

#include "variant_workload_fixture.hpp"

#include <cudf/column/column.hpp>
#include <cudf/io/variant.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void print_usage(char const* argv0)
{
  std::cout << "Usage: " << argv0 << " [num_rows] [iterations]\n"
            << "  num_rows   : number of rows per VARIANT column to synthesize (default: 1048576)\n"
            << "  iterations : number of measurement iterations after 1 warm-up (default: 3)\n"
            << "\n"
            << "The example materializes three synthetic VARIANT columns (A, B, C) with\n"
            << "nested object/array payloads, then extracts 57 JSONPath-like paths from\n"
            << "each row via extract_variant_field and reports average elapsed time +\n"
            << "extractions/s.\n";
}

}  // namespace

int main(int argc, char const** argv)
{
  using namespace cudf::io::parquet::variant_workload;

  cudf::size_type num_rows = 1 << 20;
  int iterations           = 3;

  if (argc > 1) {
    std::string const arg{argv[1]};
    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    }
    num_rows = static_cast<cudf::size_type>(std::stoll(arg));
  }
  if (argc > 2) { iterations = std::atoi(argv[2]); }

  if (num_rows <= 0 || iterations <= 0) {
    std::cerr << "num_rows and iterations must be positive\n";
    print_usage(argv[0]);
    return 1;
  }

  // columnC rows are ~765 B each, and build_variant_column stores offsets as
  // int32_t; enforce the same cap the NVBench version uses.
  constexpr cudf::size_type max_rows = 1 << 20;
  if (num_rows > max_rows) {
    std::cerr << "num_rows > " << max_rows
              << " is not supported (int32_t offset overflow in the host-side\n"
              << "synthesis of columnC).\n";
    return 1;
  }

  rmm::mr::cuda_memory_resource cuda_mr{};
  rmm::mr::pool_memory_resource pool_mr{cuda_mr, rmm::percent_of_free_device_memory(50)};
  rmm::mr::statistics_resource_adaptor stats_mr{pool_mr};
  rmm::mr::set_current_device_resource(stats_mr);

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  std::cout << "num_rows   : " << num_rows << "\n"
            << "iterations : " << iterations << "\n"
            << "paths      : " << kWorkloadPaths.size() << "\n\n";

  std::cout << "Synthesizing VARIANT columns A/B/C..." << std::flush;
  auto const t_setup_begin = std::chrono::steady_clock::now();
  auto fixture             = make_uniform_columns(num_rows, stream, mr);
  stream.synchronize();
  auto const t_setup_end                            = std::chrono::steady_clock::now();
  std::chrono::duration<double> const setup_elapsed = t_setup_end - t_setup_begin;
  std::cout << " done in " << setup_elapsed.count() << " s.  "
            << "Device bytes / row: " << (fixture.total_bytes / static_cast<double>(num_rows))
            << "\n\n";

  std::array<cudf::column_view, 3> const col_views{
    fixture.A->view(), fixture.B->view(), fixture.C->view()};

  auto run_once = [&]() {
    for (auto const& entry : kWorkloadPaths) {
      auto result = cudf::io::parquet::extract_variant_field(col_views[static_cast<int>(entry.col)],
                                                             std::string_view{entry.path},
                                                             cudf::data_type{cudf::type_id::STRING},
                                                             stream,
                                                             mr);
    }
    stream.synchronize();
  };

  std::cout << "Warm-up run..." << std::flush;
  auto const t_warm_begin = std::chrono::steady_clock::now();
  run_once();
  auto const t_warm_end                            = std::chrono::steady_clock::now();
  std::chrono::duration<double> const warm_elapsed = t_warm_end - t_warm_begin;
  std::cout << " " << warm_elapsed.count() << " s\n";

  std::cout << "Measurement runs...\n";
  double total_seconds = 0.0;
  for (int i = 0; i < iterations; ++i) {
    auto const t_iter_begin = std::chrono::steady_clock::now();
    run_once();
    auto const t_iter_end                            = std::chrono::steady_clock::now();
    std::chrono::duration<double> const iter_elapsed = t_iter_end - t_iter_begin;
    total_seconds += iter_elapsed.count();
    std::cout << "  iter " << i << ": " << iter_elapsed.count() << " s\n";
  }

  double const mean_seconds = total_seconds / iterations;
  double const extractions_per_iter =
    static_cast<double>(num_rows) * static_cast<double>(kWorkloadPaths.size());

  std::cout << "\nResults\n"
            << "-------\n"
            << "Mean elapsed / iter : " << mean_seconds << " s\n"
            << "Paths per iter      : " << kWorkloadPaths.size() << "\n"
            << "Extractions per iter: " << extractions_per_iter << "\n"
            << "Extractions / s     : " << (extractions_per_iter / mean_seconds) << "\n"
            << "Peak device memory  : " << (stats_mr.get_bytes_counter().peak / (1024.0 * 1024.0))
            << " MiB\n";

  return 0;
}
