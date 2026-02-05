/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark.hpp"
#include "hybrid_scan_commons.hpp"
#include "io_source.hpp"
#include "utils.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <ranges>
#include <stdexcept>
#include <string>
#include <thread>

/**
 * @file hybrid_scan_io_multithreaded.cpp
 *
 * @brief Demonstrates reading parquet data from the specified io source with libcudf's next-gen
 * parquet reader (hybrid scan reader) subject to a highly selective point lookup (col_name ==
 * literal) filter using multiple threads.
 *
 * The input parquet data is provided via files which are converted to the specified io source type
 * to be read using multiple threads. Optionally, the parquet data read by each thread can be
 * written to corresponding files and checked for validatity of the output files against the input
 * data.
 *
 * Run: ``hybrid_scan_io_multithreaded -h`` to see help with input args and more information.
 *
 * The following io source types are supported:
 * IO source types: HOST_BUFFER, PINNED_BUFFER
 */

/**
 * @brief Functor for multithreaded parquet reading based on the provided read_mode
 */
struct hybrid_scan_fn {
  std::vector<io_source> const& input_sources;
  cudf::ast::expression const& filter_expression;
  std::unordered_set<hybrid_scan_filter_type> const& filters;
  int const tid;
  int const num_threads;
  bool const single_step_read;
  bool const verbose;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  void operator()()
  {
    // Sweep through input sources (strided by num_threads, starting at tid)
    auto strided_indices = std::views::iota(size_t{0}, input_sources.size()) |
                           std::views::filter([&](auto idx) { return idx % num_threads == tid; });

    timer timer;

    for (auto source_idx : strided_indices) {
      if (single_step_read) {
        constexpr bool use_page_index = false;
        std::ignore                   = hybrid_scan<true, use_page_index>(
          input_sources[source_idx], filter_expression, filters, false, stream, mr);
      } else {
        constexpr bool use_page_index = true;
        std::ignore                   = hybrid_scan<false, use_page_index>(
          input_sources[source_idx], filter_expression, filters, false, stream, mr);
      }
    }

    stream.synchronize_no_throw();

    if (verbose) {
      std::cout << "Thread " << tid << " ";
      timer.print_elapsed_millis();
    }
  }
};

/**
 * @brief Helper to setup and launch multithreaded hybrid scan reading
 *
 * @param input_sources List of input sources
 * @param filter_expressions List of filter expressions (one per source; cycled)
 * @param filters Set of hybrid scan filters to apply
 * @param num_threads Number of threads
 * @param single_step_read Whether to use two-step table materialization
 * @param stream_pool CUDA stream pool
 * @param mr Memory resource to use for threads
 */
void hybrid_scan_multithreaded(std::vector<io_source> const& input_sources,
                               std::vector<cudf::ast::operation> const& filter_expressions,
                               std::unordered_set<hybrid_scan_filter_type> const& filters,
                               cudf::size_type num_threads,
                               bool single_step_read,
                               bool verbose,
                               rmm::cuda_stream_pool& stream_pool,
                               rmm::device_async_resource_ref mr)
{
  std::vector<hybrid_scan_fn> read_tasks;
  read_tasks.reserve(num_threads);
  auto const num_filter_expressions = filter_expressions.size();

  // Emplace parquet read tasks
  std::for_each(
    thrust::counting_iterator(0), thrust::counting_iterator(num_threads), [&](auto tid) {
      read_tasks.emplace_back(
        hybrid_scan_fn{.input_sources     = input_sources,
                       .filter_expression = filter_expressions[tid % num_filter_expressions],
                       .filters           = filters,
                       .tid               = tid,
                       .num_threads       = num_threads,
                       .single_step_read  = single_step_read,
                       .verbose           = verbose,
                       .stream            = stream_pool.get_stream(),
                       .mr                = mr});
    });

  // Create and launch threads
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (auto& c : read_tasks) {
    threads.emplace_back(c);
  }

  // Wait for all threads to complete
  for (auto& t : threads) {
    t.join();
  }
}

/**
 * @brief Function to print example usage and argument information
 */
void inline print_usage()
{
  std::cout
    << "\nUsage: hybrid_scan_io_multithreaded <comma delimited list of dirs and/or files> "
       "<input multiplier>\n"
       "                                    <thread count> <single step read:Y/N> "
       "<io source type>\n"
       "                                    <iterations> <column name> <literal> <verbose:Y/N>\n\n"

       "Available IO source types: FILEPATH (Default), HOST_BUFFER, PINNED_BUFFER\n\n"
       "Note: Provide as many arguments as you like in the above order. Default values\n"
       "      for the unprovided arguments will be used. All input parquet files will\n"
       "      be converted to the specified IO source type before reading\n\n";
}

/**
 * @brief The main function
 */
int main(int argc, char const** argv)
{
  auto const max_threads = std::thread::hardware_concurrency();

  // Set arguments to defaults
  auto input_paths      = std::string{"example.parquet"};
  auto input_multiplier = 1;
  auto num_threads      = 2;
  auto single_step_read = true;
  auto io_source_type   = io_source_type::FILEPATH;
  auto iterations       = 1;
  auto column_name      = std::string{"string_col"};
  auto literal_value    = std::string{"0000001"};
  auto verbose          = false;

  // Set to the provided args
  switch (argc) {
    case 10: verbose = get_boolean(argv[9]); [[fallthrough]];
    case 9: literal_value = argv[8]; [[fallthrough]];
    case 8: column_name = argv[7]; [[fallthrough]];
    case 7: iterations = std::stoi(argv[6]); [[fallthrough]];
    case 6: io_source_type = get_io_source_type(argv[5]); [[fallthrough]];
    case 5: single_step_read = get_boolean(argv[4]); [[fallthrough]];
    case 4:
      num_threads =
        std::min<int>(max_threads, std::max(num_threads, std::stoi(std::string{argv[3]})));
      [[fallthrough]];
    case 3:
      input_multiplier = std::min<int>(
        max_threads, std::max<int>(input_multiplier, std::stoi(std::string{argv[2]})));
      [[fallthrough]];
    case 2:
      // Check if instead of input_paths, the first argument is `-h` or `--help`
      if (auto arg = std::string{argv[1]};
          arg != "-h" and arg != "--help" and not arg.starts_with("-")) {
        input_paths = std::move(arg);
        break;
      }
      [[fallthrough]];
    default: print_usage(); throw std::runtime_error("Exiting...");
  }

  // Initialize mr, stream pool and default stream
  auto constexpr is_pool_used = false;
  auto resource               = create_memory_resource(is_pool_used);
  auto stream_pool = rmm::cuda_stream_pool(1 + num_threads, rmm::cuda_stream::flags::non_blocking);
  auto default_stream = stream_pool.get_stream();
  auto stats_mr =
    rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>(resource.get());
  rmm::mr::set_current_device_resource(&stats_mr);

  // Create filter expressions (one per thread; reused circularly if needed)
  auto const column_reference = cudf::ast::column_name_reference(column_name);
  auto scalar                 = cudf::string_scalar(literal_value, true, default_stream);
  default_stream.synchronize();
  auto literal = cudf::ast::literal(scalar);

  std::vector<cudf::ast::operation> filter_expressions;
  filter_expressions.emplace_back(cudf::ast::ast_operator::EQUAL, column_reference, literal);

  // Insert which hybrid scan filters to apply
  std::unordered_set<hybrid_scan_filter_type> filters;
  {
    filters.insert(hybrid_scan_filter_type::ROW_GROUPS_WITH_STATS);
    filters.insert(hybrid_scan_filter_type::ROW_GROUPS_WITH_DICT_PAGES);
    filters.insert(hybrid_scan_filter_type::ROW_GROUPS_WITH_BLOOM_FILTERS);
    // Deliberately disabled as it has a high cost to benefit ratio
    // filters.insert(hybrid_scan_filter_type::FILTER_COLUMN_PAGES_WITH_PAGE_INDEX);
    filters.insert(hybrid_scan_filter_type::PAYLOAD_COLUMN_PAGES_WITH_ROW_MASK);
  }

  // List of input sources from the input_paths string.
  auto const input_sources = [&]() {
    try {
      return extract_input_sources(
        input_paths, input_multiplier, num_threads, io_source_type, default_stream);
      default_stream.synchronize();
    } catch (const std::exception& e) {
      print_usage();
      throw std::runtime_error(e.what());
    }
  }();
  if (input_sources.empty()) {
    print_usage();
    throw std::runtime_error("No input files to read. Exiting early.\n");
  }

  // Read the input parquet sources specified times with multiple threads and discard the output
  std::cout << "\nReading " << input_sources.size() << " input sources " << iterations
            << " time(s) using " << num_threads
            << " threads and discarding output "
               "tables..\n";

  std::cout << "Note that the first read may include times for nvcomp, cufile loading and RMM "
               "growth.\n\n";

  benchmark(
    [&] {
      hybrid_scan_multithreaded(input_sources,
                                filter_expressions,
                                filters,
                                num_threads,
                                single_step_read,
                                verbose,
                                stream_pool,
                                stats_mr);
    },
    iterations);

  // Print peak memory
  std::cout << "Peak memory: " << (stats_mr.get_bytes_counter().peak / 1'048'576.0) << " MB\n\n";

  return 0;
}
