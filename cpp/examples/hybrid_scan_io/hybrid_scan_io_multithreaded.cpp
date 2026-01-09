/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.hpp"
#include "io_source.hpp"
#include "timer.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

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
  std::unordered_set<parquet_filter_type> const& filters;
  int const thread_id;
  int const thread_count;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;
  void operator()()
  {
    // Sweep the available input files
    for (auto curr_file_idx = thread_id; curr_file_idx < input_sources.size();
         curr_file_idx += thread_count) {
      timer timer;
      constexpr bool print_progress = false;
      constexpr bool single_step_materialize = true;
      std::ignore = hybrid_scan<print_progress, single_step_materialize>(
         input_sources[curr_file_idx], filter_expression, filters, stream, mr);
      std::cout << "Thread " << thread_id << " ";
      timer.print_elapsed_millis();
    }

    // Just synchronize this stream and exit
    stream.synchronize_no_throw();
  }
};

/**
 * @brief Function to setup and launch multithreaded hybrid scan reading.
 *
 * @param input_sources List of input sources to read
 * @param filter_expression Filter expression
 * @param filters Filters to apply
 * @param thread_count Number of threads
 * @param stream_pool CUDA stream pool to use for threads
 */
void hybrid_scan_multithreaded(
  std::vector<io_source> const& input_sources,
  std::vector<cudf::ast::operation> const& filter_expressions,
  std::unordered_set<parquet_filter_type> const& filters,
  int32_t thread_count,
  rmm::cuda_stream_pool& stream_pool,
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  // Table reading tasks
  std::vector<hybrid_scan_fn> read_tasks;
  read_tasks.reserve(thread_count);
  auto const num_filter_expressions = filter_expressions.size();
  // Create the read tasks
  std::for_each(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(thread_count), [&](auto tid) {
      read_tasks.emplace_back(
        hybrid_scan_fn{.input_sources     = input_sources,
                       .filter_expression = filter_expressions[tid % num_filter_expressions],
                       .filters           = filters,
                       .thread_id         = tid,
                       .thread_count      = thread_count,
                       .stream            = stream_pool.get_stream(),
                       .mr                = mr});
    });

  // Create threads with tasks
  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (auto& c : read_tasks) {
    threads.emplace_back(c);
  }
  for (auto& t : threads) {
    t.join();
  }
}

/**
 * @brief Function to print example usage and argument information
 */
void inline print_usage()
{
  std::cout << "\nUsage: hybrid_scan_io_multithreaded <comma delimited list of dirs and/or files> "
               "<input multiplier>\n"
               "                                    <thread count> <number of times to read> "
               "<column name>\n"
               "                                    <literal> <io source type> \n\n"

               "Available IO source types: HOST_BUFFER, PINNED_BUFFER (Default)\n\n"
               "Note: Provide as many arguments as you like in the above order. Default values\n"
               "      for the unprovided arguments will be used. All input parquet files will\n"
               "      be converted to the specified IO source type before reading\n\n";
}

/**
 * @brief The main function
 */
int32_t main(int argc, char const** argv)
{
  constexpr int32_t max_thread_count = 64;

  // Set arguments to defaults
  std::string input_paths       = "example.parquet";
  int32_t input_multiplier      = 1;
  int32_t thread_count          = 1;
  int32_t num_reads             = 1;
  auto column_name              = std::string{"string_col"};
  auto literal_value            = std::string{"0000001"};
  io_source_type io_source_type = io_source_type::HOST_BUFFER;

  // Set to the provided args
  switch (argc) {
    case 8: io_source_type = get_io_source_type(argv[7]); [[fallthrough]];
    case 7: literal_value = argv[6]; [[fallthrough]];
    case 6: column_name = argv[5]; [[fallthrough]];
    case 5: num_reads = std::max(1, std::stoi(argv[4])); [[fallthrough]];
    case 4:
      thread_count =
        std::min(max_thread_count, std::max(thread_count, std::stoi(std::string{argv[3]})));
      [[fallthrough]];
    case 3:
      input_multiplier =
        std::min(max_thread_count, std::max(input_multiplier, std::stoi(std::string{argv[2]})));
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

  // Initialize mr, default stream and stream pool
  bool constexpr is_pool_used = false;
  auto resource               = create_memory_resource(is_pool_used);
  auto default_stream         = cudf::get_default_stream();
  auto stats_mr =
    rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>(resource.get());
  rmm::mr::set_current_device_resource(&stats_mr);

  // List of input sources from the input_paths string.
  auto const input_sources = [&]() {
    try {
      return extract_input_sources(
        input_paths, input_multiplier, thread_count, io_source_type, default_stream);
      default_stream.synchronize();
    } catch (const std::exception& e) {
      print_usage();
      throw std::runtime_error(e.what());
    }
  }();

  // Check if there is nothing to do
  if (input_sources.empty()) {
    print_usage();
    throw std::runtime_error("No input files to read. Exiting early.\n");
  }

  // Create filter expressions (one per thread; reused circularly if needed)
  auto const column_reference = cudf::ast::column_name_reference(column_name);
  auto scalar                 = cudf::string_scalar(literal_value);
  auto literal                = cudf::ast::literal(scalar);

  std::vector<cudf::ast::operation> filter_expressions;
  filter_expressions.emplace_back(cudf::ast::ast_operator::EQUAL, column_reference, literal);

  // Insert which filters to apply
  std::unordered_set<parquet_filter_type> filters;
  {
    filters.insert(parquet_filter_type::ROW_GROUPS_WITH_STATS);
    filters.insert(parquet_filter_type::ROW_GROUPS_WITH_DICT_PAGES);
    filters.insert(parquet_filter_type::ROW_GROUPS_WITH_BLOOM_FILTERS);
    // Deliberately disabled as it has a high cost to benefit ratio
    // filters.insert(parquet_filter_type::FILTER_COLUMN_PAGES_WITH_PAGE_INDEX);
    filters.insert(parquet_filter_type::PAYLOAD_COLUMN_PAGES_WITH_ROW_MASK);
  }

  // Read the same parquet files specified times with multiple threads and discard the read tables
  {
    std::cout << "\nReading " << input_sources.size() << " input sources " << num_reads
              << " time(s) using " << thread_count
              << " threads and discarding output "
                 "tables..\n";

    std::cout << "Note that the first read may include times for nvcomp, cufile loading and RMM "
                 "growth.\n\n";

    auto stream_pool = rmm::cuda_stream_pool(thread_count, rmm::cuda_stream::flags::non_blocking);

    timer timer;
    std::for_each(thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(num_reads),
                  [&](auto i) {  // Read parquet files and discard the tables
                    hybrid_scan_multithreaded(
                      input_sources, filter_expressions, filters, thread_count, stream_pool);
                  });
    std::cout << "Total ";
    timer.print_elapsed_millis();
  }

  // Print peak memory
  std::cout << "Peak memory: " << (stats_mr.get_bytes_counter().peak / 1'048'576.0) << " MB\n\n";

  return 0;
}
