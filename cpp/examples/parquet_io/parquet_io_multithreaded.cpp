/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "common.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>

#include <fmt/chrono.h>

/**
 * @file parquet_io_multithreaded.cpp
 * @brief Demonstrates multithreaded read of parquet files and optionally
 * multithreaded writing the read tables to the specified io sink source type.
 *
 * Run: ``parquet_io_multithreaded -h`` to see help with input args and more.
 *
 * The following io sink types are supported:
 * IO sink types: FILEPATH, HOST_BUFFER, PINNED_BUFFER, DEVICE_BUFFER
 *
 */

// Type alias for unique ptr to cudf table
using table_t = std::unique_ptr<cudf::table>;

/**
 * @brief Behavior when handling the read tables by multiple threads
 */
enum class read_mode {
  NOWORK,              ///< Only read and discard tables
  CONCATENATE_THREAD,  ///< Read and concatenate tables from each thread
  CONCATENATE_ALL,     ///< Read and concatenate everything to a single table
};

/**
 * @brief Functor for multithreaded parquet reading based on the provided read_mode
 */
template <read_mode READ_FN>
struct read_fn {
  std::vector<std::string> const& input_files;
  std::vector<table_t>& tables;
  int const thread_id;
  int const thread_count;
  rmm::cuda_stream_view stream;

  void operator()()
  {
    // Tables read by this thread
    std::vector<table_t> tables_this_thread;

    // Sweep the available input files
    for (auto curr_file_idx = thread_id; curr_file_idx < input_files.size();
         curr_file_idx += thread_count) {
      auto const source_info = cudf::io::source_info(input_files[curr_file_idx]);
      auto builder           = cudf::io::parquet_reader_options::builder(source_info);
      auto const options     = builder.build();
      if constexpr (READ_FN != read_mode::NOWORK) {
        tables_this_thread.push_back(cudf::io::read_parquet(options, stream).tbl);
      } else {
        cudf::io::read_parquet(options, stream);
      }
    }

    // Concatenate the tables read by this thread if not NOWORK read_mode.
    if constexpr (READ_FN != read_mode::NOWORK) {
      auto table = concatenate_tables(std::move(tables_this_thread), stream);
      stream.synchronize_no_throw();
      tables[thread_id] = std::move(table);
    } else {
      // Just synchronize this stream and exit
      stream.synchronize_no_throw();
    }
  }
};

/**
 * @brief Function to setup and launch multithreaded parquet reading.
 *
 * @tparam read_mode Specifies if to concatenate and return the actual
 *                    tables or discard them and return an empty vector
 *
 * @param files List of files to read
 * @param thread_count Number of threads
 * @param stream_pool CUDA stream pool to use for threads
 *
 * @return Vector of read tables.
 */
template <read_mode read_mode>
std::vector<table_t> read_parquet_multithreaded(std::vector<std::string> const& files,
                                                int32_t thread_count,
                                                rmm::cuda_stream_pool& stream_pool)
{
  // Tables read by each thread
  std::vector<table_t> tables(thread_count);

  // Table reading tasks
  std::vector<read_fn<read_mode>> read_tasks;
  read_tasks.reserve(thread_count);

  // Create the read tasks
  std::for_each(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(thread_count), [&](auto tid) {
      read_tasks.emplace_back(
        read_fn<read_mode>{files, tables, tid, thread_count, stream_pool.get_stream()});
    });

  // Create threads with tasks
  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (auto& c : read_tasks) {
    threads.emplace_back(std::thread{c});
  }
  for (auto& t : threads) {
    t.join();
  }

  // If CONCATENATE_ALL mode, then concatenate to a vector of one final table.
  if (read_mode == read_mode::CONCATENATE_ALL) {
    auto stream    = stream_pool.get_stream();
    auto final_tbl = concatenate_tables(std::move(tables), stream);
    stream.synchronize();
    tables.clear();
    tables.emplace_back(std::move(final_tbl));
  }

  return tables;
}

/**
 * @brief Functor for multithreaded parquet writing
 */
struct write_fn {
  cudf::io::io_type io_sink_type;
  std::vector<cudf::table_view> const& table_views;
  int const thread_id;
  rmm::cuda_stream_view stream;

  void operator()()
  {
    // Create a sink
    auto const sink_info = [io_sink_type = io_sink_type, thread_id = thread_id]() {
      return cudf::io::sink_info(get_default_output_path() + "/table_" + std::to_string(thread_id) +
                                 ".parquet");
    }();
    // Writer options builder
    auto builder = cudf::io::parquet_writer_options::builder(sink_info, table_views[thread_id]);
    // Create a new metadata for the table
    auto table_metadata = cudf::io::table_input_metadata{table_views[thread_id]};

    builder.metadata(table_metadata);
    auto options = builder.build();

    // Write parquet data
    cudf::io::write_parquet(options, stream);

    // Done with this stream
    stream.synchronize_no_throw();
  }
};

/**
 * @brief The main function
 */
int32_t main(int argc, char const** argv)
{
  // Set arguments to defaults
  std::string input_paths                  = "example.parquet";
  int32_t input_multiplier                 = 1;
  int32_t num_reads                        = 1;
  int32_t thread_count                     = 2;
  std::optional<cudf::io::io_type> io_type = std::nullopt;
  bool validate_output                     = false;

  // Function to print example usage
  auto const print_usage = [] {
    fmt::print(
      fg(fmt::color::yellow),
      "\nUsage: parquet_io_multithreaded <comma delimited list of dirs and/or files>\n"
      "                                <input files multiplier> <number of times to reads>\n"
      "                                <thread count> <io sink type> <validate output: "
      "yes/no>\n\n");
    fmt::print(
      fg(fmt::color::light_sky_blue),
      "Note: Provide as many arguments as you like in the above order. Default values\n"
      "      for the unprovided arguments will be used. No output parquet will be written\n"
      "      if <io sink type> isn't provided.\n\n");
  };

  // Set to the provided args
  switch (argc) {
    case 7: validate_output = get_boolean(argv[6]); [[fallthrough]];
    case 6: io_type = get_io_sink_type(argv[5]); [[fallthrough]];
    case 5: thread_count = std::max(thread_count, std::stoi(std::string{argv[4]})); [[fallthrough]];
    case 4: num_reads = std::max(1, std::stoi(std::string{argv[3]})); [[fallthrough]];
    case 3:
      input_multiplier = std::max(input_multiplier, std::stoi(std::string{argv[2]}));
      [[fallthrough]];
    case 2:
      if (auto arg = std::string{argv[1]}; arg == "-h" or arg == "--help") {
        print_usage();
        return 0;
      } else
        input_paths = std::string{argv[1]};
      [[fallthrough]];
    case 1: break;
    default: print_usage(); throw std::runtime_error("");
  }

  // Lambda function to process and extract all input files
  auto const extract_input_files = [thread_count, input_multiplier](std::string const& paths) {
    std::vector<std::string> const delimited_paths = [&]() {
      std::vector<std::string> paths_list;
      std::stringstream stream{paths};
      std::string path;
      // Extract the delimited paths.
      while (std::getline(stream, path, char{','})) {
        paths_list.push_back(path);
      }
      return paths_list;
    }();

    // The final list of parquet files to be read.
    std::vector<std::string> parquet_files;
    parquet_files.reserve(
      std::max<size_t>(thread_count, input_multiplier * delimited_paths.size()));
    // Append the input files by input_multiplier times
    std::for_each(
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(input_multiplier),
      [&](auto i) {
        std::for_each(
          delimited_paths.cbegin(), delimited_paths.cend(), [&](auto const& path_string) {
            std::filesystem::path path{path_string};
            // If this is a parquet file, add it.
            if (std::filesystem::is_regular_file(path)) {
              parquet_files.emplace_back(path_string);
            }
            // If this is a directory, add all files at this path
            else if (std::filesystem::is_directory(path)) {
              for (auto const& file : std::filesystem::directory_iterator(path)) {
                if (std::filesystem::is_regular_file(file.path())) {
                  parquet_files.emplace_back(file.path().string());
                }
              }
            } else {
              throw std::runtime_error("Encountered an invalid input path\n");
            }
          });
      });

    // Cycle append parquet files from the existing ones if less than the thread_count
    for (size_t idx = 0, initial_size = parquet_files.size();
         thread_count > static_cast<int>(parquet_files.size());
         idx++) {
      parquet_files.emplace_back(parquet_files[idx % initial_size]);
    }

    return parquet_files;
  };

  // Lambda function to setup and launch multithreaded parquet writes
  auto const write_parquet_multithreaded = [&](std::vector<cudf::table_view> const& tables,
                                               int32_t thread_count,
                                               rmm::cuda_stream_pool& stream_pool) {
    // Table writing tasks
    std::vector<write_fn> write_tasks;
    write_tasks.reserve(thread_count);
    std::for_each(
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(thread_count),
      [&](auto tid) {
        write_tasks.emplace_back(write_fn{io_type.value(), tables, tid, stream_pool.get_stream()});
      });

    // Writer threads
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (auto& c : write_tasks) {
      threads.emplace_back(std::thread{c});
    }
    for (auto& t : threads) {
      t.join();
    }
  };

  // Make a list of input files from the input_paths string.
  auto const input_files  = extract_input_files(input_paths);
  auto const is_pool_used = true;
  auto resource           = create_memory_resource(is_pool_used);
  auto default_stream     = cudf::get_default_stream();
  auto stream_pool        = rmm::cuda_stream_pool(thread_count);
  auto stats_mr =
    rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>(resource.get());
  rmm::mr::set_current_device_resource(&stats_mr);

  // Exit early if nothing to do.
  if (not input_files.size()) {
    std::cerr << "No input files to read. Exiting early.\n";
    return 0;
  }

  // Read the same parquet files specified times with multiple threads and discard the read tables
  {
    fmt::print(
      "\nReading {} input files {} times using {} threads and discarding output "
      "tables..\n",
      input_files.size(),
      num_reads,
      thread_count);
    fmt::print(
      fg(fmt::color::yellow),
      "Note that the first read may include times for nvcomp, cufile loading and RMM growth.\n\n");
    cudf::examples::timer timer;
    std::for_each(thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(num_reads),
                  [&](auto i) {  // Read parquet files and discard the tables
                    std::ignore = read_parquet_multithreaded<read_mode::NOWORK>(
                      input_files, thread_count, stream_pool);
                  });
    default_stream.synchronize();
    timer.print_elapsed_millis();
  }

  // Do we need to write parquet as well?
  if (io_type.has_value()) {
    // Read input files with CONCATENATE_THREADS mode
    auto const tables = read_parquet_multithreaded<read_mode::CONCATENATE_THREAD>(
      input_files, thread_count, stream_pool);
    default_stream.synchronize();
    // Initialize the default output path to avoid race condition with multiple writer threads.
    std::ignore = get_default_output_path();

    // Construct a vector of table views for write_parquet_multithreaded
    auto const table_views = [&tables]() {
      std::vector<cudf::table_view> table_views;
      table_views.reserve(tables.size());

      std::transform(
        tables.cbegin(), tables.cend(), std::back_inserter(table_views), [](auto const& tbl) {
          return tbl->view();
        });
      return table_views;
    }();

    // Write tables to parquet
    fmt::print("Writing parquet output to sink type: {}..\n", std::string{argv[5]});
    cudf::examples::timer timer;
    write_parquet_multithreaded(table_views, thread_count, stream_pool);
    default_stream.synchronize();
    timer.print_elapsed_millis();

    // Verify the output if requested
    if (validate_output) {
      fmt::print("Verifying output..\n");

      // CONCATENATE_ALL returns a vector of 1 table
      auto const input_table = cudf::concatenate(table_views, default_stream);

      auto const transcoded_table =
        std::move(read_parquet_multithreaded<read_mode::CONCATENATE_ALL>(
                    extract_input_files(get_default_output_path()), thread_count, stream_pool)
                    .back());
      default_stream.synchronize();

      // Check if the tables are identical
      check_identical_tables(input_table->view(), transcoded_table->view());
    }
  }

  // Print peak memory
  fmt::print(fmt::emphasis::bold | fg(fmt::color::medium_purple),
             "Peak memory: {} MB\n\n",
             (stats_mr.get_bytes_counter().peak / 1048576.0));

  return 0;
}
