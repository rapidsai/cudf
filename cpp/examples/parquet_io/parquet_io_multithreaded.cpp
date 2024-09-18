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

#include <rmm/mr/device/statistics_resource_adaptor.hpp>

#include <filesystem>

/**
 * @file parquet_io_multithreaded.cpp
 * @brief Demonstrates usage of the libcudf APIs to read and write
 * parquet file format with different encodings and compression types
 * using multiple threads.
 *
 * The following encoding and compression ztypes are demonstrated:
 * Encoding Types: DEFAULT, DICTIONARY, PLAIN, DELTA_BINARY_PACKED,
 *                 DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY
 *
 * Compression Types: NONE, AUTO, SNAPPY, LZ4, ZSTD
 *
 */

using table_t = std::unique_ptr<cudf::table>;

struct read_fn {
  std::vector<std::string> const& input_files;
  std::vector<table_t>& tables;
  int const thread_id;
  int const thread_count;
  rmm::cuda_stream_view stream;

  void operator()()
  {
    std::vector<table_t> tables_this_thread;
    for (auto curr_file_idx = thread_id; curr_file_idx < input_files.size();
         curr_file_idx += thread_count) {
      auto const source_info = cudf::io::source_info(input_files[curr_file_idx]);
      auto builder           = cudf::io::parquet_reader_options::builder(source_info);
      auto const options     = builder.build();
      tables_this_thread.push_back(cudf::io::read_parquet(options, stream).tbl);
    }

    // Concatenate all tables read by this thread.
    auto table = std::move(tables_this_thread[0]);
    std::for_each(tables_this_thread.begin() + 1, tables_this_thread.end(), [&](auto& tbl) {
      std::vector<cudf::table_view> const table_views{table->view(), tbl->view()};
      table = cudf::concatenate(table_views, stream);
    });

    // Done with this stream
    stream.synchronize_no_throw();

    tables[thread_id] = std::move(table);
  }
};

struct write_fn {
  std::string const& output_path;
  std::vector<table_t> const& tables;
  cudf::io::column_encoding const encoding;
  cudf::io::compression_type const compression;
  std::optional<cudf::io::statistics_freq> const stats_level;
  int const thread_id;

  void operator()()
  {
    // write the data for inspection
    auto sink_info =
      cudf::io::sink_info(output_path + "/table_" + std::to_string(thread_id) + ".parquet");
    auto builder = cudf::io::parquet_writer_options::builder(sink_info, tables[thread_id]->view())
                     .compression(compression)
                     .stats_level(stats_level.value_or(cudf::io::statistics_freq::STATISTICS_NONE));
    auto table_metadata = cudf::io::table_input_metadata{tables[thread_id]->view()};

    std::for_each(table_metadata.column_metadata.begin(),
                  table_metadata.column_metadata.end(),
                  [=](auto& col_meta) { col_meta.set_encoding(encoding); });

    builder.metadata(table_metadata);
    auto options = builder.build();
    // Write parquet data
    cudf::io::write_parquet(options);
  }
};

int main(int argc, char const** argv)
{
  std::string input_paths;
  std::string output_path;
  cudf::io::column_encoding encoding;
  cudf::io::compression_type compression;
  std::optional<cudf::io::statistics_freq> page_stats;
  int thread_count;

  switch (argc) {
    case 1:
      input_paths  = "example.parquet";
      output_path  = std::filesystem::current_path().string();
      encoding     = get_encoding_type("DELTA_BINARY_PACKED");
      compression  = get_compression_type("ZSTD");
      thread_count = 2;
      break;
    case 7: page_stats = get_page_size_stats(argv[6]); [[fallthrough]];
    case 6:
      input_paths  = std::string{argv[1]};
      output_path  = std::string{argv[2]};
      encoding     = get_encoding_type(argv[3]);
      compression  = get_compression_type(argv[4]);
      thread_count = std::stoi(std::string(argv[5]));
      break;
    default:
      throw std::runtime_error(
        "Either provide all command-line arguments, or none to use defaults\n"
        "Use: parquet_io_multithreaded <comma delimited directories or parquet files>"
        "<output path> <encoding type> <compression type> <thread count> "
        "<write_page_stats? yes/no>\n");
  }

  // Process and extract all input files
  auto const input_files = [&]() {
    std::vector<std::string> parquet_files;
    std::vector<std::string> delimited_paths = [&]() {
      std::vector<std::string> paths_list;
      std::stringstream stream{input_paths};
      std::string path;
      // extract the delimited paths.
      while (std::getline(stream, path, char{','})) {
        paths_list.push_back(path);  // Add each token to the vector
      }
      return paths_list;
    }();

    std::for_each(delimited_paths.cbegin(), delimited_paths.cend(), [&](auto const& path_string) {
      std::filesystem::path path{path_string};
      // If this is a parquet file, add it.
      if (std::filesystem::is_regular_file(path)) {
        parquet_files.push_back(path_string);
      }
      // If this is a directory, add all files at this path
      else if (std::filesystem::is_directory(path)) {
        for (auto const& file : std::filesystem::directory_iterator(path)) {
          if (std::filesystem::is_regular_file(file.path())) {
            parquet_files.push_back(file.path().string());
          }
        }
      } else {
        throw std::runtime_error("Encountered an invalid input path\n");
      }
    });

    // Add parquet files from existing ones if less than thread_count
    for (size_t idx = 0, initial_size = parquet_files.size();
         thread_count > static_cast<int>(parquet_files.size());
         idx++) {
      parquet_files.push_back(parquet_files[idx % initial_size]);
    }

    return parquet_files;
  }();

  // Exit early if nothing to do.
  if (not input_files.size()) { return 0; }

  // Check if output path is a directory.
  if (not std::filesystem::is_directory(std::filesystem::path{output_path})) {
    throw std::runtime_error("The provided output path is not a directory\n");
  }

  auto const is_pool_used = true;
  auto resource           = create_memory_resource(is_pool_used);
  auto default_stream     = cudf::get_default_stream();
  auto stream_pool        = rmm::cuda_stream_pool(thread_count);
  auto stats_mr =
    rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>(resource.get());
  rmm::mr::set_current_device_resource(&stats_mr);

  // Lambda function to setup and launch multithread parquet read
  auto const read_parquet_multithreaded = [&]() {
    // Tables read by each thread
    std::vector<table_t> tables(thread_count);

    // Tasks to read each parquet file
    std::vector<read_fn> read_tasks;
    read_tasks.reserve(thread_count);
    std::for_each(thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(thread_count),
                  [&](auto tid) {
                    read_tasks.emplace_back(
                      read_fn{input_files, tables, tid, thread_count, stream_pool.get_stream()});
                  });

    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (auto& c : read_tasks) {
      threads.emplace_back(std::thread{c});
    }
    for (auto& t : threads) {
      t.join();
    }
    return tables;
  };

  // Lambda function to setup and launch multithread parquet write
  auto const write_parquet_multithreaded = [&](std::vector<table_t> const& tables) {
    // Tasks to read each parquet file
    std::vector<write_fn> write_tasks;
    write_tasks.reserve(thread_count);
    std::for_each(thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(thread_count),
                  [&](auto tid) {
                    write_tasks.emplace_back(
                      write_fn{output_path, tables, encoding, compression, page_stats, tid});
                  });

    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (auto& c : write_tasks) {
      threads.emplace_back(std::thread{c});
    }
    for (auto& t : threads) {
      t.join();
    }
  };

  // Read the parquet files with multiple threads
  {
    std::cout << "Note: Not timing the initial parquet read as it may include\n"
                 "times for nvcomp, cufile loading and RMM growth."
              << std::endl
              << std::endl;

    // tables read by each thread
    auto const tables = read_parquet_multithreaded();

    // In case some kernels are still running on the default stre
    default_stream.synchronize();

    // Write parquet file with the specified encoding and compression
    auto const page_stat_string = (page_stats.has_value()) ? "page stats" : "no page stats";
    std::cout << "Writing at: " << output_path << " with encoding, compression and "
              << page_stat_string << ".." << std::endl;

    // Write tables using multiple threads
    cudf::examples::timer timer;
    write_parquet_multithreaded(tables);

    // In case some kernels are still running on the default stream
    default_stream.synchronize();

    // Print elapsed time
    timer.print_elapsed_millis();
  }

  // Re-read the parquet files with multiple threads
  {
    std::cout << "Reading for the second time using " << thread_count << " threads..." << std::endl;
    cudf::examples::timer timer;
    auto tables = read_parquet_multithreaded();

    // Construct the final table
    auto table = std::move(tables[0]);
    std::for_each(tables.begin() + 1, tables.end(), [&](auto& tbl) {
      std::vector<cudf::table_view> const table_views{table->view(), tbl->view()};
      table = cudf::concatenate(table_views, default_stream);
    });

    // In case some kernels are still running on the default stream
    default_stream.synchronize();

    // Print elapsed time and peak memory
    timer.print_elapsed_millis();
    std::cout << "Peak memory: " << (stats_mr.get_bytes_counter().peak / 1048576.0) << " MB\n";
  }

  return 0;
}
