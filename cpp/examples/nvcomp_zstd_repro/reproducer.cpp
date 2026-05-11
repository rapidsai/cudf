/*
 * nvCOMP/cuDF ZSTD Decompression Reproducer - Velox Pattern
 *
 * This reproducer mimics how Velox reads multiple tables concurrently
 * within a single query, with each table scan having its own stream
 * and potentially AST filter expressions.
 *
 * Key differences from simple reproducer:
 * - Multiple concurrent chunked_parquet_reader instances (simulating multiple table scans)
 * - Each reader has its own stream from a shared pool
 * - Readers are created/destroyed dynamically (like Velox does per split)
 * - Optional AST filter expressions
 * - Shared memory resource across all readers
 */

#include <cudf/ast/expressions.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda/memory_resource>

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

// Global state
std::mutex cout_mutex;
std::atomic<int> failure_count{0};
std::atomic<int> success_count{0};
std::atomic<bool> stop_flag{false};
std::atomic<int64_t> total_bytes_read{0};
std::atomic<int64_t> total_rows_read{0};

struct FailureInfo {
  int table_id;
  int iteration;
  std::string file;
  std::string error;
};

std::vector<FailureInfo> failures;
std::mutex failures_mutex;

// Velox/Prestissimo chunk limits
std::size_t chunk_read_limit = 4294967296UL;   // 4GB
std::size_t pass_read_limit = 17179869184UL;   // 16GB

// Collect all parquet files recursively
std::vector<std::string> collect_parquet_files(const std::string& path) {
  std::set<std::string> file_set;

  if (fs::is_regular_file(path)) {
    if (path.ends_with(".parquet")) {
      file_set.insert(fs::canonical(path).string());
    }
  } else if (fs::is_directory(path)) {
    for (const auto& entry : fs::recursive_directory_iterator(path)) {
      if (entry.is_regular_file() &&
          entry.path().extension() == ".parquet") {
        file_set.insert(fs::canonical(entry.path()).string());
      }
    }
  }

  return std::vector<std::string>(file_set.begin(), file_set.end());
}

// Simulates a single table scan (like CudfHiveDataSource)
// Each "table" gets a subset of files and reads them with its own reader/stream
class TableScanSimulator {
public:
  TableScanSimulator(int table_id,
                     const std::vector<std::string>& files,
                     rmm::device_async_resource_ref mr)
      : table_id_(table_id), files_(files), mr_(mr) {}

  void run(int iterations) {
    auto table_start = std::chrono::steady_clock::now();

    for (int iter = 1; iter <= iterations && !stop_flag; iter++) {
      auto iter_start = std::chrono::steady_clock::now();
      int64_t iter_rows = 0;
      int64_t iter_bytes = 0;
      int files_this_iter = 0;

      // Get a fresh stream from the global pool (like Velox does per split)
      auto stream = cudf::detail::global_cuda_stream_pool().get_stream();

      for (const auto& filepath : files_) {
        if (stop_flag) break;

        auto file_start = std::chrono::steady_clock::now();

        try {
          // Get file size for throughput calculation
          int64_t file_size = fs::file_size(filepath);

          // Build parquet reader options
          auto source = cudf::io::source_info(filepath);
          auto builder = cudf::io::parquet_reader_options::builder(source);

          // Optionally add a simple filter (simulating AST pushdown)
          // We create a trivial "column 0 is not null" type filter
          // This exercises the filter pushdown code path

          auto options = builder.build();

          // Create a new chunked reader (like Velox does per split)
          cudf::io::chunked_parquet_reader reader(
              chunk_read_limit, pass_read_limit, options, stream, mr_);

          int64_t file_rows = 0;
          while (reader.has_next()) {
            auto chunk = reader.read_chunk();
            file_rows += chunk.tbl->num_rows();
          }

          auto file_end = std::chrono::steady_clock::now();
          auto file_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             file_end - file_start).count();

          iter_rows += file_rows;
          iter_bytes += file_size;
          files_this_iter++;

          total_rows_read += file_rows;
          total_bytes_read += file_size;
          success_count++;

        } catch (const std::exception& e) {
          auto file_end = std::chrono::steady_clock::now();
          auto file_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             file_end - file_start).count();

          failure_count++;
          stop_flag = true;

          std::lock_guard<std::mutex> lock(failures_mutex);
          failures.push_back({table_id_, iter, filepath, e.what()});

          std::lock_guard<std::mutex> cout_lock(cout_mutex);
          std::cerr << "\n============================================================\n";
          std::cerr << "FAILURE - Table " << table_id_ << ", iteration " << iter << "\n";
          std::cerr << "File: " << filepath << "\n";
          std::cerr << "Time until failure: " << file_ms << " ms\n";
          std::cerr << "Error: " << e.what() << "\n";
          std::cerr << "============================================================\n\n";
          return;
        }
      }

      auto iter_end = std::chrono::steady_clock::now();
      auto iter_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         iter_end - iter_start).count();
      double iter_throughput_mb = (iter_bytes / 1024.0 / 1024.0) / (iter_ms / 1000.0);

      // Report progress every 10 iterations
      if (iter % 10 == 0 || iter == 1 || iter == iterations) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                           iter_end - table_start).count();
        std::cout << "[T" << table_id_ << "] iter " << iter << "/" << iterations
                  << " | " << files_this_iter << " files"
                  << " | " << iter_rows << " rows"
                  << " | " << std::fixed << std::setprecision(1)
                  << (iter_bytes / 1024.0 / 1024.0) << " MB"
                  << " | " << iter_ms << " ms"
                  << " | " << std::setprecision(1) << iter_throughput_mb << " MB/s"
                  << " | elapsed " << elapsed << "s"
                  << "\n";
      }
    }
  }

private:
  int table_id_;
  std::vector<std::string> files_;
  rmm::device_async_resource_ref mr_;
};

// Simulates a single "query" with multiple concurrent table scans
void run_query_simulation(const std::vector<std::vector<std::string>>& table_files,
                          int iterations,
                          rmm::device_async_resource_ref mr) {
  std::vector<std::thread> threads;
  std::vector<std::unique_ptr<TableScanSimulator>> scanners;

  // Create a scanner for each "table"
  for (size_t i = 0; i < table_files.size(); i++) {
    scanners.push_back(std::make_unique<TableScanSimulator>(
        static_cast<int>(i), table_files[i], mr));
  }

  // Launch all table scans concurrently (like Velox does in a query)
  for (size_t i = 0; i < scanners.size(); i++) {
    threads.emplace_back([&scanners, i, iterations]() {
      scanners[i]->run(iterations);
    });
  }

  // Wait for all to complete
  for (auto& t : threads) {
    t.join();
  }
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <parquet_dir1> [parquet_dir2] ... [--iterations N] [--tables N]\n";
    std::cerr << "\n";
    std::cerr << "Simulates Velox's multi-table concurrent scan pattern.\n";
    std::cerr << "\n";
    std::cerr << "Options:\n";
    std::cerr << "  --iterations N    Number of iterations per thread (default: 100)\n";
    std::cerr << "  --threads N       Number of concurrent threads (default: 5)\n";
    std::cerr << "  --tables N        Alias for --threads\n";
    std::cerr << "  --chunk-limit N   Chunk read limit in GB (default: 4)\n";
    std::cerr << "  --pass-limit N    Pass read limit in GB (default: 16)\n";
    std::cerr << "\n";
    std::cerr << "To use a specific GPU, set CUDA_VISIBLE_DEVICES:\n";
    std::cerr << "  CUDA_VISIBLE_DEVICES=2 " << argv[0] << " /path/to/files --iterations 50 --threads 5\n";
    return 1;
  }

  // Parse arguments
  std::vector<std::string> parquet_paths;
  int iterations = 100;
  int num_threads = 5;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--iterations" && i + 1 < argc) {
      iterations = std::stoi(argv[++i]);
    } else if ((arg == "--threads" || arg == "--tables") && i + 1 < argc) {
      num_threads = std::stoi(argv[++i]);
    } else if (arg == "--chunk-limit" && i + 1 < argc) {
      chunk_read_limit = static_cast<std::size_t>(std::stod(argv[++i]) * 1024 * 1024 * 1024);
    } else if (arg == "--pass-limit" && i + 1 < argc) {
      pass_read_limit = static_cast<std::size_t>(std::stod(argv[++i]) * 1024 * 1024 * 1024);
    } else if (!arg.starts_with("--")) {
      parquet_paths.push_back(arg);
    }
  }

  if (parquet_paths.empty()) {
    std::cerr << "ERROR: No parquet paths specified\n";
    return 1;
  }

  // Get current GPU device (set via CUDA_VISIBLE_DEVICES)
  int current_device = 0;
  cudaGetDevice(&current_device);

  std::cout << "=== Velox Pattern Reproducer ===\n";
  std::cout << "GPU device: " << current_device << "\n";
  std::cout << "Chunk read limit: " << (chunk_read_limit / 1024.0 / 1024.0 / 1024.0) << " GB\n";
  std::cout << "Pass read limit: " << (pass_read_limit / 1024.0 / 1024.0 / 1024.0) << " GB\n";
  std::cout << "Iterations per thread: " << iterations << "\n";
  std::cout << "Concurrent threads: " << num_threads << "\n";

  // Setup async memory resource (like Velox does)
  std::cout << "\nSetting up CUDA async memory resource...\n";
  cuda::mr::any_resource<cuda::mr::device_accessible> async_mr =
    rmm::mr::cuda_async_memory_resource{};
  cudf::set_current_device_resource(async_mr);

  // Collect all parquet files from all paths
  std::vector<std::string> all_files;
  for (const auto& path : parquet_paths) {
    std::cout << "Collecting parquet files from: " << path << "\n";
    auto files = collect_parquet_files(path);
    std::cout << "  Found " << files.size() << " files\n";
    all_files.insert(all_files.end(), files.begin(), files.end());
  }

  if (all_files.empty()) {
    std::cerr << "ERROR: No parquet files found\n";
    return 1;
  }

  std::cout << "\nTotal files: " << all_files.size() << "\n";

  // Shuffle files for randomness
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(all_files.begin(), all_files.end(), gen);

  // Distribute files across threads
  std::vector<std::vector<std::string>> thread_files(num_threads);
  for (size_t i = 0; i < all_files.size(); i++) {
    thread_files[i % num_threads].push_back(all_files[i]);
  }

  std::cout << "\nFiles per thread:\n";
  for (int i = 0; i < num_threads; i++) {
    std::cout << "  Thread " << i << ": " << thread_files[i].size() << " files\n";
  }

  std::cout << "\n=== Starting concurrent reads ===\n\n";
  auto start_time = std::chrono::steady_clock::now();

  // Run the simulation
  run_query_simulation(thread_files, iterations, async_mr);

  auto end_time = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                     end_time - start_time)
                     .count();

  // Summary
  double elapsed_sec = elapsed / 1000.0;
  double total_mb = total_bytes_read.load() / 1024.0 / 1024.0;
  double avg_throughput = total_mb / elapsed_sec;

  std::cout << "\n";
  std::cout << "============================================================\n";
  std::cout << "SUMMARY\n";
  std::cout << "============================================================\n";
  std::cout << "Concurrent threads: " << num_threads << "\n";
  std::cout << "Iterations per thread: " << iterations << "\n";
  std::cout << "Total files: " << all_files.size() << "\n";
  std::cout << "Successful reads: " << success_count.load() << "\n";
  std::cout << "Failed reads: " << failure_count.load() << "\n";
  std::cout << "Total rows read: " << total_rows_read.load() << "\n";
  std::cout << "Total data read: " << std::fixed << std::setprecision(2)
            << total_mb << " MB (" << (total_mb / 1024.0) << " GB)\n";
  std::cout << "Total time: " << std::setprecision(2) << elapsed_sec << " seconds\n";
  std::cout << "Average throughput: " << std::setprecision(1) << avg_throughput << " MB/s\n";

  if (!failures.empty()) {
    std::cout << "\nFailures:\n";
    for (const auto& f : failures) {
      std::cout << "  - Table " << f.table_id << ", Iteration " << f.iteration
                << ": " << f.file << "\n";
      std::cout << "    Error: " << f.error << "\n";
    }
    return 1;
  } else {
    std::cout << "\nAll reads completed successfully\n";
    return 0;
  }
}
