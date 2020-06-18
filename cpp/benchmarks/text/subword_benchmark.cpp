#include <benchmark/benchmark.h>
#include <cudf/strings/strings_column_view.hpp>
#include <nvtext/subword_tokenize.hpp>

#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <fstream>
#include <iostream>
#include <vector>

#define MAX_NUM_SENTENCES 101
#define MAX_NUM_CHARS 150000
#define MAX_ROWS_TENSOR 300

static std::string create_hash_vocab_file()
{
  std::string dir_template("/tmp");
  if (const char* env_p = std::getenv("WORKSPACE")) dir_template = env_p;
  std::string hash_file = dir_template + "/fake_hash_vocab.txt";
  std::vector<std::pair<int, int>> coefficients(23, {65559, 0});
  std::ofstream outfile(hash_file, std::ofstream::out);
  outfile << "1\n0\n" << coefficients.size() << "\n";
  for (auto c : coefficients) outfile << c.first << " " << c.second << "\n";
  std::vector<uint64_t> hash_table(23, 0);
  outfile << hash_table.size() << "\n";
  hash_table[0]  = 3015668L;
  hash_table[1]  = 6205475701751155871L;
  hash_table[5]  = 6358029;
  hash_table[16] = 451412625363L;
  hash_table[20] = 6206321707968235495L;
  for (auto h : hash_table) outfile << h << "\n";
  outfile << "100\n101\n102\n\n";
  return hash_file;
}

static void BM_cuda_tokenizer_cudf(benchmark::State& state)
{
  cudf::test::strings_column_wrapper sentences{"This is a test."};
  std::string hash_file = create_hash_vocab_file();
  std::vector<uint32_t> offsets{14};
  uint32_t max_sequence_length = 64;
  uint32_t stride              = 48;
  uint32_t do_truncate         = 0;
  uint32_t do_lower            = 1;
  for (auto _ : state) {
    auto result = nvtext::subword_tokenize(cudf::strings_column_view{sentences},
                                           hash_file,
                                           max_sequence_length,
                                           stride,
                                           do_lower,
                                           do_truncate,
                                           MAX_NUM_SENTENCES,
                                           MAX_NUM_CHARS,
                                           MAX_ROWS_TENSOR);
    // not sure how these are freed by the caller
    cudaFree(result->device_attention_mask);
    cudaFree(result->device_tensor_metadata);
    cudaFree(result->device_tensor_tokenIDS);
  }
}
BENCHMARK(BM_cuda_tokenizer_cudf);

BENCHMARK_MAIN();