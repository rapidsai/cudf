
#include <cudf_test/base_fixture.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

// Chunked reader params.
std::size_t constexpr READER_INPUT_LIMIT  = 128 * 1024 * 1024;
std::size_t constexpr READER_OUTPUT_LIMIT = 128 * 1024 * 1024;

std::vector<std::string> list_parquet_files(std::string const& folder_path)
{
  std::vector<std::string> parquet_files;
  namespace fs = std::filesystem;

  for (const auto& entry : fs::directory_iterator(folder_path)) {
    if (entry.is_regular_file() && entry.path().extension() == ".parquet") {
      parquet_files.push_back(entry.path().string());
    }
  }

  return parquet_files;
}

std::string get_data_path()
{
  char const* env_path = std::getenv("PARQUET_PATH");

  // Local debug.
  auto const path = env_path ? env_path : "/home/nghiat/Devel/data/";

  return std::string{path};
}

struct ParquetTest : public cudf::test::BaseFixture {};

TEST_F(ParquetTest, ListAllFile)
{
  auto const path  = get_data_path();
  auto const files = list_parquet_files(path);

  std::cout << "List all files in '" << path << "':\n";
  for (const auto& file : files) {
    std::cout << file << '\n';
  }
  std::cout << std::endl;
}

void chunked_read(std::string const& filepath)
{
  auto const stream    = cudf::get_default_stream();
  auto const read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .convert_strings_to_categories(false)
                           .timestamp_type(cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS})
                           .build();
  auto reader =
    cudf::io::chunked_parquet_reader(READER_OUTPUT_LIMIT, READER_INPUT_LIMIT, read_opts, stream);

  std::size_t num_chunks = 0;
  std::size_t num_rows   = 0;

  do {
    [[maybe_unused]] auto const chunk = reader.read_chunk();
    ++num_chunks;
    num_rows += chunk.tbl->num_rows();
    stream.synchronize();

    std::cout << "  ..Read chunk #" << num_chunks << ", rows = " << chunk.tbl->num_rows() << "\n";
  } while (reader.has_next());
  std::cout << "  ..Read end, num total chunks: " << num_chunks << ", num total rows: " << num_rows
            << std::endl;
}

TEST_F(ParquetTest, ReadAllFiles)
{
  auto const path  = get_data_path();
  auto const files = list_parquet_files(path);

  std::cout << "Reading all fine in '" << path << "':\n";

  for (const auto& file : files) {
    std::cout << "  Reading file '" << file << "':\n";
    chunked_read(file);
  }
  std::cout << std::endl;
}
