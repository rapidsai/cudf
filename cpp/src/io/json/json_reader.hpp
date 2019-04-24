# pragma once

#include <vector>
#include <memory>

#include <cudf.h>

#include "../csv/type_conversion.cuh"
#include "io/utilities/wrapper_utils.hpp"


// TODO move to common?
class MappedFile{
  int fd_ = -1;
  size_t size_ = 0;
  void * map_data_ = nullptr;
  size_t map_size_ = 0;
  size_t map_offset_ = 0;
public:
  MappedFile(const char *path, int oflag);
  MappedFile() noexcept = default;
  ~MappedFile();

  auto size() {return size_;}
  auto data() {return map_data_;}

  void map(size_t size, off_t offset);
};

// DOXY
class JsonReader {
public:
  struct ColumnInfo {
    gdf_size_type float_count;
    gdf_size_type datetime_count;
    gdf_size_type string_count;
    gdf_size_type int_count;
    gdf_size_type null_count;
  };

private:
  const json_read_arg* args_ = nullptr;

  std::unique_ptr<MappedFile> map_file_;
  const char *input_data_ = nullptr;
  size_t input_size_ = 0;
  const char *uncomp_data_ = nullptr;
  size_t uncomp_size_ = 0;
  // Used when the input data is compressed, to ensure the allocated uncompressed data is freed
  std::vector<char> uncomp_data_owner_;
  device_buffer<char> d_data_;

  std::vector<std::string> column_names_;
  std::vector<gdf_dtype> dtypes_;
  std::vector<gdf_column_wrapper> columns_;

  // parsing options
  const bool allow_newlines_in_strings_ = false;
  const ParseOptions opts_{',', '\n', '\"','.'};

  const size_t byte_range_offset_ = 0;
  const size_t byte_range_size_ = 0;

  device_buffer<uint64_t> rec_starts_;

  void ingestInput();
  device_buffer<uint64_t> enumerateNewlinesAndQuotes();
  device_buffer<uint64_t> filterNewlines(device_buffer<uint64_t> newlines_and_quotes);
  void uploadDataToDevice();
  void setDataTypes();
  void detectDataTypes(ColumnInfo *d_columnData);
  void convertDataToColumns();
  void convertJsonToColumns(gdf_dtype * const dtypes, void **gdf_columns,
                            gdf_valid_type **valid, gdf_size_type *num_valid);

public:
  JsonReader(json_read_arg* args): args_(args){}

  void parse();

  void storeColumns(json_read_arg *out_args);
};