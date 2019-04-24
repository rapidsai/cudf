# pragma once

#include <vector>

#include <cudf.h>

#include "../csv/type_conversion.cuh"
#include "io/utilities/wrapper_utils.hpp"

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

  // TODO munmap in destructor
	void * 	map_data = nullptr;
	size_t	map_size = 0;

  const char *h_uncomp_data_ = nullptr;
  size_t h_uncomp_size_ = 0;
	// Used when the input data is compressed, to ensure the allocated uncompressed data is freed
  std::vector<char> h_uncomp_data_owner;
  device_buffer<char> d_uncomp_data_;

  std::vector<std::string> column_names_;
  std::vector<gdf_dtype> dtypes_;
  std::vector<gdf_column_wrapper> columns_;

  // tweaks/corner cases
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