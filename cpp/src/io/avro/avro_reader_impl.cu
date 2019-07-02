/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "avro.h"
#include "avro_gpu.h"
#include "avro_reader_impl.hpp"

#include <cudf/cudf.h>
#include <nvstrings/NVStrings.h>

#include <cuda_runtime.h>

namespace cudf {
namespace io {
namespace avro {

#if 1
#define LOG_PRINTF(...) std::printf(__VA_ARGS__)
#else
#define LOG_PRINTF(...) (void)0
#endif

/**
 * @brief Function that translates Avro datatype to GDF dtype
 **/
constexpr gdf_dtype to_dtype(const avro::schema_entry *col) {
  switch (col->kind) {
    case avro::type_boolean:
      return GDF_BOOL8;
    case avro::type_int:
      return GDF_INT32;
    case avro::type_long:
      return GDF_INT64;
    case avro::type_float:
      return GDF_FLOAT32;
    case avro::type_double:
      return GDF_FLOAT64;
    case avro::type_bytes:
    case avro::type_string:
      return GDF_STRING;
    case avro::type_enum:
      return (not col->symbols.empty()) ? GDF_STRING : GDF_INT32;
    default:
      return GDF_invalid;
  }
}

/**
 * @brief A helper wrapper for Avro file metadata. Provides some additional
 * convenience methods for initializing and accessing the metadata and schema
 **/
class avro_metadata : public avro::file_metadata {
 public:
  explicit avro_metadata(DataSource *const src) : source(src) {}

  /**
   * @brief Initializes the parser and filters down to a subset of rows
   *
   * @param[in,out] row_start Starting row of the selection
   * @param[in,out] row_count Total number of rows selected
   **/
  void init_and_select_rows(int &row_start, int &row_count) {
    const auto buffer = source->get_buffer(0, source->size());
    avro::container pod(buffer->data(), buffer->size());
    CUDF_EXPECTS(pod.parse(this, row_count, row_start), "Cannot parse metadata");
    print_metadata();
    row_start = skip_rows;
    row_count = num_rows;
  }

  /**
   * @brief Filters and reduces down to a selection of columns
   *
   * @param[in] use_names List of column names to select
   *
   * @return List of column names
   **/
  auto select_columns(std::vector<std::string> use_names) {
    std::vector<std::pair<int, std::string>> selection;

    if (not use_names.empty()) {
      const int num_avro_columns = static_cast<int>(columns.size());
      for (int i = 0, column_id = 0; i < (int)use_names.size(); i++) {
        for (int j = 0; j < num_avro_columns; j++, column_id++) {
          if (column_id >= num_avro_columns) {
            column_id = 0;
          }
          if (columns[column_id].name == use_names[i] &&
              GDF_invalid != to_dtype(&schema[columns[i].schema_data_idx])) {
            selection.emplace_back(column_id, columns[column_id].name);
            column_id++;
            break;
          }
        }
      }
    } else {
      // Iterate backwards as fastavro returns from last-to-first?!
      for (auto it = columns.rbegin(); it != columns.rend(); ++it) {
        const auto dtype = to_dtype(&schema[it->schema_data_idx]);
        CUDF_EXPECTS(dtype != GDF_invalid, "Unsupported data type");
        selection.emplace_back(selection.size(), it->name);
      }
    }
    CUDF_EXPECTS(selection.size() > 0, "Filtered out all columns");

    return selection;
  }

  void print_metadata() const {
    LOG_PRINTF("\n[+] Metadata:\n");
    LOG_PRINTF(" size = %zd\n", metadata_size);
    LOG_PRINTF(" codec = \"%s\"\n", codec.c_str());
    LOG_PRINTF(" sync marker = 0x%016llX%016llX\n",
               (long long unsigned int)sync_marker[1],
               (long long unsigned int)sync_marker[0]);
    LOG_PRINTF(" schema (%zd entries):\n", schema.size());
    for (size_t i = 0; i < schema.size(); ++i) {
      LOG_PRINTF(
          "  [%zd] num_children=%d, parent_idx=%d, type=%d, name=\"%s\"\n", i,
          schema[i].num_children, schema[i].parent_idx, schema[i].kind,
          schema[i].name.c_str());
    }
    LOG_PRINTF(" datablocks (%zd entries):\n", block_list.size());
    LOG_PRINTF(" num rows = %zd (max block size = %d, total_data_size = %zd)\n",
               num_rows, max_block_size, total_data_size);
    LOG_PRINTF(" num columns = %zd\n", columns.size());
    LOG_PRINTF(" user data entries = %zd\n", user_data.size());
    for (const auto &entry : user_data) {
      LOG_PRINTF("  key: %s, value: %s\n", entry.first.c_str(),
                 entry.second.c_str());
    }
  }

 private:
  DataSource *const source;
};

reader::Impl::Impl(std::unique_ptr<DataSource> source,
                   reader_options const &options)
    : source_(std::move(source)), columns_(options.columns) {

  // Open and parse the source Avro dataset metadata
  md_ = std::make_unique<avro_metadata>(source_.get());
}

table reader::Impl::read(int skip_rows, int num_rows) {

  // Select and read partial metadata / schema within the subset of rows
  md_->init_and_select_rows(skip_rows, num_rows);

  // Select only columns required by the options
  selected_cols_ = md_->select_columns(columns_);
  if (selected_cols_.empty()) {
    return table();
  }

  // Initialize gdf_columns, but hold off on allocating storage space
  LOG_PRINTF("[+] Selected columns: %zd\n", selected_cols_.size());
  LOG_PRINTF("[+] Selected skip_rows: %d, num_rows: %d\n", skip_rows, num_rows);
  std::vector<gdf_column_wrapper> columns;
  for (const auto &col : selected_cols_) {
    auto &col_schema = md_->schema[md_->columns[col.first].schema_data_idx];

    columns.emplace_back(static_cast<gdf_size_type>(num_rows),
                         to_dtype(&col_schema),
                         gdf_dtype_extra_info{TIME_UNIT_NONE}, col.second);

    LOG_PRINTF(" %2zd: name=%s size=%zd type=%d data=%lx valid=%lx\n",
               columns.size() - 1, columns.back()->col_name,
               (size_t)columns.back()->size, columns.back()->dtype,
               (uint64_t)columns.back()->data, (uint64_t)columns.back()->valid);
  }

  // TODO: decompress block data
  for (auto &column : columns) {
    CUDF_EXPECTS(column.allocate() == GDF_SUCCESS, "Cannot allocate columns");
    if (column->dtype == GDF_STRING) {
      // Kernel doesn't init invalid entries but NvStrings expects zero length
      CUDA_TRY(cudaMemsetAsync(
          column->data, 0,
          column->size * sizeof(std::pair<const char *, size_t>)));
    }
  }
  // TODO: decode block data

  // For string dtype, allocate an NvStrings container instance, deallocating
  // the original string list memory in the process.
  // This container takes a list of string pointers and lengths, and copies
  // into its own memory so the source memory must not be released yet.
  for (auto &column : columns) {
    if (column->dtype == GDF_STRING) {
      using str_pair = std::pair<const char *, size_t>;
      using str_ptr = std::unique_ptr<NVStrings, decltype(&NVStrings::destroy)>;

      auto str_list = static_cast<str_pair *>(column->data);
      str_ptr str_data(NVStrings::create_from_index(str_list, num_rows),
                       &NVStrings::destroy);
      RMM_FREE(std::exchange(column->data, str_data.release()), 0);
    }
  }

  // Transfer ownership to raw pointer output arguments
  std::vector<gdf_column *> out_cols(columns.size());
  for (size_t i = 0; i < columns.size(); ++i) {
    out_cols[i] = columns[i].release();
  }

  return table(out_cols.data(), out_cols.size());
}

reader::reader(std::string filepath, reader_options const &options)
    : impl_(std::make_unique<Impl>(
          std::make_unique<DataSource>(filepath.c_str()), options)) {}

reader::reader(const char *buffer, size_t length, reader_options const &options)
    : impl_(std::make_unique<Impl>(std::make_unique<DataSource>(buffer, length),
                                   options)) {}

reader::reader(std::shared_ptr<arrow::io::RandomAccessFile> file,
               reader_options const &options)
    : impl_(std::make_unique<Impl>(std::make_unique<DataSource>(file),
                                   options)) {}

table reader::read_all() { return impl_->read(0, -1); }

table reader::read_rows(size_t skip_rows, size_t num_rows) {
  return impl_->read(skip_rows, (num_rows != 0) ? (int)num_rows : -1);
}

reader::~reader() = default;

}  // namespace avro
}  // namespace io
}  // namespace cudf
