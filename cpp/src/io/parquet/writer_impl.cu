/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

/**
 * @file writer_impl.cu
 * @brief cuDF-IO parquet writer class implementation
 */

#include "compact_protocol_reader.hpp"
#include "compact_protocol_writer.hpp"
#include "parquet_common.hpp"
#include "parquet_gpu.cuh"
#include "writer_impl.hpp"

#include <io/comp/nvcomp_adapter.hpp>
#include <io/statistics/column_statistics.cuh>
#include <io/utilities/column_utils.cuh>
#include <io/utilities/config_utils.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/utilities/linked_column.hpp>
#include <cudf/detail/utilities/pinned_host_vector.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/detail/dremel.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/fill.h>
#include <thrust/for_each.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <utility>

namespace cudf {
namespace io {
namespace detail {
namespace parquet {
using namespace cudf::io::parquet;
using namespace cudf::io;

struct aggregate_writer_metadata {
  aggregate_writer_metadata(host_span<partition_info const> partitions,
                            host_span<std::map<std::string, std::string> const> kv_md,
                            host_span<SchemaElement const> tbl_schema,
                            size_type num_columns,
                            statistics_freq stats_granularity)
    : version(1),
      schema(std::vector<SchemaElement>(tbl_schema.begin(), tbl_schema.end())),
      files(partitions.size())
  {
    for (size_t i = 0; i < partitions.size(); ++i) {
      this->files[i].num_rows = partitions[i].num_rows;
    }
    this->column_order_listsize =
      (stats_granularity != statistics_freq::STATISTICS_NONE) ? num_columns : 0;

    for (size_t p = 0; p < kv_md.size(); ++p) {
      std::transform(kv_md[p].begin(),
                     kv_md[p].end(),
                     std::back_inserter(this->files[p].key_value_metadata),
                     [](auto const& kv) {
                       return KeyValue{kv.first, kv.second};
                     });
    }
  }

  aggregate_writer_metadata(aggregate_writer_metadata const&) = default;

  void update_files(host_span<partition_info const> partitions)
  {
    CUDF_EXPECTS(partitions.size() == this->files.size(),
                 "New partitions must be same size as previously passed number of partitions");
    for (size_t i = 0; i < partitions.size(); ++i) {
      this->files[i].num_rows += partitions[i].num_rows;
    }
  }

  FileMetaData get_metadata(size_t part)
  {
    CUDF_EXPECTS(part < files.size(), "Invalid part index queried");
    FileMetaData meta{};
    meta.version               = this->version;
    meta.schema                = this->schema;
    meta.num_rows              = this->files[part].num_rows;
    meta.row_groups            = this->files[part].row_groups;
    meta.key_value_metadata    = this->files[part].key_value_metadata;
    meta.created_by            = this->created_by;
    meta.column_order_listsize = this->column_order_listsize;
    return meta;
  }

  void set_file_paths(host_span<std::string const> column_chunks_file_path)
  {
    for (size_t p = 0; p < this->files.size(); ++p) {
      auto& file            = this->files[p];
      auto const& file_path = column_chunks_file_path[p];
      for (auto& rowgroup : file.row_groups) {
        for (auto& col : rowgroup.columns) {
          col.file_path = file_path;
        }
      }
    }
  }

  FileMetaData get_merged_metadata()
  {
    FileMetaData merged_md;
    for (size_t p = 0; p < this->files.size(); ++p) {
      auto& file = this->files[p];
      if (p == 0) {
        merged_md = this->get_metadata(0);
      } else {
        merged_md.row_groups.insert(merged_md.row_groups.end(),
                                    std::make_move_iterator(file.row_groups.begin()),
                                    std::make_move_iterator(file.row_groups.end()));
        merged_md.num_rows += file.num_rows;
      }
    }
    return merged_md;
  }

  std::vector<size_t> num_row_groups_per_file()
  {
    std::vector<size_t> global_rowgroup_base;
    std::transform(this->files.begin(),
                   this->files.end(),
                   std::back_inserter(global_rowgroup_base),
                   [](auto const& part) { return part.row_groups.size(); });
    return global_rowgroup_base;
  }

  [[nodiscard]] bool schema_matches(std::vector<SchemaElement> const& schema) const
  {
    return this->schema == schema;
  }
  auto& file(size_t p) { return files[p]; }
  [[nodiscard]] size_t num_files() const { return files.size(); }

 private:
  int32_t version = 0;
  std::vector<SchemaElement> schema;
  struct per_file_metadata {
    int64_t num_rows = 0;
    std::vector<RowGroup> row_groups;
    std::vector<KeyValue> key_value_metadata;
    std::vector<OffsetIndex> offset_indexes;
    std::vector<std::vector<uint8_t>> column_indexes;
  };
  std::vector<per_file_metadata> files;
  std::string created_by         = "";
  uint32_t column_order_listsize = 0;
};

namespace {

/**
 * @brief Function that translates GDF compression to parquet compression.
 *
 * @param compression The compression type
 * @return The supported Parquet compression
 */
parquet::Compression to_parquet_compression(compression_type compression)
{
  switch (compression) {
    case compression_type::AUTO:
    case compression_type::SNAPPY: return parquet::Compression::SNAPPY;
    case compression_type::ZSTD: return parquet::Compression::ZSTD;
    case compression_type::NONE: return parquet::Compression::UNCOMPRESSED;
    default: CUDF_FAIL("Unsupported compression type");
  }
}

/**
 * @brief Compute size (in bytes) of the data stored in the given column.
 *
 * @param column The input column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return The data size of the input
 */
size_t column_size(column_view const& column, rmm::cuda_stream_view stream)
{
  if (column.size() == 0) { return 0; }

  if (is_fixed_width(column.type())) {
    return size_of(column.type()) * column.size();
  } else if (column.type().id() == type_id::STRING) {
    auto const scol = strings_column_view(column);
    return cudf::detail::get_value<size_type>(scol.offsets(), column.size(), stream) -
           cudf::detail::get_value<size_type>(scol.offsets(), 0, stream);
  } else if (column.type().id() == type_id::STRUCT) {
    auto const scol = structs_column_view(column);
    size_t ret      = 0;
    for (int i = 0; i < scol.num_children(); i++) {
      ret += column_size(scol.get_sliced_child(i), stream);
    }
    return ret;
  } else if (column.type().id() == type_id::LIST) {
    auto const lcol = lists_column_view(column);
    return column_size(lcol.get_sliced_child(stream), stream);
  }

  CUDF_FAIL("Unexpected compound type");
}

// checks to see if the given column has a fixed size.  This doesn't
// check every row, so assumes string and list columns are not fixed, even
// if each row is the same width.
// TODO: update this if FIXED_LEN_BYTE_ARRAY is ever supported for writes.
bool is_col_fixed_width(column_view const& column)
{
  if (column.type().id() == type_id::STRUCT) {
    return std::all_of(column.child_begin(), column.child_end(), is_col_fixed_width);
  }

  return is_fixed_width(column.type());
}

/**
 * @brief Extends SchemaElement to add members required in constructing parquet_column_view
 *
 * Added members are:
 * 1. leaf_column: Pointer to leaf linked_column_view which points to the corresponding data stream
 *    of a leaf schema node. For non-leaf struct node, this is nullptr.
 * 2. stats_dtype: datatype for statistics calculation required for the data stream of a leaf node.
 * 3. ts_scale: scale to multiply or divide timestamp by in order to convert timestamp to parquet
 *    supported types
 */
struct schema_tree_node : public SchemaElement {
  cudf::detail::LinkedColPtr leaf_column;
  statistics_dtype stats_dtype;
  int32_t ts_scale;

  // TODO(fut): Think about making schema a class that holds a vector of schema_tree_nodes. The
  // function construct_schema_tree could be its constructor. It can have method to get the per
  // column nullability given a schema node index corresponding to a leaf schema. Much easier than
  // that is a method to get path in schema, given a leaf node
};

struct leaf_schema_fn {
  schema_tree_node& col_schema;
  cudf::detail::LinkedColPtr const& col;
  column_in_metadata const& col_meta;
  bool timestamp_is_int96;

  template <typename T>
  std::enable_if_t<std::is_same_v<T, bool>, void> operator()()
  {
    col_schema.type        = Type::BOOLEAN;
    col_schema.stats_dtype = statistics_dtype::dtype_bool;
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, int8_t>, void> operator()()
  {
    col_schema.type           = Type::INT32;
    col_schema.converted_type = ConvertedType::INT_8;
    col_schema.stats_dtype    = statistics_dtype::dtype_int8;
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, int16_t>, void> operator()()
  {
    col_schema.type           = Type::INT32;
    col_schema.converted_type = ConvertedType::INT_16;
    col_schema.stats_dtype    = statistics_dtype::dtype_int16;
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, int32_t>, void> operator()()
  {
    col_schema.type        = Type::INT32;
    col_schema.stats_dtype = statistics_dtype::dtype_int32;
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, int64_t>, void> operator()()
  {
    col_schema.type        = Type::INT64;
    col_schema.stats_dtype = statistics_dtype::dtype_int64;
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, uint8_t>, void> operator()()
  {
    col_schema.type           = Type::INT32;
    col_schema.converted_type = ConvertedType::UINT_8;
    col_schema.stats_dtype    = statistics_dtype::dtype_int8;
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, uint16_t>, void> operator()()
  {
    col_schema.type           = Type::INT32;
    col_schema.converted_type = ConvertedType::UINT_16;
    col_schema.stats_dtype    = statistics_dtype::dtype_int16;
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, uint32_t>, void> operator()()
  {
    col_schema.type           = Type::INT32;
    col_schema.converted_type = ConvertedType::UINT_32;
    col_schema.stats_dtype    = statistics_dtype::dtype_int32;
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, uint64_t>, void> operator()()
  {
    col_schema.type           = Type::INT64;
    col_schema.converted_type = ConvertedType::UINT_64;
    col_schema.stats_dtype    = statistics_dtype::dtype_int64;
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, float>, void> operator()()
  {
    col_schema.type        = Type::FLOAT;
    col_schema.stats_dtype = statistics_dtype::dtype_float32;
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, double>, void> operator()()
  {
    col_schema.type        = Type::DOUBLE;
    col_schema.stats_dtype = statistics_dtype::dtype_float64;
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::string_view>, void> operator()()
  {
    col_schema.type = Type::BYTE_ARRAY;
    if (col_meta.is_enabled_output_as_binary()) {
      col_schema.converted_type = ConvertedType::UNKNOWN;
      col_schema.stats_dtype    = statistics_dtype::dtype_byte_array;
    } else {
      col_schema.converted_type = ConvertedType::UTF8;
      col_schema.stats_dtype    = statistics_dtype::dtype_string;
    }
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::timestamp_D>, void> operator()()
  {
    col_schema.type           = Type::INT32;
    col_schema.converted_type = ConvertedType::DATE;
    col_schema.stats_dtype    = statistics_dtype::dtype_int32;
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::timestamp_s>, void> operator()()
  {
    col_schema.type = (timestamp_is_int96) ? Type::INT96 : Type::INT64;
    col_schema.converted_type =
      (timestamp_is_int96) ? ConvertedType::UNKNOWN : ConvertedType::TIMESTAMP_MILLIS;
    col_schema.stats_dtype = statistics_dtype::dtype_timestamp64;
    col_schema.ts_scale    = 1000;
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::timestamp_ms>, void> operator()()
  {
    col_schema.type = (timestamp_is_int96) ? Type::INT96 : Type::INT64;
    col_schema.converted_type =
      (timestamp_is_int96) ? ConvertedType::UNKNOWN : ConvertedType::TIMESTAMP_MILLIS;
    col_schema.stats_dtype = statistics_dtype::dtype_timestamp64;
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::timestamp_us>, void> operator()()
  {
    col_schema.type = (timestamp_is_int96) ? Type::INT96 : Type::INT64;
    col_schema.converted_type =
      (timestamp_is_int96) ? ConvertedType::UNKNOWN : ConvertedType::TIMESTAMP_MICROS;
    col_schema.stats_dtype = statistics_dtype::dtype_timestamp64;
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::timestamp_ns>, void> operator()()
  {
    col_schema.type           = (timestamp_is_int96) ? Type::INT96 : Type::INT64;
    col_schema.converted_type = ConvertedType::UNKNOWN;
    col_schema.stats_dtype    = statistics_dtype::dtype_timestamp64;
    if (timestamp_is_int96) {
      col_schema.ts_scale = -1000;  // negative value indicates division by absolute value
    }
    // set logical type if it's not int96
    else {
      col_schema.logical_type.isset.TIMESTAMP            = true;
      col_schema.logical_type.TIMESTAMP.unit.isset.NANOS = true;
    }
  }

  //  unsupported outside cudf for parquet 1.0.
  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::duration_D>, void> operator()()
  {
    col_schema.type                                = Type::INT32;
    col_schema.converted_type                      = ConvertedType::TIME_MILLIS;
    col_schema.stats_dtype                         = statistics_dtype::dtype_int32;
    col_schema.ts_scale                            = 24 * 60 * 60 * 1000;
    col_schema.logical_type.isset.TIME             = true;
    col_schema.logical_type.TIME.unit.isset.MILLIS = true;
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::duration_s>, void> operator()()
  {
    col_schema.type                                = Type::INT32;
    col_schema.converted_type                      = ConvertedType::TIME_MILLIS;
    col_schema.stats_dtype                         = statistics_dtype::dtype_int32;
    col_schema.ts_scale                            = 1000;
    col_schema.logical_type.isset.TIME             = true;
    col_schema.logical_type.TIME.unit.isset.MILLIS = true;
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::duration_ms>, void> operator()()
  {
    col_schema.type                                = Type::INT32;
    col_schema.converted_type                      = ConvertedType::TIME_MILLIS;
    col_schema.stats_dtype                         = statistics_dtype::dtype_int32;
    col_schema.logical_type.isset.TIME             = true;
    col_schema.logical_type.TIME.unit.isset.MILLIS = true;
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::duration_us>, void> operator()()
  {
    col_schema.type                                = Type::INT64;
    col_schema.converted_type                      = ConvertedType::TIME_MICROS;
    col_schema.stats_dtype                         = statistics_dtype::dtype_int64;
    col_schema.logical_type.isset.TIME             = true;
    col_schema.logical_type.TIME.unit.isset.MICROS = true;
  }

  //  unsupported outside cudf for parquet 1.0.
  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::duration_ns>, void> operator()()
  {
    col_schema.type                               = Type::INT64;
    col_schema.stats_dtype                        = statistics_dtype::dtype_int64;
    col_schema.logical_type.isset.TIME            = true;
    col_schema.logical_type.TIME.unit.isset.NANOS = true;
  }

  template <typename T>
  std::enable_if_t<cudf::is_fixed_point<T>(), void> operator()()
  {
    if (std::is_same_v<T, numeric::decimal32>) {
      col_schema.type              = Type::INT32;
      col_schema.stats_dtype       = statistics_dtype::dtype_int32;
      col_schema.decimal_precision = MAX_DECIMAL32_PRECISION;
    } else if (std::is_same_v<T, numeric::decimal64>) {
      col_schema.type              = Type::INT64;
      col_schema.stats_dtype       = statistics_dtype::dtype_decimal64;
      col_schema.decimal_precision = MAX_DECIMAL64_PRECISION;
    } else if (std::is_same_v<T, numeric::decimal128>) {
      col_schema.type              = Type::FIXED_LEN_BYTE_ARRAY;
      col_schema.type_length       = sizeof(__int128_t);
      col_schema.stats_dtype       = statistics_dtype::dtype_decimal128;
      col_schema.decimal_precision = MAX_DECIMAL128_PRECISION;
    } else {
      CUDF_FAIL("Unsupported fixed point type for parquet writer");
    }
    col_schema.converted_type = ConvertedType::DECIMAL;
    col_schema.decimal_scale = -col->type().scale();  // parquet and cudf disagree about scale signs
    if (col_meta.is_decimal_precision_set()) {
      CUDF_EXPECTS(col_meta.get_decimal_precision() >= col_schema.decimal_scale,
                   "Precision must be equal to or greater than scale!");
      if (col_schema.type == Type::INT64 and col_meta.get_decimal_precision() < 10) {
        CUDF_LOG_WARN("Parquet writer: writing a decimal column with precision < 10 as int64");
      }
      col_schema.decimal_precision = col_meta.get_decimal_precision();
    }
  }

  template <typename T>
  std::enable_if_t<cudf::is_nested<T>(), void> operator()()
  {
    CUDF_FAIL("This functor is only meant for physical data types");
  }

  template <typename T>
  std::enable_if_t<cudf::is_dictionary<T>(), void> operator()()
  {
    CUDF_FAIL("Dictionary columns are not supported for writing");
  }
};

inline bool is_col_nullable(cudf::detail::LinkedColPtr const& col,
                            column_in_metadata const& col_meta,
                            single_write_mode write_mode)
{
  if (col_meta.is_nullability_defined()) {
    CUDF_EXPECTS(col_meta.nullable() || !col->nullable(),
                 "Mismatch in metadata prescribed nullability and input column nullability. "
                 "Metadata for nullable input column cannot prescribe nullability = false");
    return col_meta.nullable();
  }
  // For chunked write, when not provided nullability, we assume the worst case scenario
  // that all columns are nullable.
  return write_mode == single_write_mode::NO or col->nullable();
}

/**
 * @brief Construct schema from input columns and per-column input options
 *
 * Recursively traverses through linked_columns and corresponding metadata to construct schema tree.
 * The resulting schema tree is stored in a vector in pre-order traversal order.
 */
std::vector<schema_tree_node> construct_schema_tree(
  cudf::detail::LinkedColVector const& linked_columns,
  table_input_metadata& metadata,
  single_write_mode write_mode,
  bool int96_timestamps)
{
  std::vector<schema_tree_node> schema;
  schema_tree_node root{};
  root.type            = UNDEFINED_TYPE;
  root.repetition_type = NO_REPETITION_TYPE;
  root.name            = "schema";
  root.num_children    = linked_columns.size();
  root.parent_idx      = -1;  // root schema has no parent
  schema.push_back(std::move(root));

  std::function<void(cudf::detail::LinkedColPtr const&, column_in_metadata&, size_t)> add_schema =
    [&](cudf::detail::LinkedColPtr const& col, column_in_metadata& col_meta, size_t parent_idx) {
      bool col_nullable = is_col_nullable(col, col_meta, write_mode);

      auto set_field_id = [&schema, parent_idx](schema_tree_node& s,
                                                column_in_metadata const& col_meta) {
        if (schema[parent_idx].name != "list" and col_meta.is_parquet_field_id_set()) {
          s.field_id = col_meta.get_parquet_field_id();
        }
      };

      auto is_last_list_child = [](cudf::detail::LinkedColPtr col) {
        if (col->type().id() != type_id::LIST) { return false; }
        auto const child_col_type =
          col->children[lists_column_view::child_column_index]->type().id();
        return child_col_type == type_id::UINT8;
      };

      // There is a special case for a list<int8> column with one byte column child. This column can
      // have a special flag that indicates we write this out as binary instead of a list. This is a
      // more efficient storage mechanism for a single-depth list of bytes, but is a departure from
      // original cuIO behavior so it is locked behind the option. If the option is selected on a
      // column that isn't a single-depth list<int8> the code will throw.
      if (col_meta.is_enabled_output_as_binary() && is_last_list_child(col)) {
        CUDF_EXPECTS(col_meta.num_children() == 2 or col_meta.num_children() == 0,
                     "Binary column's corresponding metadata should have zero or two children!");
        if (col_meta.num_children() > 0) {
          CUDF_EXPECTS(col->children[lists_column_view::child_column_index]->children.size() == 0,
                       "Binary column must not be nested!");
        }

        schema_tree_node col_schema{};
        col_schema.type            = Type::BYTE_ARRAY;
        col_schema.converted_type  = ConvertedType::UNKNOWN;
        col_schema.stats_dtype     = statistics_dtype::dtype_byte_array;
        col_schema.repetition_type = col_nullable ? OPTIONAL : REQUIRED;
        col_schema.name = (schema[parent_idx].name == "list") ? "element" : col_meta.get_name();
        col_schema.parent_idx  = parent_idx;
        col_schema.leaf_column = col;
        set_field_id(col_schema, col_meta);
        col_schema.output_as_byte_array = col_meta.is_enabled_output_as_binary();
        schema.push_back(col_schema);
      } else if (col->type().id() == type_id::STRUCT) {
        // if struct, add current and recursively call for all children
        schema_tree_node struct_schema{};
        struct_schema.repetition_type =
          col_nullable ? FieldRepetitionType::OPTIONAL : FieldRepetitionType::REQUIRED;

        struct_schema.name = (schema[parent_idx].name == "list") ? "element" : col_meta.get_name();
        struct_schema.num_children = col->children.size();
        struct_schema.parent_idx   = parent_idx;
        set_field_id(struct_schema, col_meta);
        schema.push_back(std::move(struct_schema));

        auto struct_node_index = schema.size() - 1;
        // for (auto child_it = col->children.begin(); child_it < col->children.end(); child_it++) {
        //   add_schema(*child_it, struct_node_index);
        // }
        CUDF_EXPECTS(col->children.size() == static_cast<size_t>(col_meta.num_children()),
                     "Mismatch in number of child columns between input table and metadata");
        for (size_t i = 0; i < col->children.size(); ++i) {
          add_schema(col->children[i], col_meta.child(i), struct_node_index);
        }
      } else if (col->type().id() == type_id::LIST && !col_meta.is_map()) {
        // List schema is denoted by two levels for each nesting level and one final level for leaf.
        // The top level is the same name as the column name.
        // So e.g. List<List<int>> is denoted in the schema by
        // "col_name" : { "list" : { "element" : { "list" : { "element" } } } }

        schema_tree_node list_schema_1{};
        list_schema_1.converted_type = ConvertedType::LIST;
        list_schema_1.repetition_type =
          col_nullable ? FieldRepetitionType::OPTIONAL : FieldRepetitionType::REQUIRED;
        list_schema_1.name = (schema[parent_idx].name == "list") ? "element" : col_meta.get_name();
        list_schema_1.num_children = 1;
        list_schema_1.parent_idx   = parent_idx;
        set_field_id(list_schema_1, col_meta);
        schema.push_back(std::move(list_schema_1));

        schema_tree_node list_schema_2{};
        list_schema_2.repetition_type = FieldRepetitionType::REPEATED;
        list_schema_2.name            = "list";
        list_schema_2.num_children    = 1;
        list_schema_2.parent_idx      = schema.size() - 1;  // Parent is list_schema_1, last added.
        schema.push_back(std::move(list_schema_2));

        CUDF_EXPECTS(col_meta.num_children() == 2,
                     "List column's metadata should have exactly two children");

        add_schema(col->children[lists_column_view::child_column_index],
                   col_meta.child(lists_column_view::child_column_index),
                   schema.size() - 1);
      } else if (col->type().id() == type_id::LIST && col_meta.is_map()) {
        // Map schema is denoted by a list of struct
        // e.g. List<Struct<String,String>> will be
        // "col_name" : { "key_value" : { "key", "value" } }

        // verify the List child structure is a struct<left_child, right_child>
        column_view struct_col = *col->children[lists_column_view::child_column_index];
        CUDF_EXPECTS(struct_col.type().id() == type_id::STRUCT, "Map should be a List of struct");
        CUDF_EXPECTS(struct_col.num_children() == 2,
                     "Map should be a List of struct with two children only but found " +
                       std::to_string(struct_col.num_children()));

        schema_tree_node map_schema{};
        map_schema.converted_type = ConvertedType::MAP;
        map_schema.repetition_type =
          col_nullable ? FieldRepetitionType::OPTIONAL : FieldRepetitionType::REQUIRED;
        map_schema.name = col_meta.get_name();
        if (col_meta.is_parquet_field_id_set()) {
          map_schema.field_id = col_meta.get_parquet_field_id();
        }
        map_schema.num_children = 1;
        map_schema.parent_idx   = parent_idx;
        schema.push_back(std::move(map_schema));

        schema_tree_node repeat_group{};
        repeat_group.repetition_type = FieldRepetitionType::REPEATED;
        repeat_group.name            = "key_value";
        repeat_group.num_children    = 2;
        repeat_group.parent_idx      = schema.size() - 1;  // Parent is map_schema, last added.
        schema.push_back(std::move(repeat_group));

        CUDF_EXPECTS(col_meta.num_children() == 2,
                     "List column's metadata should have exactly two children");
        CUDF_EXPECTS(col_meta.child(lists_column_view::child_column_index).num_children() == 2,
                     "Map struct column should have exactly two children");
        // verify the col meta of children of the struct have name key and value
        auto& left_child_meta = col_meta.child(lists_column_view::child_column_index).child(0);
        left_child_meta.set_name("key");
        left_child_meta.set_nullability(false);

        auto& right_child_meta = col_meta.child(lists_column_view::child_column_index).child(1);
        right_child_meta.set_name("value");
        // check the repetition type of key is required i.e. the col should be non-nullable
        auto key_col = col->children[lists_column_view::child_column_index]->children[0];
        CUDF_EXPECTS(!is_col_nullable(key_col, left_child_meta, write_mode),
                     "key column cannot be nullable. For chunked writing, explicitly set the "
                     "nullability to false in metadata");
        // process key
        size_type struct_col_index = schema.size() - 1;
        add_schema(key_col, left_child_meta, struct_col_index);
        // process value
        add_schema(col->children[lists_column_view::child_column_index]->children[1],
                   right_child_meta,
                   struct_col_index);

      } else {
        // if leaf, add current
        if (col->type().id() == type_id::STRING) {
          CUDF_EXPECTS(col_meta.num_children() == 2 or col_meta.num_children() == 0,
                       "String column's corresponding metadata should have zero or two children");
        } else {
          CUDF_EXPECTS(col_meta.num_children() == 0,
                       "Leaf column's corresponding metadata cannot have children");
        }

        schema_tree_node col_schema{};

        bool timestamp_is_int96 = int96_timestamps or col_meta.is_enabled_int96_timestamps();

        cudf::type_dispatcher(col->type(),
                              leaf_schema_fn{col_schema, col, col_meta, timestamp_is_int96});

        col_schema.repetition_type = col_nullable ? OPTIONAL : REQUIRED;
        col_schema.name = (schema[parent_idx].name == "list") ? "element" : col_meta.get_name();
        col_schema.parent_idx  = parent_idx;
        col_schema.leaf_column = col;
        set_field_id(col_schema, col_meta);
        schema.push_back(col_schema);
      }
    };

  CUDF_EXPECTS(metadata.column_metadata.size() == linked_columns.size(),
               "Mismatch in the number of columns and the corresponding metadata elements");
  // Add all linked_columns to schema using parent_idx = 0 (root)
  for (size_t i = 0; i < linked_columns.size(); ++i) {
    add_schema(linked_columns[i], metadata.column_metadata[i], 0);
  }

  return schema;
}

/**
 * @brief Class to store parquet specific information for one data stream.
 *
 * Contains information about a single data stream. In case of struct columns, a data stream is one
 * of the child leaf columns that contains data.
 * e.g. A column Struct<int, List<float>> contains 2 data streams:
 * - Struct<int>
 * - Struct<List<float>>
 *
 */
struct parquet_column_view {
  parquet_column_view(schema_tree_node const& schema_node,
                      std::vector<schema_tree_node> const& schema_tree,
                      rmm::cuda_stream_view stream);

  [[nodiscard]] gpu::parquet_column_device_view get_device_view(rmm::cuda_stream_view stream) const;

  [[nodiscard]] column_view cudf_column_view() const { return cudf_col; }
  [[nodiscard]] parquet::Type physical_type() const { return schema_node.type; }
  [[nodiscard]] parquet::ConvertedType converted_type() const { return schema_node.converted_type; }

  std::vector<std::string> const& get_path_in_schema() { return path_in_schema; }

  // LIST related member functions
  [[nodiscard]] uint8_t max_def_level() const noexcept { return _max_def_level; }
  [[nodiscard]] uint8_t max_rep_level() const noexcept { return _max_rep_level; }
  [[nodiscard]] bool is_list() const noexcept { return _is_list; }

 private:
  // Schema related members
  schema_tree_node schema_node;
  std::vector<std::string> path_in_schema;
  uint8_t _max_def_level = 0;
  uint8_t _max_rep_level = 0;
  rmm::device_uvector<uint8_t> _d_nullability;

  column_view cudf_col;

  // List-related members
  bool _is_list;
  rmm::device_uvector<size_type>
    _dremel_offsets;  ///< For each row, the absolute offset into the repetition and definition
                      ///< level vectors. O(num rows)
  rmm::device_uvector<uint8_t> _rep_level;
  rmm::device_uvector<uint8_t> _def_level;
  std::vector<uint8_t> _nullability;
  size_type _data_count = 0;
};

parquet_column_view::parquet_column_view(schema_tree_node const& schema_node,
                                         std::vector<schema_tree_node> const& schema_tree,
                                         rmm::cuda_stream_view stream)
  : schema_node(schema_node),
    _d_nullability(0, stream),
    _dremel_offsets(0, stream),
    _rep_level(0, stream),
    _def_level(0, stream)
{
  // Construct single inheritance column_view from linked_column_view
  auto curr_col                           = schema_node.leaf_column.get();
  column_view single_inheritance_cudf_col = *curr_col;
  while (curr_col->parent) {
    auto const& parent = *curr_col->parent;

    // For list columns, we still need to retain the offset child column.
    auto children =
      (parent.type().id() == type_id::LIST)
        ? std::vector<column_view>{*parent.children[lists_column_view::offsets_column_index],
                                   single_inheritance_cudf_col}
        : std::vector<column_view>{single_inheritance_cudf_col};

    single_inheritance_cudf_col = column_view(parent.type(),
                                              parent.size(),
                                              parent.head(),
                                              parent.null_mask(),
                                              parent.null_count(),
                                              parent.offset(),
                                              children);

    curr_col = curr_col->parent;
  }
  cudf_col = single_inheritance_cudf_col;

  // Construct path_in_schema by travelling up in the schema_tree
  std::vector<std::string> path;
  auto curr_schema_node = schema_node;
  do {
    path.push_back(curr_schema_node.name);
    if (curr_schema_node.parent_idx != -1) {
      curr_schema_node = schema_tree[curr_schema_node.parent_idx];
    }
  } while (curr_schema_node.parent_idx != -1);
  path_in_schema = std::vector<std::string>(path.crbegin(), path.crend());

  // Calculate max definition level by counting the number of levels that are optional (nullable)
  // and max repetition level by counting the number of REPEATED levels in this column's hierarchy
  uint16_t max_def_level = 0;
  uint16_t max_rep_level = 0;
  curr_schema_node       = schema_node;
  while (curr_schema_node.parent_idx != -1) {
    if (curr_schema_node.repetition_type == parquet::REPEATED or
        curr_schema_node.repetition_type == parquet::OPTIONAL) {
      ++max_def_level;
    }
    if (curr_schema_node.repetition_type == parquet::REPEATED) { ++max_rep_level; }
    curr_schema_node = schema_tree[curr_schema_node.parent_idx];
  }
  CUDF_EXPECTS(max_def_level < 256, "Definition levels above 255 are not supported");
  CUDF_EXPECTS(max_rep_level < 256, "Definition levels above 255 are not supported");

  _max_def_level = max_def_level;
  _max_rep_level = max_rep_level;

  // Construct nullability vector using repetition_type from schema.
  std::vector<uint8_t> r_nullability;
  curr_schema_node = schema_node;
  while (curr_schema_node.parent_idx != -1) {
    if (not curr_schema_node.is_stub()) {
      r_nullability.push_back(curr_schema_node.repetition_type == FieldRepetitionType::OPTIONAL);
    }
    curr_schema_node = schema_tree[curr_schema_node.parent_idx];
  }
  _nullability = std::vector<uint8_t>(r_nullability.crbegin(), r_nullability.crend());
  // TODO(cp): Explore doing this for all columns in a single go outside this ctor. Maybe using
  // hostdevice_vector. Currently this involves a cudaMemcpyAsync for each column.
  _d_nullability = cudf::detail::make_device_uvector_async(
    _nullability, stream, rmm::mr::get_current_device_resource());

  _is_list = (_max_rep_level > 0);

  if (cudf_col.size() == 0) { return; }

  if (_is_list) {
    // Top level column's offsets are not applied to all children. Get the effective offset and
    // size of the leaf column
    // Calculate row offset into dremel data (repetition/definition values) and the respective
    // definition and repetition levels
    cudf::detail::dremel_data dremel =
      get_dremel_data(cudf_col, _nullability, schema_node.output_as_byte_array, stream);
    _dremel_offsets = std::move(dremel.dremel_offsets);
    _rep_level      = std::move(dremel.rep_level);
    _def_level      = std::move(dremel.def_level);
    _data_count     = dremel.leaf_data_size;  // Needed for knowing what size dictionary to allocate

    stream.synchronize();
  } else {
    // For non-list struct, the size of the root column is the same as the size of the leaf column
    _data_count = cudf_col.size();
  }
}

gpu::parquet_column_device_view parquet_column_view::get_device_view(rmm::cuda_stream_view) const
{
  auto desc        = gpu::parquet_column_device_view{};  // Zero out all fields
  desc.stats_dtype = schema_node.stats_dtype;
  desc.ts_scale    = schema_node.ts_scale;

  if (is_list()) {
    desc.level_offsets = _dremel_offsets.data();
    desc.rep_values    = _rep_level.data();
    desc.def_values    = _def_level.data();
  }
  desc.num_rows             = cudf_col.size();
  desc.physical_type        = physical_type();
  desc.converted_type       = converted_type();
  desc.output_as_byte_array = schema_node.output_as_byte_array;

  desc.level_bits = CompactProtocolReader::NumRequiredBits(max_rep_level()) << 4 |
                    CompactProtocolReader::NumRequiredBits(max_def_level());
  desc.nullability = _d_nullability.data();
  return desc;
}

/**
 * @brief Gather row group fragments
 *
 * This calculates fragments to be used in determining row group boundaries.
 *
 * @param frag Destination row group fragments
 * @param col_desc column description array
 * @param partitions Information about partitioning of table
 * @param part_frag_offset A Partition's offset into fragment array
 * @param fragment_size Number of rows per fragment
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void init_row_group_fragments(cudf::detail::hostdevice_2dvector<gpu::PageFragment>& frag,
                              device_span<gpu::parquet_column_device_view const> col_desc,
                              host_span<partition_info const> partitions,
                              device_span<int const> part_frag_offset,
                              uint32_t fragment_size,
                              rmm::cuda_stream_view stream)
{
  auto d_partitions = cudf::detail::make_device_uvector_async(
    partitions, stream, rmm::mr::get_current_device_resource());
  gpu::InitRowGroupFragments(frag, col_desc, d_partitions, part_frag_offset, fragment_size, stream);
  frag.device_to_host(stream, true);
}

/**
 * @brief Recalculate page fragments
 *
 * This calculates fragments to be used to determine page boundaries within
 * column chunks.
 *
 * @param frag Destination page fragments
 * @param frag_sizes Array of fragment sizes for each column
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void calculate_page_fragments(device_span<gpu::PageFragment> frag,
                              host_span<size_type const> frag_sizes,
                              rmm::cuda_stream_view stream)
{
  auto d_frag_sz = cudf::detail::make_device_uvector_async(
    frag_sizes, stream, rmm::mr::get_current_device_resource());
  gpu::CalculatePageFragments(frag, d_frag_sz, stream);
}

/**
 * @brief Gather per-fragment statistics
 *
 * @param frag_stats output statistics
 * @param frags Input page fragments
 * @param int96_timestamps Flag to indicate if timestamps will be written as INT96
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void gather_fragment_statistics(device_span<statistics_chunk> frag_stats,
                                device_span<gpu::PageFragment const> frags,
                                bool int96_timestamps,
                                rmm::cuda_stream_view stream)
{
  rmm::device_uvector<statistics_group> frag_stats_group(frag_stats.size(), stream);

  gpu::InitFragmentStatistics(frag_stats_group, frags, stream);
  detail::calculate_group_statistics<detail::io_file_format::PARQUET>(
    frag_stats.data(), frag_stats_group.data(), frag_stats.size(), stream, int96_timestamps);
  stream.synchronize();
}

auto to_nvcomp_compression_type(Compression codec)
{
  if (codec == Compression::SNAPPY) return nvcomp::compression_type::SNAPPY;
  if (codec == Compression::ZSTD) return nvcomp::compression_type::ZSTD;
  CUDF_FAIL("Unsupported compression type");
}

auto page_alignment(Compression codec)
{
  if (codec == Compression::UNCOMPRESSED or
      nvcomp::is_compression_disabled(to_nvcomp_compression_type(codec))) {
    return 1u;
  }

  return 1u << nvcomp::compress_input_alignment_bits(to_nvcomp_compression_type(codec));
}

size_t max_compression_output_size(Compression codec, uint32_t compression_blocksize)
{
  if (codec == Compression::UNCOMPRESSED) return 0;

  return compress_max_output_chunk_size(to_nvcomp_compression_type(codec), compression_blocksize);
}

auto init_page_sizes(hostdevice_2dvector<gpu::EncColumnChunk>& chunks,
                     device_span<gpu::parquet_column_device_view const> col_desc,
                     uint32_t num_columns,
                     size_t max_page_size_bytes,
                     size_type max_page_size_rows,
                     Compression compression_codec,
                     rmm::cuda_stream_view stream)
{
  if (chunks.is_empty()) { return hostdevice_vector<size_type>{}; }

  chunks.host_to_device(stream);
  // Calculate number of pages and store in respective chunks
  gpu::InitEncoderPages(chunks,
                        {},
                        {},
                        {},
                        col_desc,
                        num_columns,
                        max_page_size_bytes,
                        max_page_size_rows,
                        page_alignment(compression_codec),
                        nullptr,
                        nullptr,
                        stream);
  chunks.device_to_host(stream, true);

  int num_pages = 0;
  for (auto& chunk : chunks.host_view().flat_view()) {
    chunk.first_page = num_pages;
    num_pages += chunk.num_pages;
  }
  chunks.host_to_device(stream);

  // Now that we know the number of pages, allocate an array to hold per page size and get it
  // populated
  hostdevice_vector<size_type> page_sizes(num_pages, stream);
  gpu::InitEncoderPages(chunks,
                        {},
                        page_sizes,
                        {},
                        col_desc,
                        num_columns,
                        max_page_size_bytes,
                        max_page_size_rows,
                        page_alignment(compression_codec),
                        nullptr,
                        nullptr,
                        stream);
  page_sizes.device_to_host(stream, true);

  // Get per-page max compressed size
  hostdevice_vector<size_type> comp_page_sizes(num_pages, stream);
  std::transform(page_sizes.begin(),
                 page_sizes.end(),
                 comp_page_sizes.begin(),
                 [compression_codec](auto page_size) {
                   return max_compression_output_size(compression_codec, page_size);
                 });
  comp_page_sizes.host_to_device(stream);

  // Use per-page max compressed size to calculate chunk.compressed_size
  gpu::InitEncoderPages(chunks,
                        {},
                        {},
                        comp_page_sizes,
                        col_desc,
                        num_columns,
                        max_page_size_bytes,
                        max_page_size_rows,
                        page_alignment(compression_codec),
                        nullptr,
                        nullptr,
                        stream);
  chunks.device_to_host(stream, true);
  return comp_page_sizes;
}

size_t max_page_bytes(Compression compression, size_t max_page_size_bytes)
{
  if (compression == parquet::Compression::UNCOMPRESSED) { return max_page_size_bytes; }

  auto const ncomp_type   = to_nvcomp_compression_type(compression);
  auto const nvcomp_limit = nvcomp::is_compression_disabled(ncomp_type)
                              ? std::nullopt
                              : nvcomp::compress_max_allowed_chunk_size(ncomp_type);

  auto max_size = std::min(nvcomp_limit.value_or(max_page_size_bytes), max_page_size_bytes);
  // page size must fit in a 32-bit signed integer
  return std::min<size_t>(max_size, std::numeric_limits<int32_t>::max());
}

std::pair<std::vector<rmm::device_uvector<size_type>>, std::vector<rmm::device_uvector<size_type>>>
build_chunk_dictionaries(hostdevice_2dvector<gpu::EncColumnChunk>& chunks,
                         host_span<gpu::parquet_column_device_view const> col_desc,
                         device_2dspan<gpu::PageFragment const> frags,
                         Compression compression,
                         dictionary_policy dict_policy,
                         size_t max_dict_size,
                         rmm::cuda_stream_view stream)
{
  // At this point, we know all chunks and their sizes. We want to allocate dictionaries for each
  // chunk that can have dictionary

  auto h_chunks = chunks.host_view().flat_view();

  std::vector<rmm::device_uvector<size_type>> dict_data;
  std::vector<rmm::device_uvector<size_type>> dict_index;

  if (h_chunks.size() == 0) { return std::pair(std::move(dict_data), std::move(dict_index)); }

  if (dict_policy == dictionary_policy::NEVER) {
    thrust::for_each(
      h_chunks.begin(), h_chunks.end(), [](auto& chunk) { chunk.use_dictionary = false; });
    chunks.host_to_device(stream);
    return std::pair(std::move(dict_data), std::move(dict_index));
  }

  // Allocate slots for each chunk
  std::vector<rmm::device_uvector<gpu::slot_type>> hash_maps_storage;
  hash_maps_storage.reserve(h_chunks.size());
  for (auto& chunk : h_chunks) {
    if (col_desc[chunk.col_desc_id].physical_type == Type::BOOLEAN ||
        (col_desc[chunk.col_desc_id].output_as_byte_array &&
         col_desc[chunk.col_desc_id].physical_type == Type::BYTE_ARRAY)) {
      chunk.use_dictionary = false;
    } else {
      chunk.use_dictionary = true;
      // cuCollections suggests using a hash map of size N * (1/0.7) = num_values * 1.43
      // https://github.com/NVIDIA/cuCollections/blob/3a49fc71/include/cuco/static_map.cuh#L190-L193
      auto& inserted_map   = hash_maps_storage.emplace_back(chunk.num_values * 1.43, stream);
      chunk.dict_map_slots = inserted_map.data();
      chunk.dict_map_size  = inserted_map.size();
    }
  }

  chunks.host_to_device(stream);

  gpu::initialize_chunk_hash_maps(chunks.device_view().flat_view(), stream);
  gpu::populate_chunk_hash_maps(frags, stream);

  chunks.device_to_host(stream, true);

  // Make decision about which chunks have dictionary
  for (auto& ck : h_chunks) {
    if (not ck.use_dictionary) { continue; }
    std::tie(ck.use_dictionary, ck.dict_rle_bits) = [&]() -> std::pair<bool, uint8_t> {
      // calculate size of chunk if dictionary is used

      // If we have N unique values then the idx for the last value is N - 1 and nbits is the number
      // of bits required to encode indices into the dictionary
      auto max_dict_index = (ck.num_dict_entries > 0) ? ck.num_dict_entries - 1 : 0;
      auto nbits          = std::max(CompactProtocolReader::NumRequiredBits(max_dict_index), 1);

      // We don't use dictionary if the indices are > MAX_DICT_BITS bits because that's the maximum
      // bitpacking bitsize we efficiently support
      if (nbits > MAX_DICT_BITS) { return {false, 0}; }

      auto rle_byte_size = util::div_rounding_up_safe(ck.num_values * nbits, 8);
      auto dict_enc_size = ck.uniq_data_size + rle_byte_size;
      if (ck.plain_data_size <= dict_enc_size) { return {false, 0}; }

      // don't use dictionary if it gets too large for the given compression codec
      if (dict_policy == dictionary_policy::ADAPTIVE) {
        auto const unique_size = static_cast<size_t>(ck.uniq_data_size);
        if (unique_size > max_page_bytes(compression, max_dict_size)) { return {false, 0}; }
      }

      return {true, nbits};
    }();
  }

  // TODO: (enh) Deallocate hash map storage for chunks that don't use dict and clear pointers.

  dict_data.reserve(h_chunks.size());
  dict_index.reserve(h_chunks.size());
  for (auto& chunk : h_chunks) {
    if (not chunk.use_dictionary) { continue; }

    size_t dict_data_size     = std::min(MAX_DICT_SIZE, chunk.dict_map_size);
    auto& inserted_dict_data  = dict_data.emplace_back(dict_data_size, stream);
    auto& inserted_dict_index = dict_index.emplace_back(chunk.num_values, stream);
    chunk.dict_data           = inserted_dict_data.data();
    chunk.dict_index          = inserted_dict_index.data();
  }
  chunks.host_to_device(stream);
  gpu::collect_map_entries(chunks.device_view().flat_view(), stream);
  gpu::get_dictionary_indices(frags, stream);

  return std::pair(std::move(dict_data), std::move(dict_index));
}

/**
 * @brief Initialize encoder pages.
 *
 * @param chunks Column chunk array
 * @param col_desc Column description array
 * @param pages Encoder pages array
 * @param comp_page_sizes Per-page max compressed size
 * @param page_stats Page statistics array
 * @param frag_stats Fragment statistics array
 * @param num_columns Total number of columns
 * @param num_pages Total number of pages
 * @param num_stats_bfr Number of statistics buffers
 * @param compression Compression format
 * @param max_page_size_bytes Maximum uncompressed page size, in bytes
 * @param max_page_size_rows Maximum page size, in rows
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void init_encoder_pages(hostdevice_2dvector<gpu::EncColumnChunk>& chunks,
                        device_span<gpu::parquet_column_device_view const> col_desc,
                        device_span<gpu::EncPage> pages,
                        hostdevice_vector<size_type>& comp_page_sizes,
                        statistics_chunk* page_stats,
                        statistics_chunk* frag_stats,
                        uint32_t num_columns,
                        uint32_t num_pages,
                        uint32_t num_stats_bfr,
                        Compression compression,
                        size_t max_page_size_bytes,
                        size_type max_page_size_rows,
                        rmm::cuda_stream_view stream)
{
  rmm::device_uvector<statistics_merge_group> page_stats_mrg(num_stats_bfr, stream);
  chunks.host_to_device(stream);
  InitEncoderPages(chunks,
                   pages,
                   {},
                   comp_page_sizes,
                   col_desc,
                   num_columns,
                   max_page_size_bytes,
                   max_page_size_rows,
                   page_alignment(compression),
                   (num_stats_bfr) ? page_stats_mrg.data() : nullptr,
                   (num_stats_bfr > num_pages) ? page_stats_mrg.data() + num_pages : nullptr,
                   stream);
  if (num_stats_bfr > 0) {
    detail::merge_group_statistics<detail::io_file_format::PARQUET>(
      page_stats, frag_stats, page_stats_mrg.data(), num_pages, stream);
    if (num_stats_bfr > num_pages) {
      detail::merge_group_statistics<detail::io_file_format::PARQUET>(
        page_stats + num_pages,
        page_stats,
        page_stats_mrg.data() + num_pages,
        num_stats_bfr - num_pages,
        stream);
    }
  }
  stream.synchronize();
}

/**
 * @brief Encode a batch of pages.
 *
 * @throws rmm::bad_alloc if there is insufficient space for temporary buffers
 *
 * @param chunks column chunk array
 * @param pages encoder pages array
 * @param pages_in_batch number of pages in this batch
 * @param first_page_in_batch first page in batch
 * @param rowgroups_in_batch number of rowgroups in this batch
 * @param first_rowgroup first rowgroup in batch
 * @param page_stats optional page-level statistics (nullptr if none)
 * @param chunk_stats optional chunk-level statistics (nullptr if none)
 * @param column_stats optional page-level statistics for column index (nullptr if none)
 * @param compression compression format
 * @param column_index_truncate_length maximum length of min or max values in column index, in bytes
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void encode_pages(hostdevice_2dvector<gpu::EncColumnChunk>& chunks,
                  device_span<gpu::EncPage> pages,
                  uint32_t pages_in_batch,
                  uint32_t first_page_in_batch,
                  uint32_t rowgroups_in_batch,
                  uint32_t first_rowgroup,
                  const statistics_chunk* page_stats,
                  const statistics_chunk* chunk_stats,
                  const statistics_chunk* column_stats,
                  Compression compression,
                  int32_t column_index_truncate_length,
                  rmm::cuda_stream_view stream)
{
  auto batch_pages = pages.subspan(first_page_in_batch, pages_in_batch);

  auto batch_pages_stats =
    (page_stats != nullptr)
      ? device_span<statistics_chunk const>(page_stats + first_page_in_batch, pages_in_batch)
      : device_span<statistics_chunk const>();

  uint32_t max_comp_pages =
    (compression != parquet::Compression::UNCOMPRESSED) ? pages_in_batch : 0;

  rmm::device_uvector<device_span<uint8_t const>> comp_in(max_comp_pages, stream);
  rmm::device_uvector<device_span<uint8_t>> comp_out(max_comp_pages, stream);
  rmm::device_uvector<compression_result> comp_res(max_comp_pages, stream);
  thrust::fill(rmm::exec_policy(stream),
               comp_res.begin(),
               comp_res.end(),
               compression_result{0, compression_status::FAILURE});

  gpu::EncodePages(batch_pages, comp_in, comp_out, comp_res, stream);
  switch (compression) {
    case parquet::Compression::SNAPPY:
      if (nvcomp::is_compression_disabled(nvcomp::compression_type::SNAPPY)) {
        gpu_snap(comp_in, comp_out, comp_res, stream);
      } else {
        nvcomp::batched_compress(
          nvcomp::compression_type::SNAPPY, comp_in, comp_out, comp_res, stream);
      }
      break;
    case parquet::Compression::ZSTD: {
      if (auto const reason = nvcomp::is_compression_disabled(nvcomp::compression_type::ZSTD);
          reason) {
        CUDF_FAIL("Compression error: " + reason.value());
      }
      nvcomp::batched_compress(nvcomp::compression_type::ZSTD, comp_in, comp_out, comp_res, stream);

      break;
    }
    case parquet::Compression::UNCOMPRESSED: break;
    default: CUDF_FAIL("invalid compression type");
  }

  // TBD: Not clear if the official spec actually allows dynamically turning off compression at the
  // chunk-level

  auto d_chunks_in_batch = chunks.device_view().subspan(first_rowgroup, rowgroups_in_batch);
  DecideCompression(d_chunks_in_batch.flat_view(), stream);
  EncodePageHeaders(batch_pages, comp_res, batch_pages_stats, chunk_stats, stream);
  GatherPages(d_chunks_in_batch.flat_view(), pages, stream);

  if (column_stats != nullptr) {
    EncodeColumnIndexes(d_chunks_in_batch.flat_view(),
                        {column_stats, pages.size()},
                        column_index_truncate_length,
                        stream);
  }

  auto h_chunks_in_batch = chunks.host_view().subspan(first_rowgroup, rowgroups_in_batch);
  CUDF_CUDA_TRY(cudaMemcpyAsync(h_chunks_in_batch.data(),
                                d_chunks_in_batch.data(),
                                d_chunks_in_batch.flat_view().size_bytes(),
                                cudaMemcpyDefault,
                                stream.value()));
  stream.synchronize();
}

/**
 * @brief Function to calculate the memory needed to encode the column index of the given
 * column chunk.
 *
 * @param ck pointer to column chunk
 * @param column_index_truncate_length maximum length of min or max values in column index, in bytes
 * @return Computed buffer size needed to encode the column index
 */
size_t column_index_buffer_size(gpu::EncColumnChunk* ck, int32_t column_index_truncate_length)
{
  // encoding the column index for a given chunk requires:
  //   each list (4 of them) requires 6 bytes of overhead
  //     (1 byte field header, 1 byte type, 4 bytes length)
  //   1 byte overhead for boundary_order
  //   1 byte overhead for termination
  //   sizeof(char) for boundary_order
  //   sizeof(bool) * num_pages for null_pages
  //   (ck_max_stats_len + 4) * num_pages * 2 for min/max values
  //     (each binary requires 4 bytes length + ck_max_stats_len)
  //   sizeof(int64_t) * num_pages for null_counts
  //
  // so 26 bytes overhead + sizeof(char) +
  //    (sizeof(bool) + sizeof(int64_t) + 2 * (4 + ck_max_stats_len)) * num_pages
  //
  // we already have ck->ck_stat_size = 48 + 2 * ck_max_stats_len
  // all of the overhead and non-stats data can fit in under 48 bytes
  //
  // so we can simply use ck_stat_size * num_pages
  //
  // add on some extra padding at the end (plus extra 7 bytes of alignment padding)
  // for scratch space to do stats truncation.
  //
  // calculating this per-chunk because the sizes can be wildly different.
  constexpr size_t padding = 7;
  return ck->ck_stat_size * ck->num_pages + column_index_truncate_length + padding;
}

/**
 * @brief Fill the table metadata with default column names.
 *
 * @param table_meta The table metadata to fill
 */
void fill_table_meta(std::unique_ptr<table_input_metadata> const& table_meta)
{
  // Fill unnamed columns' names in table_meta
  std::function<void(column_in_metadata&, std::string)> add_default_name =
    [&](column_in_metadata& col_meta, std::string default_name) {
      if (col_meta.get_name().empty()) col_meta.set_name(default_name);
      for (size_type i = 0; i < col_meta.num_children(); ++i) {
        add_default_name(col_meta.child(i), col_meta.get_name() + "_" + std::to_string(i));
      }
    };
  for (size_t i = 0; i < table_meta->column_metadata.size(); ++i) {
    add_default_name(table_meta->column_metadata[i], "_col" + std::to_string(i));
  }
}

/**
 * @brief Perform the processing steps needed to convert the input table into the output Parquet
 * data for writing, such as compression and encoding.
 *
 * @param[in,out] table_meta The table metadata
 * @param input The input table
 * @param partitions Optional partitions to divide the table into, if specified then must be same
 *        size as number of sinks
 * @param kv_meta Optional user metadata
 * @param curr_agg_meta The current aggregate writer metadata
 * @param max_page_fragment_size_opt Optional maximum number of rows in a page fragment
 * @param max_row_group_size Maximum row group size, in bytes
 * @param max_page_size_bytes Maximum uncompressed page size, in bytes
 * @param max_row_group_rows Maximum row group size, in rows
 * @param max_page_size_rows Maximum page size, in rows
 * @param column_index_truncate_length maximum length of min or max values in column index, in bytes
 * @param stats_granularity Level of statistics requested in output file
 * @param compression Compression format
 * @param dict_policy Policy for dictionary use
 * @param max_dictionary_size Maximum dictionary size, in bytes
 * @param single_write_mode Flag to indicate that we are guaranteeing a single table write
 * @param int96_timestamps Flag to indicate if timestamps will be written as INT96
 * @param out_sink Sink for checking if device write is supported, should not be used to write any
 *        data in this function
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A tuple of the intermediate results containing the processed data
 */
auto convert_table_to_parquet_data(table_input_metadata& table_meta,
                                   table_view const& input,
                                   host_span<partition_info const> partitions,
                                   host_span<std::map<std::string, std::string> const> kv_meta,
                                   std::unique_ptr<aggregate_writer_metadata> const& curr_agg_meta,
                                   std::optional<size_type> max_page_fragment_size_opt,
                                   size_t max_row_group_size,
                                   size_t max_page_size_bytes,
                                   size_type max_row_group_rows,
                                   size_type max_page_size_rows,
                                   int32_t column_index_truncate_length,
                                   statistics_freq stats_granularity,
                                   Compression compression,
                                   dictionary_policy dict_policy,
                                   size_t max_dictionary_size,
                                   single_write_mode write_mode,
                                   bool int96_timestamps,
                                   host_span<std::unique_ptr<data_sink> const> out_sink,
                                   rmm::cuda_stream_view stream)
{
  auto vec         = table_to_linked_columns(input);
  auto schema_tree = construct_schema_tree(vec, table_meta, write_mode, int96_timestamps);
  // Construct parquet_column_views from the schema tree leaf nodes.
  std::vector<parquet_column_view> parquet_columns;

  for (schema_tree_node const& schema_node : schema_tree) {
    if (schema_node.leaf_column) { parquet_columns.emplace_back(schema_node, schema_tree, stream); }
  }

  // Mass allocation of column_device_views for each parquet_column_view
  std::vector<column_view> cudf_cols;
  cudf_cols.reserve(parquet_columns.size());
  for (auto const& parq_col : parquet_columns) {
    cudf_cols.push_back(parq_col.cudf_column_view());
  }
  table_view single_streams_table(cudf_cols);
  size_type num_columns = single_streams_table.num_columns();

  std::vector<SchemaElement> this_table_schema(schema_tree.begin(), schema_tree.end());

  // Initialize column description
  hostdevice_vector<gpu::parquet_column_device_view> col_desc(parquet_columns.size(), stream);
  std::transform(
    parquet_columns.begin(), parquet_columns.end(), col_desc.host_ptr(), [&](auto const& pcol) {
      return pcol.get_device_view(stream);
    });

  // Init page fragments
  // 5000 is good enough for up to ~200-character strings. Longer strings and deeply nested columns
  // will start producing fragments larger than the desired page size, so calculate fragment sizes
  // for each leaf column.  Skip if the fragment size is not the default.
  size_type max_page_fragment_size =
    max_page_fragment_size_opt.value_or(default_max_page_fragment_size);

  std::vector<size_type> column_frag_size(num_columns, max_page_fragment_size);

  if (input.num_rows() > 0 && not max_page_fragment_size_opt.has_value()) {
    std::vector<size_t> column_sizes;
    std::transform(single_streams_table.begin(),
                   single_streams_table.end(),
                   std::back_inserter(column_sizes),
                   [&](auto const& column) { return column_size(column, stream); });

    // adjust global fragment size if a single fragment will overrun a rowgroup
    auto const table_size  = std::reduce(column_sizes.begin(), column_sizes.end());
    auto const avg_row_len = util::div_rounding_up_safe<size_t>(table_size, input.num_rows());
    if (avg_row_len > 0) {
      auto const rg_frag_size = util::div_rounding_up_safe(max_row_group_size, avg_row_len);
      max_page_fragment_size  = std::min<size_type>(rg_frag_size, max_page_fragment_size);
    }

    // dividing page size by average row length will tend to overshoot the desired
    // page size when there's high variability in the row lengths. instead, shoot
    // for multiple fragments per page to smooth things out. using 2 was too
    // unbalanced in final page sizes, so using 4 which seems to be a good
    // compromise at smoothing things out without getting fragment sizes too small.
    auto frag_size_fn = [&](auto const& col, size_type col_size) {
      const int target_frags_per_page = is_col_fixed_width(col) ? 1 : 4;
      auto const avg_len =
        target_frags_per_page * util::div_rounding_up_safe<size_type>(col_size, input.num_rows());
      if (avg_len > 0) {
        auto const frag_size = util::div_rounding_up_safe<size_type>(max_page_size_bytes, avg_len);
        return std::min<size_type>(max_page_fragment_size, frag_size);
      } else {
        return max_page_fragment_size;
      }
    };

    std::transform(single_streams_table.begin(),
                   single_streams_table.end(),
                   column_sizes.begin(),
                   column_frag_size.begin(),
                   frag_size_fn);
  }

  // Fragments are calculated in two passes. In the first pass, a uniform number of fragments
  // per column is used. This is done to satisfy the requirement that each column chunk within
  // a row group has the same number of rows. After the row group (and thus column chunk)
  // boundaries are known, a second pass is done to calculate fragments to be used in determining
  // page boundaries within each column chunk.
  std::vector<int> num_frag_in_part;
  std::transform(partitions.begin(),
                 partitions.end(),
                 std::back_inserter(num_frag_in_part),
                 [max_page_fragment_size](auto const& part) {
                   return util::div_rounding_up_unsafe(part.num_rows, max_page_fragment_size);
                 });

  size_type num_fragments = std::reduce(num_frag_in_part.begin(), num_frag_in_part.end());

  std::vector<int> part_frag_offset;  // Store the idx of the first fragment in each partition
  std::exclusive_scan(
    num_frag_in_part.begin(), num_frag_in_part.end(), std::back_inserter(part_frag_offset), 0);
  part_frag_offset.push_back(part_frag_offset.back() + num_frag_in_part.back());

  auto d_part_frag_offset = cudf::detail::make_device_uvector_async(
    part_frag_offset, stream, rmm::mr::get_current_device_resource());
  cudf::detail::hostdevice_2dvector<gpu::PageFragment> row_group_fragments(
    num_columns, num_fragments, stream);

  // Create table_device_view so that corresponding column_device_view data
  // can be written into col_desc members
  // These are unused but needs to be kept alive.
  auto parent_column_table_device_view = table_device_view::create(single_streams_table, stream);
  rmm::device_uvector<column_device_view> leaf_column_views(0, stream);

  if (num_fragments != 0) {
    // Move column info to device
    col_desc.host_to_device(stream);
    leaf_column_views = create_leaf_column_device_views<gpu::parquet_column_device_view>(
      col_desc, *parent_column_table_device_view, stream);

    init_row_group_fragments(row_group_fragments,
                             col_desc,
                             partitions,
                             d_part_frag_offset,
                             max_page_fragment_size,
                             stream);
  }

  std::unique_ptr<aggregate_writer_metadata> agg_meta;
  if (!curr_agg_meta) {
    agg_meta = std::make_unique<aggregate_writer_metadata>(
      partitions, kv_meta, this_table_schema, num_columns, stats_granularity);
  } else {
    agg_meta = std::make_unique<aggregate_writer_metadata>(*curr_agg_meta);

    // verify the user isn't passing mismatched tables
    CUDF_EXPECTS(agg_meta->schema_matches(this_table_schema),
                 "Mismatch in schema between multiple calls to write_chunk");

    agg_meta->update_files(partitions);
  }

  auto global_rowgroup_base = agg_meta->num_row_groups_per_file();

  // Decide row group boundaries based on uncompressed data size
  size_type num_rowgroups = 0;

  std::vector<int> num_rg_in_part(partitions.size());
  for (size_t p = 0; p < partitions.size(); ++p) {
    size_type curr_rg_num_rows = 0;
    size_t curr_rg_data_size   = 0;
    int first_frag_in_rg       = part_frag_offset[p];
    int last_frag_in_part      = part_frag_offset[p + 1] - 1;
    for (auto f = first_frag_in_rg; f <= last_frag_in_part; ++f) {
      size_t fragment_data_size = 0;
      for (auto c = 0; c < num_columns; c++) {
        fragment_data_size += row_group_fragments[c][f].fragment_data_size;
      }
      size_type fragment_num_rows = row_group_fragments[0][f].num_rows;

      // If the fragment size gets larger than rg limit then break off a rg
      if (f > first_frag_in_rg &&  // There has to be at least one fragment in row group
          (curr_rg_data_size + fragment_data_size > max_row_group_size ||
           curr_rg_num_rows + fragment_num_rows > max_row_group_rows)) {
        auto& rg    = agg_meta->file(p).row_groups.emplace_back();
        rg.num_rows = curr_rg_num_rows;
        num_rowgroups++;
        num_rg_in_part[p]++;
        curr_rg_num_rows  = 0;
        curr_rg_data_size = 0;
        first_frag_in_rg  = f;
      }
      curr_rg_num_rows += fragment_num_rows;
      curr_rg_data_size += fragment_data_size;

      // TODO: (wishful) refactor to consolidate with above if block
      if (f == last_frag_in_part) {
        auto& rg    = agg_meta->file(p).row_groups.emplace_back();
        rg.num_rows = curr_rg_num_rows;
        num_rowgroups++;
        num_rg_in_part[p]++;
      }
    }
  }

  std::vector<int> first_rg_in_part;
  std::exclusive_scan(
    num_rg_in_part.begin(), num_rg_in_part.end(), std::back_inserter(first_rg_in_part), 0);

  // Initialize row groups and column chunks
  auto const num_chunks = num_rowgroups * num_columns;
  hostdevice_2dvector<gpu::EncColumnChunk> chunks(num_rowgroups, num_columns, stream);

  // total fragments per column (in case they are non-uniform)
  std::vector<size_type> frags_per_column(num_columns, 0);

  for (size_t p = 0; p < partitions.size(); ++p) {
    int f               = part_frag_offset[p];
    size_type start_row = partitions[p].start_row;
    for (int r = 0; r < num_rg_in_part[p]; r++) {
      size_t global_r = global_rowgroup_base[p] + r;  // Number of rowgroups already in file/part
      auto& row_group = agg_meta->file(p).row_groups[global_r];
      uint32_t fragments_in_chunk =
        util::div_rounding_up_unsafe(row_group.num_rows, max_page_fragment_size);
      row_group.total_byte_size = 0;
      row_group.columns.resize(num_columns);
      for (int c = 0; c < num_columns; c++) {
        gpu::EncColumnChunk& ck = chunks[r + first_rg_in_part[p]][c];

        ck                   = {};
        ck.col_desc          = col_desc.device_ptr() + c;
        ck.col_desc_id       = c;
        ck.fragments         = &row_group_fragments.device_view()[c][f];
        ck.stats             = nullptr;
        ck.start_row         = start_row;
        ck.num_rows          = (uint32_t)row_group.num_rows;
        ck.first_fragment    = c * num_fragments + f;
        auto chunk_fragments = row_group_fragments[c].subspan(f, fragments_in_chunk);
        // In fragment struct, add a pointer to the chunk it belongs to
        // In each fragment in chunk_fragments, update the chunk pointer here.
        for (auto& frag : chunk_fragments) {
          frag.chunk = &chunks.device_view()[r + first_rg_in_part[p]][c];
        }
        ck.num_values = std::accumulate(
          chunk_fragments.begin(), chunk_fragments.end(), 0, [](uint32_t l, auto r) {
            return l + r.num_values;
          });
        ck.plain_data_size = std::accumulate(
          chunk_fragments.begin(), chunk_fragments.end(), 0, [](int sum, gpu::PageFragment frag) {
            return sum + frag.fragment_data_size;
          });
        auto& column_chunk_meta          = row_group.columns[c].meta_data;
        column_chunk_meta.type           = parquet_columns[c].physical_type();
        column_chunk_meta.encodings      = {Encoding::PLAIN, Encoding::RLE};
        column_chunk_meta.path_in_schema = parquet_columns[c].get_path_in_schema();
        column_chunk_meta.codec          = UNCOMPRESSED;
        column_chunk_meta.num_values     = ck.num_values;

        frags_per_column[c] += util::div_rounding_up_unsafe(
          row_group.num_rows, std::min(column_frag_size[c], max_page_fragment_size));
      }
      f += fragments_in_chunk;
      start_row += (uint32_t)row_group.num_rows;
    }
  }

  row_group_fragments.host_to_device(stream);
  [[maybe_unused]] auto dict_info_owner = build_chunk_dictionaries(
    chunks, col_desc, row_group_fragments, compression, dict_policy, max_dictionary_size, stream);
  for (size_t p = 0; p < partitions.size(); p++) {
    for (int rg = 0; rg < num_rg_in_part[p]; rg++) {
      size_t global_rg = global_rowgroup_base[p] + rg;
      for (int col = 0; col < num_columns; col++) {
        if (chunks.host_view()[rg][col].use_dictionary) {
          agg_meta->file(p).row_groups[global_rg].columns[col].meta_data.encodings.push_back(
            Encoding::PLAIN_DICTIONARY);
        }
      }
    }
  }

  // The code preceding this used a uniform fragment size for all columns. Now recompute
  // fragments with a (potentially) varying number of fragments per column.

  // first figure out the total number of fragments and calculate the start offset for each column
  std::vector<size_type> frag_offsets;
  size_type const total_frags = [&]() {
    if (frags_per_column.size() > 0) {
      std::exclusive_scan(frags_per_column.data(),
                          frags_per_column.data() + num_columns + 1,
                          std::back_inserter(frag_offsets),
                          0);
      return frag_offsets[num_columns];
    } else {
      return 0;
    }
  }();

  rmm::device_uvector<statistics_chunk> frag_stats(0, stream);
  hostdevice_vector<gpu::PageFragment> page_fragments(total_frags, stream);

  // update fragments and/or prepare for fragment statistics calculation if necessary
  if (total_frags != 0) {
    if (stats_granularity != statistics_freq::STATISTICS_NONE) {
      frag_stats.resize(total_frags, stream);
    }

    for (int c = 0; c < num_columns; c++) {
      auto frag_offset     = frag_offsets[c];
      auto const frag_size = column_frag_size[c];

      for (size_t p = 0; p < partitions.size(); ++p) {
        for (int r = 0; r < num_rg_in_part[p]; r++) {
          auto const global_r   = global_rowgroup_base[p] + r;
          auto const& row_group = agg_meta->file(p).row_groups[global_r];
          uint32_t const fragments_in_chunk =
            util::div_rounding_up_unsafe(row_group.num_rows, frag_size);
          gpu::EncColumnChunk& ck = chunks[r + first_rg_in_part[p]][c];
          ck.fragments            = page_fragments.device_ptr(frag_offset);
          ck.first_fragment       = frag_offset;

          // update the chunk pointer here for each fragment in chunk.fragments
          for (uint32_t i = 0; i < fragments_in_chunk; i++) {
            page_fragments[frag_offset + i].chunk =
              &chunks.device_view()[r + first_rg_in_part[p]][c];
          }

          if (not frag_stats.is_empty()) { ck.stats = frag_stats.data() + frag_offset; }
          frag_offset += fragments_in_chunk;
        }
      }
    }

    chunks.host_to_device(stream);

    // re-initialize page fragments
    page_fragments.host_to_device(stream);
    calculate_page_fragments(page_fragments, column_frag_size, stream);

    // and gather fragment statistics
    if (not frag_stats.is_empty()) {
      gather_fragment_statistics(frag_stats,
                                 {page_fragments.device_ptr(), static_cast<size_t>(total_frags)},
                                 int96_timestamps,
                                 stream);
    }
  }

  // Build chunk dictionaries and count pages. Sends chunks to device.
  hostdevice_vector<size_type> comp_page_sizes = init_page_sizes(
    chunks, col_desc, num_columns, max_page_size_bytes, max_page_size_rows, compression, stream);

  // Find which partition a rg belongs to
  std::vector<int> rg_to_part;
  for (size_t p = 0; p < num_rg_in_part.size(); ++p) {
    std::fill_n(std::back_inserter(rg_to_part), num_rg_in_part[p], p);
  }

  // Initialize batches of rowgroups to encode (mainly to limit peak memory usage)
  std::vector<size_type> batch_list;
  size_type num_pages          = 0;
  size_t max_bytes_in_batch    = 1024 * 1024 * 1024;  // 1GB - TODO: Tune this
  size_t max_uncomp_bfr_size   = 0;
  size_t max_comp_bfr_size     = 0;
  size_t max_chunk_bfr_size    = 0;
  size_type max_pages_in_batch = 0;
  size_t bytes_in_batch        = 0;
  size_t comp_bytes_in_batch   = 0;
  size_t column_index_bfr_size = 0;
  for (size_type r = 0, groups_in_batch = 0, pages_in_batch = 0; r <= num_rowgroups; r++) {
    size_t rowgroup_size      = 0;
    size_t comp_rowgroup_size = 0;
    if (r < num_rowgroups) {
      for (int i = 0; i < num_columns; i++) {
        gpu::EncColumnChunk* ck = &chunks[r][i];
        ck->first_page          = num_pages;
        num_pages += ck->num_pages;
        pages_in_batch += ck->num_pages;
        rowgroup_size += ck->bfr_size;
        comp_rowgroup_size += ck->compressed_size;
        max_chunk_bfr_size =
          std::max(max_chunk_bfr_size, (size_t)std::max(ck->bfr_size, ck->compressed_size));
        if (stats_granularity == statistics_freq::STATISTICS_COLUMN) {
          column_index_bfr_size += column_index_buffer_size(ck, column_index_truncate_length);
        }
      }
    }
    // TBD: We may want to also shorten the batch if we have enough pages (not just based on size)
    if ((r == num_rowgroups) ||
        (groups_in_batch != 0 && bytes_in_batch + rowgroup_size > max_bytes_in_batch)) {
      max_uncomp_bfr_size = std::max(max_uncomp_bfr_size, bytes_in_batch);
      max_comp_bfr_size   = std::max(max_comp_bfr_size, comp_bytes_in_batch);
      max_pages_in_batch  = std::max(max_pages_in_batch, pages_in_batch);
      if (groups_in_batch != 0) {
        batch_list.push_back(groups_in_batch);
        groups_in_batch = 0;
      }
      bytes_in_batch      = 0;
      comp_bytes_in_batch = 0;
      pages_in_batch      = 0;
    }
    bytes_in_batch += rowgroup_size;
    comp_bytes_in_batch += comp_rowgroup_size;
    groups_in_batch++;
  }

  // Clear compressed buffer size if compression has been turned off
  if (compression == parquet::Compression::UNCOMPRESSED) { max_comp_bfr_size = 0; }

  // Initialize data pointers in batch
  uint32_t const num_stats_bfr =
    (stats_granularity != statistics_freq::STATISTICS_NONE) ? num_pages + num_chunks : 0;
  rmm::device_buffer uncomp_bfr(max_uncomp_bfr_size, stream);
  rmm::device_buffer comp_bfr(max_comp_bfr_size, stream);
  rmm::device_buffer col_idx_bfr(column_index_bfr_size, stream);
  rmm::device_uvector<gpu::EncPage> pages(num_pages, stream);

  // This contains stats for both the pages and the rowgroups. TODO: make them separate.
  rmm::device_uvector<statistics_chunk> page_stats(num_stats_bfr, stream);
  auto bfr_i = static_cast<uint8_t*>(col_idx_bfr.data());
  for (auto b = 0, r = 0; b < static_cast<size_type>(batch_list.size()); b++) {
    auto bfr   = static_cast<uint8_t*>(uncomp_bfr.data());
    auto bfr_c = static_cast<uint8_t*>(comp_bfr.data());
    for (auto j = 0; j < batch_list[b]; j++, r++) {
      for (auto i = 0; i < num_columns; i++) {
        gpu::EncColumnChunk& ck = chunks[r][i];
        ck.uncompressed_bfr     = bfr;
        ck.compressed_bfr       = bfr_c;
        ck.column_index_blob    = bfr_i;
        bfr += ck.bfr_size;
        bfr_c += ck.compressed_size;
        if (stats_granularity == statistics_freq::STATISTICS_COLUMN) {
          ck.column_index_size = column_index_buffer_size(&ck, column_index_truncate_length);
          bfr_i += ck.column_index_size;
        }
      }
    }
  }

  if (num_pages != 0) {
    init_encoder_pages(chunks,
                       col_desc,
                       {pages.data(), pages.size()},
                       comp_page_sizes,
                       (num_stats_bfr) ? page_stats.data() : nullptr,
                       (num_stats_bfr) ? frag_stats.data() : nullptr,
                       num_columns,
                       num_pages,
                       num_stats_bfr,
                       compression,
                       max_page_size_bytes,
                       max_page_size_rows,
                       stream);
  }

  // Check device write support for all chunks and initialize bounce_buffer.
  bool all_device_write   = true;
  uint32_t max_write_size = 0;

  // Encode row groups in batches
  for (auto b = 0, r = 0; b < static_cast<size_type>(batch_list.size()); b++) {
    // Count pages in this batch
    auto const rnext               = r + batch_list[b];
    auto const first_page_in_batch = chunks[r][0].first_page;
    auto const first_page_in_next_batch =
      (rnext < num_rowgroups) ? chunks[rnext][0].first_page : num_pages;
    auto const pages_in_batch = first_page_in_next_batch - first_page_in_batch;

    encode_pages(
      chunks,
      {pages.data(), pages.size()},
      pages_in_batch,
      first_page_in_batch,
      batch_list[b],
      r,
      (stats_granularity == statistics_freq::STATISTICS_PAGE) ? page_stats.data() : nullptr,
      (stats_granularity != statistics_freq::STATISTICS_NONE) ? page_stats.data() + num_pages
                                                              : nullptr,
      (stats_granularity == statistics_freq::STATISTICS_COLUMN) ? page_stats.data() : nullptr,
      compression,
      column_index_truncate_length,
      stream);

    bool need_sync{false};

    for (; r < rnext; r++) {
      int p           = rg_to_part[r];
      int global_r    = global_rowgroup_base[p] + r - first_rg_in_part[p];
      auto& row_group = agg_meta->file(p).row_groups[global_r];

      for (auto i = 0; i < num_columns; i++) {
        auto const& ck          = chunks[r][i];
        auto const dev_bfr      = ck.is_compressed ? ck.compressed_bfr : ck.uncompressed_bfr;
        auto& column_chunk_meta = row_group.columns[i].meta_data;

        if (ck.is_compressed) { column_chunk_meta.codec = compression; }
        if (!out_sink[p]->is_device_write_preferred(ck.compressed_size)) {
          all_device_write = false;
        }
        max_write_size = std::max(max_write_size, ck.compressed_size);

        if (ck.ck_stat_size != 0) {
          column_chunk_meta.statistics_blob.resize(ck.ck_stat_size);
          CUDF_CUDA_TRY(cudaMemcpyAsync(column_chunk_meta.statistics_blob.data(),
                                        dev_bfr,
                                        ck.ck_stat_size,
                                        cudaMemcpyDefault,
                                        stream.value()));
          need_sync = true;
        }

        row_group.total_byte_size += ck.compressed_size;
        column_chunk_meta.total_uncompressed_size = ck.bfr_size;
        column_chunk_meta.total_compressed_size   = ck.compressed_size;
      }
    }

    // Sync before calling the next `encode_pages` which may alter the stats data.
    if (need_sync) { stream.synchronize(); }
  }

  auto bounce_buffer =
    cudf::detail::pinned_host_vector<uint8_t>(all_device_write ? 0 : max_write_size);

  return std::tuple{std::move(agg_meta),
                    std::move(pages),
                    std::move(chunks),
                    std::move(global_rowgroup_base),
                    std::move(first_rg_in_part),
                    std::move(batch_list),
                    std::move(rg_to_part),
                    std::move(uncomp_bfr),
                    std::move(comp_bfr),
                    std::move(col_idx_bfr),
                    std::move(bounce_buffer)};
}

}  // namespace

writer::impl::impl(std::vector<std::unique_ptr<data_sink>> sinks,
                   parquet_writer_options const& options,
                   single_write_mode mode,
                   rmm::cuda_stream_view stream)
  : _stream(stream),
    _compression(to_parquet_compression(options.get_compression())),
    _max_row_group_size{options.get_row_group_size_bytes()},
    _max_row_group_rows{options.get_row_group_size_rows()},
    _max_page_size_bytes(max_page_bytes(_compression, options.get_max_page_size_bytes())),
    _max_page_size_rows(options.get_max_page_size_rows()),
    _stats_granularity(options.get_stats_level()),
    _dict_policy(options.get_dictionary_policy()),
    _max_dictionary_size(options.get_max_dictionary_size()),
    _max_page_fragment_size(options.get_max_page_fragment_size()),
    _int96_timestamps(options.is_enabled_int96_timestamps()),
    _column_index_truncate_length(options.get_column_index_truncate_length()),
    _kv_meta(options.get_key_value_metadata()),
    _single_write_mode(mode),
    _out_sink(std::move(sinks))
{
  if (options.get_metadata()) {
    _table_meta = std::make_unique<table_input_metadata>(*options.get_metadata());
  }
  init_state();
}

writer::impl::impl(std::vector<std::unique_ptr<data_sink>> sinks,
                   chunked_parquet_writer_options const& options,
                   single_write_mode mode,
                   rmm::cuda_stream_view stream)
  : _stream(stream),
    _compression(to_parquet_compression(options.get_compression())),
    _max_row_group_size{options.get_row_group_size_bytes()},
    _max_row_group_rows{options.get_row_group_size_rows()},
    _max_page_size_bytes(max_page_bytes(_compression, options.get_max_page_size_bytes())),
    _max_page_size_rows(options.get_max_page_size_rows()),
    _stats_granularity(options.get_stats_level()),
    _dict_policy(options.get_dictionary_policy()),
    _max_dictionary_size(options.get_max_dictionary_size()),
    _max_page_fragment_size(options.get_max_page_fragment_size()),
    _int96_timestamps(options.is_enabled_int96_timestamps()),
    _column_index_truncate_length(options.get_column_index_truncate_length()),
    _kv_meta(options.get_key_value_metadata()),
    _single_write_mode(mode),
    _out_sink(std::move(sinks))
{
  if (options.get_metadata()) {
    _table_meta = std::make_unique<table_input_metadata>(*options.get_metadata());
  }
  init_state();
}

writer::impl::~impl() { close(); }

void writer::impl::init_state()
{
  _current_chunk_offset.resize(_out_sink.size());
  // Write file header
  file_header_s fhdr;
  fhdr.magic = parquet_magic;
  for (auto& sink : _out_sink) {
    sink->host_write(&fhdr, sizeof(fhdr));
  }
  std::fill_n(_current_chunk_offset.begin(), _current_chunk_offset.size(), sizeof(file_header_s));
}

void writer::impl::write(table_view const& input, std::vector<partition_info> const& partitions)
{
  _last_write_successful = false;
  CUDF_EXPECTS(not _closed, "Data has already been flushed to out and closed");

  if (not _table_meta) { _table_meta = std::make_unique<table_input_metadata>(input); }
  fill_table_meta(_table_meta);

  // All kinds of memory allocation and data compressions/encoding are performed here.
  // If any error occurs, such as out-of-memory exception, the internal state of the current
  // writer is still intact.
  [[maybe_unused]] auto [updated_agg_meta,
                         pages,
                         chunks,
                         global_rowgroup_base,
                         first_rg_in_part,
                         batch_list,
                         rg_to_part,
                         uncomp_bfr,   // unused, but contains data for later write to sink
                         comp_bfr,     // unused, but contains data for later write to sink
                         col_idx_bfr,  // unused, but contains data for later write to sink
                         bounce_buffer] = [&] {
    try {
      return convert_table_to_parquet_data(*_table_meta,
                                           input,
                                           partitions,
                                           _kv_meta,
                                           _agg_meta,
                                           _max_page_fragment_size,
                                           _max_row_group_size,
                                           _max_page_size_bytes,
                                           _max_row_group_rows,
                                           _max_page_size_rows,
                                           _column_index_truncate_length,
                                           _stats_granularity,
                                           _compression,
                                           _dict_policy,
                                           _max_dictionary_size,
                                           _single_write_mode,
                                           _int96_timestamps,
                                           _out_sink,
                                           _stream);
    } catch (...) {  // catch any exception type
      CUDF_LOG_ERROR(
        "Parquet writer encountered exception during processing. "
        "No data has been written to the sink.");
      throw;  // this throws the same exception
    }
  }();

  // Compression/encoding were all successful. Now write the intermediate results.
  write_parquet_data_to_sink(updated_agg_meta,
                             pages,
                             chunks,
                             global_rowgroup_base,
                             first_rg_in_part,
                             batch_list,
                             rg_to_part,
                             bounce_buffer);

  _last_write_successful = true;
}

void writer::impl::write_parquet_data_to_sink(
  std::unique_ptr<aggregate_writer_metadata>& updated_agg_meta,
  device_span<gpu::EncPage const> pages,
  host_2dspan<gpu::EncColumnChunk const> chunks,
  host_span<size_t const> global_rowgroup_base,
  host_span<int const> first_rg_in_part,
  host_span<size_type const> batch_list,
  host_span<int const> rg_to_part,
  host_span<uint8_t> bounce_buffer)
{
  _agg_meta              = std::move(updated_agg_meta);
  auto const num_columns = chunks.size().second;

  for (auto b = 0, r = 0; b < static_cast<size_type>(batch_list.size()); b++) {
    auto const rnext = r + batch_list[b];
    std::vector<std::future<void>> write_tasks;

    for (; r < rnext; r++) {
      int const p        = rg_to_part[r];
      int const global_r = global_rowgroup_base[p] + r - first_rg_in_part[p];
      auto& row_group    = _agg_meta->file(p).row_groups[global_r];

      for (std::size_t i = 0; i < num_columns; i++) {
        auto const& ck     = chunks[r][i];
        auto const dev_bfr = ck.is_compressed ? ck.compressed_bfr : ck.uncompressed_bfr;

        // Skip the range [0, ck.ck_stat_size) since it has already been copied to host
        // and stored in _agg_meta before.
        if (_out_sink[p]->is_device_write_preferred(ck.compressed_size)) {
          write_tasks.push_back(_out_sink[p]->device_write_async(
            dev_bfr + ck.ck_stat_size, ck.compressed_size, _stream));
        } else {
          CUDF_EXPECTS(bounce_buffer.size() >= ck.compressed_size,
                       "Bounce buffer was not properly initialized.");
          CUDF_CUDA_TRY(cudaMemcpyAsync(bounce_buffer.data(),
                                        dev_bfr + ck.ck_stat_size,
                                        ck.compressed_size,
                                        cudaMemcpyDefault,
                                        _stream.value()));
          _stream.synchronize();
          _out_sink[p]->host_write(bounce_buffer.data(), ck.compressed_size);
        }

        auto& column_chunk_meta = row_group.columns[i].meta_data;
        column_chunk_meta.data_page_offset =
          _current_chunk_offset[p] + ((ck.use_dictionary) ? ck.dictionary_size : 0);
        column_chunk_meta.dictionary_page_offset =
          (ck.use_dictionary) ? _current_chunk_offset[p] : 0;
        _current_chunk_offset[p] += ck.compressed_size;
      }
    }
    for (auto const& task : write_tasks) {
      task.wait();
    }
  }

  if (_stats_granularity == statistics_freq::STATISTICS_COLUMN) {
    // need pages on host to create offset_indexes
    auto const h_pages = cudf::detail::make_host_vector_sync(pages, _stream);

    // add column and offset indexes to metadata
    for (auto b = 0, r = 0; b < static_cast<size_type>(batch_list.size()); b++) {
      auto const rnext   = r + batch_list[b];
      auto curr_page_idx = chunks[r][0].first_page;
      for (; r < rnext; r++) {
        int const p           = rg_to_part[r];
        int const global_r    = global_rowgroup_base[p] + r - first_rg_in_part[p];
        auto const& row_group = _agg_meta->file(p).row_groups[global_r];
        for (std::size_t i = 0; i < num_columns; i++) {
          gpu::EncColumnChunk const& ck = chunks[r][i];
          auto const& column_chunk_meta = row_group.columns[i].meta_data;

          // start transfer of the column index
          std::vector<uint8_t> column_idx;
          column_idx.resize(ck.column_index_size);
          CUDF_CUDA_TRY(cudaMemcpyAsync(column_idx.data(),
                                        ck.column_index_blob,
                                        ck.column_index_size,
                                        cudaMemcpyDefault,
                                        _stream.value()));

          // calculate offsets while the column index is transferring
          int64_t curr_pg_offset = column_chunk_meta.data_page_offset;

          OffsetIndex offset_idx;
          for (uint32_t pg = 0; pg < ck.num_pages; pg++) {
            auto const& enc_page = h_pages[curr_page_idx++];

            // skip dict pages
            if (enc_page.page_type != PageType::DATA_PAGE) { continue; }

            int32_t this_page_size = enc_page.hdr_size + enc_page.max_data_size;
            // first_row_idx is relative to start of row group
            PageLocation loc{curr_pg_offset, this_page_size, enc_page.start_row - ck.start_row};
            offset_idx.page_locations.push_back(loc);
            curr_pg_offset += this_page_size;
          }

          _stream.synchronize();
          _agg_meta->file(p).offset_indexes.emplace_back(std::move(offset_idx));
          _agg_meta->file(p).column_indexes.emplace_back(std::move(column_idx));
        }
      }
    }
  }
}

std::unique_ptr<std::vector<uint8_t>> writer::impl::close(
  std::vector<std::string> const& column_chunks_file_path)
{
  if (_closed) { return nullptr; }
  _closed = true;
  if (not _last_write_successful) { return nullptr; }
  for (size_t p = 0; p < _out_sink.size(); p++) {
    std::vector<uint8_t> buffer;
    CompactProtocolWriter cpw(&buffer);
    file_ender_s fendr;

    if (_stats_granularity == statistics_freq::STATISTICS_COLUMN) {
      auto& fmd = _agg_meta->file(p);

      // write column indices, updating column metadata along the way
      int chunkidx = 0;
      for (auto& r : fmd.row_groups) {
        for (auto& c : r.columns) {
          auto const& index     = fmd.column_indexes[chunkidx++];
          c.column_index_offset = _out_sink[p]->bytes_written();
          c.column_index_length = index.size();
          _out_sink[p]->host_write(index.data(), index.size());
        }
      }

      // write offset indices, updating column metadata along the way
      chunkidx = 0;
      for (auto& r : fmd.row_groups) {
        for (auto& c : r.columns) {
          auto const& offsets = fmd.offset_indexes[chunkidx++];
          buffer.resize(0);
          int32_t len           = cpw.write(offsets);
          c.offset_index_offset = _out_sink[p]->bytes_written();
          c.offset_index_length = len;
          _out_sink[p]->host_write(buffer.data(), buffer.size());
        }
      }
    }

    buffer.resize(0);
    fendr.footer_len = static_cast<uint32_t>(cpw.write(_agg_meta->get_metadata(p)));
    fendr.magic      = parquet_magic;
    _out_sink[p]->host_write(buffer.data(), buffer.size());
    _out_sink[p]->host_write(&fendr, sizeof(fendr));
    _out_sink[p]->flush();
  }

  // Optionally output raw file metadata with the specified column chunk file path
  if (column_chunks_file_path.size() > 0) {
    CUDF_EXPECTS(column_chunks_file_path.size() == _agg_meta->num_files(),
                 "Expected one column chunk path per output file");
    _agg_meta->set_file_paths(column_chunks_file_path);
    file_header_s fhdr = {parquet_magic};
    std::vector<uint8_t> buffer;
    CompactProtocolWriter cpw(&buffer);
    buffer.insert(buffer.end(),
                  reinterpret_cast<const uint8_t*>(&fhdr),
                  reinterpret_cast<const uint8_t*>(&fhdr) + sizeof(fhdr));
    file_ender_s fendr;
    fendr.magic      = parquet_magic;
    fendr.footer_len = static_cast<uint32_t>(cpw.write(_agg_meta->get_merged_metadata()));
    buffer.insert(buffer.end(),
                  reinterpret_cast<const uint8_t*>(&fendr),
                  reinterpret_cast<const uint8_t*>(&fendr) + sizeof(fendr));
    return std::make_unique<std::vector<uint8_t>>(std::move(buffer));
  } else {
    return {nullptr};
  }
  return nullptr;
}

// Forward to implementation
writer::writer(std::vector<std::unique_ptr<data_sink>> sinks,
               parquet_writer_options const& options,
               single_write_mode mode,
               rmm::cuda_stream_view stream)
  : _impl(std::make_unique<impl>(std::move(sinks), options, mode, stream))
{
}

writer::writer(std::vector<std::unique_ptr<data_sink>> sinks,
               chunked_parquet_writer_options const& options,
               single_write_mode mode,
               rmm::cuda_stream_view stream)
  : _impl(std::make_unique<impl>(std::move(sinks), options, mode, stream))
{
}

// Destructor within this translation unit
writer::~writer() = default;

// Forward to implementation
void writer::write(table_view const& table, std::vector<partition_info> const& partitions)
{
  _impl->write(
    table, partitions.empty() ? std::vector<partition_info>{{0, table.num_rows()}} : partitions);
}

// Forward to implementation
std::unique_ptr<std::vector<uint8_t>> writer::close(
  std::vector<std::string> const& column_chunks_file_path)
{
  return _impl->close(column_chunks_file_path);
}

std::unique_ptr<std::vector<uint8_t>> writer::merge_row_group_metadata(
  std::vector<std::unique_ptr<std::vector<uint8_t>>> const& metadata_list)
{
  std::vector<uint8_t> output;
  CompactProtocolWriter cpw(&output);
  FileMetaData md;

  md.row_groups.reserve(metadata_list.size());
  for (const auto& blob : metadata_list) {
    CompactProtocolReader cpreader(
      blob.get()->data(),
      std::max<size_t>(blob.get()->size(), sizeof(file_ender_s)) - sizeof(file_ender_s));
    cpreader.skip_bytes(sizeof(file_header_s));  // Skip over file header
    if (md.num_rows == 0) {
      cpreader.read(&md);
    } else {
      FileMetaData tmp;
      cpreader.read(&tmp);
      md.row_groups.insert(md.row_groups.end(),
                           std::make_move_iterator(tmp.row_groups.begin()),
                           std::make_move_iterator(tmp.row_groups.end()));
      md.num_rows += tmp.num_rows;
    }
  }
  // Reader doesn't currently populate column_order, so infer it here
  if (md.row_groups.size() != 0) {
    uint32_t num_columns = static_cast<uint32_t>(md.row_groups[0].columns.size());
    md.column_order_listsize =
      (num_columns > 0 && md.row_groups[0].columns[0].meta_data.statistics_blob.size())
        ? num_columns
        : 0;
  }
  // Thrift-encode the resulting output
  file_header_s fhdr;
  file_ender_s fendr;
  fhdr.magic = parquet_magic;
  output.insert(output.end(),
                reinterpret_cast<const uint8_t*>(&fhdr),
                reinterpret_cast<const uint8_t*>(&fhdr) + sizeof(fhdr));
  fendr.footer_len = static_cast<uint32_t>(cpw.write(md));
  fendr.magic      = parquet_magic;
  output.insert(output.end(),
                reinterpret_cast<const uint8_t*>(&fendr),
                reinterpret_cast<const uint8_t*>(&fendr) + sizeof(fendr));
  return std::make_unique<std::vector<uint8_t>>(std::move(output));
}

}  // namespace parquet
}  // namespace detail
}  // namespace io
}  // namespace cudf
