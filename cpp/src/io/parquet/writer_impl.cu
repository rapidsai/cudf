/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include "writer_impl.hpp"

#include "compact_protocol_reader.hpp"
#include "compact_protocol_writer.hpp"

#include <io/statistics/column_statistics.cuh>
#include <io/utilities/column_utils.cuh>
#include <io/utilities/config_utils.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/column.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <nvcomp/snappy.h>

#include <thrust/binary_search.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

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

namespace {
/**
 * @brief Helper for pinned host memory
 */
template <typename T>
using pinned_buffer = std::unique_ptr<T, decltype(&cudaFreeHost)>;

/**
 * @brief Function that translates GDF compression to parquet compression
 */
parquet::Compression to_parquet_compression(compression_type compression)
{
  switch (compression) {
    case compression_type::AUTO:
    case compression_type::SNAPPY: return parquet::Compression::SNAPPY;
    case compression_type::NONE: return parquet::Compression::UNCOMPRESSED;
    default: CUDF_FAIL("Unsupported compression type");
  }
}

}  // namespace

struct aggregate_writer_metadata {
  aggregate_writer_metadata(std::vector<partition_info> const& partitions,
                            size_type num_columns,
                            std::vector<SchemaElement> schema,
                            statistics_freq stats_granularity,
                            std::vector<std::map<std::string, std::string>> const& kv_md)
    : version(1), schema(std::move(schema)), files(partitions.size())
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

  void update_files(std::vector<partition_info> const& partitions)
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

  void set_file_paths(std::vector<std::string> const& column_chunks_file_path)
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
  };
  std::vector<per_file_metadata> files;
  std::string created_by         = "";
  uint32_t column_order_listsize = 0;
};

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
    col_schema.type           = Type::BYTE_ARRAY;
    col_schema.converted_type = ConvertedType::UTF8;
    col_schema.stats_dtype    = statistics_dtype::dtype_string;
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
    col_schema.type           = Type::INT32;
    col_schema.converted_type = ConvertedType::TIME_MILLIS;
    col_schema.stats_dtype    = statistics_dtype::dtype_int64;
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::duration_s>, void> operator()()
  {
    col_schema.type           = Type::INT64;
    col_schema.converted_type = ConvertedType::TIME_MILLIS;
    col_schema.stats_dtype    = statistics_dtype::dtype_int64;
    col_schema.ts_scale       = 1000;
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::duration_ms>, void> operator()()
  {
    col_schema.type           = Type::INT64;
    col_schema.converted_type = ConvertedType::TIME_MILLIS;
    col_schema.stats_dtype    = statistics_dtype::dtype_int64;
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::duration_us>, void> operator()()
  {
    col_schema.type           = Type::INT64;
    col_schema.converted_type = ConvertedType::TIME_MICROS;
    col_schema.stats_dtype    = statistics_dtype::dtype_int64;
  }

  //  unsupported outside cudf for parquet 1.0.
  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::duration_ns>, void> operator()()
  {
    col_schema.type           = Type::INT64;
    col_schema.converted_type = ConvertedType::TIME_MICROS;
    col_schema.stats_dtype    = statistics_dtype::dtype_int64;
    col_schema.ts_scale       = -1000;  // negative value indicates division by absolute value
  }

  template <typename T>
  std::enable_if_t<cudf::is_fixed_point<T>(), void> operator()()
  {
    if (std::is_same_v<T, numeric::decimal32>) {
      col_schema.type              = Type::INT32;
      col_schema.stats_dtype       = statistics_dtype::dtype_int32;
      col_schema.decimal_precision = 9;
    } else if (std::is_same_v<T, numeric::decimal64>) {
      col_schema.type              = Type::INT64;
      col_schema.stats_dtype       = statistics_dtype::dtype_decimal64;
      col_schema.decimal_precision = 18;
    } else if (std::is_same_v<T, numeric::decimal128>) {
      col_schema.type              = Type::FIXED_LEN_BYTE_ARRAY;
      col_schema.type_length       = sizeof(__int128_t);
      col_schema.stats_dtype       = statistics_dtype::dtype_decimal128;
      col_schema.decimal_precision = 38;
    } else {
      CUDF_FAIL("Unsupported fixed point type for parquet writer");
    }
    col_schema.converted_type = ConvertedType::DECIMAL;
    col_schema.decimal_scale = -col->type().scale();  // parquet and cudf disagree about scale signs
    if (col_meta.is_decimal_precision_set()) {
      CUDF_EXPECTS(col_meta.get_decimal_precision() >= col_schema.decimal_scale,
                   "Precision must be equal to or greater than scale!");
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
                            bool single_write_mode)
{
  if (single_write_mode) {
    return col->nullable();
  } else {
    if (col_meta.is_nullability_defined()) {
      CUDF_EXPECTS(col_meta.nullable() || !col->nullable(),
                   "Mismatch in metadata prescribed nullability and input column nullability. "
                   "Metadata for nullable input column cannot prescribe nullability = false");
      return col_meta.nullable();
    } else {
      // For chunked write, when not provided nullability, we assume the worst case scenario
      // that all columns are nullable.
      return true;
    }
  }
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
  bool single_write_mode,
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
      bool col_nullable = is_col_nullable(col, col_meta, single_write_mode);

      auto set_field_id = [&schema, parent_idx](schema_tree_node& s,
                                                column_in_metadata const& col_meta) {
        if (schema[parent_idx].name != "list" and col_meta.is_parquet_field_id_set()) {
          s.field_id = col_meta.get_parquet_field_id();
        }
      };

      if (col->type().id() == type_id::STRUCT) {
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
        CUDF_EXPECTS(!is_col_nullable(key_col, left_child_meta, single_write_mode),
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

  [[nodiscard]] column_view leaf_column_view() const;
  [[nodiscard]] gpu::parquet_column_device_view get_device_view(rmm::cuda_stream_view stream) const;

  [[nodiscard]] column_view cudf_column_view() const { return cudf_col; }
  [[nodiscard]] parquet::Type physical_type() const { return schema_node.type; }

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
                                              UNKNOWN_NULL_COUNT,
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
  _d_nullability = cudf::detail::make_device_uvector_async(_nullability, stream);

  _is_list = (_max_rep_level > 0);

  if (cudf_col.size() == 0) { return; }

  if (_is_list) {
    // Top level column's offsets are not applied to all children. Get the effective offset and
    // size of the leaf column
    // Calculate row offset into dremel data (repetition/definition values) and the respective
    // definition and repetition levels
    gpu::dremel_data dremel = gpu::get_dremel_data(cudf_col, _d_nullability, _nullability, stream);
    _dremel_offsets         = std::move(dremel.dremel_offsets);
    _rep_level              = std::move(dremel.rep_level);
    _def_level              = std::move(dremel.def_level);
    _data_count = dremel.leaf_data_size;  // Needed for knowing what size dictionary to allocate

    stream.synchronize();
  } else {
    // For non-list struct, the size of the root column is the same as the size of the leaf column
    _data_count = cudf_col.size();
  }
}

column_view parquet_column_view::leaf_column_view() const
{
  auto col = cudf_col;
  while (cudf::is_nested(col.type())) {
    if (col.type().id() == type_id::LIST) {
      col = col.child(lists_column_view::child_column_index);
    } else if (col.type().id() == type_id::STRUCT) {
      col = col.child(0);  // Stored cudf_col has only one child if struct
    }
  }
  return col;
}

gpu::parquet_column_device_view parquet_column_view::get_device_view(
  rmm::cuda_stream_view stream) const
{
  column_view col  = leaf_column_view();
  auto desc        = gpu::parquet_column_device_view{};  // Zero out all fields
  desc.stats_dtype = schema_node.stats_dtype;
  desc.ts_scale    = schema_node.ts_scale;

  if (is_list()) {
    desc.level_offsets = _dremel_offsets.data();
    desc.rep_values    = _rep_level.data();
    desc.def_values    = _def_level.data();
  }
  desc.num_rows      = cudf_col.size();
  desc.physical_type = physical_type();

  desc.level_bits = CompactProtocolReader::NumRequiredBits(max_rep_level()) << 4 |
                    CompactProtocolReader::NumRequiredBits(max_def_level());
  desc.nullability = _d_nullability.data();
  return desc;
}

void writer::impl::init_page_fragments(cudf::detail::hostdevice_2dvector<gpu::PageFragment>& frag,
                                       device_span<gpu::parquet_column_device_view const> col_desc,
                                       host_span<partition_info const> partitions,
                                       device_span<int const> part_frag_offset,
                                       uint32_t fragment_size)
{
  auto d_partitions = cudf::detail::make_device_uvector_async(partitions, stream);
  gpu::InitPageFragments(frag, col_desc, d_partitions, part_frag_offset, fragment_size, stream);
  frag.device_to_host(stream, true);
}

void writer::impl::gather_fragment_statistics(
  device_2dspan<statistics_chunk> frag_stats_chunk,
  device_2dspan<gpu::PageFragment const> frag,
  device_span<gpu::parquet_column_device_view const> col_desc,
  uint32_t num_fragments)
{
  auto num_columns = col_desc.size();
  rmm::device_uvector<statistics_group> frag_stats_group(num_fragments * num_columns, stream);
  auto frag_stats_group_2dview =
    device_2dspan<statistics_group>(frag_stats_group.data(), num_columns, num_fragments);

  gpu::InitFragmentStatistics(frag_stats_group_2dview, frag, col_desc, stream);
  detail::calculate_group_statistics<detail::io_file_format::PARQUET>(frag_stats_chunk.data(),
                                                                      frag_stats_group.data(),
                                                                      num_fragments * num_columns,
                                                                      stream,
                                                                      int96_timestamps);
  stream.synchronize();
}

void writer::impl::init_page_sizes(hostdevice_2dvector<gpu::EncColumnChunk>& chunks,
                                   device_span<gpu::parquet_column_device_view const> col_desc,
                                   uint32_t num_columns)
{
  chunks.host_to_device(stream);
  gpu::InitEncoderPages(chunks,
                        {},
                        col_desc,
                        num_columns,
                        max_page_size_bytes,
                        max_page_size_rows,
                        nullptr,
                        nullptr,
                        0,
                        stream);
  chunks.device_to_host(stream, true);
}

auto build_chunk_dictionaries(hostdevice_2dvector<gpu::EncColumnChunk>& chunks,
                              host_span<gpu::parquet_column_device_view const> col_desc,
                              device_2dspan<gpu::PageFragment const> frags,
                              rmm::cuda_stream_view stream)
{
  // At this point, we know all chunks and their sizes. We want to allocate dictionaries for each
  // chunk that can have dictionary

  auto h_chunks = chunks.host_view().flat_view();

  std::vector<rmm::device_uvector<size_type>> dict_data;
  std::vector<rmm::device_uvector<uint16_t>> dict_index;

  if (h_chunks.size() == 0) { return std::pair(std::move(dict_data), std::move(dict_index)); }

  // Allocate slots for each chunk
  std::vector<rmm::device_uvector<gpu::slot_type>> hash_maps_storage;
  hash_maps_storage.reserve(h_chunks.size());
  for (auto& chunk : h_chunks) {
    if (col_desc[chunk.col_desc_id].physical_type == Type::BOOLEAN) {
      chunk.use_dictionary = false;
    } else {
      chunk.use_dictionary = true;
      auto& inserted_map   = hash_maps_storage.emplace_back(chunk.num_values, stream);
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
    std::tie(ck.use_dictionary, ck.dict_rle_bits) = [&]() {
      // calculate size of chunk if dictionary is used

      // If we have N unique values then the idx for the last value is N - 1 and nbits is the number
      // of bits required to encode indices into the dictionary
      auto max_dict_index = (ck.num_dict_entries > 0) ? ck.num_dict_entries - 1 : 0;
      auto nbits          = CompactProtocolReader::NumRequiredBits(max_dict_index);

      // We don't use dictionary if the indices are > 16 bits because that's the maximum bitpacking
      // bitsize we efficiently support
      if (nbits > 16) { return std::pair(false, 0); }

      // Only these bit sizes are allowed for RLE encoding because it's compute optimized
      constexpr auto allowed_bitsizes = std::array<size_type, 6>{1, 2, 4, 8, 12, 16};

      // ceil to (1/2/4/8/12/16)
      auto rle_bits = *std::lower_bound(allowed_bitsizes.begin(), allowed_bitsizes.end(), nbits);
      auto rle_byte_size = util::div_rounding_up_safe(ck.num_values * rle_bits, 8);

      auto dict_enc_size = ck.uniq_data_size + rle_byte_size;

      bool use_dict = (ck.plain_data_size > dict_enc_size);
      if (not use_dict) { rle_bits = 0; }
      return std::pair(use_dict, rle_bits);
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

void writer::impl::init_encoder_pages(hostdevice_2dvector<gpu::EncColumnChunk>& chunks,
                                      device_span<gpu::parquet_column_device_view const> col_desc,
                                      device_span<gpu::EncPage> pages,
                                      statistics_chunk* page_stats,
                                      statistics_chunk* frag_stats,
                                      size_t max_page_comp_data_size,
                                      uint32_t num_columns,
                                      uint32_t num_pages,
                                      uint32_t num_stats_bfr)
{
  rmm::device_uvector<statistics_merge_group> page_stats_mrg(num_stats_bfr, stream);
  chunks.host_to_device(stream);
  InitEncoderPages(chunks,
                   pages,
                   col_desc,
                   num_columns,
                   max_page_size_bytes,
                   max_page_size_rows,
                   (num_stats_bfr) ? page_stats_mrg.data() : nullptr,
                   (num_stats_bfr > num_pages) ? page_stats_mrg.data() + num_pages : nullptr,
                   max_page_comp_data_size,
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

void snappy_compress(device_span<device_span<uint8_t const> const> comp_in,
                     device_span<device_span<uint8_t> const> comp_out,
                     device_span<decompress_status> comp_stats,
                     size_t max_page_uncomp_data_size,
                     rmm::cuda_stream_view stream)
{
  size_t num_comp_pages = comp_in.size();
  try {
    size_t temp_size;
    nvcompStatus_t nvcomp_status = nvcompBatchedSnappyCompressGetTempSize(
      num_comp_pages, max_page_uncomp_data_size, nvcompBatchedSnappyDefaultOpts, &temp_size);

    CUDF_EXPECTS(nvcomp_status == nvcompStatus_t::nvcompSuccess,
                 "Error in getting snappy compression scratch size");

    // Not needed now but nvcomp API makes no promises about future
    rmm::device_buffer scratch(temp_size, stream);
    // Analogous to comp_in.srcDevice
    rmm::device_uvector<void const*> uncompressed_data_ptrs(num_comp_pages, stream);
    // Analogous to comp_in.srcSize
    rmm::device_uvector<size_t> uncompressed_data_sizes(num_comp_pages, stream);
    // Analogous to comp_in.dstDevice
    rmm::device_uvector<void*> compressed_data_ptrs(num_comp_pages, stream);
    // Analogous to comp_stat.bytes_written
    rmm::device_uvector<size_t> compressed_bytes_written(num_comp_pages, stream);
    // nvcomp does not currently use comp_in.dstSize. Cannot assume that the output will fit in
    // the space allocated unless one uses the API nvcompBatchedSnappyCompressGetOutputSize()

    // Prepare the vectors
    auto comp_it =
      thrust::make_zip_iterator(uncompressed_data_ptrs.begin(), uncompressed_data_sizes.begin());
    thrust::transform(
      rmm::exec_policy(stream),
      comp_in.begin(),
      comp_in.end(),
      comp_it,
      [] __device__(auto const& in) { return thrust::make_tuple(in.data(), in.size()); });

    thrust::transform(rmm::exec_policy(stream),
                      comp_out.begin(),
                      comp_out.end(),
                      compressed_data_ptrs.begin(),
                      [] __device__(auto const& out) { return out.data(); });
    nvcomp_status = nvcompBatchedSnappyCompressAsync(uncompressed_data_ptrs.data(),
                                                     uncompressed_data_sizes.data(),
                                                     max_page_uncomp_data_size,
                                                     num_comp_pages,
                                                     scratch.data(),  // Not needed rn but future
                                                     scratch.size(),
                                                     compressed_data_ptrs.data(),
                                                     compressed_bytes_written.data(),
                                                     nvcompBatchedSnappyDefaultOpts,
                                                     stream.value());

    CUDF_EXPECTS(nvcomp_status == nvcompStatus_t::nvcompSuccess, "Error in snappy compression");

    // nvcomp also doesn't use comp_out.status . It guarantees that given enough output space,
    // compression will succeed.
    // The other `comp_out` field is `reserved` which is for internal cuIO debugging and can be 0.
    thrust::transform(rmm::exec_policy(stream),
                      compressed_bytes_written.begin(),
                      compressed_bytes_written.end(),
                      comp_stats.begin(),
                      [] __device__(size_t size) {
                        decompress_status status{};
                        status.bytes_written = size;
                        return status;
                      });
    return;
  } catch (...) {
    // If we reach this then there was an error in compressing so set an error status for each page
    thrust::for_each(rmm::exec_policy(stream),
                     comp_stats.begin(),
                     comp_stats.end(),
                     [] __device__(decompress_status & stat) { stat.status = 1; });
  };
}

void writer::impl::encode_pages(hostdevice_2dvector<gpu::EncColumnChunk>& chunks,
                                device_span<gpu::EncPage> pages,
                                size_t max_page_uncomp_data_size,
                                uint32_t pages_in_batch,
                                uint32_t first_page_in_batch,
                                uint32_t rowgroups_in_batch,
                                uint32_t first_rowgroup,
                                const statistics_chunk* page_stats,
                                const statistics_chunk* chunk_stats)
{
  auto batch_pages = pages.subspan(first_page_in_batch, pages_in_batch);

  auto batch_pages_stats =
    (page_stats != nullptr)
      ? device_span<statistics_chunk const>(page_stats + first_page_in_batch, pages_in_batch)
      : device_span<statistics_chunk const>();

  uint32_t max_comp_pages =
    (compression_ != parquet::Compression::UNCOMPRESSED) ? pages_in_batch : 0;

  rmm::device_uvector<device_span<uint8_t const>> comp_in(max_comp_pages, stream);
  rmm::device_uvector<device_span<uint8_t>> comp_out(max_comp_pages, stream);
  rmm::device_uvector<decompress_status> comp_stats(max_comp_pages, stream);

  gpu::EncodePages(batch_pages, comp_in, comp_out, comp_stats, stream);
  switch (compression_) {
    case parquet::Compression::SNAPPY:
      if (nvcomp_integration::is_stable_enabled()) {
        snappy_compress(comp_in, comp_out, comp_stats, max_page_uncomp_data_size, stream);
      } else {
        gpu_snap(comp_in, comp_out, comp_stats, stream);
      }
      break;
    default: break;
  }
  // TBD: Not clear if the official spec actually allows dynamically turning off compression at the
  // chunk-level
  auto d_chunks_in_batch = chunks.device_view().subspan(first_rowgroup, rowgroups_in_batch);
  DecideCompression(d_chunks_in_batch.flat_view(), stream);
  EncodePageHeaders(batch_pages, comp_stats, batch_pages_stats, chunk_stats, stream);
  GatherPages(d_chunks_in_batch.flat_view(), pages, stream);

  auto h_chunks_in_batch = chunks.host_view().subspan(first_rowgroup, rowgroups_in_batch);
  CUDF_CUDA_TRY(cudaMemcpyAsync(h_chunks_in_batch.data(),
                                d_chunks_in_batch.data(),
                                d_chunks_in_batch.flat_view().size_bytes(),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();
}

writer::impl::impl(std::vector<std::unique_ptr<data_sink>> sinks,
                   parquet_writer_options const& options,
                   SingleWriteMode mode,
                   rmm::cuda_stream_view stream,
                   rmm::mr::device_memory_resource* mr)
  : _mr(mr),
    stream(stream),
    max_row_group_size{options.get_row_group_size_bytes()},
    max_row_group_rows{options.get_row_group_size_rows()},
    max_page_size_bytes(options.get_max_page_size_bytes()),
    max_page_size_rows(options.get_max_page_size_rows()),
    compression_(to_parquet_compression(options.get_compression())),
    stats_granularity_(options.get_stats_level()),
    int96_timestamps(options.is_enabled_int96_timestamps()),
    kv_md(options.get_key_value_metadata()),
    single_write_mode(mode == SingleWriteMode::YES),
    out_sink_(std::move(sinks))
{
  if (options.get_metadata()) {
    table_meta = std::make_unique<table_input_metadata>(*options.get_metadata());
  }
  init_state();
}

writer::impl::impl(std::vector<std::unique_ptr<data_sink>> sinks,
                   chunked_parquet_writer_options const& options,
                   SingleWriteMode mode,
                   rmm::cuda_stream_view stream,
                   rmm::mr::device_memory_resource* mr)
  : _mr(mr),
    stream(stream),
    max_row_group_size{options.get_row_group_size_bytes()},
    max_row_group_rows{options.get_row_group_size_rows()},
    max_page_size_bytes(options.get_max_page_size_bytes()),
    max_page_size_rows(options.get_max_page_size_rows()),
    compression_(to_parquet_compression(options.get_compression())),
    stats_granularity_(options.get_stats_level()),
    int96_timestamps(options.is_enabled_int96_timestamps()),
    kv_md(options.get_key_value_metadata()),
    single_write_mode(mode == SingleWriteMode::YES),
    out_sink_(std::move(sinks))
{
  if (options.get_metadata()) {
    table_meta = std::make_unique<table_input_metadata>(*options.get_metadata());
  }
  init_state();
}

writer::impl::~impl() { close(); }

void writer::impl::init_state()
{
  current_chunk_offset.resize(out_sink_.size());
  // Write file header
  file_header_s fhdr;
  fhdr.magic = parquet_magic;
  for (auto& sink : out_sink_) {
    sink->host_write(&fhdr, sizeof(fhdr));
  }
  std::fill_n(current_chunk_offset.begin(), current_chunk_offset.size(), sizeof(file_header_s));
}

void writer::impl::write(table_view const& table, std::vector<partition_info> const& partitions)
{
  last_write_successful = false;
  CUDF_EXPECTS(not closed, "Data has already been flushed to out and closed");

  if (not table_meta) { table_meta = std::make_unique<table_input_metadata>(table); }

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

  auto vec         = table_to_linked_columns(table);
  auto schema_tree = construct_schema_tree(vec, *table_meta, single_write_mode, int96_timestamps);
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

  if (!md) {
    md = std::make_unique<aggregate_writer_metadata>(
      partitions, num_columns, std::move(this_table_schema), stats_granularity_, kv_md);
  } else {
    // verify the user isn't passing mismatched tables
    CUDF_EXPECTS(md->schema_matches(this_table_schema),
                 "Mismatch in schema between multiple calls to write_chunk");

    md->update_files(partitions);
  }
  // Create table_device_view so that corresponding column_device_view data
  // can be written into col_desc members
  auto parent_column_table_device_view = table_device_view::create(single_streams_table, stream);
  rmm::device_uvector<column_device_view> leaf_column_views(0, stream);

  // Initialize column description
  hostdevice_vector<gpu::parquet_column_device_view> col_desc(parquet_columns.size(), stream);
  std::transform(
    parquet_columns.begin(), parquet_columns.end(), col_desc.host_ptr(), [&](auto const& pcol) {
      return pcol.get_device_view(stream);
    });

  // Init page fragments
  // 5000 is good enough for up to ~200-character strings. Longer strings will start producing
  // fragments larger than the desired page size -> TODO: keep track of the max fragment size, and
  // iteratively reduce this value if the largest fragment exceeds the max page size limit (we
  // ideally want the page size to be below 1MB so as to have enough pages to get good
  // compression/decompression performance).
  using cudf::io::parquet::gpu::max_page_fragment_size;

  std::vector<int> num_frag_in_part;
  std::transform(partitions.begin(),
                 partitions.end(),
                 std::back_inserter(num_frag_in_part),
                 [](auto const& part) {
                   return util::div_rounding_up_unsafe(part.num_rows, max_page_fragment_size);
                 });

  size_type num_fragments = std::reduce(num_frag_in_part.begin(), num_frag_in_part.end());

  std::vector<int> part_frag_offset;  // Store the idx of the first fragment in each partition
  std::exclusive_scan(
    num_frag_in_part.begin(), num_frag_in_part.end(), std::back_inserter(part_frag_offset), 0);
  part_frag_offset.push_back(part_frag_offset.back() + num_frag_in_part.back());

  auto d_part_frag_offset = cudf::detail::make_device_uvector_async(part_frag_offset, stream);
  cudf::detail::hostdevice_2dvector<gpu::PageFragment> fragments(
    num_columns, num_fragments, stream);

  if (num_fragments != 0) {
    // Move column info to device
    col_desc.host_to_device(stream);
    leaf_column_views = create_leaf_column_device_views<gpu::parquet_column_device_view>(
      col_desc, *parent_column_table_device_view, stream);

    init_page_fragments(
      fragments, col_desc, partitions, d_part_frag_offset, max_page_fragment_size);
  }

  std::vector<size_t> const global_rowgroup_base = md->num_row_groups_per_file();

  // Decide row group boundaries based on uncompressed data size
  int num_rowgroups = 0;

  std::vector<int> num_rg_in_part(partitions.size());
  for (size_t p = 0; p < partitions.size(); ++p) {
    size_type curr_rg_num_rows = 0;
    size_t curr_rg_data_size   = 0;
    int first_frag_in_rg       = part_frag_offset[p];
    int last_frag_in_part      = part_frag_offset[p + 1] - 1;
    for (auto f = first_frag_in_rg; f <= last_frag_in_part; ++f) {
      size_t fragment_data_size = 0;
      for (auto c = 0; c < num_columns; c++) {
        fragment_data_size += fragments[c][f].fragment_data_size;
      }
      size_type fragment_num_rows = fragments[0][f].num_rows;

      // If the fragment size gets larger than rg limit then break off a rg
      if (f > first_frag_in_rg &&  // There has to be at least one fragment in row group
          (curr_rg_data_size + fragment_data_size > max_row_group_size ||
           curr_rg_num_rows + fragment_num_rows > max_row_group_rows)) {
        auto& rg    = md->file(p).row_groups.emplace_back();
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
        auto& rg    = md->file(p).row_groups.emplace_back();
        rg.num_rows = curr_rg_num_rows;
        num_rowgroups++;
        num_rg_in_part[p]++;
      }
    }
  }

  // Allocate column chunks and gather fragment statistics
  rmm::device_uvector<statistics_chunk> frag_stats(0, stream);
  if (stats_granularity_ != statistics_freq::STATISTICS_NONE) {
    frag_stats.resize(num_fragments * num_columns, stream);
    if (not frag_stats.is_empty()) {
      auto frag_stats_2dview =
        device_2dspan<statistics_chunk>(frag_stats.data(), num_columns, num_fragments);
      gather_fragment_statistics(frag_stats_2dview, fragments, col_desc, num_fragments);
    }
  }

  std::vector<int> first_rg_in_part;
  std::exclusive_scan(
    num_rg_in_part.begin(), num_rg_in_part.end(), std::back_inserter(first_rg_in_part), 0);

  // Initialize row groups and column chunks
  auto const num_chunks = num_rowgroups * num_columns;
  hostdevice_2dvector<gpu::EncColumnChunk> chunks(num_rowgroups, num_columns, stream);

  for (size_t p = 0; p < partitions.size(); ++p) {
    int f               = part_frag_offset[p];
    size_type start_row = partitions[p].start_row;
    for (int r = 0; r < num_rg_in_part[p]; r++) {
      size_t global_r = global_rowgroup_base[p] + r;  // Number of rowgroups already in file/part
      auto& row_group = md->file(p).row_groups[global_r];
      uint32_t fragments_in_chunk =
        util::div_rounding_up_unsafe(row_group.num_rows, max_page_fragment_size);
      row_group.total_byte_size = 0;
      row_group.columns.resize(num_columns);
      for (int c = 0; c < num_columns; c++) {
        gpu::EncColumnChunk& ck = chunks[r + first_rg_in_part[p]][c];

        ck             = {};
        ck.col_desc    = col_desc.device_ptr() + c;
        ck.col_desc_id = c;
        ck.fragments   = &fragments.device_view()[c][f];
        ck.stats =
          (not frag_stats.is_empty()) ? frag_stats.data() + c * num_fragments + f : nullptr;
        ck.start_row         = start_row;
        ck.num_rows          = (uint32_t)row_group.num_rows;
        ck.first_fragment    = c * num_fragments + f;
        auto chunk_fragments = fragments[c].subspan(f, fragments_in_chunk);
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
      }
      f += fragments_in_chunk;
      start_row += (uint32_t)row_group.num_rows;
    }
  }

  fragments.host_to_device(stream);
  auto dict_info_owner = build_chunk_dictionaries(chunks, col_desc, fragments, stream);
  for (size_t p = 0; p < partitions.size(); p++) {
    for (int rg = 0; rg < num_rg_in_part[p]; rg++) {
      size_t global_rg = global_rowgroup_base[p] + rg;
      for (int col = 0; col < num_columns; col++) {
        if (chunks.host_view()[rg][col].use_dictionary) {
          md->file(p).row_groups[global_rg].columns[col].meta_data.encodings.push_back(
            Encoding::PLAIN_DICTIONARY);
        }
      }
    }
  }

  // Build chunk dictionaries and count pages
  if (num_chunks != 0) { init_page_sizes(chunks, col_desc, num_columns); }

  // Get the maximum page size across all chunks
  size_type max_page_uncomp_data_size =
    std::accumulate(chunks.host_view().flat_view().begin(),
                    chunks.host_view().flat_view().end(),
                    0,
                    [](uint32_t max_page_size, gpu::EncColumnChunk const& chunk) {
                      return std::max(max_page_size, chunk.max_page_data_size);
                    });

  size_t max_page_comp_data_size = 0;
  if (compression_ != parquet::Compression::UNCOMPRESSED) {
    auto status = nvcompBatchedSnappyCompressGetMaxOutputChunkSize(
      max_page_uncomp_data_size, nvcompBatchedSnappyDefaultOpts, &max_page_comp_data_size);
    CUDF_EXPECTS(status == nvcompStatus_t::nvcompSuccess,
                 "Error in getting compressed size from nvcomp");
  }

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
        ck->compressed_size =
          ck->ck_stat_size + ck->page_headers_size + max_page_comp_data_size * ck->num_pages;
        comp_rowgroup_size += ck->compressed_size;
        max_chunk_bfr_size =
          std::max(max_chunk_bfr_size, (size_t)std::max(ck->bfr_size, ck->compressed_size));
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
  if (compression_ == parquet::Compression::UNCOMPRESSED) { max_comp_bfr_size = 0; }

  // Initialize data pointers in batch
  uint32_t num_stats_bfr =
    (stats_granularity_ != statistics_freq::STATISTICS_NONE) ? num_pages + num_chunks : 0;
  rmm::device_buffer uncomp_bfr(max_uncomp_bfr_size, stream);
  rmm::device_buffer comp_bfr(max_comp_bfr_size, stream);
  rmm::device_uvector<gpu::EncPage> pages(num_pages, stream);

  // This contains stats for both the pages and the rowgroups. TODO: make them separate.
  rmm::device_uvector<statistics_chunk> page_stats(num_stats_bfr, stream);
  for (auto b = 0, r = 0; b < static_cast<size_type>(batch_list.size()); b++) {
    auto bfr   = static_cast<uint8_t*>(uncomp_bfr.data());
    auto bfr_c = static_cast<uint8_t*>(comp_bfr.data());
    for (auto j = 0; j < batch_list[b]; j++, r++) {
      for (auto i = 0; i < num_columns; i++) {
        gpu::EncColumnChunk& ck = chunks[r][i];
        ck.uncompressed_bfr     = bfr;
        ck.compressed_bfr       = bfr_c;
        bfr += ck.bfr_size;
        bfr_c += ck.compressed_size;
      }
    }
  }

  if (num_pages != 0) {
    init_encoder_pages(chunks,
                       col_desc,
                       {pages.data(), pages.size()},
                       (num_stats_bfr) ? page_stats.data() : nullptr,
                       (num_stats_bfr) ? frag_stats.data() : nullptr,
                       max_page_comp_data_size,
                       num_columns,
                       num_pages,
                       num_stats_bfr);
  }

  pinned_buffer<uint8_t> host_bfr{nullptr, cudaFreeHost};

  // Encode row groups in batches
  for (auto b = 0, r = 0; b < static_cast<size_type>(batch_list.size()); b++) {
    // Count pages in this batch
    auto const rnext               = r + batch_list[b];
    auto const first_page_in_batch = chunks[r][0].first_page;
    auto const first_page_in_next_batch =
      (rnext < num_rowgroups) ? chunks[rnext][0].first_page : num_pages;
    auto const pages_in_batch = first_page_in_next_batch - first_page_in_batch;
    // device_span<gpu::EncPage> batch_pages{pages.data() + first_page_in_batch, }
    encode_pages(
      chunks,
      {pages.data(), pages.size()},
      max_page_uncomp_data_size,
      pages_in_batch,
      first_page_in_batch,
      batch_list[b],
      r,
      (stats_granularity_ == statistics_freq::STATISTICS_PAGE) ? page_stats.data() : nullptr,
      (stats_granularity_ != statistics_freq::STATISTICS_NONE) ? page_stats.data() + num_pages
                                                               : nullptr);
    std::vector<std::future<void>> write_tasks;
    for (; r < rnext; r++) {
      int p           = rg_to_part[r];
      int global_r    = global_rowgroup_base[p] + r - first_rg_in_part[p];
      auto& row_group = md->file(p).row_groups[global_r];
      for (auto i = 0; i < num_columns; i++) {
        gpu::EncColumnChunk& ck = chunks[r][i];
        auto& column_chunk_meta = row_group.columns[i].meta_data;
        uint8_t* dev_bfr;
        if (ck.is_compressed) {
          column_chunk_meta.codec = compression_;
          dev_bfr                 = ck.compressed_bfr;
        } else {
          dev_bfr = ck.uncompressed_bfr;
        }

        if (out_sink_[p]->is_device_write_preferred(ck.compressed_size)) {
          // let the writer do what it wants to retrieve the data from the gpu.
          write_tasks.push_back(out_sink_[p]->device_write_async(
            dev_bfr + ck.ck_stat_size, ck.compressed_size, stream));
          // we still need to do a (much smaller) memcpy for the statistics.
          if (ck.ck_stat_size != 0) {
            column_chunk_meta.statistics_blob.resize(ck.ck_stat_size);
            CUDF_CUDA_TRY(cudaMemcpyAsync(column_chunk_meta.statistics_blob.data(),
                                          dev_bfr,
                                          ck.ck_stat_size,
                                          cudaMemcpyDeviceToHost,
                                          stream.value()));
            stream.synchronize();
          }
        } else {
          if (!host_bfr) {
            host_bfr = pinned_buffer<uint8_t>{[](size_t size) {
                                                uint8_t* ptr = nullptr;
                                                CUDF_CUDA_TRY(cudaMallocHost(&ptr, size));
                                                return ptr;
                                              }(max_chunk_bfr_size),
                                              cudaFreeHost};
          }
          // copy the full data
          CUDF_CUDA_TRY(cudaMemcpyAsync(host_bfr.get(),
                                        dev_bfr,
                                        ck.ck_stat_size + ck.compressed_size,
                                        cudaMemcpyDeviceToHost,
                                        stream.value()));
          stream.synchronize();
          out_sink_[p]->host_write(host_bfr.get() + ck.ck_stat_size, ck.compressed_size);
          if (ck.ck_stat_size != 0) {
            column_chunk_meta.statistics_blob.resize(ck.ck_stat_size);
            memcpy(column_chunk_meta.statistics_blob.data(), host_bfr.get(), ck.ck_stat_size);
          }
        }
        row_group.total_byte_size += ck.compressed_size;
        column_chunk_meta.data_page_offset =
          current_chunk_offset[p] + ((ck.use_dictionary) ? ck.dictionary_size : 0);
        column_chunk_meta.dictionary_page_offset =
          (ck.use_dictionary) ? current_chunk_offset[p] : 0;
        column_chunk_meta.total_uncompressed_size = ck.bfr_size;
        column_chunk_meta.total_compressed_size   = ck.compressed_size;
        current_chunk_offset[p] += ck.compressed_size;
      }
    }
    for (auto const& task : write_tasks) {
      task.wait();
    }
  }
  last_write_successful = true;
}

std::unique_ptr<std::vector<uint8_t>> writer::impl::close(
  std::vector<std::string> const& column_chunks_file_path)
{
  if (closed) { return nullptr; }
  closed = true;
  if (not last_write_successful) { return nullptr; }
  for (size_t p = 0; p < out_sink_.size(); p++) {
    std::vector<uint8_t> buffer;
    CompactProtocolWriter cpw(&buffer);
    file_ender_s fendr;
    buffer.resize(0);
    fendr.footer_len = static_cast<uint32_t>(cpw.write(md->get_metadata(p)));
    fendr.magic      = parquet_magic;
    out_sink_[p]->host_write(buffer.data(), buffer.size());
    out_sink_[p]->host_write(&fendr, sizeof(fendr));
    out_sink_[p]->flush();
  }

  // Optionally output raw file metadata with the specified column chunk file path
  if (column_chunks_file_path.size() > 0) {
    CUDF_EXPECTS(column_chunks_file_path.size() == md->num_files(),
                 "Expected one column chunk path per output file");
    md->set_file_paths(column_chunks_file_path);
    file_header_s fhdr = {parquet_magic};
    std::vector<uint8_t> buffer;
    CompactProtocolWriter cpw(&buffer);
    buffer.insert(buffer.end(),
                  reinterpret_cast<const uint8_t*>(&fhdr),
                  reinterpret_cast<const uint8_t*>(&fhdr) + sizeof(fhdr));
    file_ender_s fendr;
    fendr.magic      = parquet_magic;
    fendr.footer_len = static_cast<uint32_t>(cpw.write(md->get_merged_metadata()));
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
               SingleWriteMode mode,
               rmm::cuda_stream_view stream,
               rmm::mr::device_memory_resource* mr)
  : _impl(std::make_unique<impl>(std::move(sinks), options, mode, stream, mr))
{
}

writer::writer(std::vector<std::unique_ptr<data_sink>> sinks,
               chunked_parquet_writer_options const& options,
               SingleWriteMode mode,
               rmm::cuda_stream_view stream,
               rmm::mr::device_memory_resource* mr)
  : _impl(std::make_unique<impl>(std::move(sinks), options, mode, stream, mr))
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
