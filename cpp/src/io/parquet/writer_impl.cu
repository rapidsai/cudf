/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <io/statistics/column_statistics.cuh>
#include "writer_impl.hpp"

#include <io/utilities/column_utils.cuh>
#include "compact_protocol_writer.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

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
    default:
      CUDF_EXPECTS(false, "Unsupported compression type");
      return parquet::Compression::UNCOMPRESSED;
  }
}

}  // namespace

struct linked_column_view;

using LinkedColPtr    = std::shared_ptr<linked_column_view>;
using LinkedColVector = std::vector<LinkedColPtr>;

/**
 * @brief column_view with the added member pointer to the parent of this column.
 *
 */
struct linked_column_view : public column_view {
  // TODO(cp): we are currently keeping all column_view children info multiple times - once for each
  //       copy of this object. Options:
  // 1. Inherit from column_view_base. Only lose out on children vector. That is not needed.
  // 2. Don't inherit at all. make linked_column_view keep a reference wrapper to its column_view
  linked_column_view(column_view const& col) : column_view(col), parent(nullptr)
  {
    for (auto child_it = col.child_begin(); child_it < col.child_end(); ++child_it) {
      children.push_back(std::make_shared<linked_column_view>(this, *child_it));
    }
  }

  linked_column_view(linked_column_view* parent, column_view const& col)
    : column_view(col), parent(parent)
  {
    for (auto child_it = col.child_begin(); child_it < col.child_end(); ++child_it) {
      children.push_back(std::make_shared<linked_column_view>(this, *child_it));
    }
  }

  linked_column_view* parent;  //!< Pointer to parent of this column. Nullptr if root
  LinkedColVector children;
};

/**
 * @brief Converts all column_views of a table into linked_column_views
 *
 * @param table table of columns to convert
 * @return Vector of converted linked_column_views
 */
LinkedColVector input_table_to_linked_columns(table_view const& table)
{
  LinkedColVector result;
  for (column_view const& col : table) {
    result.emplace_back(std::make_shared<linked_column_view>(col));
  }

  return result;
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
  LinkedColPtr leaf_column;
  statistics_dtype stats_dtype;
  int32_t ts_scale;

  // TODO(fut): Think about making schema a class that holds a vector of schema_tree_nodes. The
  // function construct_schema_tree could be its constructor. It can have method to get the per
  // column nullability given a schema node index corresponding to a leaf schema. Much easier than
  // that is a method to get path in schema, given a leaf node
};

struct leaf_schema_fn {
  schema_tree_node& col_schema;
  LinkedColPtr const& col;
  column_in_metadata const& col_meta;
  bool timestamp_is_int96;

  template <typename T>
  std::enable_if_t<std::is_same<T, bool>::value, void> operator()()
  {
    col_schema.type        = Type::BOOLEAN;
    col_schema.stats_dtype = statistics_dtype::dtype_bool;
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, int8_t>::value, void> operator()()
  {
    col_schema.type           = Type::INT32;
    col_schema.converted_type = ConvertedType::INT_8;
    col_schema.stats_dtype    = statistics_dtype::dtype_int8;
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, int16_t>::value, void> operator()()
  {
    col_schema.type           = Type::INT32;
    col_schema.converted_type = ConvertedType::INT_16;
    col_schema.stats_dtype    = statistics_dtype::dtype_int16;
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, int32_t>::value, void> operator()()
  {
    col_schema.type        = Type::INT32;
    col_schema.stats_dtype = statistics_dtype::dtype_int32;
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, int64_t>::value, void> operator()()
  {
    col_schema.type        = Type::INT64;
    col_schema.stats_dtype = statistics_dtype::dtype_int64;
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, uint8_t>::value, void> operator()()
  {
    col_schema.type           = Type::INT32;
    col_schema.converted_type = ConvertedType::UINT_8;
    col_schema.stats_dtype    = statistics_dtype::dtype_int8;
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, uint16_t>::value, void> operator()()
  {
    col_schema.type           = Type::INT32;
    col_schema.converted_type = ConvertedType::UINT_16;
    col_schema.stats_dtype    = statistics_dtype::dtype_int16;
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, uint32_t>::value, void> operator()()
  {
    col_schema.type           = Type::INT32;
    col_schema.converted_type = ConvertedType::UINT_32;
    col_schema.stats_dtype    = statistics_dtype::dtype_int32;
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, uint64_t>::value, void> operator()()
  {
    col_schema.type           = Type::INT64;
    col_schema.converted_type = ConvertedType::UINT_64;
    col_schema.stats_dtype    = statistics_dtype::dtype_int64;
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, float>::value, void> operator()()
  {
    col_schema.type        = Type::FLOAT;
    col_schema.stats_dtype = statistics_dtype::dtype_float32;
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, double>::value, void> operator()()
  {
    col_schema.type        = Type::DOUBLE;
    col_schema.stats_dtype = statistics_dtype::dtype_float64;
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, cudf::string_view>::value, void> operator()()
  {
    col_schema.type           = Type::BYTE_ARRAY;
    col_schema.converted_type = ConvertedType::UTF8;
    col_schema.stats_dtype    = statistics_dtype::dtype_string;
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, cudf::timestamp_D>::value, void> operator()()
  {
    col_schema.type           = Type::INT32;
    col_schema.converted_type = ConvertedType::DATE;
    col_schema.stats_dtype    = statistics_dtype::dtype_int32;
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, cudf::timestamp_s>::value, void> operator()()
  {
    col_schema.type = (timestamp_is_int96) ? Type::INT96 : Type::INT64;
    col_schema.converted_type =
      (timestamp_is_int96) ? ConvertedType::UNKNOWN : ConvertedType::TIMESTAMP_MILLIS;
    col_schema.stats_dtype = statistics_dtype::dtype_timestamp64;
    col_schema.ts_scale    = 1000;
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, cudf::timestamp_ms>::value, void> operator()()
  {
    col_schema.type = (timestamp_is_int96) ? Type::INT96 : Type::INT64;
    col_schema.converted_type =
      (timestamp_is_int96) ? ConvertedType::UNKNOWN : ConvertedType::TIMESTAMP_MILLIS;
    col_schema.stats_dtype = statistics_dtype::dtype_timestamp64;
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, cudf::timestamp_us>::value, void> operator()()
  {
    col_schema.type = (timestamp_is_int96) ? Type::INT96 : Type::INT64;
    col_schema.converted_type =
      (timestamp_is_int96) ? ConvertedType::UNKNOWN : ConvertedType::TIMESTAMP_MICROS;
    col_schema.stats_dtype = statistics_dtype::dtype_timestamp64;
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, cudf::timestamp_ns>::value, void> operator()()
  {
    col_schema.type = (timestamp_is_int96) ? Type::INT96 : Type::INT64;
    col_schema.converted_type =
      (timestamp_is_int96) ? ConvertedType::UNKNOWN : ConvertedType::TIMESTAMP_MICROS;
    col_schema.stats_dtype = statistics_dtype::dtype_timestamp64;
    col_schema.ts_scale    = -1000;  // negative value indicates division by absolute value
  }

  //  unsupported outside cudf for parquet 1.0.
  template <typename T>
  std::enable_if_t<std::is_same<T, cudf::duration_D>::value, void> operator()()
  {
    col_schema.type           = Type::INT32;
    col_schema.converted_type = ConvertedType::TIME_MILLIS;
    col_schema.stats_dtype    = statistics_dtype::dtype_int64;
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, cudf::duration_s>::value, void> operator()()
  {
    col_schema.type           = Type::INT64;
    col_schema.converted_type = ConvertedType::TIME_MILLIS;
    col_schema.stats_dtype    = statistics_dtype::dtype_int64;
    col_schema.ts_scale       = 1000;
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, cudf::duration_ms>::value, void> operator()()
  {
    col_schema.type           = Type::INT64;
    col_schema.converted_type = ConvertedType::TIME_MILLIS;
    col_schema.stats_dtype    = statistics_dtype::dtype_int64;
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, cudf::duration_us>::value, void> operator()()
  {
    col_schema.type           = Type::INT64;
    col_schema.converted_type = ConvertedType::TIME_MICROS;
    col_schema.stats_dtype    = statistics_dtype::dtype_int64;
  }

  //  unsupported outside cudf for parquet 1.0.
  template <typename T>
  std::enable_if_t<std::is_same<T, cudf::duration_ns>::value, void> operator()()
  {
    col_schema.type           = Type::INT64;
    col_schema.converted_type = ConvertedType::TIME_MICROS;
    col_schema.stats_dtype    = statistics_dtype::dtype_int64;
    col_schema.ts_scale       = -1000;  // negative value indicates division by absolute value
  }

  template <typename T>
  std::enable_if_t<cudf::is_fixed_point<T>(), void> operator()()
  {
    if (std::is_same<T, numeric::decimal32>::value) {
      col_schema.type        = Type::INT32;
      col_schema.stats_dtype = statistics_dtype::dtype_int32;
    } else if (std::is_same<T, numeric::decimal64>::value) {
      col_schema.type        = Type::INT64;
      col_schema.stats_dtype = statistics_dtype::dtype_decimal64;
    } else {
      CUDF_FAIL("Unsupported fixed point type for parquet writer");
    }
    col_schema.converted_type = ConvertedType::DECIMAL;
    col_schema.decimal_scale = -col->type().scale();  // parquet and cudf disagree about scale signs
    CUDF_EXPECTS(col_meta.is_decimal_precision_set(),
                 "Precision must be specified for decimal columns");
    CUDF_EXPECTS(col_meta.get_decimal_precision() >= col_schema.decimal_scale,
                 "Precision must be equal to or greater than scale!");
    col_schema.decimal_precision = col_meta.get_decimal_precision();
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

/**
 * @brief Construct schema from input columns and per-column input options
 *
 * Recursively traverses through linked_columns and corresponding metadata to construct schema tree.
 * The resulting schema tree is stored in a vector in pre-order traversal order.
 */
std::vector<schema_tree_node> construct_schema_tree(LinkedColVector const& linked_columns,
                                                    table_input_metadata const& metadata,
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

  std::function<void(LinkedColPtr const&, column_in_metadata const&, size_t)> add_schema =
    [&](LinkedColPtr const& col, column_in_metadata const& col_meta, size_t parent_idx) {
      bool col_nullable = [&]() {
        if (single_write_mode) {
          return col->nullable();
        } else {
          if (col_meta.is_nullability_defined()) {
            if (col_meta.nullable() == false) {
              CUDF_EXPECTS(
                col->nullable() == false,
                "Mismatch in metadata prescribed nullability and input column nullability. "
                "Metadata for nullable input column cannot prescribe nullability = false");
            }
            return col_meta.nullable();
          } else {
            // For chunked write, when not provided nullability, we assume the worst case scenario
            // that all columns are nullable.
            return true;
          }
        }
      }();

      if (col->type().id() == type_id::STRUCT) {
        // if struct, add current and recursively call for all children
        schema_tree_node struct_schema{};
        struct_schema.repetition_type =
          col_nullable ? FieldRepetitionType::OPTIONAL : FieldRepetitionType::REQUIRED;

        struct_schema.name = (schema[parent_idx].name == "list") ? "element" : col_meta.get_name();
        struct_schema.num_children = col->num_children();
        struct_schema.parent_idx   = parent_idx;
        schema.push_back(std::move(struct_schema));

        auto struct_node_index = schema.size() - 1;
        // for (auto child_it = col->children.begin(); child_it < col->children.end(); child_it++) {
        //   add_schema(*child_it, struct_node_index);
        // }
        CUDF_EXPECTS(col->num_children() == static_cast<int>(col_meta.num_children()),
                     "Mismatch in number of child columns between input table and metadata");
        for (size_t i = 0; i < col->children.size(); ++i) {
          add_schema(col->children[i], col_meta.child(i), struct_node_index);
        }
      } else if (col->type().id() == type_id::LIST) {
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

  column_view leaf_column_view() const;
  gpu::parquet_column_device_view get_device_view(rmm::cuda_stream_view stream);

  column_view cudf_column_view() const { return cudf_col; }
  parquet::Type physical_type() const { return schema_node.type; }

  std::vector<std::string> const& get_path_in_schema() { return path_in_schema; }

  // LIST related member functions
  uint8_t max_def_level() const noexcept { return _max_def_level; }
  uint8_t max_rep_level() const noexcept { return _max_rep_level; }
  bool is_list() const noexcept { return _is_list; }

  // Dictionary related member functions
  uint32_t* get_dict_data() { return (_dict_data.size()) ? _dict_data.data() : nullptr; }
  uint32_t* get_dict_index() { return (_dict_index.size()) ? _dict_index.data() : nullptr; }
  void use_dictionary(bool use_dict) { _dictionary_used = use_dict; }
  void alloc_dictionary(size_t max_num_rows, rmm::cuda_stream_view stream)
  {
    _dict_data.resize(max_num_rows, stream);
    _dict_index.resize(max_num_rows, stream);
  }
  bool check_dictionary_used(rmm::cuda_stream_view stream)
  {
    if (!_dictionary_used) {
      _dict_data.resize(0, stream);
      _dict_data.shrink_to_fit(stream);
      _dict_index.resize(0, stream);
      _dict_index.shrink_to_fit(stream);
    }
    return _dictionary_used;
  }

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

  // Dictionary related members
  bool _dictionary_used = false;
  rmm::device_uvector<uint32_t> _dict_data;
  rmm::device_uvector<uint32_t> _dict_index;
};

parquet_column_view::parquet_column_view(schema_tree_node const& schema_node,
                                         std::vector<schema_tree_node> const& schema_tree,
                                         rmm::cuda_stream_view stream)
  : schema_node(schema_node),
    _d_nullability(0, stream),
    _dremel_offsets(0, stream),
    _rep_level(0, stream),
    _def_level(0, stream),
    _dict_data(0, stream),
    _dict_index(0, stream)
{
  // Construct single inheritance column_view from linked_column_view
  auto curr_col                           = schema_node.leaf_column.get();
  column_view single_inheritance_cudf_col = *curr_col;
  while (curr_col->parent) {
    auto const& parent = *curr_col->parent;

    // For list columns, we still need to retain the offset child column.
    auto children =
      (parent.type().id() == type_id::LIST)
        ? std::vector<column_view>{parent.child(lists_column_view::offsets_column_index),
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
  _d_nullability = rmm::device_uvector<uint8_t>(_nullability.size(), stream);
  CUDA_TRY(cudaMemcpyAsync(_d_nullability.data(),
                           _nullability.data(),
                           _nullability.size() * sizeof(uint8_t),
                           cudaMemcpyHostToDevice,
                           stream.value()));

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

gpu::parquet_column_device_view parquet_column_view::get_device_view(rmm::cuda_stream_view stream)
{
  column_view col  = leaf_column_view();
  auto desc        = gpu::parquet_column_device_view{};  // Zero out all fields
  desc.stats_dtype = schema_node.stats_dtype;
  desc.ts_scale    = schema_node.ts_scale;

  // TODO (dm): Enable dictionary for list and struct after refactor
  if (physical_type() != BOOLEAN && physical_type() != UNDEFINED_TYPE &&
      !is_nested(cudf_col.type())) {
    alloc_dictionary(_data_count, stream);
    desc.dict_index = get_dict_index();
    desc.dict_data  = get_dict_data();
  }

  if (is_list()) {
    desc.level_offsets = _dremel_offsets.data();
    desc.rep_values    = _rep_level.data();
    desc.def_values    = _def_level.data();
  }
  desc.num_rows      = cudf_col.size();
  desc.physical_type = static_cast<uint8_t>(physical_type());
  auto count_bits    = [](uint16_t number) {
    int16_t nbits = 0;
    while (number > 0) {
      nbits++;
      number >>= 1;
    }
    return nbits;
  };
  desc.level_bits  = count_bits(max_rep_level()) << 4 | count_bits(max_def_level());
  desc.nullability = _d_nullability.data();
  return desc;
}

void writer::impl::init_page_fragments(cudf::detail::hostdevice_2dvector<gpu::PageFragment>& frag,
                                       device_span<gpu::parquet_column_device_view const> col_desc,
                                       uint32_t num_rows,
                                       uint32_t fragment_size)
{
  gpu::InitPageFragments(frag, col_desc, fragment_size, num_rows, stream);
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
  detail::calculate_group_statistics<detail::io_file_format::PARQUET>(
    frag_stats_chunk.data(), frag_stats_group.data(), num_fragments * num_columns, stream);
  stream.synchronize();
}

void writer::impl::build_chunk_dictionaries(
  hostdevice_2dvector<gpu::EncColumnChunk>& chunks,
  device_span<gpu::parquet_column_device_view const> col_desc,
  uint32_t num_columns,
  uint32_t num_dictionaries)
{
  chunks.host_to_device(stream);
  if (num_dictionaries > 0) {
    size_t dict_scratch_size = (size_t)num_dictionaries * gpu::kDictScratchSize;
    auto dict_scratch        = cudf::detail::make_zeroed_device_uvector_async<uint32_t>(
      dict_scratch_size / sizeof(uint32_t), stream);

    gpu::BuildChunkDictionaries(chunks.device_view().flat_view(), dict_scratch.data(), stream);
  }
  gpu::InitEncoderPages(chunks, {}, col_desc, num_columns, nullptr, nullptr, stream);
  chunks.device_to_host(stream, true);
}

void writer::impl::init_encoder_pages(hostdevice_2dvector<gpu::EncColumnChunk>& chunks,
                                      device_span<gpu::parquet_column_device_view const> col_desc,
                                      device_span<gpu::EncPage> pages,
                                      statistics_chunk* page_stats,
                                      statistics_chunk* frag_stats,
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

void writer::impl::encode_pages(hostdevice_2dvector<gpu::EncColumnChunk>& chunks,
                                device_span<gpu::EncPage> pages,
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

  rmm::device_uvector<gpu_inflate_input_s> compression_input(max_comp_pages, stream);
  rmm::device_uvector<gpu_inflate_status_s> compression_status(max_comp_pages, stream);

  device_span<gpu_inflate_input_s> comp_in{compression_input.data(), compression_input.size()};
  device_span<gpu_inflate_status_s> comp_stat{compression_status.data(), compression_status.size()};

  gpu::EncodePages(batch_pages, comp_in, comp_stat, stream);
  switch (compression_) {
    case parquet::Compression::SNAPPY:
      CUDA_TRY(gpu_snap(comp_in.data(), comp_stat.data(), pages_in_batch, stream));
      break;
    default: break;
  }
  // TBD: Not clear if the official spec actually allows dynamically turning off compression at the
  // chunk-level
  auto d_chunks_in_batch = chunks.device_view().subspan(first_rowgroup, rowgroups_in_batch);
  DecideCompression(d_chunks_in_batch.flat_view(), stream);
  EncodePageHeaders(batch_pages, comp_stat, batch_pages_stats, chunk_stats, stream);
  GatherPages(d_chunks_in_batch.flat_view(), pages, stream);

  auto h_chunks_in_batch = chunks.host_view().subspan(first_rowgroup, rowgroups_in_batch);
  CUDA_TRY(cudaMemcpyAsync(h_chunks_in_batch.data(),
                           d_chunks_in_batch.data(),
                           d_chunks_in_batch.flat_view().size_bytes(),
                           cudaMemcpyDeviceToHost,
                           stream.value()));
  stream.synchronize();
}

writer::impl::impl(std::unique_ptr<data_sink> sink,
                   parquet_writer_options const& options,
                   SingleWriteMode mode,
                   rmm::cuda_stream_view stream,
                   rmm::mr::device_memory_resource* mr)
  : _mr(mr),
    stream(stream),
    compression_(to_parquet_compression(options.get_compression())),
    stats_granularity_(options.get_stats_level()),
    int96_timestamps(options.is_enabled_int96_timestamps()),
    out_sink_(std::move(sink)),
    single_write_mode(mode == SingleWriteMode::YES)
{
  if (options.get_metadata()) {
    table_meta = std::make_unique<table_input_metadata>(*options.get_metadata());
  }
  init_state();
}

writer::impl::impl(std::unique_ptr<data_sink> sink,
                   chunked_parquet_writer_options const& options,
                   SingleWriteMode mode,
                   rmm::cuda_stream_view stream,
                   rmm::mr::device_memory_resource* mr)
  : _mr(mr),
    stream(stream),
    compression_(to_parquet_compression(options.get_compression())),
    stats_granularity_(options.get_stats_level()),
    int96_timestamps(options.is_enabled_int96_timestamps()),
    single_write_mode(mode == SingleWriteMode::YES),
    out_sink_(std::move(sink))
{
  if (options.get_metadata()) {
    table_meta = std::make_unique<table_input_metadata>(*options.get_metadata());
  }
  init_state();
}

writer::impl::~impl() { close(); }

void writer::impl::init_state()
{
  // Write file header
  file_header_s fhdr;
  fhdr.magic = parquet_magic;
  out_sink_->host_write(&fhdr, sizeof(fhdr));
  current_chunk_offset = sizeof(file_header_s);
}

void writer::impl::write(table_view const& table)
{
  CUDF_EXPECTS(not closed, "Data has already been flushed to out and closed");

  size_type num_rows = table.num_rows();

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

  auto vec         = input_table_to_linked_columns(table);
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

  if (md.version == 0) {
    md.version  = 1;
    md.num_rows = num_rows;
    md.column_order_listsize =
      (stats_granularity_ != statistics_freq::STATISTICS_NONE) ? num_columns : 0;
    std::transform(table_meta->user_data.begin(),
                   table_meta->user_data.end(),
                   std::back_inserter(md.key_value_metadata),
                   [](auto const& kv) {
                     return KeyValue{kv.first, kv.second};
                   });
    md.schema = this_table_schema;
  } else {
    // verify the user isn't passing mismatched tables
    CUDF_EXPECTS(md.schema == this_table_schema,
                 "Mismatch in schema between multiple calls to write_chunk");

    // increment num rows
    md.num_rows += num_rows;
  }
  // Create table_device_view so that corresponding column_device_view data
  // can be written into col_desc members
  auto parent_column_table_device_view = table_device_view::create(single_streams_table, stream);
  rmm::device_uvector<column_device_view> leaf_column_views(0, stream);

  // Initialize column description
  hostdevice_vector<gpu::parquet_column_device_view> col_desc(parquet_columns.size(), stream);
  // This should've been `auto const&` but isn't since dictionary space is allocated when calling
  // get_device_view(). Fix during dictionary refactor.
  std::transform(
    parquet_columns.begin(), parquet_columns.end(), col_desc.host_ptr(), [&](auto& pcol) {
      return pcol.get_device_view(stream);
    });

  // Init page fragments
  // 5000 is good enough for up to ~200-character strings. Longer strings will start producing
  // fragments larger than the desired page size -> TODO: keep track of the max fragment size, and
  // iteratively reduce this value if the largest fragment exceeds the max page size limit (we
  // ideally want the page size to be below 1MB so as to have enough pages to get good
  // compression/decompression performance).
  using cudf::io::parquet::gpu::max_page_fragment_size;
  constexpr uint32_t fragment_size = 5000;
  static_assert(fragment_size <= max_page_fragment_size,
                "fragment size cannot be greater than max_page_fragment_size");

  uint32_t num_fragments = (uint32_t)((num_rows + fragment_size - 1) / fragment_size);
  cudf::detail::hostdevice_2dvector<gpu::PageFragment> fragments(
    num_columns, num_fragments, stream);

  if (num_fragments != 0) {
    // Move column info to device
    col_desc.host_to_device(stream);
    leaf_column_views = create_leaf_column_device_views<gpu::parquet_column_device_view>(
      col_desc, *parent_column_table_device_view, stream);

    init_page_fragments(fragments, col_desc, num_rows, fragment_size);
  }

  size_t global_rowgroup_base = md.row_groups.size();

  // Decide row group boundaries based on uncompressed data size
  size_t rowgroup_size   = 0;
  uint32_t num_rowgroups = 0;
  for (uint32_t f = 0, global_r = global_rowgroup_base, rowgroup_start = 0; f < num_fragments;
       f++) {
    size_t fragment_data_size = 0;
    // Replace with STL algorithm to transform and sum
    for (auto i = 0; i < num_columns; i++) {
      fragment_data_size += fragments[i][f].fragment_data_size;
    }
    if (f > rowgroup_start && (rowgroup_size + fragment_data_size > max_rowgroup_size_ ||
                               (f + 1 - rowgroup_start) * fragment_size > max_rowgroup_rows_)) {
      // update schema
      md.row_groups.resize(md.row_groups.size() + 1);
      md.row_groups[global_r++].num_rows = (f - rowgroup_start) * fragment_size;
      num_rowgroups++;
      rowgroup_start = f;
      rowgroup_size  = 0;
    }
    rowgroup_size += fragment_data_size;
    if (f + 1 == num_fragments) {
      // update schema
      md.row_groups.resize(md.row_groups.size() + 1);
      md.row_groups[global_r++].num_rows = num_rows - rowgroup_start * fragment_size;
      num_rowgroups++;
    }
  }

  // Allocate column chunks and gather fragment statistics
  rmm::device_uvector<statistics_chunk> frag_stats(0, stream);
  if (stats_granularity_ != statistics_freq::STATISTICS_NONE) {
    frag_stats.resize(num_fragments * num_columns, stream);
    if (frag_stats.size() != 0) {
      auto frag_stats_2dview =
        device_2dspan<statistics_chunk>(frag_stats.data(), num_columns, num_fragments);
      gather_fragment_statistics(frag_stats_2dview, fragments, col_desc, num_fragments);
    }
  }
  // Initialize row groups and column chunks
  uint32_t num_chunks = num_rowgroups * num_columns;
  hostdevice_2dvector<gpu::EncColumnChunk> chunks(num_rowgroups, num_columns, stream);
  uint32_t num_dictionaries = 0;
  for (uint32_t r = 0, global_r = global_rowgroup_base, f = 0, start_row = 0; r < num_rowgroups;
       r++, global_r++) {
    uint32_t fragments_in_chunk =
      (uint32_t)((md.row_groups[global_r].num_rows + fragment_size - 1) / fragment_size);
    md.row_groups[global_r].total_byte_size = 0;
    md.row_groups[global_r].columns.resize(num_columns);
    for (int i = 0; i < num_columns; i++) {
      gpu::EncColumnChunk* ck = &chunks[r][i];
      bool dict_enable        = false;

      *ck           = {};
      ck->col_desc  = col_desc.device_ptr() + i;
      ck->fragments = &fragments.device_view()[i][f];
      ck->stats = (frag_stats.size() != 0) ? frag_stats.data() + i * num_fragments + f : nullptr;
      ck->start_row        = start_row;
      ck->num_rows         = (uint32_t)md.row_groups[global_r].num_rows;
      ck->first_fragment   = i * num_fragments + f;
      auto chunk_fragments = fragments[i].subspan(f, fragments_in_chunk);
      ck->num_values =
        std::accumulate(chunk_fragments.begin(), chunk_fragments.end(), 0, [](uint32_t l, auto r) {
          return l + r.num_values;
        });
      ck->dictionary_id = num_dictionaries;
      if (col_desc[i].dict_data) {
        size_t plain_size      = 0;
        size_t dict_size       = 1;
        uint32_t num_dict_vals = 0;
        for (uint32_t j = 0; j < fragments_in_chunk && num_dict_vals < 65536; j++) {
          plain_size += chunk_fragments[j].fragment_data_size;
          dict_size += chunk_fragments[j].dict_data_size +
                       ((num_dict_vals > 256) ? 2 : 1) * chunk_fragments[j].non_nulls;
          num_dict_vals += chunk_fragments[j].num_dict_vals;
        }
        if (dict_size < plain_size) {
          parquet_columns[i].use_dictionary(true);
          dict_enable = true;
          num_dictionaries++;
        }
      }
      ck->has_dictionary                                     = dict_enable;
      md.row_groups[global_r].columns[i].meta_data.type      = parquet_columns[i].physical_type();
      md.row_groups[global_r].columns[i].meta_data.encodings = {Encoding::PLAIN, Encoding::RLE};
      if (dict_enable) {
        md.row_groups[global_r].columns[i].meta_data.encodings.push_back(
          Encoding::PLAIN_DICTIONARY);
      }
      md.row_groups[global_r].columns[i].meta_data.path_in_schema =
        parquet_columns[i].get_path_in_schema();
      md.row_groups[global_r].columns[i].meta_data.codec      = UNCOMPRESSED;
      md.row_groups[global_r].columns[i].meta_data.num_values = ck->num_values;
    }
    f += fragments_in_chunk;
    start_row += (uint32_t)md.row_groups[global_r].num_rows;
  }

  // Free unused dictionaries
  for (auto& col : parquet_columns) {
    col.check_dictionary_used(stream);
  }

  // Build chunk dictionaries and count pages
  if (num_chunks != 0) {
    build_chunk_dictionaries(chunks, col_desc, num_columns, num_dictionaries);
  }

  // Initialize batches of rowgroups to encode (mainly to limit peak memory usage)
  std::vector<uint32_t> batch_list;
  uint32_t num_pages          = 0;
  size_t max_bytes_in_batch   = 1024 * 1024 * 1024;  // 1GB - TBD: Tune this
  size_t max_uncomp_bfr_size  = 0;
  size_t max_chunk_bfr_size   = 0;
  uint32_t max_pages_in_batch = 0;
  size_t bytes_in_batch       = 0;
  for (uint32_t r = 0, groups_in_batch = 0, pages_in_batch = 0; r <= num_rowgroups; r++) {
    size_t rowgroup_size = 0;
    if (r < num_rowgroups) {
      for (int i = 0; i < num_columns; i++) {
        gpu::EncColumnChunk* ck = &chunks[r][i];
        ck->first_page          = num_pages;
        num_pages += ck->num_pages;
        pages_in_batch += ck->num_pages;
        rowgroup_size += ck->bfr_size;
        max_chunk_bfr_size =
          std::max(max_chunk_bfr_size, (size_t)std::max(ck->bfr_size, ck->compressed_size));
      }
    }
    // TBD: We may want to also shorten the batch if we have enough pages (not just based on size)
    if ((r == num_rowgroups) ||
        (groups_in_batch != 0 && bytes_in_batch + rowgroup_size > max_bytes_in_batch)) {
      max_uncomp_bfr_size = std::max(max_uncomp_bfr_size, bytes_in_batch);
      max_pages_in_batch  = std::max(max_pages_in_batch, pages_in_batch);
      if (groups_in_batch != 0) {
        batch_list.push_back(groups_in_batch);
        groups_in_batch = 0;
      }
      bytes_in_batch = 0;
      pages_in_batch = 0;
    }
    bytes_in_batch += rowgroup_size;
    groups_in_batch++;
  }

  // Initialize data pointers in batch
  size_t max_comp_bfr_size =
    (compression_ != parquet::Compression::UNCOMPRESSED)
      ? gpu::GetMaxCompressedBfrSize(max_uncomp_bfr_size, max_pages_in_batch)
      : 0;
  uint32_t num_stats_bfr =
    (stats_granularity_ != statistics_freq::STATISTICS_NONE) ? num_pages + num_chunks : 0;
  rmm::device_buffer uncomp_bfr(max_uncomp_bfr_size, stream);
  rmm::device_buffer comp_bfr(max_comp_bfr_size, stream);
  rmm::device_uvector<gpu::EncPage> pages(num_pages, stream);

  // This contains stats for both the pages and the rowgroups. TODO: make them separate.
  rmm::device_uvector<statistics_chunk> page_stats(num_stats_bfr, stream);
  for (uint32_t b = 0, r = 0; b < (uint32_t)batch_list.size(); b++) {
    uint8_t* bfr   = static_cast<uint8_t*>(uncomp_bfr.data());
    uint8_t* bfr_c = static_cast<uint8_t*>(comp_bfr.data());
    for (uint32_t j = 0; j < batch_list[b]; j++, r++) {
      for (int i = 0; i < num_columns; i++) {
        gpu::EncColumnChunk* ck = &chunks[r][i];
        ck->uncompressed_bfr    = bfr;
        ck->compressed_bfr      = bfr_c;
        bfr += ck->bfr_size;
        bfr_c += ck->compressed_size;
      }
    }
  }

  if (num_pages != 0) {
    init_encoder_pages(chunks,
                       col_desc,
                       {pages.data(), pages.size()},
                       (num_stats_bfr) ? page_stats.data() : nullptr,
                       (num_stats_bfr) ? frag_stats.data() : nullptr,
                       num_columns,
                       num_pages,
                       num_stats_bfr);
  }

  pinned_buffer<uint8_t> host_bfr{nullptr, cudaFreeHost};

  // Encode row groups in batches
  for (uint32_t b = 0, r = 0, global_r = global_rowgroup_base; b < (uint32_t)batch_list.size();
       b++) {
    // Count pages in this batch
    uint32_t rnext               = r + batch_list[b];
    uint32_t first_page_in_batch = chunks[r][0].first_page;
    uint32_t first_page_in_next_batch =
      (rnext < num_rowgroups) ? chunks[rnext][0].first_page : num_pages;
    uint32_t pages_in_batch = first_page_in_next_batch - first_page_in_batch;
    // device_span<gpu::EncPage> batch_pages{pages.data() + first_page_in_batch, }
    encode_pages(
      chunks,
      {pages.data(), pages.size()},
      pages_in_batch,
      first_page_in_batch,
      batch_list[b],
      r,
      (stats_granularity_ == statistics_freq::STATISTICS_PAGE) ? page_stats.data() : nullptr,
      (stats_granularity_ != statistics_freq::STATISTICS_NONE) ? page_stats.data() + num_pages
                                                               : nullptr);
    for (; r < rnext; r++, global_r++) {
      for (auto i = 0; i < num_columns; i++) {
        gpu::EncColumnChunk* ck = &chunks[r][i];
        uint8_t* dev_bfr;
        if (ck->is_compressed) {
          md.row_groups[global_r].columns[i].meta_data.codec = compression_;
          dev_bfr                                            = ck->compressed_bfr;
        } else {
          dev_bfr = ck->uncompressed_bfr;
        }

        if (out_sink_->is_device_write_preferred(ck->compressed_size)) {
          // let the writer do what it wants to retrieve the data from the gpu.
          out_sink_->device_write(dev_bfr + ck->ck_stat_size, ck->compressed_size, stream);
          // we still need to do a (much smaller) memcpy for the statistics.
          if (ck->ck_stat_size != 0) {
            md.row_groups[global_r].columns[i].meta_data.statistics_blob.resize(ck->ck_stat_size);
            CUDA_TRY(
              cudaMemcpyAsync(md.row_groups[global_r].columns[i].meta_data.statistics_blob.data(),
                              dev_bfr,
                              ck->ck_stat_size,
                              cudaMemcpyDeviceToHost,
                              stream.value()));
            stream.synchronize();
          }
        } else {
          if (!host_bfr) {
            host_bfr = pinned_buffer<uint8_t>{[](size_t size) {
                                                uint8_t* ptr = nullptr;
                                                CUDA_TRY(cudaMallocHost(&ptr, size));
                                                return ptr;
                                              }(max_chunk_bfr_size),
                                              cudaFreeHost};
          }
          // copy the full data
          CUDA_TRY(cudaMemcpyAsync(host_bfr.get(),
                                   dev_bfr,
                                   ck->ck_stat_size + ck->compressed_size,
                                   cudaMemcpyDeviceToHost,
                                   stream.value()));
          stream.synchronize();
          out_sink_->host_write(host_bfr.get() + ck->ck_stat_size, ck->compressed_size);
          if (ck->ck_stat_size != 0) {
            md.row_groups[global_r].columns[i].meta_data.statistics_blob.resize(ck->ck_stat_size);
            memcpy(md.row_groups[global_r].columns[i].meta_data.statistics_blob.data(),
                   host_bfr.get(),
                   ck->ck_stat_size);
          }
        }
        md.row_groups[global_r].total_byte_size += ck->compressed_size;
        md.row_groups[global_r].columns[i].meta_data.data_page_offset =
          current_chunk_offset + ((ck->has_dictionary) ? ck->dictionary_size : 0);
        md.row_groups[global_r].columns[i].meta_data.dictionary_page_offset =
          (ck->has_dictionary) ? current_chunk_offset : 0;
        md.row_groups[global_r].columns[i].meta_data.total_uncompressed_size = ck->bfr_size;
        md.row_groups[global_r].columns[i].meta_data.total_compressed_size   = ck->compressed_size;
        current_chunk_offset += ck->compressed_size;
      }
    }
  }
}

std::unique_ptr<std::vector<uint8_t>> writer::impl::close(
  std::string const& column_chunks_file_path)
{
  if (closed) { return nullptr; }
  closed = true;
  CompactProtocolWriter cpw(&buffer_);
  file_ender_s fendr;
  buffer_.resize(0);
  fendr.footer_len = static_cast<uint32_t>(cpw.write(md));
  fendr.magic      = parquet_magic;
  out_sink_->host_write(buffer_.data(), buffer_.size());
  out_sink_->host_write(&fendr, sizeof(fendr));
  out_sink_->flush();

  // Optionally output raw file metadata with the specified column chunk file path
  if (column_chunks_file_path.length() > 0) {
    file_header_s fhdr = {parquet_magic};
    buffer_.resize(0);
    buffer_.insert(buffer_.end(),
                   reinterpret_cast<const uint8_t*>(&fhdr),
                   reinterpret_cast<const uint8_t*>(&fhdr) + sizeof(fhdr));
    for (auto& rowgroup : md.row_groups) {
      for (auto& col : rowgroup.columns) {
        col.file_path = column_chunks_file_path;
      }
    }
    fendr.footer_len = static_cast<uint32_t>(cpw.write(md));
    buffer_.insert(buffer_.end(),
                   reinterpret_cast<const uint8_t*>(&fendr),
                   reinterpret_cast<const uint8_t*>(&fendr) + sizeof(fendr));
    return std::make_unique<std::vector<uint8_t>>(std::move(buffer_));
  } else {
    return {nullptr};
  }
}

// Forward to implementation
writer::writer(std::unique_ptr<data_sink> sink,
               parquet_writer_options const& options,
               SingleWriteMode mode,
               rmm::cuda_stream_view stream,
               rmm::mr::device_memory_resource* mr)
  : _impl(std::make_unique<impl>(std::move(sink), options, mode, stream, mr))
{
}

writer::writer(std::unique_ptr<data_sink> sink,
               chunked_parquet_writer_options const& options,
               SingleWriteMode mode,
               rmm::cuda_stream_view stream,
               rmm::mr::device_memory_resource* mr)
  : _impl(std::make_unique<impl>(std::move(sink), options, mode, stream, mr))
{
}

// Destructor within this translation unit
writer::~writer() = default;

// Forward to implementation
void writer::write(table_view const& table) { _impl->write(table); }

// Forward to implementation
std::unique_ptr<std::vector<uint8_t>> writer::close(std::string const& column_chunks_file_path)
{
  return _impl->close(column_chunks_file_path);
}

std::unique_ptr<std::vector<uint8_t>> writer::merge_rowgroup_metadata(
  const std::vector<std::unique_ptr<std::vector<uint8_t>>>& metadata_list)
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
