/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
 * @file parquet_metadata.hpp
 * @brief cuDF-IO freeform API
 */

#pragma once

#include <cudf/io/types.hpp>

#include <optional>
#include <string_view>
#include <variant>
#include <vector>

namespace cudf {
namespace io {
/**
 * @addtogroup io_types
 * @{
 * @file
 */

//! Parquet I/O interfaces
namespace parquet {
/**
 * @brief Basic data types in Parquet, determines how data is physically stored
 */
enum class TypeKind : int8_t {
  UNDEFINED_TYPE       = -1,  // Undefined for non-leaf nodes
  BOOLEAN              = 0,
  INT32                = 1,
  INT64                = 2,
  INT96                = 3,  // Deprecated
  FLOAT                = 4,
  DOUBLE               = 5,
  BYTE_ARRAY           = 6,
  FIXED_LEN_BYTE_ARRAY = 7,
};
}  // namespace parquet

/**
 * @brief Schema of a parquet column, including the nested columns.
 */
struct parquet_column_schema {
 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit parquet_column_schema() = default;

  /**
   * @brief constructor
   *
   * @param name column name
   * @param type parquet type
   * @param children child columns (empty for non-nested types)
   */
  parquet_column_schema(std::string_view name,
                        parquet::TypeKind type,
                        std::vector<parquet_column_schema> children)
    : _name{name}, _type_kind{type}, _children{std::move(children)}
  {
  }

  /**
   * @brief Returns parquet column name; can be empty.
   *
   * @return Column name
   */
  [[nodiscard]] auto name() const { return _name; }

  /**
   * @brief Returns parquet type of the column.
   *
   * @return Column parquet type
   */
  [[nodiscard]] auto type_kind() const { return _type_kind; }

  /**
   * @brief Returns schemas of all child columns.
   *
   * @return Children schemas
   */
  [[nodiscard]] auto const& children() const& { return _children; }

  /** @copydoc children
   * Children array is moved out of the object (rvalues only).
   *
   */
  [[nodiscard]] auto children() && { return std::move(_children); }

  /**
   * @brief Returns schema of the child with the given index.
   *
   * @param idx child index
   *
   * @return Child schema
   */
  [[nodiscard]] auto const& child(int idx) const& { return children().at(idx); }

  /** @copydoc child
   * Child is moved out of the object (rvalues only).
   *
   */
  [[nodiscard]] auto child(int idx) && { return std::move(children().at(idx)); }

  /**
   * @brief Returns the number of child columns.
   *
   * @return Children count
   */
  [[nodiscard]] auto num_children() const { return children().size(); }

 private:
  std::string _name;
  // 3 types available: Physical, Converted, Logical.
  parquet::TypeKind _type_kind;  // Physical
  std::vector<parquet_column_schema> _children;
};

/**
 * @brief Schema of a parquet file.
 */
struct parquet_schema {
 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit parquet_schema() = default;

  /**
   * @brief constructor
   *
   * @param root_column_schema root column
   */
  parquet_schema(parquet_column_schema root_column_schema) : _root{std::move(root_column_schema)} {}

  /**
   * @brief Returns the schema of the struct column that contains all columns as fields.
   *
   * @return Root column schema
   */
  [[nodiscard]] auto const& root() const& { return _root; }

  /** @copydoc root
   * Root column schema is moved out of the object (rvalues only).
   *
   */
  [[nodiscard]] auto root() && { return std::move(_root); }

 private:
  parquet_column_schema _root;
};

/**
 * @brief Information about content of a parquet file.
 */
class parquet_metadata {
 public:
  /// Key-value metadata in the file footer.
  using key_value_metadata = std::unordered_map<std::string, std::string>;
  /// row group metadata from each RowGroup element.
  using row_group_metadata = std::unordered_map<std::string, int64_t>;

  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit parquet_metadata() = default;

  /**
   * @brief constructor
   *
   * @param schema parquet schema
   * @param num_rows number of rows
   * @param num_rowgroups number of row groups
   * @param file_metadata key-value metadata in the file footer
   * @param rg_metadata vector of maps containing metadata for each row group
   */
  parquet_metadata(parquet_schema schema,
                   int64_t num_rows,
                   size_type num_rowgroups,
                   key_value_metadata file_metadata,
                   std::vector<row_group_metadata> rg_metadata)
    : _schema{std::move(schema)},
      _num_rows{num_rows},
      _num_rowgroups{num_rowgroups},
      _file_metadata{std::move(file_metadata)},
      _rowgroup_metadata{std::move(rg_metadata)}
  {
  }

  /**
   * @brief Returns the parquet schema.
   *
   * @return parquet schema
   */
  [[nodiscard]] auto const& schema() const { return _schema; }

  /**
   * @brief Returns the number of rows of the root column.
   *
   * If a file contains list columns, nested columns can have a different number of rows.
   *
   * @return Number of rows
   */
  [[nodiscard]] auto num_rows() const { return _num_rows; }

  /**
   * @brief Returns the number of rowgroups in the file.
   *
   * @return Number of row groups
   */
  [[nodiscard]] auto num_rowgroups() const { return _num_rowgroups; }

  /**
   * @brief Returns the Key value metadata in the file footer.
   *
   * @return Key value metadata as a map
   */
  [[nodiscard]] auto const& metadata() const { return _file_metadata; }

  /**
   * @brief Returns the row group metadata in the file footer.
   *
   * @return vector of row group metadata as maps
   */
  [[nodiscard]] auto const& rowgroup_metadata() const { return _rowgroup_metadata; }

 private:
  parquet_schema _schema;
  int64_t _num_rows;
  size_type _num_rowgroups;
  key_value_metadata _file_metadata;
  std::vector<row_group_metadata> _rowgroup_metadata;
};

/**
 * @brief Reads metadata of parquet dataset.
 *
 * @ingroup io_readers
 *
 * @param src_info Dataset source
 *
 * @return parquet_metadata with parquet schema, number of rows, number of row groups and key-value
 * metadata.
 */
parquet_metadata read_parquet_metadata(source_info const& src_info);

/** @} */  // end of group
}  // namespace io
}  // namespace cudf
