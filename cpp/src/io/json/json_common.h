/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/cudf.h>
// #include <io/csv/legacy/type_conversion.cuh>

class SerialTrieNode;

namespace cudf {
namespace io {
namespace json {

struct ColumnInfo {
  cudf::size_type float_count;
  cudf::size_type datetime_count;
  cudf::size_type string_count;
  cudf::size_type int_count;
  cudf::size_type bool_count;
  cudf::size_type null_count;
};

/**
 * @brief Structure for holding various options used when parsing and
 * converting CSV data to cuDF data type values.
 */
struct ParseOptions {
  char delimiter;
  char terminator;
  char quotechar;
  char decimal;
  char thousands;
  char comment;
  bool keepquotes;
  bool doublequote;
  bool dayfirst;
  bool skipblanklines;
  SerialTrieNode* trueValuesTrie;
  SerialTrieNode* falseValuesTrie;
  SerialTrieNode* naValuesTrie;
  bool multi_delimiter;
};

}  // namespace json
}  // namespace io
}  // namespace cudf
