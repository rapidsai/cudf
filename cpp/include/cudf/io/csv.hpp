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

#include <cudf/io/types.hpp>

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudf {
namespace io {

/**
 *@breif Builder to build options for `read_csv()`.
 */
class csv_reader_options_builder;

/**
 * @brief Settings to use for `read_csv()`.
 *
 * @ingroup io_readers
 */

class csv_reader_options {
  source_info _source;

  // Read settings

  // Specify the compression format of the source or infer from file extension
  compression_type _compression = compression_type::AUTO;
  // Bytes to skip from the source start
  size_t _byte_range_offset = 0;
  // Bytes to read; always reads complete rows
  size_t _byte_range_size = 0;
  // Names of all the columns; if empty then names are auto-generated
  std::vector<std::string> _names;
  // If there is no header or names, prepend this to the column ID as the name
  std::string _prefix;
  // Whether to rename duplicate column names
  bool _mangle_dupe_cols = true;

  // Filter settings

  // Names of columns to read; empty is all columns
  std::vector<std::string> _use_cols_names;
  // Indexes of columns to read; empty is all columns
  std::vector<int> _use_cols_indexes;
  // Rows to read; -1 is all
  size_type _nrows = -1;
  // Rows to skip from the start
  size_type _skiprows = 0;
  // Rows to skip from the end
  size_type _skipfooter = 0;
  // Header row index
  size_type _header = 0;

  // Parsing settings

  // Line terminator
  char _lineterminator = '\n';
  // Field delimiter
  char _delimiter = ',';
  // Numeric data thousands separator; cannot match delimiter
  char _thousands = '\0';
  // Decimal point character; cannot match delimiter
  char _decimal = '.';
  // Comment line start character
  char _comment                = '\0';
  bool _windowslinetermination = false;
  // Treat whitespace as field delimiter; overrides character delimiter
  bool _delim_whitespace = false;
  // Skip whitespace after the delimiter
  bool _skipinitialspace = false;
  // Ignore empty lines or parse line values as invalid
  bool _skip_blank_lines = true;
  // Treatment of quoting behavior
  quote_style _quoting = quote_style::MINIMAL;
  // Quoting character (if `quoting` is true)
  char _quotechar = '"';
  // Whether a quote inside a value is double-quoted
  bool _doublequote = true;
  // Names of columns to read as datetime
  std::vector<std::string> _infer_date_names;
  // Indexes of columns to read as datetime
  std::vector<int> _infer_date_indexes;

  // Conversion settings

  // Per-column types; disables type inference on those columns
  std::vector<std::string> _dtypes;
  // Additional values to recognize as boolean true values
  std::vector<std::string> _true_values{"True", "TRUE", "true"};
  // Additional values to recognize as boolean false values
  std::vector<std::string> _false_values{"False", "FALSE", "false"};
  // Additional values to recognize as null values
  std::vector<std::string> _na_values{"#N/A",
                                      "#N/A N/A",
                                      "#NA",
                                      "-1.#IND",
                                      "-1.#QNAN",
                                      "-NaN",
                                      "-nan",
                                      "1.#IND",
                                      "1.#QNAN",
                                      "N/A",
                                      "NA",
                                      "NULL",
                                      "NaN",
                                      "n/a",
                                      "nan",
                                      "null"};

  // Whether to keep the built-in default NA values
  bool _keep_default_na = true;
  // Whether to disable null filter; disabling can improve performance
  bool _na_filter = true;
  // Whether to parse dates as DD/MM versus MM/DD
  bool _dayfirst = false;
  // Cast timestamp columns to a specific type
  data_type _timestamp_type{type_id::EMPTY};

  /**
   * @brief Constructor from source info.
   *
   * @param src source information used to read csv file.
   */
  explicit csv_reader_options(source_info const& src) : _source(src) {}

  friend csv_reader_options_builder;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  csv_reader_options() = default;

  /**
   * @brief Creates a `csv_reader_options_builder` which will build `csv_reader_options`.
   *
   * @param src Source information to read csv file.
   * @return Builder to build reader options.
   */
  static csv_reader_options_builder builder(source_info const& src);

  /**
   * @brief Returns source info.
   */
  source_info const& get_source() const { return _source; }

  /**
   * @brief Returns compression format of the source.
   */
  compression_type get_compression() const { return _compression; }

  /**
   * @brief Returns number of bytes to skip from source start.
   */
  size_t get_byte_range_offset() const { return _byte_range_offset; }

  /**
   * @brief Returns number of bytes to read.
   */
  size_t get_byte_range_size() const { return _byte_range_size; }

  /**
   * @brief Returns names of the columns.
   */
  std::vector<std::string> const& get_names() const { return _names; }

  /**
   * @brief Returns prefix to be used for column ID.
   */
  std::string get_prefix() const { return _prefix; }

  /**
   * @brief Whether to rename duplicate column names.
   */
  bool is_enabled_mangle_dupe_cols() const { return _mangle_dupe_cols; }

  /**
   * @brief Returns names of the columns to be read.
   */
  std::vector<std::string> const& get_use_cols_names() const { return _use_cols_names; }

  /**
   * @brief Returns indexes of columns to read.
   */
  std::vector<int> const& get_use_cols_indexes() const { return _use_cols_indexes; }

  /**
   * @brief Returns number of rows to read.
   */
  size_type get_nrows() const { return _nrows; }

  /**
   * @brief Returns number of rows to skip from start.
   */
  size_type get_skiprows() const { return _skiprows; }

  /**
   * @brief Returns number of rows to skip from end.
   */
  size_type get_skipfooter() const { return _skipfooter; }

  /**
   * @brief Returns header row index.
   */
  size_type get_header() const { return _header; }

  /**
   * @brief Returns line terminator.
   */
  char get_lineterminator() const { return _lineterminator; }

  /**
   * @brief Returns field delimiter.
   */
  char get_delimiter() const { return _delimiter; }

  /**
   * @brief Returns numeric data thousands separator.
   */
  char get_thousands() const { return _thousands; }

  /**
   * @brief Returns decimal point character.
   */
  char get_decimal() const { return _decimal; }

  /**
   * @brief Returns comment line start character.
   */
  char get_comment() const { return _comment; }

  /**
   * @brief Whether to treat `\r\n` as line terminator.
   */
  bool is_enabled_windowslinetermination() const { return _windowslinetermination; }

  /**
   * @brief Whether to treat whitespace as field delimiter.
   */
  bool is_enabled_delim_whitespace() const { return _delim_whitespace; }

  /**
   * @brief Whether to skip whitespace after the delimiter.
   */
  bool is_enabled_skipinitialspace() const { return _skipinitialspace; }

  /**
   * @brief Whether to ignore empty lines or parse line values as invalid.
   */
  bool is_enabled_skip_blank_lines() const { return _skip_blank_lines; }

  /**
   * @brief Returns quoting style.
   */
  quote_style get_quoting() const { return _quoting; }

  /**
   * @brief Returns quoting character.
   */
  char get_quotechar() const { return _quotechar; }

  /**
   * @brief Whether a quote inside a value is double-quoted.
   */
  bool is_enabled_doublequote() const { return _doublequote; }

  /**
   * @brief Returns names of columns to read as datetime.
   */
  std::vector<std::string> const& get_infer_date_names() const { return _infer_date_names; }

  /**
   * @brief Returns indexes of columns to read as datetime.
   */
  std::vector<int> const& get_infer_date_indexes() const { return _infer_date_indexes; }

  /**
   * @brief Returns per-column types.
   */
  std::vector<std::string> const& get_dtypes() const { return _dtypes; }

  /**
   * @brief Returns additional values to recognize as boolean true values.
   */
  std::vector<std::string> const& get_true_values() const { return _true_values; }

  /**
   * @brief Returns additional values to recognize as boolean false values.
   */
  std::vector<std::string> const& get_false_values() const { return _false_values; }

  /**
   * @brief Returns additional values to recognize as null values.
   */
  std::vector<std::string> const& get_na_values() const { return _na_values; }

  /**
   * @brief Whether to keep the built-in default NA values.
   */
  bool is_enabled_keep_default_na() const { return _keep_default_na; }

  /**
   * @brief Whether to disable null filter.
   */
  bool is_enabled_na_filter() const { return _na_filter; }

  /**
   * @brief Whether to parse dates as DD/MM versus MM/DD.
   */
  bool is_enabled_dayfirst() const { return _dayfirst; }

  /**
   * @brief Returns timestamp_type to which all timestamp columns will be cast.
   */
  data_type get_timestamp_type() const { return _timestamp_type; }

  /**
   * @brief Sets compression format of the source.
   *
   * @param comp Compression type.
   */
  void set_compression(compression_type comp) { _compression = comp; }

  /**
   * @brief Sets number of bytes to skip from source start.
   *
   * @param offset Number of bytes of offset.
   */
  void set_byte_range_offset(size_type offset)
  {
    if ((offset != 0) and ((_skiprows != 0) or (_skipfooter != 0) or (_nrows != -1))) {
      CUDF_FAIL(
        "When ther is valid value in skiprows or skipfooter or nrows, offset can't have non-zero "
        "value");
    }
    _byte_range_offset = offset;
  }

  /**
   * @brief Sets number of bytes to read.
   *
   * @param size Number of bytes to read.
   */
  void set_byte_range_size(size_type size)
  {
    if ((size != 0) and ((_skiprows != 0) or (_skipfooter != 0) or (_nrows != -1))) {
      CUDF_FAIL(
        "When ther is valid value in skiprows or skipfooter or nrows, range size can't have "
        "non-zero value");
    }
    _byte_range_size = size;
  }

  /**
   * @brief Sets names of the column.
   *
   * @param col_names Vector of column names.
   */
  void set_names(std::vector<std::string> col_names) { _names = std::move(col_names); }

  /**
   * @brief Sets prefix to be used for column ID.
   *
   * @param String used as prefix in for each column name.
   */
  void set_prefix(std::string pfx) { _prefix = pfx; }

  /**
   * @brief Sets whether to rename duplicate column names.
   *
   * @param val Boolean value to enable/disable.
   */
  void enable_mangle_dupe_cols(bool val) { _mangle_dupe_cols = val; }

  /**
   * @brief Sets names of the columns to be read.
   *
   * @param col_names Vector of column names that are needed.
   */
  void set_use_cols_names(std::vector<std::string> col_names)
  {
    _use_cols_names = std::move(col_names);
  }

  /**
   * @brief Sets indexes of columns to read.
   *
   * @param col_ind Vector of column indices that are needed.
   */
  void set_use_cols_indexes(std::vector<int> col_ind) { _use_cols_indexes = std::move(col_ind); }

  /**
   * @brief Sets number of rows to read.
   *
   * @param val Number of rows to read.
   */
  void set_nrows(size_type nrows)
  {
    CUDF_EXPECTS((nrows == 0) or (_skipfooter == 0), "Cannot use both `nrows` and `skipfooter`");
    if ((nrows != -1) and ((_byte_range_offset != 0) or (_byte_range_size != 0))) {
      CUDF_FAIL(
        "nrows can't be a non negative value if range offset and/or range size has been set");
    }

    _nrows = nrows;
  }

  /**
   * @brief Sets number of rows to skip from start.
   *
   * @param val Number of rows to skip.
   */
  void set_skiprows(size_type skip)
  {
    if ((skip != 0) and ((_byte_range_offset != 0) or (_byte_range_size != 0))) {
      CUDF_FAIL(
        "skiprows can't be a non zero value if range offset and/or range size has been set");
    }
    _skiprows = skip;
  }

  /**
   * @brief Sets number of rows to skip from end.
   *
   * @param skip Number of rows to skip.
   */
  void set_skipfooter(size_type skip)
  {
    CUDF_EXPECTS((skip == 0) or (_nrows == -1), "Cannot use both `nrows` and `skipfooter`");
    if ((skip != 0) and ((_byte_range_offset != 0) or (_byte_range_size != 0))) {
      CUDF_FAIL(
        "skipfooter can't be a non zero value if range offset and/or range size has been set");
    }

    _skipfooter = skip;
  }

  /**
   * @brief Sets header row index.
   *
   * @param hdr Index where header row is located.
   */
  void set_header(size_type hdr) { _header = hdr; }

  /**
   * @brief Sets line terminator
   *
   * @param term A character to indicate line termination.
   */
  void set_lineterminator(char term) { _lineterminator = term; }

  /**
   * @brief Sets field delimiter.
   *
   * @param delim A character to indicate delimiter.
   */
  void set_delimiter(char delim) { _delimiter = delim; }

  /**
   * @brief Sets numeric data thousands separator.
   *
   * @param val A character that separates thousands.
   */
  void set_thousands(char val) { _thousands = val; }

  /**
   * @brief Sets decimal point character.
   *
   * @param val A character that indicates decimal values.
   */
  void set_decimal(char val) { _decimal = val; }

  /**
   * @brief Sets comment line start character.
   *
   * @param val A character that indicates comment.
   */
  void set_comment(char val) { _comment = val; }

  /**
   * @brief Sets whether to treat `\r\n` as line terminator.
   *
   * @param val Boolean value to enable/disable.
   */
  void enable_windowslinetermination(bool val) { _windowslinetermination = val; }

  /**
   * @brief Sets whether to treat whitespace as field delimiter.
   *
   * @param val Boolean value to enable/disable.
   */
  void enable_delim_whitespace(bool val) { _delim_whitespace = val; }

  /**
   * @brief Sets whether to skip whitespace after the delimiter.
   *
   * @param val Boolean value to enable/disable.
   */
  void enable_skipinitialspace(bool val) { _skipinitialspace = val; }

  /**
   * @brief Sets whether to ignore empty lines or parse line values as invalid.
   *
   * @param val Boolean value to enable/disable.
   */
  void enable_skip_blank_lines(bool val) { _skip_blank_lines = val; }

  /**
   * @brief Sets quoting style.
   *
   * @param style Quoting style used.
   */
  void set_quoting(quote_style style) { _quoting = style; }

  /**
   * @brief Sets quoting character.
   *
   * @param ch A character to indicate quoting.
   */
  void set_quotechar(char ch) { _quotechar = ch; }

  /**
   * @brief Sets a quote inside a value is double-quoted.
   *
   * @param val Boolean value to enable/disable.
   */
  void enable_doublequote(bool val) { _doublequote = val; }

  /**
   * @brief Sets names of columns to read as datetime.
   *
   * @param col_names Vector of column names to infer as datetime.
   */
  void set_infer_date_names(std::vector<std::string> col_names)
  {
    _infer_date_names = std::move(col_names);
  }

  /**
   * @brief Sets indexes of columns to read as datetime.
   *
   * @param col_names Vector of column indices to infer as datetime.
   */
  void set_infer_date_indexes(std::vector<int> col_ind)
  {
    _infer_date_indexes = std::move(col_ind);
  }

  /**
   * @brief Sets per-column types.
   *
   * @param types Vector of dtypes in which the column needs to be read.
   */
  void set_dtypes(std::vector<std::string> types) { _dtypes = std::move(types); }

  /**
   * @brief Sets additional values to recognize as boolean true values.
   *
   * @param vals Vector of values to be considered to be `true`.
   */
  void set_true_values(std::vector<std::string> vals)
  {
    _true_values.insert(_true_values.end(), vals.begin(), vals.end());
  }

  /**
   * @brief Sets additional values to recognize as boolean false values.
   *
   * @param vals Vector of values to be considered to be `false`.
   */
  void set_false_values(std::vector<std::string> vals)
  {
    _false_values.insert(_false_values.end(), vals.begin(), vals.end());
  }

  /**
   * @brief Sets additional values to recognize as null values.
   *
   * @param vals Vector of values to be considered to be null.
   */
  void set_na_values(std::vector<std::string> vals)
  {
    if ((!vals.empty()) and (!_na_filter)) {
      CUDF_FAIL("Can't set na_values when na_filtering is disabled");
    }

    if (_keep_default_na) {
      _na_values.insert(_na_values.end(), vals.begin(), vals.end());
    } else {
      _na_values = std::move(vals);
    }
  }

  /**
   * @brief Sets whether to keep the built-in default NA values.
   *
   * @param val Boolean value to enable/disable.
   */
  void enable_keep_default_na(bool val) { _keep_default_na = val; }

  /**
   * @brief Sets whether to disable null filter.
   *
   * @param val Boolean value to enable/disable.
   */
  void enable_na_filter(bool val)
  {
    if (!val) { _na_values.clear(); }
    _na_filter = val;
  }

  /**
   * @brief Sets whether to parse dates as DD/MM versus MM/DD.
   *
   * @param val Boolean value to enable/disable.
   */
  void enable_dayfirst(bool val) { _dayfirst = val; }

  /**
   * @brief Sets timestamp_type to which all timestamp columns will be cast.
   *
   * @param type Dtype to which all timestamp column will be cast.
   */
  void set_timestamp_type(data_type type) { _timestamp_type = type; }
};

class csv_reader_options_builder {
  csv_reader_options options;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  csv_reader_options_builder() = default;

  /**
   * @brief Constructor from source info.
   *
   * @param src The source information used to read csv file.
   */
  csv_reader_options_builder(source_info const& src) : options(src) {}

  /**
   * @brief Sets compression format of the source.
   *
   * @param comp Compression type.
   * @return this for chaining.
   */
  csv_reader_options_builder& compression(compression_type comp)
  {
    options._compression = comp;
    return *this;
  }

  /**
   * @brief Sets number of bytes to skip from source start.
   *
   * @param offset Number of bytes of offset.
   * @return this for chaining.
   */
  csv_reader_options_builder& byte_range_offset(size_type offset)
  {
    options.set_byte_range_offset(offset);
    return *this;
  }

  /**
   * @brief Sets number of bytes to read.
   *
   * @param size Number of bytes to read.
   * @return this for chaining.
   */
  csv_reader_options_builder& byte_range_size(size_type size)
  {
    options.set_byte_range_size(size);
    return *this;
  }

  /**
   * @brief Sets names of the column.
   *
   * @param col_names Vector of column names.
   * @return this for chaining.
   */
  csv_reader_options_builder& names(std::vector<std::string> col_names)
  {
    options._names = std::move(col_names);
    return *this;
  }

  /**
   * @brief Sets prefix to be used for column ID.
   *
   * @param String used as prefix in for each column name.
   * @return this for chaining.
   */
  csv_reader_options_builder& prefix(std::string pfx)
  {
    options._prefix = pfx;
    return *this;
  }

  /**
   * @brief Sets whether to rename duplicate column names.
   *
   * @param val Boolean value to enable/disable.
   * @return this for chaining.
   */
  csv_reader_options_builder& mangle_dupe_cols(bool val)
  {
    options._mangle_dupe_cols = val;
    return *this;
  }

  /**
   * @brief Sets names of the columns to be read.
   *
   * @param col_names Vector of column names that are needed.
   * @return this for chaining.
   */
  csv_reader_options_builder& use_cols_names(std::vector<std::string> col_names)
  {
    options._use_cols_names = std::move(col_names);
    return *this;
  }

  /**
   * @brief Sets indexes of columns to read.
   *
   * @param col_ind Vector of column indices that are needed.
   * @return this for chaining.
   */
  csv_reader_options_builder& use_cols_indexes(std::vector<int> col_ind)
  {
    options._use_cols_indexes = std::move(col_ind);
    return *this;
  }

  /**
   * @brief Sets number of rows to read.
   *
   * @param val Number of rows to read.
   * @return this for chaining.
   */
  csv_reader_options_builder& nrows(size_type rows)
  {
    options.set_nrows(rows);
    return *this;
  }

  /**
   * @brief Sets number of rows to skip from start.
   *
   * @param val Number of rows to skip.
   * @return this for chaining.
   */
  csv_reader_options_builder& skiprows(size_type skip)
  {
    options.set_skiprows(skip);
    return *this;
  }

  /**
   * @brief Sets number of rows to skip from end.
   *
   * @param skip Number of rows to skip.
   * @return this for chaining.
   */
  csv_reader_options_builder& skipfooter(size_type skip)
  {
    options.set_skipfooter(skip);
    return *this;
  }

  /**
   * @brief Sets header row index.
   *
   * @param hdr Index where header row is located.
   * @return this for chaining.
   */
  csv_reader_options_builder& header(size_type hdr)
  {
    options._header = hdr;
    return *this;
  }

  /**
   * @brief Sets line terminator.
   *
   * @param term A character to indicate line termination.
   * @return this for chaining.
   */
  csv_reader_options_builder& lineterminator(char term)
  {
    options._lineterminator = term;
    return *this;
  }

  /**
   * @brief Sets field delimiter
   *
   * @param delim A character to indicate delimiter.
   * @return this for chaining.
   */
  csv_reader_options_builder& delimiter(char delim)
  {
    options._delimiter = delim;
    return *this;
  }

  /**
   * @brief Sets numeric data thousands separator.
   *
   * @param val A character that separates thousands.
   * @return this for chaining.
   */
  csv_reader_options_builder& thousands(char val)
  {
    options._thousands = val;
    return *this;
  }

  /**
   * @brief Sets decimal point character.
   *
   * @param val A character that indicates decimal values.
   * @return this for chaining.
   */
  csv_reader_options_builder& decimal(char val)
  {
    options._decimal = val;
    return *this;
  }

  /**
   * @brief Sets comment line start character.
   *
   * @param val A character that indicates comment.
   * @return this for chaining.
   */
  csv_reader_options_builder& comment(char val)
  {
    options._comment = val;
    return *this;
  }

  /**
   * @brief Sets whether to treat `\r\n` as line terminator.
   *
   * @param val Boolean value to enable/disable.
   * @return this for chaining.
   */
  csv_reader_options_builder& windowslinetermination(bool val)
  {
    options._windowslinetermination = val;
    return *this;
  }

  /**
   * @brief Sets whether to treat whitespace as field delimiter.
   *
   * @param val Boolean value to enable/disable.
   * @return this for chaining.
   */
  csv_reader_options_builder& delim_whitespace(bool val)
  {
    options._delim_whitespace = val;
    return *this;
  }

  /**
   * @brief Sets whether to skip whitespace after the delimiter.
   *
   * @param val Boolean value to enable/disable.
   * @return this for chaining.
   */
  csv_reader_options_builder& skipinitialspace(bool val)
  {
    options._skipinitialspace = val;
    return *this;
  }

  /**
   * @brief Sets whether to ignore empty lines or parse line values as invalid.
   *
   * @param val Boolean value to enable/disable.
   * @return this for chaining.
   */
  csv_reader_options_builder& skip_blank_lines(bool val)
  {
    options._skip_blank_lines = val;
    return *this;
  }

  /**
   * @brief Sets quoting style.
   *
   * @param style Quoting style used.
   * @return this for chaining.
   */
  csv_reader_options_builder& quoting(quote_style style)
  {
    options._quoting = style;
    return *this;
  }

  /**
   * @brief Sets quoting character.
   *
   * @param ch A character to indicate quoting.
   * @return this for chaining.
   */
  csv_reader_options_builder& quotechar(char ch)
  {
    options._quotechar = ch;
    return *this;
  }

  /**
   * @brief Sets a quote inside a value is double-quoted.
   *
   * @param val Boolean value to enable/disable.
   * @return this for chaining.
   */
  csv_reader_options_builder& doublequote(bool val)
  {
    options._doublequote = val;
    return *this;
  }

  /**
   * @brief Sets names of columns to read as datetime.
   *
   * @param col_names Vector of column names to infer as datetime.
   * @return this for chaining.
   */
  csv_reader_options_builder& infer_date_names(std::vector<std::string> col_names)
  {
    options._infer_date_names = std::move(col_names);
    return *this;
  }

  /**
   * @brief Sets indexes of columns to read as datetime.
   *
   * @param col_names Vector of column indices to infer as datetime.
   * @return this for chaining.
   */
  csv_reader_options_builder& infer_date_indexes(std::vector<int> col_ind)
  {
    options._infer_date_indexes = std::move(col_ind);
    return *this;
  }

  /**
   * @brief Sets per-column types.
   *
   * @param types Vector of dtypes in which the column needs to be read.
   * @return this for chaining.
   */
  csv_reader_options_builder& dtypes(std::vector<std::string> types)
  {
    options._dtypes = std::move(types);
    return *this;
  }

  /**
   * @brief Sets additional values to recognize as boolean true values.
   *
   * @param vals Vector of values to be considered to be `true`.
   * @return this for chaining.
   */
  csv_reader_options_builder& true_values(std::vector<std::string> vals)
  {
    options._true_values.insert(options._true_values.end(), vals.begin(), vals.end());
    return *this;
  }

  /**
   * @brief Sets additional values to recognize as boolean false values.
   *
   * @param vals Vector of values to be considered to be `false`.
   * @return this for chaining.
   */
  csv_reader_options_builder& false_values(std::vector<std::string> vals)
  {
    options._false_values.insert(options._false_values.end(), vals.begin(), vals.end());
    return *this;
  }

  /**
   * @brief Sets additional values to recognize as null values.
   *
   * @param vals Vector of values to be considered to be null.
   * @return this for chaining.
   */
  csv_reader_options_builder& na_values(std::vector<std::string> vals)
  {
    options.set_na_values(std::move(vals));
    return *this;
  }

  /**
   * @brief Sets whether to keep the built-in default NA values.
   *
   * @param val Boolean value to enable/disable.
   * @return this for chaining.
   */
  csv_reader_options_builder& keep_default_na(bool val)
  {
    options.enable_keep_default_na(val);
    return *this;
  }

  /**
   * @brief Sets whether to disable null filter.
   *
   * @param val Boolean value to enable/disable.
   * @return this for chaining.
   */
  csv_reader_options_builder& na_filter(bool val)
  {
    options.enable_na_filter(val);
    return *this;
  }

  /**
   * @brief Sets whether to parse dates as DD/MM versus MM/DD.
   *
   * @param val Boolean value to enable/disable.
   * @return this for chaining.
   */
  csv_reader_options_builder& dayfirst(bool val)
  {
    options._dayfirst = val;
    return *this;
  }

  /**
   * @brief Sets timestamp_type to which all timestamp columns will be cast.
   *
   * @param type Dtype to which all timestamp column will be cast.
   * @return this for chaining.
   */
  csv_reader_options_builder& timestamp_type(data_type type)
  {
    options._timestamp_type = type;
    return *this;
  }

  /**
   * @brief move csv_reader_options member once it's built.
   */
  operator csv_reader_options &&() { return std::move(options); }

  /**
   * @brief move csv_reader_options member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   */
  csv_reader_options&& build() { return std::move(options); }
};

/**
 * @brief Reads a CSV dataset into a set of columns.
 *
 * @ingroup io_readers
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  std::string filepath = "dataset.csv";
 *  cudf::io::csv_reader_options options =
 * cudf::io::csv_reader_options::builder(cudf::source_info(filepath));
 *  ...
 *  auto result = cudf::read_csv(options);
 * @endcode
 *
 * @param options Settings for controlling reading behavior.
 * @param mr Device memory resource used to allocate device memory of the table in the returned.
 * table_with_metadata
 *
 * @return The set of columns along with metadata.
 */
table_with_metadata read_csv(
  csv_reader_options const& options,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 *@breif Builder to build options for `writer_csv()`.
 */
class csv_writer_options_builder;

/**
 * @brief Settings to use for `write_csv()`.
 *
 * @ingroup io_writers
 */
class csv_writer_options {
  // Specify the sink to use for writer output
  sink_info _sink;
  // Set of columns to output
  table_view _table;
  // string to use for null entries
  std::string _na_rep = "";
  // Indicates whether to write headers to csv
  bool _include_header = true;
  // maximum number of rows to process for each file write
  int _rows_per_chunk = 8;
  // character to use for separating lines (default "\n")
  std::string _line_terminator = "\n";
  // character to use for separating lines (default "\n")
  char _inter_column_delimiter = ',';
  // string to use for values != 0 in INT8 types (default 'true')
  std::string _true_value = std::string{"true"};
  // string to use for values == 0 in INT8 types (default 'false')
  std::string _false_value = std::string{"false"};
  // Optional associated metadata
  table_metadata const* _metadata = nullptr;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output.
   * @param table Table to be written to output.
   */
  explicit csv_writer_options(sink_info const& sink, table_view const& table)
    : _sink(sink), _table(table)
  {
  }

  friend csv_writer_options_builder;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit csv_writer_options() = default;

  /**
   * @brief Create builder to create `csv_writer_options`.
   *
   * @param sink The sink used for writer output.
   * @param table Table to be written to output.
   *
   * @return Builder to build csv_writer_options.
   */
  static csv_writer_options_builder builder(sink_info const& sink, table_view const& table);

  /**
   * @brief Returns sink used for writer output.
   */
  sink_info const& get_sink(void) const { return _sink; }

  /**
   * @brief Returns table that would be written to output.
   */
  table_view const& get_table(void) const { return _table; }

  /**
   * @brief Returns optional associated metadata.
   */
  table_metadata const* get_metadata(void) const { return _metadata; }

  /**
   * @brief Returns string to used for null entries.
   */
  std::string get_na_rep(void) const { return _na_rep; }

  /**
   * @brief Whether to write headers to csv.
   */
  bool is_enabled_include_header(void) const { return _include_header; }

  /**
   * @brief Returns maximum number of rows to process for each file write.
   */
  int get_rows_per_chunk(void) const { return _rows_per_chunk; }

  /**
   * @brief Returns character used for separating lines.
   */
  std::string get_line_terminator(void) const { return _line_terminator; }

  /**
   * @brief Returns character used for separating lines.
   */
  char get_inter_column_delimiter(void) const { return _inter_column_delimiter; }

  /**
   * @brief Returns string used for values != 0 in INT8 types.
   */
  std::string get_true_value(void) const { return _true_value; }

  /**
   * @brief Returns string used for values == 0 in INT8 types.
   */
  std::string get_false_value(void) const { return _false_value; }

  // Setter
  /**
   * @brief Sets optional associated metadata.
   *
   @param metadata Associated metadata.
   */
  void set_metadata(table_metadata* metadata) { _metadata = metadata; }

  /**
   * @brief Sets string to used for null entries.
   *
   * @param val String to represent null value.
   */
  void set_na_rep(std::string val) { _na_rep = val; }

  /**
   * @brief Enables/Disables headers being written to csv.
   *
   * @param val Boolean value to enable/disable.
   */
  void enable_include_header(bool val) { _include_header = val; }

  /**
   * @brief Sets maximum number of rows to process for each file write.
   *
   * @param val Number of rows per chunk.
   */
  void set_rows_per_chunk(int val) { _rows_per_chunk = val; }

  /**
   * @brief Sets character used for separating lines.
   *
   * @param term Character to represent line termination.
   */
  void set_line_terminator(std::string term) { _line_terminator = term; }

  /**
   * @brief Sets character used for separating lines.
   *
   * @param delim Character to indicate delimiting.
   */
  void set_inter_column_delimiter(char delim) { _inter_column_delimiter = delim; }

  /**
   * @brief Sets string used for values != 0 in INT8 types.
   *
   * @param val String to represent values != 0 in INT8 types.
   */
  void set_true_value(std::string val) { _true_value = val; }

  /**
   * @brief Sets string used for values == 0 in INT8 types.
   *
   * @param val String to represent values == 0 in INT8 types.
   */
  void set_false_value(std::string val) { _false_value = val; }
};

class csv_writer_options_builder {
  csv_writer_options options;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit csv_writer_options_builder() = default;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output.
   * @param table Table to be written to output.
   */
  explicit csv_writer_options_builder(sink_info const& sink, table_view const& table)
    : options{sink, table}
  {
  }

  /**
   * @brief Sets optional associated metadata.
   *
   * @param metadata Associated metadata.
   * @return this for chaining.
   */
  csv_writer_options_builder& metadata(table_metadata* metadata)
  {
    options._metadata = metadata;
    return *this;
  }

  /**
   * @brief Sets string to used for null entries.
   *
   * @param val String to represent null value.
   * @return this for chaining.
   */
  csv_writer_options_builder& na_rep(std::string val)
  {
    options._na_rep = val;
    return *this;
  };

  /**
   * @brief Enables/Disables headers being written to csv.
   *
   * @param val Boolean value to enable/disable.
   * @return this for chaining.
   */
  csv_writer_options_builder& include_header(bool val)
  {
    options._include_header = val;
    return *this;
  }

  /**
   * @brief Sets maximum number of rows to process for each file write.
   *
   * @param val Number of rows per chunk.
   * @return this for chaining.
   */
  csv_writer_options_builder& rows_per_chunk(int val)
  {
    options._rows_per_chunk = val;
    return *this;
  }

  /**
   * @brief Sets character used for separating lines.
   *
   * @param term Character to represent line termination.
   * @return this for chaining.
   */
  csv_writer_options_builder& line_terminator(std::string term)
  {
    options._line_terminator = term;
    return *this;
  }

  /**
   * @brief Sets character used for separating lines.
   *
   * @param delim Character to indicate delimiting.
   * @return this for chaining.
   */
  csv_writer_options_builder& inter_column_delimiter(char delim)
  {
    options._inter_column_delimiter = delim;
    return *this;
  }

  /**
   * @brief Sets string used for values != 0 in INT8 types.
   *
   * @param val String to represent values != 0 in INT8 types.
   * @return this for chaining.
   */
  csv_writer_options_builder& true_value(std::string val)
  {
    options._true_value = val;
    return *this;
  }

  /**
   * @brief Sets string used for values == 0 in INT8 types.
   *
   * @param val String to represent values == 0 in INT8 types.
   * @return this for chaining.
   */
  csv_writer_options_builder& false_value(std::string val)
  {
    options._false_value = val;
    return *this;
  }

  /**
   * @brief move `csv_writer_options` member once it's built.
   */
  operator csv_writer_options &&() { return std::move(options); }

  /**
   * @brief move `csv_writer_options` member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   */
  csv_writer_options&& build() { return std::move(options); }
};

/**
 * @brief Writes a set of columns to CSV format.
 *
 * The following code snippet demonstrates how to write columns to a file:
 * @code
 *  std::string filepath = "dataset.csv";
 *  cudf::io::sink_info sink_info(filepath);
 *
 *  cudf::io::csv_writer_options options = cudf::io::csv_writer_options(sink_info,
 * table->view()).na_rep(na).include_header(include_header).rows_per_chunk(rows_per_chunk);
 *  ...
 *  cudf::io::write_csv(options);
 * @endcode
 *
 * @param options Settings for controlling writing behavior.
 * @param mr Device memory resource to use for device memory allocation.
 */
void write_csv(csv_writer_options const& options,
               rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());
}  // namespace io
}  // namespace cudf
