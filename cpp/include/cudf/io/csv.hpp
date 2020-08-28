/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
 * @file csv.hpp
 * @brief cuDF-IO freeform API
 */

#pragma once

#include "types.hpp"

#include <cudf/io/writers.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

//! cuDF interfaces
namespace cudf {
//! In-development features
namespace io {

/**
 *@breif Builder to build options for `read_csv()`
 */
class csv_reader_options_builder;

/**
 * @brief Settings to use for `read_csv()`
 *
 * @ingroup io_readers
 */

class csv_reader_options {
  source_info _source;

  // Read settings

  /// Specify the compression format of the source or infer from file extension
  compression_type _compression = compression_type::AUTO;
  /// Bytes to skip from the source start
  size_t _byte_range_offset = 0;
  /// Bytes to read; always reads complete rows
  size_t _byte_range_size = 0;
  /// Names of all the columns; if empty then names are auto-generated
  std::vector<std::string> _names;
  /// If there is no header or names, prepend this to the column ID as the name
  std::string _prefix;
  /// Whether to rename duplicate column names
  bool _mangle_dupe_cols = true;

  // Filter settings

  /// Names of columns to read; empty is all columns
  std::vector<std::string> _use_cols_names;
  /// Indexes of columns to read; empty is all columns
  std::vector<int> _use_cols_indexes;
  /// Rows to read; -1 is all
  size_type _nrows = -1;
  /// Rows to skip from the start; 0 is none
  size_type _skiprows = 0;
  /// Rows to skip from the end; 0 is none
  size_type _skipfooter = 0;
  /// Header row index
  size_type _header = 0;

  // Parsing settings

  /// Line terminator
  char _lineterminator = '\n';
  /// Field delimiter
  char _delimiter = ',';
  /// Numeric data thousands seperator; cannot match delimiter
  char _thousands = '\0';
  /// Decimal point character; cannot match delimiter
  char _decimal = '.';
  /// Comment line start character
  char _comment = '\0';
  bool _windowslinetermination = false;
  /// Treat whitespace as field delimiter; overrides character delimiter
  bool _delim_whitespace = false;
  /// Skip whitespace after the delimiter
  bool _skipinitialspace = false;
  /// Ignore empty lines or parse line values as invalid
  bool _skip_blank_lines = true;
  /// Treatment of quoting behavior
  quote_style _quoting = quote_style::MINIMAL;
  /// Quoting character (if `quoting` is true)
  char _quotechar = '"';
  /// Whether a quote inside a value is double-quoted
  bool _doublequote = true;
  /// Names of columns to read as datetime
  std::vector<std::string> _infer_date_names;
  /// Indexes of columns to read as datetime
  std::vector<int> _infer_date_indexes;

  // Conversion settings

  /// Per-column types; disables type inference on those columns
  std::vector<std::string> _dtypes;
  /// Additional values to recognize as boolean true values
  std::vector<std::string> _true_values{"True", "TRUE", "true"};
  /// Additional values to recognize as boolean false values
  std::vector<std::string> _false_values{"False", "FALSE", "false"};
  /// Additional values to recognize as null values
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

  /// Whether to keep the built-in default NA values
  bool _keep_default_na = true;
  /// Whether to disable null filter; disabling can improve performance
  bool _na_filter = true;
  /// Whether to parse dates as DD/MM versus MM/DD
  bool _dayfirst = false;
  /// Cast timestamp columns to a specific type
  data_type _timestamp_type{type_id::EMPTY};

  explicit csv_reader_options(source_info const& src) : _source(src) {}

  friend csv_reader_options_builder;

  public:

  csv_reader_options() = default;

  /**
   * @brief Returns source info
   */
  source_info const& source() const {return _source;}

  /**
   * @brief Returns compression format of the source
   */
  compression_type compression() const {return _compression;}

  /**
   * @brief Returns number of bytes to skip from source start
   */
  size_t byte_range_offset() const {return _byte_range_offset;}

  /**
   * @brief Returns number of bytes to read
   */
  size_t byte_range_size() const {return _byte_range_size;}

  /**
   * @brief Returns names of the column
   */
  std::vector<std::string>const& names() const {return _names;}

  /**
   * @brief Returns prefix to be used for column ID
   */
  std::string prefix() const {return _prefix;}

  /**
   * @brief Whether to rename duplicate column names
   */
  bool mangle_dupe_cols() const {return _mangle_dupe_cols;}
  
  /**
   * @brief Returns names of the columns to be read
   */
  std::vector<std::string>const& use_cols_names() const {return _use_cols_names;}

  /**
   * @brief Returns indexes of columns to read
   */
  std::vector<int>const& use_cols_indexes() const {return _use_cols_indexes;}

  /**
   * @brief Returns number of rows to read
   */
  size_type nrows() const {return _nrows;}
  
  /**
   * @brief Returns number of rows to skip from start
   */
  size_type skiprows() const {return _skiprows;}

  /**
   * @brief Returns number of rows to skip from end
   */
  size_type skipfooter() const {return _skipfooter;}

  /**
   * @brief Returns header row index
   */
  size_type header() const {return _header;}

  /**
   * @brief Returns line terminator
   */
  char lineterminator() const {return _lineterminator;}

  /**
   * @brief Returns field delimiter
   */
  char delimiter() const {return _delimiter;}

  /**
   * @brief Returns numeric data thousands seperator
   */
  char thousands() const {return _thousands;}
 
  /**
   * @brief Returns decimal point character
   */
  char decimal() const {return _decimal;}

  /**
   * @brief Returns comment line start character
   */
  char comment() const {return _comment;}

  /**
   * @brief Whether to treat `\r\n` as line terminator
   */
  bool windowslinetermination() const {return _windowslinetermination;}

  /**
   * @brief Whether to treat whitespace as field delimiter
   */
  bool delim_whitespace() const {return _delim_whitespace;}

  /**
   * @brief Whether to skip whitespace after the delimiter
   */
  bool skipinitialspace() const {return _skipinitialspace;}

  /**
   * @brief Whether to ignore empty lines or parse line values as invalid
   */
  bool skip_blank_lines() const {return _skip_blank_lines;}

  /**
   * @brief Returns quoting style
   */
  quote_style quoting() const {return _quoting;}

  /**
   * @brief Returns quoting character
   */
  char quotechar() const {return _quotechar;}

  /**
   * @brief Whether a quote inside a value is double-quoted
   */
  bool doublequote() const {return _doublequote;}

  /**
   * @brief Returns names of columns to read as datetime
   */
  std::vector<std::string>const& infer_date_names() const {return _infer_date_names;}

  /**
   * @brief Returns indexes of columns to read as datetime
   */
  std::vector<int>const& infer_date_indexes() const {return _infer_date_indexes;}

  /**
   * @brief Returns per-column types
   */
  std::vector<std::string>const& dtypes() const {return _dtypes;}

  /**
   * @brief Returns additional values to recognize as boolean true values
   */
  std::vector<std::string>const& true_values() const {return _true_values;}

  /**
   * @brief Returns additional values to recognize as boolean false values
   */
  std::vector<std::string>const& false_values() const {return _false_values;}

  /**
   * @brief Returns additional values to recognize as null values
   */
  std::vector<std::string>const& na_values() const {
      return _na_values;
  }

  /**
   * @brief Whether to keep the built-in default NA values
   */
  bool keep_default_na() const {return _keep_default_na;}

  /**
   * @brief Whether to disable null filter
   */
  bool na_filter() const {return _na_filter;}

  /**
   * @brief Whether to parse dates as DD/MM versus MM/DD
   */
  bool dayfirst() const {return _dayfirst;}

  /**
   * @brief Returns timestamp_type to which all timestamp columns will be cast
   */
  data_type timestamp_type() const {return _timestamp_type;}

  /**
   * @brief Set source info
   */
  void source(source_info src) {
      _source = src;
  }

  /**
   * @brief Set compression format of the source
   */
  void compression(compression_type comp) {
      _compression = comp;
  }

  /**
   * @brief Set number of bytes to skip from source start
   */
  void byte_range_offset(size_type offset) {
      if((offset != 0) and ((_skiprows != 0) or (_skipfooter != 0) or (_nrows != -1))) {
          CUDF_FAIL("When ther is valid value in skiprows or skipfooter or nrows, offset can't have non-zero value");
      }
     _byte_range_offset = offset;
  }

  /**
   * @brief Set number of bytes to read
   */
  void byte_range_size(size_type size) {
      if((size != 0) and ((_skiprows != 0) or (_skipfooter != 0) or (_nrows != -1))) {
          CUDF_FAIL("When ther is valid value in skiprows or skipfooter or nrows, range size can't have non-zero value");
      }
      _byte_range_size = size;
  }

  /**
   * @brief Set names of the column
   */
  void names(std::vector<std::string> col_names) {
      _names = std::move(col_names);
  }

  /**
   * @brief Set prefix to be used for column ID
   */
  void prefix(std::string pfx) {
      _prefix = pfx;
  }

  /**
   * @brief Set whether to rename duplicate column names
   */
  void mangle_dupe_cols(bool val) {
      _mangle_dupe_cols = val;
  }
  
  /**
   * @brief Set names of the columns to be read
   */
  void use_cols_names(std::vector<std::string> col_names) {
      _use_cols_names = std::move(col_names);
  }

  /**
   * @brief Set indexes of columns to read
   */
  void use_cols_indexes(std::vector<int> col_ind) {
      _use_cols_indexes = std::move(col_ind);
  }

  /**
   * @brief Set number of rows to read
   */
  void nrows(size_type val){
      CUDF_EXPECTS((val == 0) or (_nrows == -1), "Cannot use both `_nrows` and `_skipfooter`");
      if ((val != 0) and ((_byte_range_offset != 0) or (_byte_range_size != 0))) {
          CUDF_FAIL("nrows can't be a non negative value if range offset and/or range size has been set");
      }
      
      _nrows = val;
  }
  
  /**
   * @brief Set number of rows to skip from start
   */
  void skiprows(size_type val) {
      if ((val != 0) and ((_byte_range_offset != 0) or (_byte_range_size != 0))) {
          CUDF_FAIL("skiprows can't be a non zero value if range offset and/or range size has been set");
      }
      _skiprows = val;
  }

  /**
   * @brief Set number of rows to skip from end
   */
  void skipfooter(size_type skip) {
      CUDF_EXPECTS((skip == 0) or (_nrows == -1), "Cannot use both `_nrows` and `_skipfooter`");
      if ((skip != 0) and ((_byte_range_offset != 0) or (_byte_range_size != 0))) {
          CUDF_FAIL("skipfooter can't be a non zero value if range offset and/or range size has been set");
      }
      
      _skipfooter = skip;
  }

  /**
   * @brief Set header row index
   */
  void header(size_type hdr) {
      _header = hdr;
  }

  /**
   * @brief Set line terminator
   */
  void lineterminator(char term) {
      _lineterminator = term;
  }

  /**
   * @brief Set field delimiter
   */
  void delimiter(char delim) {
      _delimiter = delim;
  }

  /**
   * @brief Set numeric data thousands seperator
   */
  void thousands(char val) {
      _thousands = val;
  }
 
  /**
   * @brief Set decimal point character
   */
  void decimal(char val) {
      _decimal = val;
  }

  /**
   * @brief Set comment line start character
   */
  void comment(char val) {
      _comment = val;
  }

  /**
   * @brief Set whether to treat `\r\n` as line terminator
   */
  void windowslinetermination(bool val) {
      _windowslinetermination = val;
  }

  /**
   * @brief Set whether to treat whitespace as field delimiter
   */
  void delim_whitespace(bool val) {
      _delim_whitespace = val;
  }

  /**
   * @brief Set whether to skip whitespace after the delimiter
   */
  void skipinitialspace(bool val) {
      _skipinitialspace = val;
  }

  /**
   * @brief Set whether to ignore empty lines or parse line values as invalid
   */
  void skip_blank_lines(bool val) {
      _skip_blank_lines = val;
  }

  /**
   * @brief Set quoting style
   */
  void quoting(quote_style style) {
      _quoting = style;
  }

  /**
   * @brief Set quoting character
   */
  void quotechar(char ch) {
      _quotechar = ch;
  }

  /**
   * @brief Set a quote inside a value is double-quoted
   */
  void doublequote(bool val) {
      _doublequote = val;
  }

  /**
   * @brief Set names of columns to read as datetime
   */
  void infer_date_names(std::vector<std::string> col_names) {
      _infer_date_names = std::move(col_names);
  }

  /**
   * @brief Set indexes of columns to read as datetime
   */
  void infer_date_indexes(std::vector<int> col_ind) {
      _infer_date_indexes = std::move(col_ind);
  }

  /**
   * @brief Set per-column types
   */
  void dtypes(std::vector<std::string> types) {
      _dtypes = std::move(types);
  }

  /**
   * @brief Set additional values to recognize as boolean true values
   */
  void true_values(std::vector<std::string> vals) {
      _true_values.insert(_true_values.end(), vals.begin(), vals.end());
  }

  /**
   * @brief Set additional values to recognize as boolean false values
   */
  void false_values(std::vector<std::string> vals) {
      _false_values.insert(_false_values.end(), vals.begin(), vals.end());
  }

  /**
   * @brief Set additional values to recognize as null values
   */
  void na_values(std::vector<std::string> vals) {
      if ((!vals.empty()) and (!_na_filter)){
          CUDF_FAIL("Can't set na_values when na_filtering is disabled");
      }

      if (_keep_default_na) {
          _na_values.insert(_na_values.end(), vals.begin(), vals.end());
      } else {
          _na_values = std::move(vals);
      }
  }

  /**
   * @brief Set whether to keep the built-in default NA values
   */
  void keep_default_na(bool val) {
      _keep_default_na = val;
  }

  /**
   * @brief Set whether to disable null filter
   */
  void na_filter(bool val) {
      if(!val) {
          _na_values.clear();
      }
      _na_filter = val;
  }

  /**
   * @brief Set whether to parse dates as DD/MM versus MM/DD
   */
  void dayfirst(bool val) {
      _dayfirst = val;
  }
  
  /**
   * @brief Set timestamp_type to which all timestamp columns will be cast
   */
  void timestamp_type(data_type type) {_timestamp_type = type;}

  static csv_reader_options_builder builder(source_info const& src);
};

class csv_reader_options_builder {
    csv_reader_options options;

    public:
    csv_reader_options_builder() = default;

    csv_reader_options_builder(source_info const& src): options(src) {}

  /**
   * @brief Set source info
   */
  csv_reader_options_builder& source(source_info src) {
      options._source = src;
      return *this;
  }

  /**
   * @brief Set compression format of the source
   */
  csv_reader_options_builder& compression(compression_type comp) {
      options._compression = comp;
      return *this;
  }

  /**
   * @brief Set number of bytes to skip from source start
   */
  csv_reader_options_builder& byte_range_offset(size_type offset) {
      options.byte_range_offset(offset);
      return *this;
  }

  /**
   * @brief Set number of bytes to read
   */
  csv_reader_options_builder& byte_range_size(size_type size) {
      options.byte_range_size(size);
      return *this;
  }

  /**
   * @brief Set names of the column
   */
  csv_reader_options_builder& names(std::vector<std::string> col_names) {
      options._names = std::move(col_names);
      return *this;
  }

  /**
   * @brief Set prefix to be used for column ID
   */
  csv_reader_options_builder& prefix(std::string pfx) {
      options._prefix = pfx;
      return *this;
  }


  /**
   * @brief Set whether to rename duplicate column names
   */
  csv_reader_options_builder& mangle_dupe_cols(bool val) {
      options._mangle_dupe_cols = val;
      return *this;
  }
  
  /**
   * @brief Set names of the columns to be read
   */
  csv_reader_options_builder& use_cols_names(std::vector<std::string> col_names) {
      options._use_cols_names = std::move(col_names);
      return *this;
  }

  /**
   * @brief Set indexes of columns to read
   */
  csv_reader_options_builder& use_cols_indexes(std::vector<int> col_ind) {
      options._use_cols_indexes = std::move(col_ind);
      return *this;
  }

  /**
   * @brief Set number of rows to read
   */
  csv_reader_options_builder& nrows(size_type val){
      options.nrows(val);
      return *this;
  }
  
  /**
   * @brief Set number of rows to skip from start
   */
  csv_reader_options_builder& skiprows(size_type val) {
      options.skiprows(val);
      return *this;
  }

  /**
   * @brief Set number of rows to skip from end
   */
  csv_reader_options_builder& skipfooter(size_type skip) {
      options.skipfooter(skip);
      return *this;
  }

  /**
   * @brief Set header row index
   */
  csv_reader_options_builder& header(size_type hdr) {
      options._header = hdr;
      return *this;
  }

  /**
   * @brief Set line terminator
   */
  csv_reader_options_builder& lineterminator(char term) {
      options._lineterminator = term;
      return *this;
  }

  /**
   * @brief Set field delimiter
   */
  csv_reader_options_builder& delimiter(char delim) {
      options._delimiter = delim;
      return *this;
  }

  /**
   * @brief Set numeric data thousands seperator
   */
  csv_reader_options_builder& thousands(char val) {
      options._thousands = val;
      return *this;
  }
 
  /**
   * @brief Set decimal point character
   */
  csv_reader_options_builder& decimal(char val) {
      options._decimal = val;
      return *this;
  }

  /**
   * @brief Set comment line start character
   */
  csv_reader_options_builder& comment(char val) {
      options._comment = val;
      return *this;
  }

  /**
   * @brief Set whether to treat `\r\n` as line terminator
   */
  csv_reader_options_builder& windowslinetermination(bool val) {
      options._windowslinetermination = val;
      return *this;
  }

  /**
   * @brief Set whether to treat whitespace as field delimiter
   */
  csv_reader_options_builder& delim_whitespace(bool val) {
      options._delim_whitespace = val;
      return *this;
  }

  /**
   * @brief Set whether to skip whitespace after the delimiter
   */
  csv_reader_options_builder& skipinitialspace(bool val) {
      options._skipinitialspace = val;
      return *this;
  }

  /**
   * @brief Set whether to ignore empty lines or parse line values as invalid
   */
  csv_reader_options_builder& skip_blank_lines(bool val) {
      options._skip_blank_lines = val;
      return *this;
  }

  /**
   * @brief Set quoting style
   */
  csv_reader_options_builder& quoting(quote_style style) {
      options._quoting = style;
      return *this;
  }

  /**
   * @brief Set quoting character
   */
  csv_reader_options_builder& quotechar(char ch) {
      options._quotechar = ch;
      return *this;
  }

  /**
   * @brief Set a quote inside a value is double-quoted
   */
  csv_reader_options_builder& doublequote(bool val) {
      options._doublequote = val;
      return *this;
  }

  /**
   * @brief Set names of columns to read as datetime
   */
  csv_reader_options_builder& infer_date_names(std::vector<std::string> col_names) {
      options._infer_date_names = std::move(col_names);
      return *this;
  }

  /**
   * @brief Set indexes of columns to read as datetime
   */
  csv_reader_options_builder& infer_date_indexes(std::vector<int> col_ind) {
      options._infer_date_indexes = std::move(col_ind);
      return *this;
  }

  /**
   * @brief Set per-column types
   */
  csv_reader_options_builder& dtypes(std::vector<std::string> types) {
      options._dtypes = std::move(types);
      return *this;
  }

  /**
   * @brief Set additional values to recognize as boolean true values
   */
  csv_reader_options_builder& true_values(std::vector<std::string> vals) {
      options._true_values.insert(options._true_values.end(), vals.begin(), vals.end());
      return *this;
  }

  /**
   * @brief Set additional values to recognize as boolean false values
   */
  csv_reader_options_builder& false_values(std::vector<std::string> vals) {
      options._false_values.insert(options._false_values.end(), vals.begin(), vals.end());
      return *this;
  }

  /**
   * @brief Set additional values to recognize as null values
   */
  csv_reader_options_builder& na_values(std::vector<std::string> vals) {
      options.na_values(std::move(vals));
      return *this;
  }

  /**
   * @brief Set whether to keep the built-in default NA values
   */
  csv_reader_options_builder& keep_default_na(bool val) {
      options.keep_default_na(val);
      return *this;
  }

  /**
   * @brief Set whether to disable null filter
   */
  csv_reader_options_builder& na_filter(bool val) {
      options.na_filter(val);
      return *this;
  }

  /**
   * @brief Set whether to parse dates as DD/MM versus MM/DD
   */
  csv_reader_options_builder& dayfirst(bool val) {
      options._dayfirst = val;
      return *this;
  }
  
  /**
   * @brief Set timestamp_type to which all timestamp columns will be cast
   */
  csv_reader_options_builder& timestamp_type(data_type type) {
      options._timestamp_type = type;
      return *this;
  }

  /**
   * @brief move csv_reader_options member once options is built
   */
  operator csv_reader_options &&() { return std::move(options); }

  /**
   * @brief move csv_reader_options member once options is built
   */
  csv_reader_options&& build() { return std::move(options); }

};

/**
 * @brief Reads a CSV dataset into a set of columns
 *
 * @ingroup io_readers
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  #include <cudf/io/functions.hpp>
 *  ...
 *  std::string filepath = "dataset.csv";
 *  cudf::io::csv_reader_options options{cudf::source_info(filepath)};
 *  ...
 *  auto result = cudf::read_csv(options);
 * @endcode
 *
 * @param options Settings for controlling reading behavior
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata
 *
 * @return The set of columns along with metadata
 */
table_with_metadata read_csv(csv_reader_options const& options,
                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

} // namespace io
} // namespace cudf
