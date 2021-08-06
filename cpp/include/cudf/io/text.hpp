// THIS FILE'S CONTENT IS 100% MEANT TO BE MOVED TO multibyte_split.hpp ONCE THE UPSTREAM PR IS DONE

#pragma once

#include <cudf/io/detail/text.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudf {
namespace io {
/**
 * @addtogroup io_readers
 * @{
 * @file
 */

/**
 * @brief Builds settings to use for `read_text()`.
 */
class text_reader_options_builder;

/**
 * @brief Settings to use for `read_text()`.
 */
class text_reader_options {
  source_info _source;

  // Names of column to read; empty is all
  std::vector<std::string> _columns;

  // delimiter, multi-byte is supported, for splitting the text into individual column rows
  std::string _delimiter;

  friend text_reader_options_builder;

  /**
   * @brief Constructor from source info.
   *
   * @param src source information used to read orc file.
   */
  explicit text_reader_options(source_info const& src) : _source(src) {}

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  text_reader_options() = default;

  /**
   * @brief Creates `text_reader_options_builder` which will build `text_reader_options`.
   *
   * @param src Source information to read orc file.
   * @return Builder to build reader options.
   */
  static text_reader_options_builder builder(source_info const& src);

  /**
   * @brief Returns source info.
   */
  source_info const& get_source() const { return _source; }

  /**
   * @brief Delimiter for splitting the text into individual column rows
   */
  std::string get_delimiter() const { return _delimiter; }

  // Setters

  /**
   * @brief Sets the text delimiter
   *
   * @param delimiter String that should be used for splitting the text.
   */
  void set_delimiter(std::string delimiter) { _delimiter = delimiter; }
};

class text_reader_options_builder {
  text_reader_options options;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit text_reader_options_builder() = default;

  /**
   * @brief Constructor from source info.
   *
   * @param src The source information used to read text data.
   */
  explicit text_reader_options_builder(source_info const& src) : options{src} {};

  /**
   * @brief Sets the delimiter for splitting text data.
   *
   * @param delimiter String for splitting the text data.
   * @return this for chaining.
   */
  text_reader_options_builder& delimiter(std::string delimiter)
  {
    options._delimiter = delimiter;
    return *this;
  }

  /**
   * @brief move text_reader_options member once it's built.
   */
  operator text_reader_options&&() { return std::move(options); }

  /**
   * @brief move text_reader_options member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   */
  text_reader_options&& build() { return std::move(options); }
};

/**
 * @brief Reads an text dataset into a set of columns.
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  ...
 *  std::string filepath = "dataset.txt";
 *  cudf::text_reader_options options =
 * cudf::text_reader_options::builder(cudf::source_info(filepath));
 *  ...
 *  auto result = cudf::read_text(options);
 * @endcode
 *
 * @param options Settings for controlling reading behavior.
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata.
 *
 * @return The set of columns.
 */
table_with_metadata read_text(
  text_reader_options const& options,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace io
}  // namespace cudf
