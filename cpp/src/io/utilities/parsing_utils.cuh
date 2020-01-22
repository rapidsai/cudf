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

#include <cudf/detail/utilities/trie.cuh>
#include <cudf/io/types.hpp>

namespace cudf {
namespace experimental {
namespace io {

/**
 * @brief Structure for holding various options used when parsing and
 * converting CSV/json data to cuDF data type values.
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

namespace gpu {
/**
 * @brief CUDA kernel iterates over the data until the end of the current field
 *
 * Also iterates over (one or more) delimiter characters after the field.
 * Function applies to formats with field delimiters and line terminators.
 *
 * @param data The entire plain text data to read
 * @param opts A set of parsing options
 * @param pos Offset to start the seeking from
 * @param stop Offset of the end of the row
 *
 * @return long The position of the last character in the field, including the
 *  delimiter(s) following the field data
 */
__device__ __inline__ long seek_field_end(const char *data,
                                          ParseOptions const &opts, long pos,
                                          long stop) {
  bool quotation = false;
  while (true) {
    // Use simple logic to ignore control chars between any quote seq
    // Handles nominal cases including doublequotes within quotes, but
    // may not output exact failures as PANDAS for malformed fields
    if (data[pos] == opts.quotechar) {
      quotation = !quotation;
    } else if (quotation == false) {
      if (data[pos] == opts.delimiter) {
        while (opts.multi_delimiter && pos < stop &&
               data[pos + 1] == opts.delimiter) {
          ++pos;
        }
        break;
      } else if (data[pos] == opts.terminator) {
        break;
      } else if (data[pos] == '\r' &&
                 (pos + 1 < stop && data[pos + 1] == '\n')) {
        stop--;
        break;
      }
    }
    if (pos >= stop) break;
    pos++;
  }
  return pos;
}

} // namespace gpu

/**
 * @brief Searches the input character array for each of characters in a set.
 * Sums up the number of occurrences. If the 'positions' parameter is not void*,
 * positions of all occurrences are stored in the output device array.
 * 
 * Does not load the entire file into the GPU memory at any time, so it can 
 * be used to parse large files. Output array needs to be preallocated.
 * 
 * @param[in] h_data Pointer to the input character array
 * @param[in] h_size Number of bytes in the input array
 * @param[in] keys Vector containing the keys to count in the buffer
 * @param[in] result_offset Offset to add to the output positions
 * @param[out] positions Array containing the output positions
 * 
 * @return cudf::size_type total number of occurrences
 **/
template<class T>
cudf::size_type find_all_from_set(const char *h_data, size_t h_size, const std::vector<char>& keys, uint64_t result_offset,
	T *positions);

/**
 * @brief Searches the input character array for each of characters in a set
 * and sums up the number of occurrences.
 *
 * Does not load the entire buffer into the GPU memory at any time, so it can 
 * be used with buffers of any size.
 *
 * @param[in] h_data Pointer to the data in host memory
 * @param[in] h_size Size of the input data, in bytes
 * @param[in] keys Vector containing the keys to count in the buffer
 *
 * @return cudf::size_type total number of occurrences
 **/
cudf::size_type count_all_from_set(const char *h_data, size_t h_size, const std::vector<char>& keys);

/**
 * @brief Infer file compression type based on user supplied arguments.
 *
 * If the user specifies a valid compression_type for compression arg,
 * compression type will be computed based on that.  Otherwise the filename
 * and ext_to_comp_map will be used.
 *
 * @param[in] compression_arg User specified compression type (if any)
 * @param[in] filename Filename to base compression type (by extension) on
 * @param[in] ext_to_comp_map User supplied mapping of file extension to compression type
 *
 * @return string representing compression type ("gzip, "bz2", etc)
 **/
std::string infer_compression_type(
    const compression_type &compression_arg, const std::string &filename,
    const std::vector<std::pair<std::string, std::string>> &ext_to_comp_map);

}  // namespace io
}  // namespace experimental
}  // namespace cudf
