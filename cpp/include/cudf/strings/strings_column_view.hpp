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
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/column/column.hpp>

#include <rmm/thrust_rmm_allocator.h>

namespace cudf {

/**---------------------------------------------------------------------------*
 * @brief Given a column-view of strings type, an instance of this class
 * provides a wrapper on this compound column for strings operations.
 *---------------------------------------------------------------------------**/
class strings_column_view : private column_view
{
public:
    strings_column_view( column_view strings_column );
    strings_column_view( strings_column_view&& strings_view ) = default;
    strings_column_view( const strings_column_view& strings_view ) = default;
    ~strings_column_view() = default;

    /**---------------------------------------------------------------------------*
     * @brief Returns the number of strings in the column
     *---------------------------------------------------------------------------**/
    size_type size() const;

    /**---------------------------------------------------------------------------*
     * @brief Returns the internal parent string column
     *---------------------------------------------------------------------------**/
    column_view parent() const;

    /**---------------------------------------------------------------------------*
    * @brief Returns the internal column of offsets
    *---------------------------------------------------------------------------**/
    column_view offsets() const;

    /**---------------------------------------------------------------------------*
    * @brief Returns the internal column of chars
    *---------------------------------------------------------------------------**/
    column_view chars() const;

    /**---------------------------------------------------------------------------*
    * @brief Returns a pointer to the internal null mask memory
    *---------------------------------------------------------------------------**/
    const bitmask_type* null_mask() const;

    /**---------------------------------------------------------------------------*
    * @brief Returns the number of nulls in this column
    *---------------------------------------------------------------------------**/
    size_type null_count() const;

private:
    const column_view _parent;
};

namespace strings
{

/**---------------------------------------------------------------------------*
 * @brief Prints the strings to stdout.
 *
 * @param strings Strings instance for this operation.
 * @param start Index of first string to print.
 * @param end Index of last string to print. Specify -1 for all strings.
 * @param max_width Maximum number of characters to print per string.
 *        Specify -1 to print all characters.
 * @param delimiter The chars to print between each string.
 *        Default is new-line character.
 *---------------------------------------------------------------------------**/
void print( strings_column_view strings,
            size_type start=0, size_type end=-1,
            size_type max_width=-1, const char* delimiter = "\n" );


/**---------------------------------------------------------------------------*
 * @brief Create output per Arrow strings format.
 * The return pair is the array of chars and the array of offsets.
 *
 * @param strings Strings instance for this operation.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return Pair containing a contiguous array of chars and an array of offsets.
 *---------------------------------------------------------------------------**/
std::pair<rmm::device_vector<char>, rmm::device_vector<size_type>>
    create_offsets( strings_column_view strings,
                    cudaStream_t stream=0,
                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

/**---------------------------------------------------------------------------*
 * @brief Returns a new strings column created from a subset of
 * of this instance's strings column.
 *
 * ```
 * s1 = ["a", "b", "c", "d", "e", "f"]
 * s2 = sublist( s1, 2 )
 * s2 is ["c", "d", "e", "f"]
 * ```
 *
 * @param strings Strings instance for this operation.
 * @param start Index of first string to use.
 * @param end Index of last string to use.
 *        Default -1 indicates the last element.
 * @param step Increment value between indexes.
 *        Default step is 1.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return New strings column of size (end-start)/step.
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> sublist( strings_column_view strings,
                                       size_type start, size_type end=-1,
                                       size_type step=1,
                                       cudaStream_t stream=0,
                                       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

/**---------------------------------------------------------------------------*
 * @brief Returns a new strings column using the specified indices to select
 * elements from the specified strings column.
 *
 * ```
 * s1 = ["a", "b", "c", "d", "e", "f"]
 * map = [0, 2]
 * s2 = gather( s1, map )
 * s2 is ["a", "c"]
 * ```
 *
 * @param strings Strings instance for this operation.
 * @param gather_map The indices with which to select strings for the new column.
 *        Values must be within [0,size()) range.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return New strings column of size indices.size()
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> gather( strings_column_view strings,
                                      cudf::column_view gather_map,
                                      cudaStream_t stream=0,
                                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

/**---------------------------------------------------------------------------*
 * @brief Sort types for the sort method.
 *---------------------------------------------------------------------------**/
enum sort_type {
    none=0,    ///< no sorting
    length=1,  ///< sort by string length
    name=2     ///< sort by characters code-points
};

/**---------------------------------------------------------------------------*
 * @brief Returns a new strings column that is a sorted version of the
 * strings in this instance.
 *
 * @param strings Strings instance for this operation.
 * @param stype Specify what attribute of the string to sort on.
 * @param order Sort strings in ascending or descending order.
 * @param null_order Sort nulls to the beginning or the end of the new column.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return New strings column with sorted elements of this instance.
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> sort( strings_column_view strings,
                                    sort_type stype,
                                    cudf::order order=cudf::order::ASCENDING,
                                    cudf::null_order null_order=cudf::null_order::BEFORE,
                                    cudaStream_t stream=0,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

/**
 * @brief Returns new instance using the provided map values and strings.
 * The map values specify the location in the new strings instance.
 * Missing values pass through from the column at those positions.
 *
 * ```
 * s1 = ["a", "b", "c", "d"]
 * s2 = ["e", "f"]
 * map = [1, 3]
 * s3 = scatter( s1, s2, m1 )
 * s3 is ["a", "e", "c", "f"]
 * ```
 *
 * @param strings Strings instance for this operation.
 * @param values The instance for which to retrieve the strings
 *        specified in map column.
 * @param scatter_map The 0-based index values to retrieve from the
 *        strings parameter. Number of values must equal the number
 *        of elements in strings pararameter (strings.size()).
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return New instance with the specified strings.
 */
std::unique_ptr<cudf::column> scatter( strings_column_view strings,
                                       strings_column_view values,
                                       cudf::column_view scatter_map,
                                       cudaStream_t stream=0,
                                       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );
/**
 * @brief Returns new instance using the provided index values and a
 * single string. The map values specify where to place the string
 * in the new strings instance. Missing values pass through from
 * the column at those positions.
 *
 * ```
 * s1 = ["a", "b", "c", "d"]
 * map = [1, 3]
 * s2 = scatter( s1, "e", m1 )
 * s2 is ["a", "e", "c", "e"]
 * ```
 *
 * @param strings Strings instance for this operation.
 * @param value Null-terminated encoded string in host memory to use with
 *        the scatter_map.
 * @param scatter_map The 0-based index values to place the given string.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return New instance with the specified strings.
 */
std::unique_ptr<cudf::column> scatter( strings_column_view strings,
                                       const char* value,
                                       cudf::column_view scatter_map,
                                       cudaStream_t stream=0,
                                       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

/**---------------------------------------------------------------------------*
 * @brief Returns the number of bytes for each string in a strings column.
 * Null strings will have a byte count of 0.
 *
 * @param strings Strings instance for this operation.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return Numeric column of type int32.
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> bytes_counts( strings_column_view strings,
                                            cudaStream_t stream=0,
                                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

/**---------------------------------------------------------------------------*
 * @brief Returns the number of characters for each string in a strings column.
 * Null strings will have a count of 0. The number of characters is not the
 * same as the number of bytes if multi-byte encoded characters make up a
 * string.
 *
 * @param strings Strings instance for this operation.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return Numeric column of type int32.
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> characters_counts( strings_column_view strings,
                                                 cudaStream_t stream=0,
                                                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

/**---------------------------------------------------------------------------*
 * @brief Creates a column with code point values (integers) for each string.
 * A code point is the integer value representation of a character.
 * For example, in UTF-8 the code point value for the character 'A' is 65.
 * The column is an array of variable-length integer arrays each with length
 * as returned by characters_counts().
 *
 * ```
 * s = ["a","xyz", "éee"]
 * v = code_points(s)
 * v is [97, 120, 121, 122, 50089, 101, 101]
 * ```
 *
 * @param strings Strings instance for this operation.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return Numeric column of type int32. TODO: need uint32 here
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> code_points( strings_column_view strings,
                                           cudaStream_t stream=0,
                                           rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

enum character_attribute {
    DECIMAL=0,
    NUMERIC=1,
    DIGIT=2,
    ALPHA=3,
    SPACE=4,
    UPPER=5,
    LOWER=6,
    ALPHANUM=7,
    EMPTY=8
};
/**---------------------------------------------------------------------------*
 * @brief Returns true for strings that have only characters of the specified
 * type.
 * @param strings Strings instance for this operation.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return Column of type bool.
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> is_of_type( strings_column_view strings,
                                          character_attribute ca_type,
                                          cudaStream_t stream=0,
                                          rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

/**---------------------------------------------------------------------------*
 * @brief Row-wise concatenates two columns of strings into a new a column.
 * The number of strings in both columns must match.
 * @param strings 1st string column.
 * @param others 2nd string column.
 * @param separator Null-terminated CPU string that should appear between each element.
 * @param narep Null-terminated CPU string that should represent any null strings found.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return New column with concatenated results
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> concatenate( strings_column_view strings,
                                           strings_column_view others,
                                           const char* separator="", const char* narep=nullptr,
                                           cudaStream_t stream=0,
                                           rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

/**---------------------------------------------------------------------------*
 * @brief Row-wise concatenates the given list of strings columns with the first column.
 *
 * ```
 * s1 = ['aa', null, '', 'aa']
 * s2 = ['', 'bb', 'bb', null]
 * r = concatenate(s1,s2)
 * r is ['aa', null, 'bb', null]
 * ```
 *
 * @param strings 1st string column.
 * @param others List of string columns to concatenate.
 * @param separator Null-terminated CPU string that should appear between each instance.
 *        Default is empty string.
 * @param narep Null-terminated CPU string that should represent any null strings found.
 *        Default of null means any null operand produces a null result.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return New column with concatenated results
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> concatenate( std::vector<strings_column_view>& strings,
                                           const char* separator="",
                                           const char* narep=nullptr,
                                           cudaStream_t stream=0,
                                           rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

/**---------------------------------------------------------------------------*
 * @brief Concatenates all strings in the column into one new string.
 * This provides the Pandas strings equivalent of join().
 * @param strings Strings for this operation.
 * @param separator Null-terminated CPU string that should appear between each string.
 * @param narep Null-terminated CPU string that should represent any null strings found.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return New column containing one string.
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> join_strings( strings_column_view strings,
                                            const char* separator="",
                                            const char* narep=nullptr,
                                            cudaStream_t stream=0,
                                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

/**---------------------------------------------------------------------------*
 * @brief Split strings vertically creating new columns of strings.
 * The number of columns will be equal to the string with the most splits.
 * The delimiter is searched starting from the beginning of each string.
 *
 * ```
 * s = ["a b c", "d e f", "g h"]
 * r = split(s," ")
 * r is vector of 3 columns:
 * r[0] = ["a", "d", "g"]
 * r[1] = ["b", "e", "h"]
 * r[2] = ["c", "f", nullptr]
 * ```
 *
 * @param delimiter Null-terminated CPU string identifying the split points within each string.
 *        Default of null splits on whitespace.
 * @param maxsplit Maximum number of splits to perform searching from the beginning.
 *        Default -1 indicates all delimiters are processed.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return List of strings columns.
 *---------------------------------------------------------------------------**/
std::vector<std::unique_ptr<cudf::column>> split( strings_column_view strings,
                                                  const char* delimiter=nullptr,
                                                  int maxsplit=-1,
                                                  cudaStream_t stream=0,
                                                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );
/**---------------------------------------------------------------------------*
 * @brief Split strings vertically creating new columns of strings.
 * The number of columns will be equal to the string with the most splits.
 * The delimiter is searched starting from the end of each string.
 *
 * ```
 * s = ["a b c", "d e f", "g h"]
 * r = split(s," ",1)
 * r is vector of 2 columns:
 * r[0] = ["a b", "d e", "g h"]
 * r[1] = ["c", "f", nullptr]
 * ```
 *
 * @param delimiter Null-terminated CPU string identifying the split points within each string.
 *        Default of null splits on whitespace.
 * @param maxsplit Maximum number of splits to perform searching right to left.
 *        Default -1 indicates all delimiters are processed.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return List of strings columns.
 *---------------------------------------------------------------------------**/
std::vector<std::unique_ptr<cudf::column>> rsplit( strings_column_view strings,
                                                   const char* delimiter=nullptr,
                                                   int maxsplit=-1,
                                                   cudaStream_t stream=0,
                                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );
/**---------------------------------------------------------------------------*
 * @brief Each string is split into a list of new column of strings.
 * The delimiter is searched from the beginning of each string.
 * Each string results in a new strings column.
 *
 * ```
 * s = ["a b c", "d e f", "g h", "i j"]
 * r = split_record(s," ")
 * r is vector of 4 columns:
 * r[0] = ["a", "b", "c"]
 * r[1] = ["d", "e", "f"]
 * r[2] = ["g", "h", nullptr]
 * r[3] = ["i", "j", nullptr]
 * ```
 *
 * @param strings Strings for this operation.
 * @param delimiter Null-terminated CPU string identifying the split points within each string.
 *        Default of null splits on whitespace.
 * @param maxsplit Maximum number of splits to perform searching from the beginning.
 *        Default -1 indicates all delimiters are processed.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return List of columns for each string.
 *---------------------------------------------------------------------------**/
std::vector<std::unique_ptr<cudf::column>> split_record( strings_column_view strings,
                                                         const char* delimiter=nullptr,
                                                         int maxsplit=-1,
                                                         cudaStream_t stream=0,
                                                         rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );
/**---------------------------------------------------------------------------*
 * @brief Each string is split into a list of new strings.
 * The delimiter is searched from the end of each string.
 * Each string results in a new strings column.
 *
 * ```
 * s = ["a b c", "d e f", "g h", "i j"]
 * r = rsplit_record(s," ",1)
 * r is vector of 4 columns:
 * r[0] = ["a b", "c"]
 * r[1] = ["d e", "f"]
 * r[2] = ["g", "h"]
 * r[3] = ["i", "j"]
 * ```
 *
 * @param strings Strings for this operation.
 * @param delimiter Null-terminated CPU string identifying the split points within each string.
 *        Default of null splits on whitespace.
 * @param maxsplit Maximum number of splits to perform searching from the end.
 *        Default -1 indicates all delimiters are processed.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return List of columns for each string.
 *---------------------------------------------------------------------------**/
std::vector<std::unique_ptr<cudf::column>> rsplit_record( strings_column_view strings,
                                                          const char* delimiter=nullptr,
                                                          int maxsplit=-1,
                                                          cudaStream_t stream=0,
                                                          rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );
/**---------------------------------------------------------------------------*
 * @brief Each string is split into two strings on the first delimiter found.
 * Three strings are always created for each string: left-half, delimiter itself, right-half.
 * The result is 3 strings columns representing the 3 partitions.
 *
 * ```
 * s = ["a:b:c", "d:e:f", "g:h", "i:j"]
 * r = partition(s,":")
 * r is vector of 4 columns:
 * r[0] = ["a", ":", "b:c"]
 * r[1] = ["d", ":", "e:f"]
 * r[2] = ["g", ":", "h"]
 * r[3] = ["i", ":", "j"]
 * ```
 *
 * @param delimiter Null-terminated CPU string identifying the split points within each string.
 * @param results The list of instances for each string.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return List of columns for each partition.
 *---------------------------------------------------------------------------**/
std::vector<std::unique_ptr<cudf::column>> partition( strings_column_view strings,
                                                      const char* delimiter,
                                                      cudaStream_t stream=0,
                                                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );
/**---------------------------------------------------------------------------*
 * @brief Each string is split into two strings on the last delimiter found.
 * Three strings are always created for each string: left-half, delimiter itself, right-half.
 * The result is 3 strings columns representing the 3 partitions.
 *
 * ```
 * s = ["a:b:c", "d:e:f", "g:h", "i:j"]
 * r = rpartition(s,":")
 * r is vector of 4 columns:
 * r[0] = ["a:b", ":", "c"]
 * r[1] = ["d:e", ":", "f"]
 * r[2] = ["g", ":", "h"]
 * r[3] = ["i", ":", "j"]
 * ```
 *
 * @param delimiter Null-terminated CPU string identifying the split points within each string.
 * @param results The list of instances for each string.
 * @param stream CUDA stream to use kernels in this method.
 * @param mr Resource for allocating device memory.
 * @return List of columns for each partition.
 *---------------------------------------------------------------------------**/
std::vector<std::unique_ptr<cudf::column>> rpartition( strings_column_view strings,
                                                       const char* delimiter,
                                                       cudaStream_t stream=0,
                                                       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

} // namespace strings
} // namespace cudf
