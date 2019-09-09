/*
* Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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
#include <cstddef>
#include <vector>

/**
 * @file NVStrings.h
 * @brief Class definition for NVStrings.
 */

struct nvstrings_ipc_transfer;
struct StringsStatistics;
class custring_view;
class NVStringsImpl;
/**
 * @brief This class manages a list of strings stored in device memory.
 * An instance of this class is a CPU (host) object whose methods run
 * on all the strings it manages in parallel on the GPU.
 * @par
 * The methods are meant to match more or less directly with its python
 * counterpart, @p nvstrings.py. And the operations strive to mimic the
 * behavoir of the equivalent Pandas strings methods.
 * @par
 * An instance of this class is immutable and operations that modify
 * or create new strings will return a new instance.
 * @par
 * All methods accept and return only UTF-8 encoded strings.
 * @par
 * You can use the static \p create methods to instantiate from
 * host strings -- those that are stored in CPU memory.
 * Use the \p destroy() method to free an instance of this class.
 * @nosubgrouping
*/
class NVStrings
{
    NVStringsImpl* pImpl;

    // ctors/dtor are made private to control memory allocation
    NVStrings();
    NVStrings(unsigned int count);
    NVStrings(const NVStrings&);
    NVStrings& operator=(const NVStrings&) = delete;
    ~NVStrings();

public:
    /**
     * @brief  Sorting by attributes.
     *
     * Sorting by both length and characters is allowed and sorts by length first.
     * Sorting could increase performance of other operations by reducing divergence.
     */
    enum sorttype {
        none=0,    ///< no sorting
        length=1,  ///< sort by string length
        name=2     ///< sort by characters code-points
    };

    /** @name Create NVStrings instance from strings
     *  Use these static methods to create a new instance of this class given a list of character arrays encoded in UTF-8.
     *  These methods will make a copy of strings in host/device memory so sufficient storage must be available for them to succeed.
     *  Use the \p destroy() method to free any instance created by these methods.
     */
    ///@{
    /**
     * @brief Create an instance from an array of null-terminated host strings.
     * @param[in] strs Array of character-array pointers to UTF-8 encoded strings.
     *                 Null pointers identify null strings.
     * @param count The number of pointers.
     *
     * @return Instance with the strings copied into device memory.
     */
    static NVStrings* create_from_array(const char** strs, unsigned int count);
    /**
     * @brief Create an instance from an array of string/length pairs.
     * @param[in] strs Array of pointer/length pairs to UTF-8 encoded strings.
     *                 Lengths should be in bytes and should not include null-terminator.
     *                 Null pointers identify null strings.
     *                 Zero lengths identify empty strings.
     * @param count The number of pairs in the \p strs array.
     * @param devmem Set to true (default) if pointers are to device memory.
     * @param stype Optionally sort the strings accordingly.
     * @return Instance with the strings copied into device memory.
     */
    static NVStrings* create_from_index(std::pair<const char*,size_t>* strs, unsigned int count, bool devmem=true, sorttype stype=none );
    /**
     * @brief Create an instance from single buffer of strings.
     *
     * Start of each string is identified by the offsets array.
     * The size of each string is determined by computing adjacent offset differences so null-terminators are not allowed.
     * @param[in] strs The pointer to the contiguous buffer of strings encoded in UTF-8.
     * @param count The total number of strings in the buffer.
     * @param[in] offsets Array of byte offsets to each string.
     *                    This should be 1 more than the count value with the last value specifying the length of the buffer in bytes.
     * @param[in] nullbitmask Each bit identifies which strings are to be considered null.
     *                        The bits are organized as specified in the Arrow format. If no nulls, this parameter can be null.
     *                        The size of this byte array should be at least (count+7)/8 bytes.
     * @param nulls The number of nulls identified by the \p nullbitmask.
     * @param devmem Set to true (default) if pointers are to device memory.
     *
     * @return Instance with the strings copied into device memory.
     */
    static NVStrings* create_from_offsets(const char* strs, int count, const int* offsets, const unsigned char* nullbitmask=0, int nulls=0, bool devmem=true);
    /**
     * @brief Create an instance from other NVStrings instances.
     *
     * This can be used to create a new instance from a collection of other instances.
     * @param[in] strs A vector of NVStrings instance pointers to use.
     *                 These can be safely freed by the caller on return from this method.
     * @return Instance with copy of the strings provided.
     */
    static NVStrings* create_from_strings( std::vector<NVStrings*> strs );
    /**
     * @brief Create an instance from an IPC-transfer object built from nvstrings_ipc_transfer.
     *
     * @param[in] ipc Data needed to create a new instance.
     * @return Instance with data provided.
     */
    static NVStrings* create_from_ipc( nvstrings_ipc_transfer& ipc );
    /**
     * @brief Create an instance from a specific column in a CSV file.
     *
     * This has very limited support for CSV formats is intended for experimentation only.
     * Recommend using the cuDF read_csv method which has many more features.
     *
     * @param[in] csvfile Full-path to CSV file to parse.
     * @param[in] column  0-based column index to retrieve
     * @param[in] lines   Limit the number of lines read from the file. Default is all lines.
     * @param[in] stype   Whether to sort the strings or not.
     * @param[in] nullIsEmpty   How to handle null entries. Set to true to treat nulls as empty strings.
     * @return Instance of strings for the specified column.
     */
    static NVStrings* create_from_csv( const char* csvfile, unsigned int column, unsigned int lines=0, sorttype stype=none, bool nullIsEmpty=false);
    ///@}

    /**
     * @brief Use this method to free any instance created by methods in this class.
     *
     * All device and CPU memory used by this instance is freed.
     * Caller should no longer use this instance after calling this method.
     * @param[in] inst The instance to free.
     */
    static void destroy(NVStrings* inst);

    /**
     * @brief Returns the number of device bytes used by this instance.
     * @return Number of bytes.
     */
    size_t memsize() const;
    /**
     * @brief Returns the number of strings managed by this instance.
     * @return Number of strings.
     */
    unsigned int size() const;

    /**
     * @brief Create an index for the device strings contained in this instance.
     *
     * The pointers returned are to the internal device memory and should
     * not be modified or freed by the caller.
     * @param[in,out] strs An array of empty pairs to be filled in by this method.
     *                This array must hold at least size() elements.
     * @param devmem If the \p strs array is in device memory or CPU memory.
     *               The resulting pointers are always to device memory regardless.
     * @return 0 if successful.
     */
    int create_index(std::pair<const char*,size_t>* strs, bool devmem=true );
    /**
     * @brief Return a list to the internal \p custring_view pointers for this instance.
     *
     * The pointers returned are to the internal device memory and should not
     * be modified or freed by the caller.
     * @param[in,out] strs An empty array of pointers.
     *                This array must hold at least size() elements.
     * @param devmem If the \p strs array is in device memory or CPU memory.
     *               The resulting pointers are always to device memory regardless.
     * @return 0 if successful.
     */
    int create_custring_index( custring_view** strs, bool devmem=true );
    /**
     * @brief Copy strings into single memory buffer provided.
     *
     * Start of each string is placed into the offsets array.
     * Nulls will be specified in the nullbitmask if provided.
     * @param[in,out] strs The pointer to the contiguous buffer of strings encoded in UTF-8.
     * @param[in,out] offsets Array of 0-based byte offsets to each string.
         *                    The number of values should be count+1 with the last value specifying the length of the \p strs buffer in bytes.
     * @param[in,out] nullbitmask Byte array of bits identifies which strings are to be considered null.
     *                            The bits are organized as specified in the Arrow format. If no nulls, this parameter can be null.
     *                            The size of this byte array should be at least (count+7)/8 bytes.
     * @param devmem If the \p strs array and \p nullbitmask array is in device memory or CPU memory.
     * @return 0 if successful.
     */
    int create_offsets( char* strs, int* offsets, unsigned char* nullbitmask=0, bool devmem=true );
    /**
     * @brief Create IPC-transfer data from this instance.
     *
     * @param[in,out] ipc Structure will be set with data needed by create_from_ipc method.
     * @return 0 if successful.
     */
    int create_ipc_transfer( nvstrings_ipc_transfer& ipc );
    /**
     * @brief Set bit-array identifying the null strings.
     *
     * The bits are arranged using the Arrow format for bitmask.
     * @param[in,out] bitarray Byte array to be filled in by this method.
     *                         The array must be at least (size()+7)/8 bytes.
     * @param emptyIsNull Set to true to specify empty strings as null strings.
     * @param devmem Identifies the provided \p bitarray parameter points to device memory (default) or CPU memory.
     * @return The number of nulls found.
     */
    unsigned int set_null_bitarray( unsigned char* bitarray, bool emptyIsNull=false, bool devmem=true );
    /**
     * @brief Set int array with position of null strings.
     *
     * Returns the 0-based index positions of strings that are null in this instance.
     * @param[in,out] pos Integer array to be filled in by this method.
     * @param emptyIsNull Set to true to specify empty strings as null strings.
     * @param devmem Identifies the provided \p pos parameter points to device memory (default) or CPU memory.
     * @return The number of nulls found.
     */
    unsigned int get_nulls( unsigned int* pos, bool emptyIsNull=false, bool devmem=true );
    /**
     * @brief Create a new instance from this instance.
     * @return New instance as duplicate of this instance.
     */
    NVStrings* copy();
    /**
     * @brief Copy the list of strings into the provided host memory.
     *
     * Each pointer must point to memory large enough to hold the bytes of each corresponding string.
     * Null strings should be identified using the set_null_bitarray method.
     * @param[in,out] list The list of pointers to CPU memory to copy each string into.
     * @param start The 0-based index position of the string to copy first.
     * @param end The 0-based index position of the string to copy last.
     * @return 0 if successful.
     */
    int to_host(char** list, int start, int end);

    // array.cu
    /**
     * @brief Create a new instance containing only the strings in the specified range.
     * @param start First 0-based index to capture from.
     * @param end The last 0-based index to capture from.
     * @param step This can be used to capture indexes in intervals. Default is all strings within start and end.
     * @return New instance with the specified strings.
     */
    NVStrings* sublist( unsigned int start, unsigned int end, int step=0 );
    /**
     * @brief Returns new instance using the order of the specified index values for this instance.
     *
     * @param[in] pos The 0-based index values to retrieve from this instance. Values may be repeated.
     * @param count The number of values in the pos array.
     * @param devmem Indicates whether the pos parameter points to device memory or CPU memory.
     * @return New instance with the specified strings.
     */
    NVStrings* gather( const int* pos, unsigned int count, bool devmem=true );
    /**
     * @brief Returns new instance where the corresponding boolean array values are true.
     *
     * @param[in] mask Must have the same number of elements as this instance.
     * @param devmem Indicates whether the mask parameter points to device memory or CPU memory.
     * @return New instance with the indicated strings.
     */
    NVStrings* gather( const bool* mask, bool devmem=true );
    /**
     * @brief Returns new instance using the provided index values and strings instance.
     * The position values specify the location in the new strings instance.
     * Missing values pass through from this instance at those positions.
     *
     * @param[in] strs The instance for which to retrieve the values specified in pos array.
     * @param[in] pos The 0-based index values to retrieve from the provided instance.
     *                Number of values must equal the number of strings in strs pararameter.
     * @param devmem Indicates whether the pos parameter points to device memory or CPU memory.
     * @return New instance with the specified strings.
     */
    NVStrings* scatter( NVStrings& strs, const int* pos, bool devmem=true );
    /**
     * @brief Returns new instance using the provided index values and strings instance.
     * The position values specify the location in the new strings instance.
     * Missing values pass through from this instance at those positions.
     *
     * @param[in] str The instance for which to retrieve the values specified in pos array.
     * @param[in] pos The 0-based index values to replace with the given strings in this instance.
     * @param[in] count Number of values in pos parameter.
     * @param devmem Indicates whether the pos parameter points to device memory or CPU memory.
     * @return New instance with the specified strings.
     */
    NVStrings* scatter( const char* str, const int* pos, unsigned int count, bool devmem=true );
    /**
     * @brief Returns a new instance without the specified strings.
     *
     * @param[in] pos The 0-based index of the strings to be ignored when creating a copy of this instance.
     * @param count The number of values in the pos array.
     * @param devmem Indicates whether the pos parameter points to device memory or CPU memory.
     * @return New instance without the strings specified.
     */
    NVStrings* remove_strings( const int* pos, unsigned int count, bool devmem=true );
    /**
     * @brief Returns a sorted copy of the strings managed by this instance.
     *
     * @param stype Specify what attribute to sort: length or character code-points.
     * @param ascending Set to true (default) to sort logically lowest to highest.
     * @param nullfirst Null strings are either always placed first or last regardless of ascending parameter.
     * @return New instance with sorted strings as specified.
     */
    NVStrings* sort( sorttype stype=sorttype::name, bool ascending=true, bool nullfirst=true );
    /**
     * @brief Returns new row index positions for strings sorted in this instance.
     *
     * The strings in this instance are neither modified nor moved in position.
     * @param stype Specify what attribute to srot: length or character code-points.
     *              Sorting by both length and characters is allowed. In this case, length is sorted first.
     * @param ascending Set to true to sort logically lowest to highest.
     * @param[in,out] indexes Pointer to array to be filled in by this method.
     *                        This array must be able to hold size() values.
     * @param nullfirst Null strings are either always placed first or last regardless of ascending parameter.
     * @param devmem Indicates whether the pos parameter points to device memory or CPU memory.
     * @return 0 if successful.
     */
    int order( sorttype stype, bool ascending, unsigned int* indexes, bool nullfirst=true, bool devmem=true );

    // attrs.cu
    /**
     * @brief Retrieve the number of characters in each string.
     * @param[in,out] lengths The length in characters for each string.
     *                        This parameter must be able to hold size() values.
     *                        Null strings will have value -1.
     * @param devmem Indicates whether the indexes parameter points to device memory or CPU memory.
     * @return The total number of characters.
     */
    unsigned int len(int* lengths, bool devmem=true);
    /**
     * @brief Retrieve the number of bytes for each string.
     * @param[in,out] lengths The length in bytes for each string.
     *                        This must point to memory able to hold size() values.
     *                        Null strings will have value -1.
     * @param devmem Indicates whether the indexes parameter points to device memory or CPU memory.
     * @return The total number of bytes.
     */
    size_t byte_count(int* lengths, bool devmem=true);
    /**
     * @brief Returns true for strings that have only alphanumeric characters.
     * @param[in,out] results Array filled in by this method.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether the results parameter points to device memory or CPU memory.
     * @return The number of trues.
     */
    unsigned int isalnum( bool* results, bool devmem=true );
    /**
     * @brief Returns true for strings that have only alphabetic characters.
     * @param[in,out] results Array filled in by this method.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether the results parameter points to device memory or CPU memory.
     * @return The number of trues.
     */
    unsigned int isalpha( bool* results, bool devmem=true );
    /**
     * @brief Returns true for strings that have only digit characters.
     * @param[in,out] results Array filled in by this method.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether the results parameter points to device memory or CPU memory.
     * @return The number of trues.
     */
    unsigned int isdigit( bool* results, bool devmem=true );
    /**
     * @brief Returns true for strings that have only whitespace characters.
     * @param[in,out] results Array filled in by this method.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether the results parameter points to device memory or CPU memory.
     * @return The number of trues.
     */
    unsigned int isspace( bool* results, bool devmem=true );
    /**
     * @brief Returns true for strings that have only decimal characters.
     *        Characters that can be used to extract base10 numbers.
     * @param[in,out] results Array filled in by this method.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether the results parameter points to device memory or CPU memory.
     * @return The number of trues.
     */
    unsigned int isdecimal( bool* results, bool devmem=true );
    /**
     * @brief Returns true for strings that have only numeric characters.
     * @param[in,out] results Array filled in by this method.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether the results parameter points to device memory or CPU memory.
     * @return The number of trues.
     */
    unsigned int isnumeric( bool* results, bool devmem=true );
    /**
     * @brief Returns true for strings that have only lowercase characters.
     * @param[in,out] results Array filled in by this method.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether the results parameter points to device memory or CPU memory.
     * @return The number of trues.
     */
    unsigned int islower( bool* results, bool devmem=true );
    /**
     * @brief Returns true for strings that have only uppercase characters.
     * @param[in,out] results Array filled in by this method.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether the results parameter points to device memory or CPU memory.
     * @return The number of trues.
     */
    unsigned int isupper( bool* results, bool devmem=true );
    /**
     * @brief Returns true for non-empty strings -- non-null strings with at least one character.
     * @param[in,out] results Array filled in by this method.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether the results parameter points to device memory or CPU memory.
     * @return The number of trues.
     */
    unsigned int is_empty( bool* results, bool devmem=true );

    // combine.cu
    /**
     * @brief Concatenates the given strings to this instance of strings and returns as new instance.
     * @param[in] others The number of strings must match this instance.
     * @param[in] separator Null-terminated CPU string that should appear between each instance.
     * @param[in] narep Null-terminated CPU string that should represent any null strings found.
     * @return New instance with this instance concatentated with the provided instance.
     */
    NVStrings* cat( NVStrings* others, const char* separator, const char* narep=nullptr);
    /**
     * @brief Concatenates the given list of strings to this instance of strings and returns as new instance.
     * @param[in] others The number of strings in each item must match this instance.
     * @param[in] separator Null-terminated CPU string that should appear between each instance.
     * @param[in] narep Null-terminated CPU string that should represent any null strings found.
     * @return New instance with this instance concatentated with the provided instances.
     */
    NVStrings* cat( std::vector<NVStrings*>& others, const char* separator, const char* narep=nullptr);
    /**
     * @brief Concatenates all strings into one new string.
     * @param[in] separator Null-terminated CPU string that should appear between each string.
     * @param[in] narep Null-terminated CPU string that should represent any null strings found.
     * @return Resulting instance with one string.
     */
    NVStrings* join( const char* separator="", const char* narep=nullptr );

    // split.cu
    /**
     * @brief Each string is split into a list of new strings.
     *
     * Each string results in a new instance of this class.
     * @param[in] delimiter Null-terminated CPU string identifying the split points within each string.
     * @param maxsplit Maximum number of splits to perform searching left to right.
     * @param[out] results The list of instances for each string.
     * @return Number of strings returned in the results vector.
     */
    int split_record( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results);
    /**
     * @brief Each string is split into a list of new strings.
     *
     * Each string results in a new instance of this class.
     * @param[in] delimiter Null-terminated CPU string identifying the split points within each string.
     * @param maxsplit Maximum number of splits to perform searching right to left.
     * @param[out] results The list of instances for each string.
     * @return Number of strings returned in the results vector.
     */
    int rsplit_record( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results);
    /**
     * @brief Each string is split on whitespace into a list of new strings.
     *
     * Each string results in a new instance of this class.
     * Whitespace is identified by any character code less than equal to ASCII space (0x20).
     * @param maxsplit Maximum number of splits to perform searching left to right.
     * @param[out] results The resulting instances for each string.
     * @return Number of strings returned in the results vector.
     */
    int split_record( int maxsplit, std::vector<NVStrings*>& results);
    /**
     * @brief Each string is split on whitespace into a list of new strings.
     *
     * Each string results in a new instance of this class.
     * Whitespace is identified by any character code less than equal to ASCII space (0x20).
     * @param maxsplit Maximum number of splits to perform searching right to left.
     * @param[out] results The list new instances for each string.
     * @return Number of strings returned in the results vector.
     */
    int rsplit_record( int maxsplit, std::vector<NVStrings*>& results);
    /**
     * @brief Split strings vertically creating new columns of NVStrings instances.
     *
     * The number of columns will be equal to the string with the most splits.
     * @param[in] delimiter Null-terminated CPU string identifying the split points within each string.
     * @param maxsplit Maximum number of splits to perform searching left to right.
     * @param[out] results The list of instances for each string.
     * @return Number of strings returned in the results vector.
     */
    unsigned int split( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results);
    /**
     * @brief Split strings vertically creating new columns of NVStrings instances.
     *
     * The number of columns will be equal to the string with the most splits.
     * @param[in] delimiter Null-terminated CPU string identifying the split points within each string.
     * @param maxsplit Maximum number of splits to perform searching right to left.
     * @param[out] results The list of instances for each string.
     * @return Number of strings returned in the results vector.
     */
    unsigned int rsplit( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results);
    /**
     * @brief Split strings on whitespace vertically creating new columns of NVStrings instances.
     *
     * The number of columns will be equal to the string with the most splits.
     * Whitespace is identified by any character code less than equal to ASCII space (0x20).
     * @param maxsplit Maximum number of splits to perform searching left to right.
     * @param[out] results The list of instances for each string.
     * @return Number of strings returned in the results vector.
     */
    unsigned int split( int maxsplit, std::vector<NVStrings*>& results);
    /**
     * @brief Split strings on whitespace vertically creating new columns of NVStrings instances.
     *
     * The number of columns will be equal to the string with the most splits.
     * Whitespace is identified by any character code less than equal to ASCII space (0x20).
     * @param maxsplit Maximum number of splits to perform searching right to left.
     * @param[out] results The list of instances for each string.
     * @return Number of strings returned in the results vector.
     */
    unsigned int rsplit( int maxsplit, std::vector<NVStrings*>& results);
    /**
     * @brief Each string is split into two strings on the first delimiter found.
     *
     * Three strings are returned for each string: left-half, delimiter itself, right-half.
     * @param[in] delimiter Null-terminated CPU string identifying the split points within each string.
     * @param[out] results The list of instances for each string.
     * @return Number of strings returned in the results vector.
     */
    int partition( const char* delimiter, std::vector<NVStrings*>& results);
    /**
     * @brief Each string is split into two strings on the last delimiter found.
     *
     * Three strings are returned for each string: left-half, delimiter itself, right-half.
     * @param[in] delimiter Null-terminated CPU string identifying the split points within each string.
     * @param[out] results The list of instances for each string.
     * @return Number of strings returned in the results vector.
     */
    int rpartition( const char* delimiter, std::vector<NVStrings*>& results);

    // pad.cu
    /**
     * @brief Concatenate each string with itself the number of times specified.
     * @param count The number of times to repeat each string.
     *              Values of 0 or 1 return a copy of this instance.
     * @return New instance with each string duplicated as specified.
     */
    NVStrings* repeat(unsigned int count);
    /**
     * @brief Padding placement points.
     * Used in the pad() method.
     */
    enum padside {
        left,   ///< Add padding to the left.
        right,  ///< Add padding to the right.
        both    ///< Add padding equally to the right and left.
    };
    /**
     * @brief Add padding to each string using a provided character.
     *
     * If the string is already width or more characters, no padding is performed.
     * No strings are truncated. Null strings result in null strings.
     * @param width The minimum number of characters for each string.
     * @param side Where to place the padding characters.
     * @param[in] fillchar Single character in CPU memory to use for padding.
     *                     A UTF-8 encoded character array of single character may be provided.
     *                     Default is the ASCII space character (0x20).
     * @return New instance with each string padded appropriately.
     */
    NVStrings* pad(unsigned int width, padside side, const char* fillchar=nullptr);
    /**
     * @brief Add padding to the left of each string using a provided character.
     *
     * If the string is already width or more characters, no padding is performed.
     * No strings are truncated. Null strings result in null strings.
     * @param width The minimum number of characters for each string.
     * @param[in] fillchar Single character in CPU memory to use for padding.
     *                     A UTF-8 encoded character array of single character may be provided.
     *                     Default is the ASCII space character (0x20).
     * @return New instance with each string padded appropriately.
     */
    NVStrings* ljust( unsigned int width, const char* fillchar=nullptr );
    /**
     * @brief Add padding to the left and right of each string using a provided character.
     *
     * If the string is already width or more characters, no padding is performed.
     * No strings are truncated. Null strings result in null strings.
     * If left/right split results in odd character padding the extra fill character will added to the right.
     * @param width The minimum number of characters for each string.
     * @param[in] fillchar Single character in CPU memory to use for padding.
     *                     A UTF-8 encoded character array of single character may be provided.
     *                     Default is the ASCII space character (0x20).
     * @return New instance with each string padded appropriately.
     */
    NVStrings* center( unsigned int width, const char* fillchar=nullptr );
    /**
     * @brief Add padding to the right of each string using a provided character.
     *
     * If the string is already width or more characters, no padding is performed.
     * No strings are truncated. Null strings result in null strings.
     * @param width The minimum number of characters for each string.
     * @param[in] fillchar Single character in CPU memory to use for padding.
     *                     A UTF-8 encoded character array of single character may be provided.
     *                     Default is the ASCII space character (0x20).
     * @return New instance with each string padded appropriately.
     */
    NVStrings* rjust( unsigned int width, const char* fillchar=nullptr );
    /**
     * @brief Pads strings with leading zeros.
     *
     * If the string is already width or more characters, no padding is performed.
     * No strings are truncated. Null strings result in null strings.
     * The zeros will be filled after '+' or '-' if found in the first character position.
     * @param width The minimum number of characters for each string.
     * @return New instance with each string padded appropriately.
     */
    NVStrings* zfill( unsigned int width );
    /**
     * @brief This inserts new-line characters (ASCII 0x0A) into each string in place of spaces.
     *
     * Attempts to make each line less than or equal to width characters.
     * If a string or characters is longer than width, the line is split on the next
     * closest space character.
     * @param width The minimum number of characters for a line in each string.
     * @return New instance with each string added with new-line characters appropriately.
     */
    NVStrings* wrap( unsigned int width );

    // substr.cu
    /**
     * @brief Return a specific character (as a string) by position for each string.
     *
     * The return instance will have strings containing no more than 1 character each.
     * @param pos The 0-based index of the character location within each string.
     * @return Instance containing single-character strings.
     */
    NVStrings* get(unsigned int pos);
    /**
     * @brief Returns a substring of each string.
     * @param start First position (0-based index of characters) to start retrieving.
     * @param stop Last position (0-based index of characters) to stop retrieving.
     * @param step This can be used to retrieve interval of characters between start and stop.
     * @return New instance containing strings with characters only between start and stop.
     */
    NVStrings* slice( int start=0, int stop=-1, int step=1 );
    /**
     * @brief Returns a substring of each string.
     *
     * This method allows specifying unique start and end positions for each string.
     * Use 0 to specify the start of any specific string.
     * Use -1 to specify the end of any specific string.
     * @param[in] starts Array of positions to start retrieving.
     *                   This must point to device memory of size() values.
     * @param[in] ends Array of positions to stop retrieving.
     *                 This must point to device memory of size() values.
     * @return New instance containing strings with characters specified.
     */
    NVStrings* slice_from( const int* starts=nullptr, const int* ends=nullptr );

    // extract.cu
    /**
     * @brief Returns a list of strings for each group specified in the given regular expression pattern.
     *
     * This will create an NVStrings instance for each groups.
     * @param[in] pattern The regular expression pattern with group indicators.
     * @param[out] results The instances containing the extract strings.
     * @return The number of instances returned in results.
     */
    int extract( const char* pattern, std::vector<NVStrings*>& results );

    // extract_record.cu
    /**
     * @brief Returns a list of strings for each group specified in the given regular expression pattern.
     *
     * This will return a new instance of this class for each string in this instance.
     * @param[in] pattern The regular expression pattern with group indicators.
     * @param[out] results The instances containing the extract strings.
     * @return The number of instances returned in results.
     */
    int extract_record( const char* pattern, std::vector<NVStrings*>& results );

    // modify.cu
    /**
     * @brief Inserts the specified string (repl) into each string at the specified position.
     * @param[in] repl Null-terminated CPU string to insert into each string.
     * @param start The 0-based character position in each string to start the replace.
     * @param stop The 0-based character position to complete the replace.
     *             The default (-1) indicates replace to the end of each string.
     * @return New instance with updated strings.
     */
    NVStrings* slice_replace( const char* repl, int start=0, int stop=-1 );
    /**
     * @brief Replaces occurrences found of one string with another string in each string of this instance.
     *
     * This method does not use regular expression to search for the target \p str to replace.
     * @param[in] str Null-terminated CPU string to search for replacement.
     * @param[in] repl Null-terminated CPU string to replace any found strings.
     * @param maxrepl Maximum number of times to search and replace.
     * @return New instance with the characters replaced appropriately.
     */
    NVStrings* replace( const char* str, const char* repl, int maxrepl=-1 );
    /**
     * @brief Replaces any occurrences found in list of strings with corresponding string in each string of this instance.
     *
     * This method does not use regular expression to search for the target string to replace.
     * All occurrences found of any of the specified strings are replaced.
     * If only a single string is present in repls, it is used for replacement for all targets.
     * @param[in] strs List of strings to search for replacement.
     * @param[in] repls List of strings to substitute for the corresponding string in strs.
     *                  Must have the same number of strings as strs or contain just a single string.
     * @return New instance with the characters replaced appropriately.
     */
    NVStrings* replace( NVStrings& strs, NVStrings& repls );
    /**
     * @brief Translate characters in each string using the character-mapping table provided.
     * @param[in] table Individual Unicode characters and their replacment counterparts.
     * @param count The number of entries in table.
     * @return New instance with the appropriate characters replace.
     */
    NVStrings* translate( std::pair<unsigned,unsigned>* table, unsigned int count );
    /**
     * @brief Replace null strings with specified string.
     * @param[in] str Null-terminated CPU string to place instead of null strings.
     * @return Copy of this instance with null strings replaced.
     */
    NVStrings* fillna( const char* str );
    /**
     * @brief Replace null strings with corresponding strings from the parameter.
     *
     * Strings are matched by index. Strings that are not null are not replaced.
     * @param[in] strs Strings to replace nulls. The number of strings must match this instance.
     * @return Copy of this instance with null strings replaced.
     */
    NVStrings* fillna( NVStrings& strs );
    /**
     * @brief Inserts the specified string (repl) into each string at the specified position.
     * @param[in] repl Null-terminated CPU string to insert into each string.
     * @param pos The 0-based character position in each string to start the replace.
     * @return New instance with updated strings.
     */
    NVStrings* insert( const char* repl, int pos=0 );

    // replace.cu
    /**
     * @brief Replaces occurrences found of one string with another string in each string of this instance.
     *
     * This method uses the given regular expression pattern to search for the target \p str to replace.
     * @param[in] pattern Null-terminated CPU string with regular expression.
     * @param[in] repl Null-terminated CPU string to replace any found strings.
     * @param maxrepl Maximum number of times to search and replace.
     * @return New instance with the characters replaced appropriately.
     */
    NVStrings* replace_re( const char* pattern, const char* repl, int maxrepl=-1 );

    // replace_multi.cu
    /**
     * @brief Replaces occurrences found of string list with corresponding strings in each string of this instance.
     *
     * This method uses the given regular expression patterns to search for the target \p str to replace.
     * If only a single string is present in repls, it is used for replacement for all targets.
     * @param[in] patterns Null-terminated CPU strings with regular expressions.
     * @param[in] repls Strings to replace any found strings.
     *                  Must have the same number of strings as strs or contain just a single string.
     * @return New instance with the characters replaced appropriately.
     */
    NVStrings* replace_re( std::vector<const char*>& patterns, NVStrings& repls );

    // replace_backref.cu
    /**
     * @brief Extract values using pattern and place them repl as indicated by backref indicators.
     * @param[in] pattern Null-terminated CPU string with regular expression.
     * @param[in] repl Null-terminated CPU string with back-reference indicators.
     * @return New instance with the characters replaced appropriately.
     */
    NVStrings* replace_with_backrefs( const char* pattern, const char* repl );

    // strip.cu
    /**
     * @brief Remove the specified character(s) if found at the beginning of each string.
     * @param[in] to_strip Null-terminated CPU string of characters (UTF-8 encoded) to remove.
     * @return New instance with characters removed from each string.
     */
    NVStrings* lstrip( const char* to_strip );
    /**
     * @brief Remove the specified character(s) if found at the beginning or end of each string.
     * @param[in] to_strip Null-terminated CPU string of characters (UTF-8 encoded) to remove.
     * @return New instance with characters removed from each string.
     */
    NVStrings* strip( const char* to_strip );
    /**
     * @brief Remove the specified character(s) if found at the end of each string.
     * @param[in] to_strip Null-terminated CPU string of characters (UTF-8 encoded) to remove.
     * @return New instance with characters removed from each string.
     */
    NVStrings* rstrip( const char* to_strip );

    // case.cu
    /**
     * @brief Return new instance modifying uppercase characters to lowercase.
     * @return New instance with each string case modified.
     */
    NVStrings* lower();
    /**
     * @brief Return new instance modifying lowercase characters to uppercase.
     * @return New instance with each string case modified.
     */
    NVStrings* upper();
    /**
     * @brief Return new instance modifying the first character to uppercase and lower-casing the rest.
     * @return New instance with each string case modified.
     */
    NVStrings* capitalize();
    /**
     * @brief Return new instance modifying uppercase characters to lowercase and vice versa.
     * @return New instance with each string case modified.
     */
    NVStrings* swapcase();
    /**
     * @brief Return new instance modifying first characters after space to uppercase and lower-casing the rest.
     * @return New instance with each string case modified.
     */
    NVStrings* title();

    // find.cu
    /**
     * @brief Compare string to all the strings in this instance.
     *
     * The returned values are 0 when the string matches and positive or negative depending
     * the difference between the first non-matching characters.
     * @param[in] str Null-terminated CPU string to compare each string.
     * @param[in,out] results Array this method will fill in with the results.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return Number of matches.
     */
    unsigned int compare( const char* str, int* results, bool devmem=true );
    /**
     * @brief Search for a string within each string in this instance.
     * @param[in] str Null-terminated CPU string to search for.
     * @param start 0-based character position to start search for each string.
     * @param end 0-based character position to stop searching for each string.
     *            Value of -1 indicates search to the end of the string.
     * @param[in,out] results Array this method will fill in with the results.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return The number of positive (>=0) results.
     */
    unsigned int find( const char* str, int start, int end, int* results, bool devmem=true );
    /**
     * @brief Search from the end for a string within each string in this instance.
     * @param[in] str Null-terminated CPU string to search for.
     * @param start 0-based character position to start search for each string.
     * @param end 0-based character position to stop searching for each string.
     *            Value of -1 indicates search from the end of the string.
     * @param[in,out] results Array this method will fill in with the results.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return The number of positive (>=0) results.
     */
    unsigned int rfind( const char* str, int start, int end, int* results, bool devmem=true );
    /**
     * @brief Search for a string within each string in this instance.
     *
     * This methods allows for searching in unique ranges for each string.
     * @param[in] str Null-terminated CPU string to search for.
     * @param[in] starts Array of 0-based character position to start search for each string.
     *                   This must have size() values.
     * @param[in] ends Array of 0-based character position to stop searching for each string.
     *                This must have size() values.
     *                Values of -1 indicate search to the end of that string.
     * @param[in,out] results Array this method will fill in with the results.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return The number of positive (>=0) results.
     */
    unsigned int find_from( const char* str, int* starts, int* ends, int* results, bool devmem=true );
    /**
     * @brief Search multiple strings
     * @param[in] strs List of strings to search for.
     * @param[in,out] results Array this method will fill in with the results.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return The number of positive (>=0) results.
     */
    unsigned int find_multiple( NVStrings& strs, int* results, bool devmem=true );
    /**
     * @brief Search for string within each string of this instance.
     * @param[in] str Null-terminated CPU string to search for in each string.
     * @param[in,out] results Array this method will fill in with the results.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return Number of matches.
     */
    int contains( const char* str, bool* results, bool devmem=true );
    /**
     * @brief Check each argument string matches with the corresponding strings in this list.
     * @param[in] strs Strings to compare against. The number of strings must match with this instance.
     * @param[in,out] results Array this method will fill in with the results.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return Number of matches.
     */
    int match_strings( NVStrings& strs, bool* results, bool devmem=true );
    /**
     * @brief Compares the beginning of each string with the specified string.
     * @param[in] str Null-terminated CPU string to search for.
     * @param[in,out] results Array this method will fill in with the results.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return Number of matches.
     */
    unsigned int startswith( const char* str, bool* results, bool devmem=true );
    /**
     * @brief Compares the end of each string with the specified string.
     * @param[in] str Null-terminated CPU string to search for.
     * @param[in,out] results Array this method will fill in with the results.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return Number of matches.
     */
    unsigned int endswith( const char* str, bool* results, bool devmem=true );

    // findall.cu
    /**
     * @brief Return all occurrences of the specified regular expression pattern in each string.
     * @param[in] pattern The regulare expression pattern to search.
     * @param[out] results List of instances.
     * @return Number of strings returned in the results vector.
     */
    int findall( const char* pattern, std::vector<NVStrings*>& results );

    // findall_record.cu
    /**
     * @brief Return all occurrences of the specified regular expression pattern in each string.
     * @param[in] pattern The regular expression pattern to search.
     * @param[out] results List of instances.
     * @return Number of strings returned in the results vector.
     */
    int findall_record( const char* pattern, std::vector<NVStrings*>& results );

    // count.cu
    /**
     * @brief Search for regular expression pattern within each string of this instance.
     * @param[in] pattern Null-terminated CPU string of regular expression.
     * @param[in,out] results Array this method will fill in with the results.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return Number of matches.
     */
    int contains_re( const char* pattern, bool* results, bool devmem=true );
    /**
     * @brief Search for regular expression pattern match at the beginning of each string.
     * @param[in] pattern Null-terminated CPU string of regular expression.
     * @param[in,out] results Array this method will fill in with the results.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return Number of matches.
     */
    int match( const char* pattern, bool* results, bool devmem=true );
    /**
     * @brief Search for regular expression pattern match and count their occurrences for each string.
     * @param[in] pattern Null-terminated CPU string of regular expression.
     * @param[in,out] results Array this method will fill in with the results.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return Number of matches.
     */
    int count_re( const char* pattern, int* results, bool devmem=true );

    // convert.cu
    /**
     * @brief Returns integer values represented by each string.
     * @param[in,out] results Array this method will fill in with the results.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return Number of non-zero values.
     */
    int stoi(int* results, bool devmem=true);
    /**
     * @brief Returns long integer values represented by each string.
     * @param[in,out] results Array this method will fill in with the results.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return Number of non-zero values.
     */
    int stol(long* results, bool devmem=true);
    /**
     * @brief Returns integer values represented by each string assuming hex characters.
     * @param[in,out] results Array this method will fill in with the results.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return Number of non-zero values.
     */
    int htoi(unsigned int* results, bool devmem=true);
    /**
     * @brief Returns float values represented by each string.
     * @param[in,out] results Array this method will fill in with the results.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return Number of non-zero values.
     */
    int stof(float* results, bool devmem=true);
    /**
     * @brief Returns double values represented by each string.
     * @param[in,out] results Array this method will fill in with the results.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return Number of non-zero values.
     */
    int stod(double* results, bool devmem=true);
    /**
     * @brief Returns unsigned 32-bit hash value for each string.
     * @param[in,out] results Array this method will fill in with the results.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return Number of non-zero values.
     */
    int hash( unsigned int* results, bool devmem=true );
    /**
     * @brief Returns string representation for the provided integers.
     * @param[in] values Array of integers to convert to strings.
     * @param count The number of integers in the values parameter.
     * @param[in] nullbitmask Indicates which entries should result in a null string.
     *                        If specified, this array should be at least (count+7)/8 bytes.
     *                        The bits are expected to be organized in Arrow format.
     * @param devmem Indicates whether results and nullbitmask points to device memory or CPU memory.
     * @return New instance with string representation of the values as appropriate.
     */
    static NVStrings* itos(const int* values, unsigned int count, const unsigned char* nullbitmask=nullptr, bool devmem=true);
    /**
     * @brief Returns string representation for the provided long integers.
     * @param[in] values Array of long integers to convert to strings.
     * @param count The number of long integers in the values parameter.
     * @param[in] nullbitmask Indicates which entries should result in a null string.
     *                        If specified, this array should be at least (count+7)/8 bytes.
     *                        The bits are expected to be organized in Arrow format.
     * @param devmem Indicates whether results and nullbitmask points to device memory or CPU memory.
     * @return New instance with string representation of the values as appropriate.
     */
    static NVStrings* ltos(const long* values, unsigned int count, const unsigned char* nullbitmask=nullptr, bool devmem=true);
    /**
     * @brief Returns string representation for the provided float values.
     *
     * Upto 10 significant digits are recorded.
     * Numbers above 10^9 and numbers below 10^5 may be converted to scientific notation.
     * @param[in] values Array of float values to convert to strings.
     * @param count The number of float values in the values parameter.
     * @param[in] nullbitmask Indicates which entries should result in a null string.
     *                        If specified, this array should be at least (count+7)/8 bytes.
     *                        The bits are expected to be organized in Arrow format.
     * @param devmem Indicates whether results and nullbitmask points to device memory or CPU memory.
     * @return New instance with string representation of the values as appropriate.
     */
    static NVStrings* ftos(const float* values, unsigned int count, const unsigned char* nullbitmask=nullptr, bool devmem=true);
    /**
     * @brief Returns string representation for the provided double float values.
     *
     * Upto 10 significant digits are recorded.
     * Numbers above 10^9 and numbers below 10^5 may be converted to scientific notation.
     * @param[in] values Array of double float values to convert to strings.
     * @param count The number of doiuble float values in the values parameter.
     * @param[in] nullbitmask Indicates which entries should result in a null string.
     *                        If specified, this array should be at least (count+7)/8 bytes.
     *                        The bits are expected to be organized in Arrow format.
     * @param devmem Indicates whether results and nullbitmask points to device memory or CPU memory.
     * @return New instance with string representation of the values as appropriate.
     */
    static NVStrings* dtos(const double* values, unsigned int count, const unsigned char* nullbitmask=nullptr, bool devmem=true);
    /**
     * @brief Returns boolean representation of the strings in this instance.
     *
     * This will do a compare of the target string and return true when matched and false when not.
     * @param true_string What text to identify a string as 'true'.
     * @param[in,out] results Array this method will fill in with the results.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return Number of 'true' values.
     */
    int to_bools( bool* results, const char* true_string, bool devmem=true );
    /**
     * @brief Returns string representation for the provided booleans.
     * @param[in] values Array of booleans to convert to strings.
     * @param count The number of elements in the values parameter.
     * @param true_string What string to use for 'true'.
     * @param false_string What string to use for 'false'.
     * @param[in] nullbitmask Indicates which entries should result in a null string.
     *                        If specified, this array should be at least (count+7)/8 bytes.
     *                        The bits are expected to be organized in Arrow format.
     * @param devmem Indicates whether results and nullbitmask points to device memory or CPU memory.
     * @return New instance with string representation.
     */
    static NVStrings* create_from_bools(const bool* values, unsigned int count, const char* true_string, const char* false_string, const unsigned char* nullbitmask=nullptr, bool devmem=true);
    /**
     * @brief Returns integer representation of IPv4 address.
     * @param[in,out] results Array this method will fill in with the results.
     *                        This must point to memory able to hold size() values.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return Number of non-zero values.
     */
    int ip2int( unsigned int* results, bool devmem=true );
    /**
     * @brief Returns string representation of IPv4 address (v4) for the provided integer values.
     * @param[in] values Array of integers to convert to strings.
     * @param count The number of integers in the values parameter.
     * @param[in] nullbitmask Indicates which entries should result in a null string.
     *                        If specified, this array should be at least (count+7)/8 bytes.
     *                        The bits are expected to be organized in Arrow format.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return New instance with string representation of the integers as appropriate.
     */
    static NVStrings* int2ip( const unsigned int* values, unsigned int count, const unsigned char* nullbitmask=nullptr, bool devmem=true);
    /**
     * @brief  Units for timestamp conversion.
     */
    enum timestamp_units {
        years,           ///< precision is years
        months,          ///< precision is months
        days,            ///< precision is days
        hours,           ///< precision is hours
        minutes,         ///< precision is minutes
        seconds,         ///< precision is seconds
        ms,              ///< precision is milliseconds
        us,              ///< precision is microseconds
        ns               ///< precision is nanoseconds
    };

    // datetime.cu
    /**
     * @brief Returns integer representation date-time string.
     *
     * @param[in] format Format must include strptime format specifiers though only the following are
     *                   supported: %Y,%y,%m,%d,%H,%I,%p,%M,%S,%f,%z
     *                   Default format is "%Y-%m-%dT%H:%M:%SZ"
     * @param[in] units The values will be created in these units.
     * @param[in,out] results Array this method will fill in with the results.
     *                        This must point to memory able to hold size() values.
     * @param[in] devmem Indicates whether results points to device memory or CPU memory.
     * @return Number of non-zero values.
     */
    int timestamp2long( const char* format, timestamp_units units, unsigned long* results, bool devmem=true );
    /**
     * @brief Returns string representation of UTC timestamp in milliseconds from Epoch time.
     *
     * Each string will be created with the following format: YYYY-MM-DDThh:mm:ss.sssZ
     *
     * @param[in] values Array of integers to convert to strings.
     * @param count The number of integers in the values parameter.
     * @param units Time units of the values array.
     * @param[in] format Format must include strftime format specifiers though only the following are
     *                   supported: %Y,%y,%m,%d,%H,%I,%p,%M,%S,%f,%z
     *                   Default format is "%Y-%m-%dT%H:%M:%SZ"
     * @param[in] nullbitmask Indicates which entries should result in a null string.
     *                        If specified, this array should be at least (count+7)/8 bytes.
     *                        The bits are expected to be organized in Arrow format.
     * @param devmem Indicates whether results points to device memory or CPU memory.
     * @return New instance with string representation of the integers as appropriate.
     */
    static NVStrings* long2timestamp( const unsigned long* values, unsigned int count, timestamp_units units, const char* format, const unsigned char* nullbitmask=nullptr, bool devmem=true);

    // urlencode.cu
    /**
     * @brief URL-encodes each string and returns the new instance.
     *        This uses UTF-8 encoding style and does no error checking on the strings.
     *        All letters and digits are not encoded as well as characters '.','_','~','-'.
     *
     * @return New instance of url-encoded strings
     */
    NVStrings* url_encode();
    /**
     * @brief URL-decodes each string and returns the new instance.
     *        This expects UTF-8 encoding style and does no error checking on the strings.
     *
     * @return New instance of url-encoded strings
     */
    NVStrings* url_decode();

    /**
     * @brief Output strings to stdout.
     * @param pos First string to start printing.
     * @param end Last string to print. Default (-1) prints from pos to the last string in this instance.
     * @param maxwidth Strings longer than this are truncated in the output. Default (-1) prints the entire string.
     * @param[in] delimiter Line separator character to use between each string.
     */
    void print( int pos=0, int end=-1, int maxwidth=-1, const char* delimiter = "\n" );

    /**
     * @brief Computes a variety of statistics for the strings/characters in this instance.
     * @param[out] stats Fills in this statistics structure.
     */
    void compute_statistics(StringsStatistics& stats);

};
