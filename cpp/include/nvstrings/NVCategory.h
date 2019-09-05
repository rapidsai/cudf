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
#include "base_category.h"

/**
 * @file NVCategory.h
 * @brief Class definition for NVCategory.
 */

struct nvcategory_ipc_transfer;
class NVStrings;
class NVCategoryImpl;
/**
 * @brief Manages a list of strings for a category and their associated indexes.
 *
 * This operates like a dictionary with strings as keys and values as integers.
 * Unique strings are assigned unique integer values within this instance.
 * Each value represents the index position into the keyset.
 * The order of values is the original order when the category was created
 * from a list of strings.
 * @par
 * The methods are meant to match more or less directly with its python
 * counterpart, @p nvcategory.py. And the operations strive to mimic the
 * behavoir of the equivalent Pandas strings methods.
 * @par
 * An instance of this class is immutable and operations that modify
 * or modify keys or values will return a new instance.
 * @par
 * All methods accept and return only UTF-8 encoded strings.
 * @nosubgrouping
 */
class NVCategory : base_category_type
{
    NVCategoryImpl* pImpl;
    NVCategory();
    NVCategory(const NVCategory&);
    ~NVCategory();
    NVCategory& operator=(const NVCategory&);

public:

    /** @name Create NVCategory instance from strings
     *  Use these static methods to create a new instance of this class given a list of strings.
     *  Character array should be encoded in UTF-8.
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
    static NVCategory* create_from_array(const char** strs, unsigned int count);
    /**
     * @brief Create an instance from an array of string/length pairs.
     * @param[in] strs Array of pointer/length pairs to UTF-8 encoded strings.
     *                 Lengths should be in bytes and should not include null-terminator.
     *                 Null pointers identify null strings.
     *                 Zero lengths identify empty strings.
     * @param count The number of pairs in the \p strs array.
     * @param devmem Set to true (default) if pointers are to device memory.
     *
      @return Instance with the strings copied into device memory.
     */
    static NVCategory* create_from_index(std::pair<const char*,size_t>* strs, unsigned int count, bool devmem=true);
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
    static NVCategory* create_from_offsets(const char* strs, unsigned int count, const int* offsets, const unsigned char* nullbitmask=0, int nulls=0, bool devmem=true);
    /**
     * @brief Create an instance from an NVStrings instance.
     * @param[in] strs Strings to create this category.
     * @return Instance with copy of the strings provided.
     */
    static NVCategory* create_from_strings(NVStrings& strs);
    /**
     * @brief Create an instance from one or more NVStrings instances.
     * @param[in] strs A vector of NVStrings instance pointers to use.
     *                 These can be safely freed by the caller on return from this method.
     * @return Instance with copy of the strings provided.
     */
    static NVCategory* create_from_strings(std::vector<NVStrings*>& strs);
    /**
     * @brief Create an instance from other NVCategory instances.
     * @param[in] cats A vector of NVCategories instance pointers to use.
     *                 These can be safely freed by the caller on return from this method.
     * @return Instance with copy of the categories provided.
     */
    static NVCategory* create_from_categories(std::vector<NVCategory*>& cats);
    /**
     * @brief Create an instance from an IPC-transfer object built from nvcategory_ipc_transfer.
     *
     * @param[in] ipc Data needed to create a new instance.
     * @return Instance with data provided.
     */
    static NVCategory* create_from_ipc( nvcategory_ipc_transfer& ipc );
    ///@}

    /**
     * @brief Use this method to free any instance created by methods in this class.
     *
     * All device and CPU memory used by this instance is freed.
     * Caller should no longer use this instance after calling this method.
     * @param[in] inst The instance to free.
     */
    static void destroy(NVCategory* inst);

    /**
     * @brief Returns string name of this category.
     */
    const char* get_type_name();

    /**
     * @brief Returns the number of values.
     */
    unsigned int size();
    /**
     * @brief Returns the number of keys.
     */
    unsigned int keys_size();
    /**
     * @brief Returns true if any values point to a null key.
     */
    bool has_nulls();

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
     * @brief Create IPC-transfer data from this instance.
     *
     * @param[in,out] ipc Structure will be set with data needed by create_from_ipc method.
     * @return 0 if successful.
     */
    int create_ipc_transfer( nvcategory_ipc_transfer& ipc );
    /**
     * @brief Set bit-array identifying the null values.
     *
     * The bits are arranged using the Arrow format for bitmask.
     * @param[in,out] bitarray Byte array to be filled in by this method.
     *                         The array must be at least (size()+7)/8 bytes.
     * @param devmem Identifies the provided \p bitarray parameter points to device memory (default) or CPU memory.
     * @return The number of nulls found.
     */
    int set_null_bitarray( unsigned char* bitarray, bool devmem=true );
    /**
     * @brief Create a new instance from this instance.
     * @return New instance as duplicate of this instance.
     */
    NVCategory* copy();

    /**
     * @brief Create a new NVStrings instance of the keys in this instance.
     * @return Strings in the keyset.
     */
    NVStrings* get_keys();

    /**
     * @brief Return single keyset index given string.
     * @param index Value to map to a keyset index.
     * @return Index in keyset or -1 if not found.
     */
    int get_value(unsigned int index);
    /**
     * @brief Return single keyset index given a value.
     * @param str String in the keyset.
     * @return Index of string or -1 if not found.
     */
    int get_value(const char* str);
    /**
     * @brief This method can be used to return possible position index for unknown keys.
     * @param str Null-terminated UTF-8 encoded string to check for within the keyset.
     * @return Lower and upper bound where this string would appear in the keyset.
     *         If the string is found in the keyset both pair values will equal the key's index.
     */
    std::pair<int,int> get_value_bounds(const char* str);

    /**
     * @brief Return category values for all indexes.
     * @param[in,out] results The array will be filled with the values for this instance.
     * @param devmem Indicates whether the \p results parameter points to device or CPU memory.
     * @return Number of values.
     */
    int get_values( int* results, bool devmem=true );
    /**
     * @brief Returns pointer to internal values array.
     *
     * Caller should neither modify nor free this memory.
     * @return Pointer to internal device memory.
     */
    const int* values_cptr();

    /**
     * @brief Return index positions of values for the given key index.
     * @param index The key index to retrieve values for.
     * @param[in,out] results The array will be filled with the indexes for this instance.
     * @param devmem Indicates whether the \p results parameter points to device or CPU memory.
     * @return Number of values returned or -1 if the key does not exist.
     */
    int get_indexes_for( unsigned int index, int* results, bool devmem=true );
    /**
     * @brief Return values for given key.
     * @param str Null-terminated UTF-8 encoded string.
     * @param[in,out] results The array will be filled with the indexes for this instance.
     * @param devmem Indicates whether the \p results parameter points to device or CPU memory.
     * @return Number of values returned or -1 if the key does not exist.
     */
    int get_indexes_for( const char* str, int* results, bool devmem=true );

    /**
     * @brief Creates a new instance incorporating the new strings into the keyset.
     *
     * The values are updated like the new strings were appended to original strings list.
     * @param strs New strings to add.
     * @return New instance with new keys and values.
     */
    NVCategory* add_strings(NVStrings& strs);
    /**
     * @brief Creates a new instance without the strings specified.
     *
     * The values are updated like the new strings were removed from the original strings list.
     * @param strs Strings to remove.
     * @return New instance with new keys and values.
     */
    NVCategory* remove_strings(NVStrings& strs);

    /**
     * @brief Creates a new instance adding the specified strings as keys and remapping the values.
     *
     * The values are adjusted to the new keyset positions. No new values are added.
     * @param strs New strings to add to the keyset.
     * @return New instance with new keys and values.
     */
    NVCategory* add_keys_and_remap(NVStrings& strs);
    /**
     * @brief Creates a new instance removing the keys matching the specified strings and remapping the values.
     *
     * The values are adjusted to the new keyset positions. No values are removed.
     * Any abandoned values (their key no longer exists) are set to -1.
     * @param strs Strings to remove from the keyset.
     * @return New instance with new keys and values.
     */
    NVCategory* remove_keys_and_remap(NVStrings& strs);
    /**
     * @brief Creates a new instance using the specified strings as keys causing add/remove as appropriate.
     *
     * The values are adjusted to the new keyset positions. No new values are added or removed.
     * Any abandoned values (their key no longer exists) are set to -1.
     * @param strs Strings to be used for the new keyset.
     * @return New instance with new keys and values.
     */
    NVCategory* set_keys_and_remap(NVStrings& strs);
    /**
     * @brief Creates a new instance removing any keys that are not represented in the values.
     *
     * The values are adjusted to the new keyset positions. Any -1 values are removed.
     * @return New instance with new keys and values.
     */
    NVCategory* remove_unused_keys_and_remap();
    /**
     * @brief Merges this category with another creating a new category.
     *
     * The merge operation preserves the original keyset positions from this instance.
     * New keys are appended to this keyset and therefore may no longer be sorted.
     * The values are appended and adjusted to the new keyset positions.
     * The first set of values from this instance are unchanged.
     * @param cat Instance to merge with.
     * @return New instance with new keys and values.
     */
    NVCategory* merge_category(NVCategory& cat);
    /**
     * @brief Merges this category with another creating a new category and remapping all values.
     *
     * This merge operation creates a new combined keyset and appends the values.
     * All the values are then adjusted to the new keyset positions.
     * @param cat Instance to merge with.
     * @return New instance with new keys and values.
     */
    NVCategory* merge_and_remap(NVCategory& cat);

    /**
     * @brief Essentially converts this instance to original strings list.
     * @return New instance of the strings gathered from the values and the keyset.
     */
    NVStrings* to_strings();
    /**
     * @brief Create a new strings instance identified by the specified index values.
     * @param[in] pos The index values to build the strings instance from.
     * @param elems The number of values in the \p pos array.
     * @param devmem Indicates whether the \p results parameter points to device or CPU memory.
     * @return New strings instance.
     */
    NVStrings* gather_strings( const int* pos, unsigned int elems, bool devmem=true );
    /**
     * @brief Create a new instance identified by the specified values and remap the values to resulting keyset.
     * @param[in] pos The index values to build the strings instance from.
     * @param elems The number of values in the \p pos array.
     * @param devmem Indicates whether the \p results parameter points to device or CPU memory.
     * @return New instance with new keys and values.
     */
    NVCategory* gather_and_remap( const int* pos, unsigned int elems, bool devmem=true );
    /**
     * @brief Create new category instance using the current keys but with the specified values.
     * @param[in] pos The index values to use for the new instance.
     * @param elems The number of values in the \p pos array.
     * @param devmem Indicates whether the \p results parameter points to device or CPU memory.
     * @return New instance with new keys and values.
     */
    NVCategory* gather( const int* pos, unsigned int elems, bool devmem=true );
};
