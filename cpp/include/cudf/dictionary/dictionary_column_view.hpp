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

namespace cudf
{
/**
 * @brief Given a column-view of DICTIONARY type, an instance of this class
 * provides a wrapper on this compound column for dictionary operations.
 *
 * A dictionary column contains a dictionary and a set of indices.
 * The dictionary is a sorted set of unique values for the column.
 * The indices values are position indices into the dictionary.
 */
class dictionary_column_view : public column_view
{
public:
    dictionary_column_view( column_view dictionary_column );
    dictionary_column_view( dictionary_column_view&& dictionary_view ) = default;
    dictionary_column_view( const dictionary_column_view& dictionary_view ) = default;
    ~dictionary_column_view() = default;
    dictionary_column_view& operator=(dictionary_column_view const&) = default;
    dictionary_column_view& operator=(dictionary_column_view&&) = default;

    /**
     * @brief Returns the internal dictionary column.
     */
    column_view dictionary() const noexcept { return _dictionary; }

    /**
     * @brief Returns the column of indices
     */
    column_view indices() const;

private:
    column_view _dictionary;
};

namespace dictionary
{

/**
 * @brief Create a new dictionary column by adding the new dictionary elements
 * to the existing dictionary_column.
 *
 * The indices are updated if any of the new dictionary elements are sorted
 * before any of the existing dictionary elements.
 *
 * ```
 * d1 = {["a","c","d"],[2,0,1,0]}
 * d2 = add_dictionary(d1,["b","c"])
 * d2 is now {["a","b","c","d"],[3,0,2,0]}
 * ```
 *
 * @throw cudf_logic_error if the dictionary type does not match the dictionary type in
 * the dictionary_column.
 *
 * @param dictionary_column Existing dictionary column.
 * @param dictionary New keys to incorporate into the dictionary_column
 * @param mr Resource for allocating memory for the output.
 * @return New dictionary column.
 */
std::unique_ptr<column> add_dictionary( dictionary_column_view const& dictionary_column,
                                        column_view const& dictionary,
                                        rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Create a new dictionary column by removing the specified dictionary elements
 * from the existing dictionary_column.
 *
 * The indices are updated to the new positions of the remaining dictionary elements.
 * Any indices pointing to removed elements are set to null.
 *
 * ```
 * d1 = {["a","c","d"],[2,0,1,0]}
 * d2 = remove_dictionary(d1,["b","c"])
 * d2 is now {["a","d"],[1,0,null,0]}
 * ```
 *
 * @throw cudf_logic_error if the dictionary type does not match the dictionary type in
 * the dictionary_column.
 *
 * @param dictionary_column Existing dictionary column.
 * @param dictionary New keys to remove from the dictionary_column
 * @param mr Resource for allocating memory for the output.
 * @return New dictionary column.
 */
std::unique_ptr<column> remove_dictionary( dictionary_column_view const& dictionary_column,
                                           column_view const& dictionary,
                                           rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Create a new dictionary column by removing any dictionary elements
 * that are not referenced by any of the indices.
 *
 * The indices are updated to the new position values of the remaining dictionary elements.
 *
 * ```
 * d1 = {["a","c","d"],[2,0,2,0]}
 * d2 = remove_unused_dictionary(d1)
 * d2 is now {["a","d"],[1,0,1,0]}
 * ```
 *
 * @param dictionary_column Existing dictionary column.
 * @param mr Resource for allocating memory for the output.
 * @return New dictionary column.
 */
std::unique_ptr<column> remove_unused_dictionary( dictionary_column_view const& dictionary_column,
                                                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Create a new dictionary column by applying only the specified dictionary elements
 * to the existing dictionary_column.
 *
 * Any new elements found in the dictionary parameter are added to the output dictionary.
 * Any existing dictionary elements not in the dictionary parameter are removed.
 *
 * The indices are update to reflect the position values of the new dictionary.
 * Any indices pointing to removed elements are set to null.
 *
 * ```
 * d1 = {["a","c","d"],[2,0,1,0]}
 * d2 = set_dictionary(d1,["a","b","c"])
 * d2 is now {["a","b","c"],[null,0,2,0]}
 * ```
 *
 * @throw cudf_logic_error if the dictionary type does not match the dictionary type in
 * the dictionary_column.
 *
 * @param dictionary_column Existing dictionary column.
 * @param dictionary New keys to use for the output column.
 * @param mr Resource for allocating memory for the output.
 * @return New dictionary column.
 */
std::unique_ptr<column> set_dictionary( dictionary_column_view const& dictionary_column,
                                        column_view const& dictionary,
                                        rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Create a new dictionary column by merging the specified dictionary elements
 * from two existing dictionary_columns.
 *
 * The indices are updated to appear as though the second dictionary_column is appended to the first.
 *
 * ```
 * d1 = {["a","c","d"],[2,0,1,0]}
 * d2 = {["b","e"],[1,0,1,1,0]}
 * d3 = merge(d1,d2)
 * d3 is now {["a","b","c","d","e"],[3,0,2,0,4,1,4,4,1]}
 * ```
 *
 * @throw cudf_logic_error if the dictionary types do not match for both dictionary columns.
 *
 * @param dictionary_column Existing dictionary column.
 * @param dictionary New keys to remove from the dictionary_column
 * @param mr Resource for allocating memory for the output.
 * @return New dictionary column.
 */
std::unique_ptr<column> merge_dictionaries( dictionary_column_view const& dictionary_column1,
                                            dictionary_column_view const& dictionary_column2,
                                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Create a new dictionary column by copying the dictionary from the provided
 * dictionary_column and setting the indices from the provided indices parameter.
 *
 * ```
 * d1 = {["a","c","d"],[2,0,1,0]}
 * d2 = gather(d1,[1,0,0,2,2])
 * d2 is now {["a","c","d"],[1,0,0,2,2]}
 * ```
 *
 * @param dictionary_column Existing dictionary column.
 * @param indices Indices to use for the new dictionary column.
 * @param mr Resource for allocating memory for the output.
 * @return New dictionary column.
 */
std::unique_ptr<column> gather( dictionary_column_view const& dictionary_column,
                                column_view const& indices,
                                rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Create a column by gathering the elements from the provided
 * dictionary_column using the specified indices.
 *
 * ```
 * d1 = {["a","c","d"],[2,0,1,0]}
 * s = gather_type(d1,[1,0,0,2,2])
 * s is now ["c","a","a","d","d"]
 * ```
 *
 * @param dictionary_column Existing dictionary column.
 * @param indices Indices to use for the new dictionary column.
 * @param mr Resource for allocating memory for the output.
 * @return New column with type matching the dictionary_column's dictionary.
 */
std::unique_ptr<column> gather_type( dictionary_column_view const& dictionary_column,
                                     column_view const& indices,
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Create a column by gathering the elements from the provided
 * dictionary_column using the indices from that column.
 *
 * ```
 * d1 = {["a","c","d"],[2,0,1,0]}
 * s = to_type(d1)
 * s is now ["d","a","c","a"]
 * ```
 *
 * @param dictionary_column Existing dictionary column.
 * @param mr Resource for allocating memory for the output.
 * @return New column with type matching the dictionary_column's dictionary.
 */
std::unique_ptr<column> to_type( dictionary_column_view const& dictionary_column,
                                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

} // namespace dictionary
} // namespace cudf
