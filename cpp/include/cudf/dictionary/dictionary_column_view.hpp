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

#include <cudf/column/column_view.hpp>
#include <cudf/column/column.hpp>

namespace cudf
{
/**
 * @brief A wrapper class for operations on a dictionary column.
 *
 * A dictionary column contains a set of keys and a column of indices.
 * The keys are a sorted set of unique values for the column.
 * The indices values are position indices into the keys.
 */
class dictionary_column_view : private column_view
{
public:
    dictionary_column_view( column_view dictionary_column );
    dictionary_column_view( dictionary_column_view&& dictionary_view ) = default;
    dictionary_column_view( const dictionary_column_view& dictionary_view ) = default;
    ~dictionary_column_view() = default;
    dictionary_column_view& operator=(dictionary_column_view const&) = default;
    dictionary_column_view& operator=(dictionary_column_view&&) = default;

    using column_view::size;
    using column_view::null_mask;
    using column_view::null_count;
    using column_view::has_nulls;
    using column_view::offset;
    using column_view::keys;

    /**
     * @brief Returns the parent column.
     */
    column_view parent() const;

    /**
     * @brief Returns the column of indices
     */
    column_view indices() const;

    /**
     * @brief Returns the number of rows in the keys column.
     */
    size_type keys_size() const noexcept;

private:
    column_view _dictionary;
};

namespace dictionary
{

/**
 * @brief Create a new dictionary column by adding the new keys elements
 * to the existing dictionary_column.
 *
 * The indices are updated if any of the new keys are sorted
 * before any of the existing dictionary elements.
 *
 * ```
 * d1 = {["a","c","d"],[2,0,1,0]}
 * d2 = add_keys(d1,["b","c"])
 * d2 is now {["a","b","c","d"],[3,0,2,0]}
 * ```
 *
 * @throw cudf_logic_error if the keys type does not match the keys type in
 * the dictionary_column.
 *
 * @param dictionary_column Existing dictionary column.
 * @param keys New keys to incorporate into the dictionary_column
 * @param mr Resource for allocating memory for the output.
 * @return New dictionary column.
 */
std::unique_ptr<column> add_keys( dictionary_column_view const& dictionary_column,
                                  column_view const& keys,
                                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Create a new dictionary column by removing the specified keys
 * from the existing dictionary_column.
 *
 * The indices are updated to the new positions of the remaining keys.
 * Any indices pointing to removed keys are set to null.
 *
 * ```
 * d1 = {["a","c","d"],[2,0,1,0]}
 * d2 = remove_keys(d1,["b","c"])
 * d2 is now {["a","d"],[1,0,null,0]}
 * ```
 *
 * @throw cudf_logic_error if the keys type does not match the keys type in
 * the dictionary_column.
 *
 * @param dictionary_column Existing dictionary column.
 * @param keys_to_remove The keys to remove from the dictionary_column
 * @param mr Resource for allocating memory for the output.
 * @return New dictionary column.
 */
std::unique_ptr<column> remove_keys( dictionary_column_view const& dictionary_column,
                                     column_view const& keys_to_remove,
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Create a new dictionary column by removing any keys
 * that are not referenced by any of the indices.
 *
 * The indices are updated to the new position values of the remaining keys.
 *
 * ```
 * d1 = {["a","c","d"],[2,0,2,0]}
 * d2 = remove_unused_keys(d1)
 * d2 is now {["a","d"],[1,0,1,0]}
 * ```
 *
 * @param dictionary_column Existing dictionary column.
 * @param mr Resource for allocating memory for the output.
 * @return New dictionary column.
 */
std::unique_ptr<column> remove_unused_keys( dictionary_column_view const& dictionary_column,
                                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Create a new dictionary column by applying only the specified keys
 * to the existing dictionary_column.
 *
 * Any new elements found in the keys parameter are added to the output dictionary.
 * Any existing keys not in the keys parameter are removed.
 *
 * The indices are updated to reflect the position values of the new keys.
 * Any indices pointing to removed keys are set to null.
 *
 * ```
 * d1 = {["a","c","d"],[2,0,1,0]}
 * d2 = set_keys(d1,["a","b","c"])
 * d2 is now {["a","b","c"],[null,0,2,0]}
 * ```
 *
 * @throw cudf_logic_error if the keys type does not match the keys type in
 * the dictionary_column.
 *
 * @param dictionary_column Existing dictionary column.
 * @param keys New keys to use for the output column.
 * @param mr Resource for allocating memory for the output.
 * @return New dictionary column.
 */
std::unique_ptr<column> set_keys( dictionary_column_view const& dictionary_column,
                                  column_view const& keys,
                                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Create a new dictionary column by merging the keys and indices
 * from two existing dictionary columns.
 *
 * The indices of the resulting column are created to appear as though the
 * second dictionary_column indices is appended to the first.
 *
 * ```
 * d1 = {["a","c","d"],[2,0,1,0]}
 * d2 = {["b","e"],[1,0,1,1,0]}
 * d3 = merge(d1,d2)
 * d3 is now {["a","b","c","d","e"],[3,0,2,0,4,1,4,4,1]}
 * ```
 *
 * @throw cudf_logic_error if the keys types do not match for both dictionary columns.
 *
 * @param dictionary_column1 Existing dictionary column.
 * @param dictionary_column2 2nd existing dictionary column.
 * @param mr Resource for allocating memory for the output.
 * @return New dictionary column.
 */
std::unique_ptr<column> merge_dictionaries( dictionary_column_view const& dictionary_column1,
                                            dictionary_column_view const& dictionary_column2,
                                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Create a column by gathering the keys from the provided
 * dictionary_column into a new column using the specified indices.
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
 * @return New column with type matching the dictionary_column's keys.
 */
std::unique_ptr<column> gather_type( dictionary_column_view const& dictionary_column,
                                     column_view const& indices,
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Create a column by gathering the keys from the provided
 * dictionary_column into a new column using the indices from that column.
 *
 * ```
 * d1 = {["a","c","d"],[2,0,1,0]}
 * s = to_type(d1)
 * s is now ["d","a","c","a"]
 * ```
 *
 * @param dictionary_column Existing dictionary column.
 * @param mr Resource for allocating memory for the output.
 * @return New column with type matching the dictionary_column's keys.
 */
std::unique_ptr<column> to_type( dictionary_column_view const& dictionary_column,
                                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

} // namespace dictionary
} // namespace cudf
