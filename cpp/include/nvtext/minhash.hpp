/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/hashing.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/span.hpp>

namespace nvtext {
/**
 * @addtogroup nvtext_minhash
 * @{
 * @file
 */

/**
 * @brief Returns the minhash value for each string
 *
 * Hash values are computed from substrings of each string and the
 * minimum hash value is returned for each string.
 *
 * Any null row entries result in corresponding null output rows.
 *
 * @throw std::invalid_argument if the width < 2
 * @throw std::invalid_argument if hash_function is not HASH_MURMUR3
 *
 * @param input Strings column to compute minhash
 * @param seed  Seed value used for the MurmurHash3_32 algorithm
 * @param width The character width used for apply substrings;
 *              Default is 4 characters.
 * @param hash_function Hash algorithm to use;
 *                      Only HASH_MURMUR3 is currently supported.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Minhash values for each string in input
 */
std::unique_ptr<cudf::column> minhash(
  cudf::strings_column_view const& input,
  cudf::numeric_scalar<cudf::hash_value_type> seed = cudf::numeric_scalar(cudf::DEFAULT_HASH_SEED),
  cudf::size_type width                            = 4,
  cudf::hash_id hash_function                      = cudf::hash_id::HASH_MURMUR3,
  rmm::mr::device_memory_resource* mr              = rmm::mr::get_current_device_resource());

/**
 * @brief Returns the minhash values for each string per seed
 *
 * Hash values are computed from substrings of each string and the
 * minimum hash value is returned for each string for each seed.
 * Each row of the list column are seed results for the corresponding
 * string. The order of the elements in each row match the order of
 * the seeds provided in the `seeds` parameter.
 *
 * Any null row entries result in corresponding null output rows.
 *
 * @throw std::invalid_argument if the width < 2
 * @throw std::invalid_argument if hash_function is not HASH_MURMUR3
 * @throw std::invalid_argument if seeds is empty
 * @throw std::overflow_error if `seeds * input.size()` exceeds the column size limit
 *
 * @param input Strings column to compute minhash
 * @param seeds Seed values used for the MurmurHash3_32 algorithm
 * @param width The character width used for apply substrings;
 *              Default is 4 characters.
 * @param hash_function Hash algorithm to use;
 *                      Only HASH_MURMUR3 is currently supported.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return List column of minhash values for each string per seed
 *         or a hash_value_type column if only a single seed is specified
 */
std::unique_ptr<cudf::column> minhash(
  cudf::strings_column_view const& input,
  cudf::device_span<cudf::hash_value_type const> seeds,
  cudf::size_type width               = 4,
  cudf::hash_id hash_function         = cudf::hash_id::HASH_MURMUR3,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace nvtext
