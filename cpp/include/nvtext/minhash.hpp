/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/resource_ref.hpp>

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
 * This function uses MurmurHash3_x86_32 for the hash algorithm.
 *
 * @throw std::invalid_argument if the width < 2
 *
 * @param input Strings column to compute minhash
 * @param seed  Seed value used for the hash algorithm
 * @param width The character width used for apply substrings;
 *              Default is 4 characters.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Minhash values for each string in input
 */
std::unique_ptr<cudf::column> minhash(
  cudf::strings_column_view const& input,
  cudf::numeric_scalar<uint32_t> seed = 0,
  cudf::size_type width               = 4,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr   = rmm::mr::get_current_device_resource());

/**
 * @brief Returns the minhash values for each string per seed
 *
 * Hash values are computed from substrings of each string and the
 * minimum hash value is returned for each string for each seed.
 * Each row of the list column are seed results for the corresponding
 * string. The order of the elements in each row match the order of
 * the seeds provided in the `seeds` parameter.
 *
 * This function uses MurmurHash3_x86_32 for the hash algorithm.
 *
 * Any null row entries result in corresponding null output rows.
 *
 * @throw std::invalid_argument if the width < 2
 * @throw std::invalid_argument if seeds is empty
 * @throw std::overflow_error if `seeds * input.size()` exceeds the column size limit
 *
 * @param input Strings column to compute minhash
 * @param seeds Seed values used for the hash algorithm
 * @param width The character width used for apply substrings;
 *              Default is 4 characters.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return List column of minhash values for each string per seed
 */
std::unique_ptr<cudf::column> minhash(
  cudf::strings_column_view const& input,
  cudf::device_span<uint32_t const> seeds,
  cudf::size_type width             = 4,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns the minhash value for each string
 *
 * Hash values are computed from substrings of each string and the
 * minimum hash value is returned for each string.
 *
 * Any null row entries result in corresponding null output rows.
 *
 * This function uses MurmurHash3_x64_128 for the hash algorithm.
 * The hash function returns 2 uint64 values but only the first value
 * is used with the minhash calculation.
 *
 * @throw std::invalid_argument if the width < 2
 *
 * @param input Strings column to compute minhash
 * @param seed  Seed value used for the hash algorithm
 * @param width The character width used for apply substrings;
 *              Default is 4 characters.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Minhash values as UINT64 for each string in input
 */
std::unique_ptr<cudf::column> minhash64(
  cudf::strings_column_view const& input,
  cudf::numeric_scalar<uint64_t> seed = 0,
  cudf::size_type width               = 4,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr   = rmm::mr::get_current_device_resource());

/**
 * @brief Returns the minhash values for each string per seed
 *
 * Hash values are computed from substrings of each string and the
 * minimum hash value is returned for each string for each seed.
 * Each row of the list column are seed results for the corresponding
 * string. The order of the elements in each row match the order of
 * the seeds provided in the `seeds` parameter.
 *
 * This function uses MurmurHash3_x64_128 for the hash algorithm.
 *
 * Any null row entries result in corresponding null output rows.
 *
 * @throw std::invalid_argument if the width < 2
 * @throw std::invalid_argument if seeds is empty
 * @throw std::overflow_error if `seeds * input.size()` exceeds the column size limit
 *
 * @param input Strings column to compute minhash
 * @param seeds Seed values used for the hash algorithm
 * @param width The character width used for apply substrings;
 *              Default is 4 characters.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return List column of minhash values for each string per seed
 */
std::unique_ptr<cudf::column> minhash64(
  cudf::strings_column_view const& input,
  cudf::device_span<uint64_t const> seeds,
  cudf::size_type width             = 4,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns the minhash values for each row of strings per seed
 *
 * Hash values are computed from each string in each row and the
 * minimum hash value is returned for each row for each seed.
 * Each row of the output list column are seed results for the corresponding
 * input row. The order of the elements in each row match the order of
 * the seeds provided in the `seeds` parameter.
 *
 * This function uses MurmurHash3_x86_32 for the hash algorithm.
 *
 * Any null row entries result in corresponding null output rows.
 *
 * @throw std::invalid_argument if seeds is empty
 * @throw std::overflow_error if `seeds * input.size()` exceeds the column size limit
 *
 * @param input Lists column of strings to compute minhash
 * @param seeds Seed values used for the hash algorithm
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return List column of minhash values for each string per seed
 */
std::unique_ptr<cudf::column> word_minhash(
  cudf::lists_column_view const& input,
  cudf::device_span<uint32_t const> seeds,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns the minhash values for each row of strings per seed
 *
 * Hash values are computed from each string in each row and the
 * minimum hash value is returned for each row for each seed.
 * Each row of the output list column are seed results for the corresponding
 * input row. The order of the elements in each row match the order of
 * the seeds provided in the `seeds` parameter.
 *
 * This function uses MurmurHash3_x64_128 for the hash algorithm.
 *
 * Any null row entries result in corresponding null output rows.
 *
 * @throw std::invalid_argument if seeds is empty
 * @throw std::overflow_error if `seeds * input.size()` exceeds the column size limit
 *
 * @param input Lists column of strings to compute minhash
 * @param seeds Seed values used for the hash algorithm
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return List column of minhash values for each string per seed
 */
std::unique_ptr<cudf::column> word_minhash64(
  cudf::lists_column_view const& input,
  cudf::device_span<uint64_t const> seeds,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());
/** @} */  // end of group
}  // namespace nvtext
