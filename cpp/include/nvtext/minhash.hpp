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
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

namespace CUDF_EXPORT nvtext {
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
 * @deprecated Deprecated in 24.12
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
[[deprecated]] std::unique_ptr<cudf::column> minhash(
  cudf::strings_column_view const& input,
  cudf::numeric_scalar<uint32_t> seed = 0,
  cudf::size_type width               = 4,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr   = cudf::get_current_device_resource_ref());

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
 * @deprecated Deprecated in 24.12 - to be replaced in a future release
 *
 * @throw std::invalid_argument if the width < 2
 * @throw std::invalid_argument if seeds is empty
 * @throw std::overflow_error if `seeds.size() * input.size()` exceeds the column size limit
 *
 * @param input Strings column to compute minhash
 * @param seeds Seed values used for the hash algorithm
 * @param width The character width used for apply substrings;
 *              Default is 4 characters.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return List column of minhash values for each string per seed
 */
[[deprecated]] std::unique_ptr<cudf::column> minhash(
  cudf::strings_column_view const& input,
  cudf::device_span<uint32_t const> seeds,
  cudf::size_type width             = 4,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the minhash values for each string
 *
 * This function uses MurmurHash3_x86_32 for the hash algorithm.
 *
 * The input strings are first hashed using the given `seed` over substrings
 * of `width` characters. These hash values are then combined with the `a`
 * and `b` parameter values using the following formula:
 * ```
 *   max_hash = max of uint32
 *   mp = (1 << 61) - 1
 *   hv[i] = hash value of a substring at i
 *   pv[i] = ((hv[i] * a[i] + b[i]) % mp) & max_hash
 * ```
 *
 * This calculation is performed on each substring and the minimum value is computed
 * as follows:
 * ```
 *   mh[j,i] = min(pv[i]) for all substrings in row j
 *                        and where i=[0,a.size())
 * ```
 *
 * Any null row entries result in corresponding null output rows.
 *
 * @throw std::invalid_argument if the width < 2
 * @throw std::invalid_argument if parameter_a is empty
 * @throw std::invalid_argument if `parameter_b.size() != parameter_a.size()`
 * @throw std::overflow_error if `parameter_a.size() * input.size()` exceeds the column size limit
 *
 * @param input Strings column to compute minhash
 * @param seed Seed value used for the hash algorithm
 * @param parameter_a Values used for the permuted calculation
 * @param parameter_b Values used for the permuted calculation
 * @param width The character width of substrings to hash for each row
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return List column of minhash values for each string per seed
 */
std::unique_ptr<cudf::column> minhash_permuted(
  cudf::strings_column_view const& input,
  uint32_t seed,
  cudf::device_span<uint32_t const> parameter_a,
  cudf::device_span<uint32_t const> parameter_b,
  cudf::size_type width,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

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
 * @deprecated Deprecated in 24.12
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
[[deprecated]] std::unique_ptr<cudf::column> minhash64(
  cudf::strings_column_view const& input,
  cudf::numeric_scalar<uint64_t> seed = 0,
  cudf::size_type width               = 4,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr   = cudf::get_current_device_resource_ref());

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
 * @deprecated Deprecated in 24.12 - to be replaced in a future release
 *
 * @throw std::invalid_argument if the width < 2
 * @throw std::invalid_argument if seeds is empty
 * @throw std::overflow_error if `seeds.size() * input.size()` exceeds the column size limit
 *
 * @param input Strings column to compute minhash
 * @param seeds Seed values used for the hash algorithm
 * @param width The character width used for apply substrings;
 *              Default is 4 characters.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return List column of minhash values for each string per seed
 */
[[deprecated]] std::unique_ptr<cudf::column> minhash64(
  cudf::strings_column_view const& input,
  cudf::device_span<uint64_t const> seeds,
  cudf::size_type width             = 4,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the minhash values for each string
 *
 * This function uses MurmurHash3_x64_128 for the hash algorithm.
 *
 * The input strings are first hashed using the given `seed` over substrings
 * of `width` characters. These hash values are then combined with the `a`
 * and `b` parameter values using the following formula:
 * ```
 *   max_hash = max of uint64
 *   mp = (1 << 61) - 1
 *   hv[i] = hash value of a substring at i
 *   pv[i] = ((hv[i] * a[i] + b[i]) % mp) & max_hash
 * ```
 *
 * This calculation is performed on each substring and the minimum value is computed
 * as follows:
 * ```
 *   mh[j,i] = min(pv[i]) for all substrings in row j
 *                        and where i=[0,a.size())
 * ```
 *
 * Any null row entries result in corresponding null output rows.
 *
 * @throw std::invalid_argument if the width < 2
 * @throw std::invalid_argument if parameter_a is empty
 * @throw std::invalid_argument if `parameter_b.size() != parameter_a.size()`
 * @throw std::overflow_error if `parameter_a.size() * input.size()` exceeds the column size limit
 *
 * @param input Strings column to compute minhash
 * @param seed Seed value used for the hash algorithm
 * @param parameter_a Values used for the permuted calculation
 * @param parameter_b Values used for the permuted calculation
 * @param width The character width of substrings to hash for each row
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return List column of minhash values for each string per seed
 */
std::unique_ptr<cudf::column> minhash64_permuted(
  cudf::strings_column_view const& input,
  uint64_t seed,
  cudf::device_span<uint64_t const> parameter_a,
  cudf::device_span<uint64_t const> parameter_b,
  cudf::size_type width,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

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
 * @deprecated Deprecated in 24.12
 *
 * @throw std::invalid_argument if seeds is empty
 * @throw std::overflow_error if `seeds.size() * input.size()` exceeds the column size limit
 *
 * @param input Lists column of strings to compute minhash
 * @param seeds Seed values used for the hash algorithm
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return List column of minhash values for each string per seed
 */
[[deprecated]] std::unique_ptr<cudf::column> word_minhash(
  cudf::lists_column_view const& input,
  cudf::device_span<uint32_t const> seeds,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the minhash values for each row of strings per seed
 *
 * Hash values are computed from each string in each row and the
 * minimum hash value is returned for each row for each seed.
 * Each row of the output list column are seed results for the corresponding
 * input row. The order of the elements in each row match the order of
 * the seeds provided in the `seeds` parameter.
 *
 * This function uses MurmurHash3_x64_128 for the hash algorithm though
 * only the first 64-bits of the hash are used in computing the output.
 *
 * Any null row entries result in corresponding null output rows.
 *
 * @deprecated Deprecated in 24.12
 *
 * @throw std::invalid_argument if seeds is empty
 * @throw std::overflow_error if `seeds.size() * input.size()` exceeds the column size limit
 *
 * @param input Lists column of strings to compute minhash
 * @param seeds Seed values used for the hash algorithm
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return List column of minhash values for each string per seed
 */
[[deprecated]] std::unique_ptr<cudf::column> word_minhash64(
  cudf::lists_column_view const& input,
  cudf::device_span<uint64_t const> seeds,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
/** @} */  // end of group
}  // namespace CUDF_EXPORT nvtext
