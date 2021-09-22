/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

namespace tdigest {

// mean and weight column indices within tdigest inner struct columns
constexpr size_type mean_column_index   = 0;
constexpr size_type weight_column_index = 1;

// min and max column indices within tdigest outer struct columns
constexpr size_type centroid_column_index = 0;
constexpr size_type min_column_index      = 1;
constexpr size_type max_column_index      = 2;

/**
 * @brief Verifies that the input column is a valid tdigest column.
 *
 * struct {
 *   // centroids for the digest
 *   list {
 *    struct {
 *      double    // mean
 *      double    // weight
 *    },
 *    ...
 *   }
 *   // these are from the input stream, not the centroids. they are used
 *   // during the percentile_approx computation near the beginning or
 *   // end of the quantiles
 *   double       // min
 *   double       // max
 * }
 *
 * Each output row is a single tdigest.  The length of the row is the "size" of the
 * tdigest, each element of which represents a weighted centroid (mean, weight).
 *
 * @param col    Column to be checkeed
 *
 * @throws cudf::logic error if the column is not a valid tdigest column.
 */
void check_is_valid_tdigest_column(column_view const& col);

/**
 * @brief Create an empty tdigest column.
 *
 * An empty tdigest column contains a single row of length 0
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 *
 * @returns An empty tdigest column.
 */
std::unique_ptr<column> make_empty_tdigest_column(
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace tdigest
}  // namespace detail
}  // namespace cudf