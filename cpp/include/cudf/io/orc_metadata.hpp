/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

/**
 * @file orc_metadata.hpp
 * @brief cuDF-IO freeform API
 */

#pragma once

#include <cudf/io/types.hpp>

#include <vector>

//! cuDF interfaces
namespace cudf {
//! In-development features
namespace io {

/**
 * @brief Reads file-level and stripe-level statistics of ORC dataset
 *
 * @ingroup io_readers
 *
 * The following code snippet demonstrates how to read statistics of a dataset
 * from a file:
 * @code
 *  std::string filepath = "dataset.orc";
 *  auto result = cudf::read_orc_statistics(cudf::source_info(filepath));
 * @endcode
 *
 * @param src_info Dataset source
 *
 * @return Decompressed ColumnStatistics blobs stored in a vector of vectors. The first element of
 * result vector, which is itself a vector, contains the name of each column. The second element
 * contains statistics of each column of the whole file. Remaining elements contain statistics of
 * each column for each stripe.
 */
std::vector<std::vector<std::string>> read_orc_statistics(source_info const& src_info);

}  // namespace io
}  // namespace cudf
