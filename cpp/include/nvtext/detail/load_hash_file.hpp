/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <nvtext/subword_tokenize.hpp>

#include <cudf/column/column.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>
#include <cstring>

namespace nvtext {
namespace detail {

/**
 * @brief Load the hashed vocabulary file into device memory.
 *
 * The object here can be used to call the subword_tokenize without
 * incurring the cost of loading the same file each time.
 *
 * @param filename_hashed_vocabulary A path to the preprocessed vocab.txt file.
 *        Note that this is the file AFTER python/perfect_hash.py has been used
 *        for preprocessing.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Memory resource to allocate any returned objects.
 * @return vocabulary hash-table elements
 */
std::unique_ptr<hashed_vocabulary> load_vocabulary_file(
  std::string const& filename_hashed_vocabulary,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr);

}  // namespace detail
}  // namespace nvtext
