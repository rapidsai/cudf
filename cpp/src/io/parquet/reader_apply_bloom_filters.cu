/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
 * @file bloom_filter_reader.cu
 * @brief Bloom filter reader based row group filtration implementation
 */

#include "parquet.hpp"
#include "parquet_common.hpp"

#include <cudf/utilities/error.hpp>

#include <cuco/bloom_filter.cuh>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <tuple>

// TODO: Implement this
cuda::std::optional<std::vector<std::vector<cudf::size_type>>> apply_bloom_filters() { return {}; }