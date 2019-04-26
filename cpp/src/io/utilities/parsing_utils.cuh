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

/**
 * @file parsing_utils.cuh Declarations of utility functions for parsing plain-text files
 *
 */


#pragma once

#include <vector>

#include "cudf.h"

#include <thrust/pair.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include "rmm/rmm.h"
#include "rmm/thrust_rmm_allocator.h"

#include "io/utilities/wrapper_utils.hpp"

gdf_size_type countAllFromSet(const char *h_data, size_t h_size, const std::vector<char>& keys);

template<class T>
gdf_size_type findAllFromSet(const char *h_data, size_t h_size, const std::vector<char>& keys, uint64_t result_offset,
	T *positions);

device_buffer<int16_t> getBracketLevels(
	thrust::pair<uint64_t,char>* brackets, int count, 
	const std::string& open_chars, const std::string& close_chars);
