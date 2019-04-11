/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
 * @file memory_utils.hpp  code to provide RAII abstractions for RMM managed device memory
 *
 */

#pragma once

#include <type_traits>

#include "rmm/rmm.h"
#include "rmm/thrust_rmm_allocator.h"

#include "utilities/error_utils.hpp"

namespace rmm {

// alias for the rmm unique pointer
template<typename T>
using unique_ptr = std::unique_ptr<T, std::function<void(std::remove_extent_t<T> *)>>;

// creates a unique pointer holding a single object
template<typename T, std::enable_if_t<!std::is_array<T>::value, int> = 0>
unique_ptr<T> make_unique(cudaStream_t stream) {
	T* ptr = nullptr;
	const auto error = RMM_ALLOC(&ptr, sizeof(T), 0);
	if(error != RMM_SUCCESS) {
		cudf::detail::throw_cuda_error(cudaErrorMemoryAllocation, __FILE__, __LINE__);
	}

	auto deleter = [stream](T *p) {
		RMM_FREE(p, stream);
	};

	return { ptr, deleter };
}

// creates a unique pointer holding an array of objects
template<typename T, std::enable_if_t<std::is_array<T>::value, int> = 0>
unique_ptr<T> make_unique(size_t size, cudaStream_t stream) {
	typedef std::remove_extent_t<T> Elem;

	Elem* ptr = nullptr;
	const auto error = RMM_ALLOC(&ptr, sizeof(Elem)*size, 0);
	if(error != RMM_SUCCESS) {
		cudf::detail::throw_cuda_error(cudaErrorMemoryAllocation, __FILE__, __LINE__);
	}

	auto deleter = [stream](Elem *p) {
		RMM_FREE(p, stream);
	};

	return { ptr, deleter };
}

}  // namespace rmm