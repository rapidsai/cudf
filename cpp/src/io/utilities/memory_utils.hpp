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

#include "rmm/rmm.h"
#include "rmm/thrust_rmm_allocator.h"

#include "utilities/error_utils.hpp"

/**---------------------------------------------------------------------------*
 * @brief Base class for pointers owning RMM allocated memory
 *---------------------------------------------------------------------------**/
template <class T>
class rmm_unique_ptr_base {
protected:
	/**
	 * @brief Pointer to the owned memory
	 */
	T* ptr = nullptr;

	/**
	 * @brief Create an empty object - does not allocate any memory
	 */
	rmm_unique_ptr_base() noexcept = default;

public:
	/**
	 * @brief Frees the owned memory on destruction
	 */
	~rmm_unique_ptr_base() {
		RMM_FREE(ptr, 0);
	}

	/**
	 * @brief Delete the copy assignment operator to prevent object copying
	 */
	rmm_unique_ptr_base& operator= (const rmm_unique_ptr_base &p) = delete;

	/**
	 * @brief Takes over the ownership of the memory. Leaves the passed
	 * object in an empty state
	 */
	rmm_unique_ptr_base& operator= (rmm_unique_ptr_base &&p) noexcept {
		ptr = p.ptr;
		p.ptr = nullptr;
		return *this;
	}

	/**
	 * @brief Delete the copy constructor to prevent object copying
	 */
	rmm_unique_ptr_base(rmm_unique_ptr_base &p) = delete;

	/**
	 * @brief Takes over the ownership of the memory. Leaves the passed
	 * object in an empty state
	 */
	rmm_unique_ptr_base(rmm_unique_ptr_base &&p) noexcept {
		ptr = p.ptr;
		p.ptr = nullptr;
	}

	/**
	 * @brief Getter method for the internal pointer
	 */
	T* get() noexcept {
		return ptr;
	}
};

/**---------------------------------------------------------------------------*
 * @brief Smart pointer class for a single element RMM allocation
 *---------------------------------------------------------------------------**/
template <class T>
class rmm_unique_ptr: public rmm_unique_ptr_base<T> {
public:
	/**
	 * @brief Allocates a single element of the given type.
	 * 
	 * Throws if allocation fails.
	 */
	rmm_unique_ptr() {
		const auto error = RMM_ALLOC(&this->ptr, sizeof(T), 0);
		if(error != RMM_SUCCESS) {
			cudf::detail::throw_cuda_error(cudaErrorMemoryAllocation, __FILE__, __LINE__);
		}
	}
};

/**---------------------------------------------------------------------------*
 * @brief Smart pointer class for an RMM allocated device array
 *---------------------------------------------------------------------------**/
template <class T>
class rmm_unique_ptr<T[]>: public rmm_unique_ptr_base<T> {
public:
	/**
	 * @brief Create an empty object - does not allocate any memory
	 */
	rmm_unique_ptr() = default;

	/**
	 * @brief Allocates an array of cnt elements
	 * 
	 * Throws if allocation fails.
	 */
	explicit rmm_unique_ptr(size_t cnt) {
		resize(cnt);
	}
	
	/**
	 * @brief Resize the owned array
	 * 
	 * Does not retain the values in the owner memory.
	 * Throws if allocation fails.
	 */
	void resize(size_t cnt) {
		RMM_FREE(this->ptr, 0);

		if (cnt != 0) {
			const auto error = RMM_ALLOC(&this->ptr, sizeof(T)*cnt, 0);
			if(error != RMM_SUCCESS) {
				cudf::detail::throw_cuda_error(cudaErrorMemoryAllocation, __FILE__, __LINE__);
			}
		}
		else {
			this->ptr = nullptr;
		}
	}
};
