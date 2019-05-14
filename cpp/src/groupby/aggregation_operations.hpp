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

#ifndef AGGREGATION_OPERATIONS_H
#define AGGREGATION_OPERATIONS_H

#include "utilities/cudf_utils.h"
#include "utilities/wrapper_types.hpp"


/* --------------------------------------------------------------------------*/
/** 
 * @file aggregation_operations.hpp
 * @brief This header defines the functors that may be used as aggregation operations for 
 * the hash-based groupby implementation. Each functor must define an 'identity value'.
 * The identity value 'I' of an operation 'op' is defined as: for any x, op(x,I) == x.
 * This identity value is used to initialize the hash table values
 * Every functor accepts a 'new_value' which is the new value being inserted into the 
 * hash table and an 'old_value' which is the existing value in the hash table
 */
/* ----------------------------------------------------------------------------*/


template<typename value_type>
struct max_op_valids
{
  constexpr static value_type IDENTITY{0};

  using tuple_type = thrust::tuple<value_type, bool>;

  CUDA_HOST_DEVICE_CALLABLE
  tuple_type operator()(tuple_type tx, tuple_type ty) {

		auto data_x = thrust::get<0>(tx);
		auto valid_x = thrust::get<1>(tx);

		auto data_y = thrust::get<0>(ty);
		auto valid_y = thrust::get<1>(ty);

		if (!valid_x)
			return ty;

		if (!valid_y)
			return tx;

		return thrust::make_tuple(data_x > data_y ? data_x : data_y, true);
	}
};

template<typename value_type>
struct max_op{
  constexpr static value_type IDENTITY{std::numeric_limits<value_type>::lowest()};

  using with_valids = max_op_valids<value_type>;

  CUDA_HOST_DEVICE_CALLABLE
  value_type operator()(value_type new_value, value_type old_value)
  {
    return (new_value > old_value ? new_value : old_value);
  }
};


template<typename value_type>
struct min_op_valids
{
  constexpr static value_type IDENTITY{0};

  using tuple_type = thrust::tuple<value_type, bool>;

  CUDA_HOST_DEVICE_CALLABLE
  tuple_type operator()(tuple_type tx, tuple_type ty) {

		auto data_x = thrust::get<0>(tx);
		auto valid_x = thrust::get<1>(tx);

		auto data_y = thrust::get<0>(ty);
		auto valid_y = thrust::get<1>(ty);

		if (!valid_x)
			return ty;

		if (!valid_y)
			return tx;

		return thrust::make_tuple(data_x < data_y ? data_x : data_y, true);
	}
};



template<typename value_type>
struct min_op 
{
  constexpr static value_type IDENTITY{std::numeric_limits<value_type>::max()};

  using with_valids = min_op_valids<value_type>;

  CUDA_HOST_DEVICE_CALLABLE
  value_type operator()(value_type new_value, value_type old_value)
  {
    return (new_value < old_value ? new_value : old_value);
  }
};

template<typename value_type>
struct count_op_valids
{
  constexpr static value_type IDENTITY{0};

  using tuple_type = thrust::tuple<value_type, bool>;

  CUDA_HOST_DEVICE_CALLABLE
  tuple_type operator()(tuple_type tx, tuple_type ty) {

		auto data_x = thrust::get<0>(tx);
		auto valid_x = thrust::get<1>(tx);

		auto data_y = thrust::get<0>(ty);
		auto valid_y = thrust::get<1>(ty);

		if (!valid_x)
  		return thrust::make_tuple(data_y, true);

		return thrust::make_tuple(++data_y, true);
	}
};

template<typename value_type>
struct count_op 
{
  constexpr static value_type IDENTITY{0};

  using with_valids = count_op_valids<value_type>;

  CUDA_HOST_DEVICE_CALLABLE
  value_type operator()(value_type, value_type old_value)
  {
    old_value += value_type{1};
    return old_value;
  }
};



template<typename value_type>
struct count_distinct_op_valids
{
  constexpr static value_type IDENTITY{0};

  using tuple_type = thrust::tuple<value_type, bool>;

  CUDA_HOST_DEVICE_CALLABLE
  tuple_type operator()(tuple_type tx, tuple_type ty) {

		auto data_x = thrust::get<0>(tx);
		auto valid_x = thrust::get<1>(tx);

		auto data_y = thrust::get<0>(ty);
		auto valid_y = thrust::get<1>(ty);

		if (!valid_x)
  		return thrust::make_tuple(data_y, true);

		return thrust::make_tuple(++data_y, true);
	}
};

template<typename value_type>
struct count_distinct_op   
{
  constexpr static value_type IDENTITY{0};
  using with_valids = count_distinct_op_valids<value_type>;

  CUDA_HOST_DEVICE_CALLABLE
  value_type operator()(value_type new_value, value_type old_value)
  {
    return ++old_value;
  }
};

template<typename value_type>
struct sum_op_valids
{
  constexpr static value_type IDENTITY{0};

  using tuple_type = thrust::tuple<value_type, bool>;

  CUDA_HOST_DEVICE_CALLABLE
  tuple_type operator()(tuple_type tx, tuple_type ty) {

		auto data_x = thrust::get<0>(tx);
		auto valid_x = thrust::get<1>(tx);

		auto data_y = thrust::get<0>(ty);
		auto valid_y = thrust::get<1>(ty);

		if (!valid_x)
			return ty;

		if (!valid_y)
			return tx;

		return thrust::make_tuple(data_x + data_y, true);
	}
};


template<typename value_type>
struct sum_op 
{
  constexpr static value_type IDENTITY{0};
  using with_valids = sum_op_valids<value_type>;

  CUDA_HOST_DEVICE_CALLABLE
  value_type operator()(value_type new_value, value_type old_value)
  {
    return new_value + old_value;
  }

};

// Functor for AVG is empty. Used only for template specialization
template<typename value_type>
struct avg_op
{
  constexpr static value_type IDENTITY{};
  CUDA_HOST_DEVICE_CALLABLE
  value_type operator()(value_type new_value, value_type old_value)
  {
    return 0;
  }
};

#endif
