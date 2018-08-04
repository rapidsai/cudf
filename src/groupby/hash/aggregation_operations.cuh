#ifndef AGGREGATION_OPERATIONS_H
#define AGGREGATION_OPERATIONS_H

#include <limits>

// This header defines the functors that may be used as aggregation operations for 
// the hash-based groupby implementation. Each functor must define an 'identity value'.
// The identity value 'I' of an operation 'op' is defined as: for any x, op(x,I) == x.
// This identity value is used to initialize the hash table values
// Every functor accepts a 'new_value' which is the new value being inserted into the 
// hash table and an 'old_value' which is the existing value in the hash table

template<typename value_type>
struct max_op{

  constexpr static value_type IDENTITY{std::numeric_limits<value_type>::min()};

  __host__ __device__ value_type operator()(value_type new_value, value_type old_value)
  {
    return (new_value > old_value ? new_value : old_value);
  }
};

template<typename value_type>
struct min_op 
{

  constexpr static value_type IDENTITY{std::numeric_limits<value_type>::max()};

  __host__ __device__ value_type operator()(value_type new_value, value_type old_value)
  {
    return (new_value < old_value ? new_value : old_value);
  }
};

template<typename value_type>
struct count_op 
{

  constexpr static value_type IDENTITY{0};

  __host__ __device__ value_type operator()(value_type new_value, value_type old_value)
  {
    return ++old_value;
  }
};

template<typename value_type>
struct sum_op 
{

  constexpr static value_type IDENTITY{0};

  __host__ __device__ value_type operator()(value_type new_value, value_type old_value)
  {
    return new_value + old_value;
  }
};

#endif
