#ifndef AGGREGATION_OPERATIONS_H
#define AGGREGATION_OPERATIONS_H

#include <limits>

// This header defines the functors that may be used as aggregation operations for 
// the hash-based groupby implementation. Each functor must define an 'identity value'.
// The identity value 'I' of an operation 'op' is defined as: for any x, op(x,I) == x.
// This identity value is used to initialize the hash table values

template<typename value_type>
struct max_op{

  constexpr static value_type IDENTITY{std::numeric_limits<value_type>::min()};

  __host__ __device__
    value_type operator()(value_type a, value_type b)
    {
      return (a > b? a : b);
    }
};

template<typename value_type>
struct min_op 
{

  constexpr static value_type IDENTITY{std::numeric_limits<value_type>::max()};

  __host__ __device__
    value_type operator()(value_type a, value_type b)
    {
      return (a < b? a : b);
    }
};

#endif
