#ifndef AGGREGATION_OPERATIONS_H
#define AGGREGATION_OPERATIONS_H
template<typename value_type>
struct max_op
{
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
