#pragma once
#include <tuple>
#include <algorithm>
#include <random>
#include <type_traits>
#include <vector>
#include <map>

#include <cudf/types.h>

namespace without_agg {

template <typename... T>
using VTuple = std::tuple<std::vector<T>...>;


enum struct GroupByOutType
{
  PANDAS,//0
  SQL//1
};


template<GroupByOutType,
         typename>
struct
TestParameters {};



// helper function to tuple_each a tuple of any size
template <class Tuple, typename Func, std::size_t N>
struct TupleEach {
  static void tuple_each(Tuple &t, Func &f) {
    TupleEach<Tuple, Func, N - 1>::tuple_each(t, f);
    f(std::get<N - 1>(t));
  }
};

template <class Tuple, typename Func>
struct TupleEach<Tuple, Func, 1> {
  static void tuple_each(Tuple &t, Func &f) { f(std::get<0>(t)); }
};

template <typename Tuple, typename Func>
void tuple_each(Tuple &t, Func &&f) {
  TupleEach<Tuple, Func, std::tuple_size<Tuple>::value>::tuple_each(t, f);
}
// end helper function

// This structure is used to nest the group operations, group method and
// number/types of columns for use with Google Test type-parameterized
// tests. Here agg_operation refers to the type of aggregation eg. min,
// max etc. and group_method refers to the underlying group algorithm
//that performs it eg. GDF_HASH or GDF_SORT. VTuple<K...> expects a
//tuple of datatypes that specify the column types to be tested.
//AggOutput specifies the output type of the aggregated column.
template<GroupByOutType group_output_method, typename... K>
struct
TestParameters<group_output_method,
    VTuple<K...>>
{
  // // The method to use for the groupby
  const static GroupByOutType group_output_type {group_output_method};

  using multi_column_t = std::tuple<std::vector<K>...>;
  using vector_tuple_t = std::vector<std::tuple<K...>>;
  // The method to use for the groupby
  const static gdf_method group_type{GDF_SORT};

  using output_type = gdf_size_type;

  using tuple_t = std::tuple<K...>;

  using ref_map_type = std::multimap<std::tuple<K...>, output_type>;
};

typedef ::testing::Types<
    TestParameters< GroupByOutType::SQL, VTuple<int32_t > >,
    TestParameters< GroupByOutType::SQL, VTuple<int16_t > >,
    TestParameters< GroupByOutType::SQL, VTuple<int32_t > >,
    TestParameters< GroupByOutType::SQL, VTuple<int64_t> >,
    TestParameters< GroupByOutType::PANDAS, VTuple<float   > >,
    TestParameters< GroupByOutType::PANDAS, VTuple<double  > >,
    TestParameters< GroupByOutType::PANDAS, VTuple<int32_t, int32_t > >,
    TestParameters< GroupByOutType::SQL, VTuple<int32_t, int64_t > >
  > Implementations;

typedef ::testing::Types<
    TestParameters< GroupByOutType::SQL, VTuple<int32_t >>,
    TestParameters< GroupByOutType::SQL, VTuple<int64_t >>,
    TestParameters< GroupByOutType::SQL, VTuple<double >>,
    TestParameters< GroupByOutType::SQL, VTuple<int16_t >>,
    TestParameters< GroupByOutType::SQL, VTuple<int32_t, int64_t >>    
  > ValidTestImplementations;

} //namespace: without_agg
