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
#ifndef TEST_PARAMETERS_CUH
#define TEST_PARAMETERS_CUH

#include <tests/utilities/cudf_test_fixtures.h>

#include <cudf/types.h>

#include <tuple>
#include <algorithm>
#include <random>
#include <type_traits>
#include <vector>
#include <map>


// Selects the kind of join operation that is performed
enum struct agg_op
{
  MIN,//0
  MAX,//1
  SUM,//2
  CNT,//3
  AVG //4
};

template <agg_op op>
struct AggOp {
    template <typename T>
    T operator()(const T a, const T b) {
        return static_cast<T>(0);
    }
    template <typename T>
    T operator()(const T a) {
        return static_cast<T>(0);
    }
};

template<>
struct AggOp<agg_op::MIN> {
    template <typename T>
    T operator()(const T a, const T b) {
        return (a < b)? a : b;
    }
    template <typename T>
    T operator()(const T a) {
        return a;
    }
};

template<>
struct AggOp<agg_op::MAX> {
    template <typename T>
    T operator()(const T a, const T b) {
        return (a > b)? a : b;
    }
    template <typename T>
    T operator()(const T a) {
        return a;
    }
};

template<>
struct AggOp<agg_op::SUM> {
    template <typename T>
    T operator()(const T a, const T b) {
        return a + b;
    }
    template <typename T>
    T operator()(const T a) {
        return a;
    }
};

template<>
struct AggOp<agg_op::CNT> {
    size_t count{0};
    template <typename T>
    T operator()(const T a, const T b) {
        count = a+1;
        return count;
    }
    template <typename T>
    T operator()(const T a) {
        count = 1;
        return count;
    }
};

template <typename... T>
using VTuple = std::tuple<std::vector<T>...>;

template<agg_op,
         gdf_method,
         typename,
         typename>
struct
TestParameters {};

// This structure is used to nest the group operations, group method and
// number/types of columns for use with Google Test type-parameterized
// tests. Here agg_operation refers to the type of aggregation eg. min,
// max etc. and group_method refers to the underlying group algorithm
//that performs it eg. GDF_HASH or GDF_SORT. VTuple<K...> expects a
//tuple of datatypes that specify the column types to be tested.
//AggOutput specifies the output type of the aggregated column.
template<agg_op agg_operation, 
         gdf_method group_method, 
         typename... K,
         typename AggOutput>
struct
TestParameters<
    agg_operation,
    group_method,
    VTuple<K...>,
    AggOutput>
{
  // The method to use for the groupby
  const static agg_op op = agg_operation;

  using multi_column_t = std::tuple<std::vector<K>...>;
  using vector_tuple_t = std::vector<std::tuple<K...>>;
  // The method to use for the groupby
  const static gdf_method group_type{group_method};

  using output_type = AggOutput;

  using tuple_t = std::tuple<K...>;

  using ref_map_type = std::map<std::tuple<K...>, output_type>;
};

const static gdf_method HASH = gdf_method::GDF_HASH;
typedef ::testing::Types<
    TestParameters< agg_op::AVG, HASH, VTuple<int32_t >, int32_t>,
    TestParameters< agg_op::AVG, HASH, VTuple<int64_t >, float>,
    TestParameters< agg_op::MIN, HASH, VTuple<float   >, int32_t>,
    TestParameters< agg_op::MIN, HASH, VTuple<double  >, int32_t>,
    TestParameters< agg_op::MAX, HASH, VTuple<uint32_t>, int32_t>,
    TestParameters< agg_op::MAX, HASH, VTuple<uint64_t>, int32_t>,
    TestParameters< agg_op::SUM, HASH, VTuple<int32_t >, int32_t>,
    TestParameters< agg_op::SUM, HASH, VTuple<int64_t >, float>,
    TestParameters< agg_op::CNT, HASH, VTuple<float   >, int32_t>,
    TestParameters< agg_op::CNT, HASH, VTuple<double  >, int32_t>,
    TestParameters< agg_op::SUM, HASH, VTuple<int32_t , int32_t >, int64_t >,
    TestParameters< agg_op::SUM, HASH, VTuple<int64_t , int32_t >, uint32_t>,
    TestParameters< agg_op::MIN, HASH, VTuple<float   , double  >, double  >,
    TestParameters< agg_op::MIN, HASH, VTuple<double  , int64_t >, float   >,
    TestParameters< agg_op::MAX, HASH, VTuple<uint32_t, int32_t >, int32_t >,
    TestParameters< agg_op::MAX, HASH, VTuple<uint64_t, uint32_t>, uint64_t>,
    TestParameters< agg_op::CNT, HASH, VTuple<uint32_t, int32_t >, int32_t >,
    TestParameters< agg_op::CNT, HASH, VTuple<uint64_t, uint32_t>, uint64_t>,
    TestParameters< agg_op::MIN, HASH, VTuple<int32_t , int32_t , float   >, int64_t >,
    TestParameters< agg_op::MAX, HASH, VTuple<int64_t , int32_t , int32_t >, uint32_t>,
    TestParameters< agg_op::SUM, HASH, VTuple<float   , double  , uint64_t>, double  >,
    TestParameters< agg_op::CNT, HASH, VTuple<double  , int64_t , int32_t >, float   >,
    TestParameters< agg_op::SUM, HASH, VTuple<uint32_t, int32_t , uint64_t>, int32_t >,
    TestParameters< agg_op::SUM, HASH, VTuple<uint64_t, uint32_t, double  >, uint64_t>,
    TestParameters< agg_op::AVG, HASH, VTuple<uint32_t, int32_t , int64_t >, int32_t >,
    TestParameters< agg_op::AVG, HASH, VTuple<uint64_t, uint32_t, int32_t >, uint64_t>
  > Implementations;

typedef ::testing::Types<
    TestParameters< agg_op::AVG, HASH, VTuple<int32_t >, int32_t>
  > ValidTestImplementations;

#endif // TEST_PARAMETERS_CUH