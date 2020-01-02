/*
 * Copyright (c) 2018-19, NVIDIA CORPORATION.
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

#include <cudf/cudf.h>
#include <tests/utilities/legacy/cudf_test_fixtures.h>
#include <hash/concurrent_unordered_map.cuh>

#include <gtest/gtest.h>
#include <rmm/thrust_rmm_allocator.h>
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <unordered_map>
#include <vector>

template <typename K, typename V>
struct key_value_types {
  using key_type = K;
  using value_type = V;
  using pair_type = thrust::pair<K, V>;
  using map_type = concurrent_unordered_map<key_type, value_type>;
};

template <typename T>
struct InsertTest : public GdfTest {
  using key_type = typename T::key_type;
  using value_type = typename T::value_type;
  using pair_type = typename T::pair_type;
  using map_type = typename T::map_type;

  InsertTest() {
    // prevent overflow of small types
    const size_t input_size = std::min(static_cast<key_type>(size),
                                       std::numeric_limits<key_type>::max());
    pairs.resize(input_size);
    map = std::move(map_type::create(compute_hash_table_size(size)));
    CUDA_TRY(cudaStreamSynchronize(0));
  }

  const cudf::size_type size{10000};
  rmm::device_vector<pair_type> pairs;
  std::unique_ptr<map_type, std::function<void(map_type*)>> map;
};

using TestTypes = ::testing::Types<
    key_value_types<int32_t, int32_t>, key_value_types<int64_t, int64_t>,
    key_value_types<int8_t, int8_t>, key_value_types<int16_t, int16_t>,
    key_value_types<int8_t, float>, key_value_types<int16_t, double>,
    key_value_types<int32_t, float>, key_value_types<int64_t, double>>;

TYPED_TEST_CASE(InsertTest, TestTypes);

template <typename map_type, typename pair_type>
struct insert_pair {
  insert_pair(map_type _map) : map{_map} {}

  __device__ bool operator()(pair_type const& pair) {
    auto result = map.insert(pair);
    if (result.first == map.end()) {
      return false;
    }
    return result.second;
  }

  map_type map;
};

template <typename map_type, typename pair_type>
struct find_pair {
  find_pair(map_type _map) : map{_map} {}

  __device__ bool operator()(pair_type const& pair) {
    auto result = map.find(pair.first);
    if (result == map.end()) {
      return false;
    }
    return *result == pair;
  }
  map_type map;
};

template <typename pair_type,
          typename key_type = typename pair_type::first_type,
          typename value_type = typename pair_type::second_type>
struct unique_pair_generator {
  __device__ pair_type operator()(cudf::size_type i) {
    return thrust::make_pair(key_type(i), value_type(i));
  }
};

template <typename pair_type,
          typename key_type = typename pair_type::first_type,
          typename value_type = typename pair_type::second_type>
struct identical_pair_generator {
  identical_pair_generator(key_type k = 42, value_type v = 42)
      : key{k}, value{v} {}
  __device__ pair_type operator()(cudf::size_type i) {
    return thrust::make_pair(key, value);
  }
  key_type key;
  value_type value;
};

template <typename pair_type,
          typename key_type = typename pair_type::first_type,
          typename value_type = typename pair_type::second_type>
struct identical_key_generator {
  identical_key_generator(key_type k = 42) : key{k} {}
  __device__ pair_type operator()(cudf::size_type i) {
    return thrust::make_pair(key, value_type(i));
  }
  key_type key;
};

TYPED_TEST(InsertTest, UniqueKeysUniqueValues) {
  using map_type = typename TypeParam::map_type;
  using pair_type = typename TypeParam::pair_type;
  thrust::tabulate(this->pairs.begin(), this->pairs.end(),
                   unique_pair_generator<pair_type>{});
  // All pairs should be new inserts
  EXPECT_TRUE(thrust::all_of(this->pairs.begin(), this->pairs.end(),
                             insert_pair<map_type, pair_type>{*this->map}));

  // All pairs should be present in the map
  EXPECT_TRUE(thrust::all_of(this->pairs.begin(), this->pairs.end(),
                             find_pair<map_type, pair_type>{*this->map}));
}

TYPED_TEST(InsertTest, IdenticalKeysIdenticalValues) {
  using map_type = typename TypeParam::map_type;
  using pair_type = typename TypeParam::pair_type;
  thrust::tabulate(this->pairs.begin(), this->pairs.end(),
                   identical_pair_generator<pair_type>{});
  // Insert a single pair
  EXPECT_TRUE(thrust::all_of(this->pairs.begin(), this->pairs.begin() + 1,
                             insert_pair<map_type, pair_type>{*this->map}));
  // Identical inserts should all return false (no new insert)
  EXPECT_FALSE(thrust::all_of(this->pairs.begin(), this->pairs.end(),
                              insert_pair<map_type, pair_type>{*this->map}));

  // All pairs should be present in the map
  EXPECT_TRUE(thrust::all_of(this->pairs.begin(), this->pairs.end(),
                             find_pair<map_type, pair_type>{*this->map}));
}

TYPED_TEST(InsertTest, IdenticalKeysUniqueValues) {
  using map_type = typename TypeParam::map_type;
  using pair_type = typename TypeParam::pair_type;
  thrust::tabulate(this->pairs.begin(), this->pairs.end(),
                   identical_key_generator<pair_type>{});

  // Insert a single pair
  EXPECT_TRUE(thrust::all_of(this->pairs.begin(), this->pairs.begin() + 1,
                             insert_pair<map_type, pair_type>{*this->map}));

  // Identical key inserts should all return false (no new insert)
  EXPECT_FALSE(thrust::all_of(this->pairs.begin() + 1, this->pairs.end(),
                              insert_pair<map_type, pair_type>{*this->map}));

  // Only first pair is present in map
  EXPECT_TRUE(thrust::all_of(this->pairs.begin(), this->pairs.begin() + 1,
                             find_pair<map_type, pair_type>{*this->map}));

  EXPECT_FALSE(thrust::all_of(this->pairs.begin() + 1, this->pairs.end(),
                              find_pair<map_type, pair_type>{*this->map}));
}
