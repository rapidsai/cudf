/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

/*
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/hashing/detail/xxhash_64.cuh>
#include <cudf/utilities/default_stream.hpp>

#include <cuco/bloom_filter.cuh>

using StringType = cudf::string_view;

template <typename T>
class BloomFilter_TestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(BloomFilter_TestTyped, StringType);

TYPED_TEST(BloomFilter_TestTyped, TestStrings)
{
  using key_type    = StringType;
  using hasher_type = cudf::hashing::detail::XXHash_64<key_type>;
  using policy_type = cuco::arrow_filter_policy<key_type, hasher_type>;

  std::size_t constexpr num_filter_blocks = 4;
  std::size_t constexpr num_keys          = 50;
  auto stream                             = cudf::get_default_stream();

  // strings data
  auto data = cudf::test::strings_column_wrapper(
    {"seventh",    "fifteenth",   "second",      "tenth",      "fifth",       "first",
     "seventh",    "tenth",       "ninth",       "ninth",      "seventeenth", "eighteenth",
     "thirteenth", "fifth",       "fourth",      "twelfth",    "second",      "second",
     "fourth",     "seventh",     "seventh",     "tenth",      "thirteenth",  "seventeenth",
     "fifth",      "seventeenth", "eighth",      "fourth",     "second",      "eighteenth",
     "fifteenth",  "second",      "seventeenth", "thirteenth", "eighteenth",  "fifth",
     "seventh",    "tenth",       "fourteenth",  "first",      "fifth",       "fifth",
     "tenth",      "thirteenth",  "fourteenth",  "third",      "third",       "sixth",
     "first",      "third"});
  auto d_column = cudf::column_device_view::create(data);

  // Spawn a bloom filter
  cuco::bloom_filter<key_type,
                     cuco::extent<size_t>,
                     cuda::thread_scope_device,
                     policy_type,
                     cudf::detail::cuco_allocator<char>>
    filter{num_filter_blocks,
           cuco::thread_scope_device,
           {hasher_type{0}},
           cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
           stream};

  // Add strings to the bloom filter
  auto col = table.column(0);
  filter.add(d_column->begin<key_type>(), d_column->end<key_type>(), stream);

  // Number of words in the filter
  cudf::size_type const num_words = filter.block_extent() * filter.words_per_block;

  auto const output = cudf::column_view{
    cudf::data_type{cudf::type_id::UINT32}, num_words, filter.data(), nullptr, 0, 0, {}};

  using word_type = filter_type::word_type;

  // Expected filter bitset words computed using Arrow implementation here:
  // https://godbolt.org/z/oKfqcPWbY
  auto expected = cudf::test::fixed_width_column_wrapper<word_type>(
    {4194306,    4194305,  2359296,   1073774592, 524544,     1024,       268443648,  8519680,
     2147500040, 8421380,  269500416, 4202624,    8396802,    100665344,  2147747840, 5243136,
     131146,     655364,   285345792, 134222340,  545390596,  2281717768, 51201,      41943553,
     1619656708, 67441680, 8462730,   361220,     2216738864, 587333888,  4219272,    873463873});
  auto d_expected = cudf::column_device_view::create(expected);

  // Check
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output, d_expected);
}
*/