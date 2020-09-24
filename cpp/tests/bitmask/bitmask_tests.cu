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
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

struct BitmaskUtilitiesTest : public cudf::test::BaseFixture {
};

TEST_F(BitmaskUtilitiesTest, StateNullCount)
{
  EXPECT_EQ(0, cudf::state_null_count(cudf::mask_state::UNALLOCATED, 42));
  EXPECT_EQ(cudf::UNKNOWN_NULL_COUNT, cudf::state_null_count(cudf::mask_state::UNINITIALIZED, 42));
  EXPECT_EQ(42, cudf::state_null_count(cudf::mask_state::ALL_NULL, 42));
  EXPECT_EQ(0, cudf::state_null_count(cudf::mask_state::ALL_VALID, 42));
}

TEST_F(BitmaskUtilitiesTest, BitmaskAllocationSize)
{
  EXPECT_EQ(0u, cudf::bitmask_allocation_size_bytes(0));
  EXPECT_EQ(64u, cudf::bitmask_allocation_size_bytes(1));
  EXPECT_EQ(64u, cudf::bitmask_allocation_size_bytes(512));
  EXPECT_EQ(128u, cudf::bitmask_allocation_size_bytes(513));
  EXPECT_EQ(128u, cudf::bitmask_allocation_size_bytes(1023));
  EXPECT_EQ(128u, cudf::bitmask_allocation_size_bytes(1024));
  EXPECT_EQ(192u, cudf::bitmask_allocation_size_bytes(1025));
}

TEST_F(BitmaskUtilitiesTest, NumBitmaskWords)
{
  EXPECT_EQ(0, cudf::num_bitmask_words(0));
  EXPECT_EQ(1, cudf::num_bitmask_words(1));
  EXPECT_EQ(1, cudf::num_bitmask_words(31));
  EXPECT_EQ(1, cudf::num_bitmask_words(32));
  EXPECT_EQ(2, cudf::num_bitmask_words(33));
  EXPECT_EQ(2, cudf::num_bitmask_words(63));
  EXPECT_EQ(2, cudf::num_bitmask_words(64));
  EXPECT_EQ(3, cudf::num_bitmask_words(65));
}

struct CountBitmaskTest : public cudf::test::BaseFixture {
};

TEST_F(CountBitmaskTest, NullMask)
{
  EXPECT_EQ(0, cudf::count_set_bits(nullptr, 0, 32));

  std::vector<cudf::size_type> indices = {0, 32, 7, 25};
  auto counts                          = cudf::segmented_count_set_bits(nullptr, indices);
  EXPECT_EQ(indices.size(), counts.size() * 2);
  for (size_t i = 0; i < counts.size(); i++) {
    EXPECT_EQ(indices[2 * i + 1] - indices[2 * i], counts[i]);
  }
}

TEST_F(CountBitmaskTest, NegativeStart)
{
  thrust::device_vector<cudf::bitmask_type> mask(1, 0);
  EXPECT_THROW(cudf::count_set_bits(mask.data().get(), -1, 32), cudf::logic_error);

  std::vector<cudf::size_type> indices = {0, 16, -1, 32};
  EXPECT_THROW(cudf::segmented_count_set_bits(mask.data().get(), indices), cudf::logic_error);
}

TEST_F(CountBitmaskTest, StartLargerThanStop)
{
  thrust::device_vector<cudf::bitmask_type> mask(1, 0);
  EXPECT_THROW(cudf::count_set_bits(mask.data().get(), 32, 31), cudf::logic_error);

  std::vector<cudf::size_type> indices = {0, 16, 31, 30};
  EXPECT_THROW(cudf::segmented_count_set_bits(mask.data().get(), indices), cudf::logic_error);
}

TEST_F(CountBitmaskTest, EmptyRange)
{
  thrust::device_vector<cudf::bitmask_type> mask(1, 0);
  EXPECT_EQ(0, cudf::count_set_bits(mask.data().get(), 17, 17));

  std::vector<cudf::size_type> indices = {0, 0, 17, 17};
  auto counts                          = cudf::segmented_count_set_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{0, 0}));
}

TEST_F(CountBitmaskTest, SingleWordAllZero)
{
  thrust::device_vector<cudf::bitmask_type> mask(1, 0);
  EXPECT_EQ(0, cudf::count_set_bits(mask.data().get(), 0, 32));

  std::vector<cudf::size_type> indices = {0, 32, 0, 32};
  auto counts                          = cudf::segmented_count_set_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{0, 0}));
}

TEST_F(CountBitmaskTest, SingleBitAllZero)
{
  thrust::device_vector<cudf::bitmask_type> mask(1, 0);
  EXPECT_EQ(0, cudf::count_set_bits(mask.data().get(), 17, 18));

  std::vector<cudf::size_type> indices = {17, 18, 7, 8};
  auto counts                          = cudf::segmented_count_set_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{0, 0}));
}

TEST_F(CountBitmaskTest, SingleBitAllSet)
{
  thrust::device_vector<cudf::bitmask_type> mask(1, ~cudf::bitmask_type{0});
  EXPECT_EQ(1, cudf::count_set_bits(mask.data().get(), 13, 14));

  std::vector<cudf::size_type> indices = {13, 14, 0, 1};
  auto counts                          = cudf::segmented_count_set_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{1, 1}));
}

TEST_F(CountBitmaskTest, SingleWordAllBitsSet)
{
  thrust::device_vector<cudf::bitmask_type> mask(1, ~cudf::bitmask_type{0});
  EXPECT_EQ(32, cudf::count_set_bits(mask.data().get(), 0, 32));

  std::vector<cudf::size_type> indices = {0, 32, 0, 32};
  auto counts                          = cudf::segmented_count_set_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{32, 32}));
}

TEST_F(CountBitmaskTest, SingleWordPreSlack)
{
  thrust::device_vector<cudf::bitmask_type> mask(1, ~cudf::bitmask_type{0});
  EXPECT_EQ(25, cudf::count_set_bits(mask.data().get(), 7, 32));

  std::vector<cudf::size_type> indices = {7, 32, 8, 32};
  auto counts                          = cudf::segmented_count_set_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{25, 24}));
}

TEST_F(CountBitmaskTest, SingleWordPostSlack)
{
  thrust::device_vector<cudf::bitmask_type> mask(1, ~cudf::bitmask_type{0});
  EXPECT_EQ(17, cudf::count_set_bits(mask.data().get(), 0, 17));

  std::vector<cudf::size_type> indices = {0, 17, 0, 18};
  auto counts                          = cudf::segmented_count_set_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{17, 18}));
}

TEST_F(CountBitmaskTest, SingleWordSubset)
{
  thrust::device_vector<cudf::bitmask_type> mask(1, ~cudf::bitmask_type{0});
  EXPECT_EQ(30, cudf::count_set_bits(mask.data().get(), 1, 31));

  std::vector<cudf::size_type> indices = {1, 31, 7, 17};
  auto counts                          = cudf::segmented_count_set_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{30, 10}));
}

TEST_F(CountBitmaskTest, SingleWordSubset2)
{
  thrust::device_vector<cudf::bitmask_type> mask(1, ~cudf::bitmask_type{0});
  EXPECT_EQ(28, cudf::count_set_bits(mask.data().get(), 2, 30));

  std::vector<cudf::size_type> indices = {4, 16, 2, 30};
  auto counts                          = cudf::segmented_count_set_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{12, 28}));
}

TEST_F(CountBitmaskTest, MultipleWordsAllBits)
{
  thrust::device_vector<cudf::bitmask_type> mask(10, ~cudf::bitmask_type{0});
  EXPECT_EQ(320, cudf::count_set_bits(mask.data().get(), 0, 320));

  std::vector<cudf::size_type> indices = {0, 320, 0, 320};
  auto counts                          = cudf::segmented_count_set_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{320, 320}));
}

TEST_F(CountBitmaskTest, MultipleWordsSubsetWordBoundary)
{
  thrust::device_vector<cudf::bitmask_type> mask(10, ~cudf::bitmask_type{0});
  EXPECT_EQ(256, cudf::count_set_bits(mask.data().get(), 32, 288));

  std::vector<cudf::size_type> indices = {32, 192, 32, 288};
  auto counts                          = cudf::segmented_count_set_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{160, 256}));
}

TEST_F(CountBitmaskTest, MultipleWordsSplitWordBoundary)
{
  thrust::device_vector<cudf::bitmask_type> mask(10, ~cudf::bitmask_type{0});
  EXPECT_EQ(2, cudf::count_set_bits(mask.data().get(), 31, 33));

  std::vector<cudf::size_type> indices = {31, 33, 60, 67};
  auto counts                          = cudf::segmented_count_set_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{2, 7}));
}

TEST_F(CountBitmaskTest, MultipleWordsSubset)
{
  thrust::device_vector<cudf::bitmask_type> mask(10, ~cudf::bitmask_type{0});
  EXPECT_EQ(226, cudf::count_set_bits(mask.data().get(), 67, 293));

  std::vector<cudf::size_type> indices = {67, 293, 37, 319};
  auto counts                          = cudf::segmented_count_set_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{226, 282}));
}

TEST_F(CountBitmaskTest, MultipleWordsSingleBit)
{
  thrust::device_vector<cudf::bitmask_type> mask(10, ~cudf::bitmask_type{0});
  EXPECT_EQ(1, cudf::count_set_bits(mask.data().get(), 67, 68));

  std::vector<cudf::size_type> indices = {67, 68, 31, 32, 192, 193};
  auto counts                          = cudf::segmented_count_set_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{1, 1, 1}));
}

using CountUnsetBitsTest = CountBitmaskTest;

TEST_F(CountUnsetBitsTest, SingleBitAllSet)
{
  thrust::device_vector<cudf::bitmask_type> mask(1, ~cudf::bitmask_type{0});
  EXPECT_EQ(0, cudf::count_unset_bits(mask.data().get(), 13, 14));

  std::vector<cudf::size_type> indices = {13, 14, 31, 32};
  auto counts = cudf::segmented_count_unset_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{0, 0}));
}

TEST_F(CountUnsetBitsTest, NullMask)
{
  EXPECT_EQ(0, cudf::count_unset_bits(nullptr, 0, 32));

  std::vector<cudf::size_type> indices = {0, 32, 7, 25};
  auto counts                          = cudf::segmented_count_unset_bits(nullptr, indices);
  EXPECT_EQ(indices.size(), counts.size() * 2);
  for (size_t i = 0; i < counts.size(); i++) { EXPECT_EQ(0, counts[i]); }
}

TEST_F(CountUnsetBitsTest, SingleWordAllBits)
{
  thrust::device_vector<cudf::bitmask_type> mask(1, cudf::bitmask_type{0});
  EXPECT_EQ(32, cudf::count_unset_bits(mask.data().get(), 0, 32));

  std::vector<cudf::size_type> indices = {0, 32, 0, 32};
  auto counts = cudf::segmented_count_unset_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{32, 32}));
}

TEST_F(CountUnsetBitsTest, SingleWordPreSlack)
{
  thrust::device_vector<cudf::bitmask_type> mask(1, cudf::bitmask_type{0});
  EXPECT_EQ(25, cudf::count_unset_bits(mask.data().get(), 7, 32));

  std::vector<cudf::size_type> indices = {7, 32, 8, 32};
  auto counts = cudf::segmented_count_unset_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{25, 24}));
}

TEST_F(CountUnsetBitsTest, SingleWordPostSlack)
{
  thrust::device_vector<cudf::bitmask_type> mask(1, cudf::bitmask_type{0});
  EXPECT_EQ(17, cudf::count_unset_bits(mask.data().get(), 0, 17));

  std::vector<cudf::size_type> indices = {0, 17, 0, 18};
  auto counts = cudf::segmented_count_unset_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{17, 18}));
}

TEST_F(CountUnsetBitsTest, SingleWordSubset)
{
  thrust::device_vector<cudf::bitmask_type> mask(1, cudf::bitmask_type{0});
  EXPECT_EQ(30, cudf::count_unset_bits(mask.data().get(), 1, 31));

  std::vector<cudf::size_type> indices = {1, 31, 7, 17};
  auto counts = cudf::segmented_count_unset_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{30, 10}));
}

TEST_F(CountUnsetBitsTest, SingleWordSubset2)
{
  thrust::device_vector<cudf::bitmask_type> mask(1, cudf::bitmask_type{0});
  EXPECT_EQ(28, cudf::count_unset_bits(mask.data().get(), 2, 30));

  std::vector<cudf::size_type> indices = {4, 16, 2, 30};
  auto counts = cudf::segmented_count_unset_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{12, 28}));
}

TEST_F(CountUnsetBitsTest, MultipleWordsAllBits)
{
  thrust::device_vector<cudf::bitmask_type> mask(10, cudf::bitmask_type{0});
  EXPECT_EQ(320, cudf::count_unset_bits(mask.data().get(), 0, 320));

  std::vector<cudf::size_type> indices = {0, 320, 0, 320};
  auto counts = cudf::segmented_count_unset_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{320, 320}));
}

TEST_F(CountUnsetBitsTest, MultipleWordsSubsetWordBoundary)
{
  thrust::device_vector<cudf::bitmask_type> mask(10, cudf::bitmask_type{0});
  EXPECT_EQ(256, cudf::count_unset_bits(mask.data().get(), 32, 288));

  std::vector<cudf::size_type> indices = {32, 192, 32, 288};
  auto counts = cudf::segmented_count_unset_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, testing::ContainerEq(std::vector<cudf::size_type>{160, 256}));
}

TEST_F(CountUnsetBitsTest, MultipleWordsSplitWordBoundary)
{
  thrust::device_vector<cudf::bitmask_type> mask(10, cudf::bitmask_type{0});
  EXPECT_EQ(2, cudf::count_unset_bits(mask.data().get(), 31, 33));

  std::vector<cudf::size_type> indices = {31, 33, 60, 67};
  auto counts = cudf::segmented_count_unset_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, ::testing::ContainerEq(std::vector<cudf::size_type>{2, 7}));
}

TEST_F(CountUnsetBitsTest, MultipleWordsSubset)
{
  thrust::device_vector<cudf::bitmask_type> mask(10, cudf::bitmask_type{0});
  EXPECT_EQ(226, cudf::count_unset_bits(mask.data().get(), 67, 293));

  std::vector<cudf::size_type> indices = {67, 293, 37, 319};
  auto counts = cudf::segmented_count_unset_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, ::testing::ContainerEq(std::vector<cudf::size_type>{226, 282}));
}

TEST_F(CountUnsetBitsTest, MultipleWordsSingleBit)
{
  thrust::device_vector<cudf::bitmask_type> mask(10, cudf::bitmask_type{0});
  EXPECT_EQ(1, cudf::count_unset_bits(mask.data().get(), 67, 68));

  std::vector<cudf::size_type> indices = {67, 68, 31, 32, 192, 193};
  auto counts = cudf::segmented_count_unset_bits(mask.data().get(), indices);
  EXPECT_THAT(counts, ::testing::ContainerEq(std::vector<cudf::size_type>{1, 1, 1}));
}

struct CopyBitmaskTest : public cudf::test::BaseFixture, cudf::test::UniformRandomGenerator<int> {
  CopyBitmaskTest() : cudf::test::UniformRandomGenerator<int>{0, 1} {}
};

void cleanEndWord(rmm::device_buffer &mask, int begin_bit, int end_bit)
{
  thrust::device_ptr<cudf::bitmask_type> ptr(static_cast<cudf::bitmask_type *>(mask.data()));

  auto number_of_mask_words = cudf::num_bitmask_words(static_cast<size_t>(end_bit - begin_bit));
  auto number_of_bits       = end_bit - begin_bit;
  if (number_of_bits % 32 != 0) {
    auto end_mask = ptr[number_of_mask_words - 1];
    end_mask      = end_mask & ((1 << (number_of_bits % 32)) - 1);
  }
}

TEST_F(CopyBitmaskTest, NegativeStart)
{
  thrust::device_vector<cudf::bitmask_type> mask(1, 0);
  EXPECT_THROW(cudf::copy_bitmask(mask.data().get(), -1, 32), cudf::logic_error);
}

TEST_F(CopyBitmaskTest, StartLargerThanStop)
{
  thrust::device_vector<cudf::bitmask_type> mask(1, 0);
  EXPECT_THROW(cudf::copy_bitmask(mask.data().get(), 32, 31), cudf::logic_error);
}

TEST_F(CopyBitmaskTest, EmptyRange)
{
  thrust::device_vector<cudf::bitmask_type> mask(1, 0);
  auto buff = cudf::copy_bitmask(mask.data().get(), 17, 17);
  EXPECT_EQ(0, static_cast<int>(buff.size()));
}

TEST_F(CopyBitmaskTest, NullPtr)
{
  auto buff = cudf::copy_bitmask(nullptr, 17, 17);
  EXPECT_EQ(0, static_cast<int>(buff.size()));
}

TEST_F(CopyBitmaskTest, TestZeroOffset)
{
  thrust::host_vector<int> validity_bit(1000);
  for (auto &m : validity_bit) { m = this->generate(); }
  auto input_mask = cudf::test::detail::make_null_mask(validity_bit.begin(), validity_bit.end());

  int begin_bit         = 0;
  int end_bit           = 800;
  auto gold_splice_mask = cudf::test::detail::make_null_mask(validity_bit.begin() + begin_bit,
                                                             validity_bit.begin() + end_bit);

  auto splice_mask = cudf::copy_bitmask(
    static_cast<const cudf::bitmask_type *>(input_mask.data()), begin_bit, end_bit);

  cleanEndWord(splice_mask, begin_bit, end_bit);
  auto number_of_bits = end_bit - begin_bit;
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(
    gold_splice_mask.data(), splice_mask.data(), number_of_bits / CHAR_BIT);
}

TEST_F(CopyBitmaskTest, TestNonZeroOffset)
{
  thrust::host_vector<int> validity_bit(1000);
  for (auto &m : validity_bit) { m = this->generate(); }
  auto input_mask = cudf::test::detail::make_null_mask(validity_bit.begin(), validity_bit.end());

  int begin_bit         = 321;
  int end_bit           = 998;
  auto gold_splice_mask = cudf::test::detail::make_null_mask(validity_bit.begin() + begin_bit,
                                                             validity_bit.begin() + end_bit);

  auto splice_mask = cudf::copy_bitmask(
    static_cast<const cudf::bitmask_type *>(input_mask.data()), begin_bit, end_bit);

  cleanEndWord(splice_mask, begin_bit, end_bit);
  auto number_of_bits = end_bit - begin_bit;
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(
    gold_splice_mask.data(), splice_mask.data(), number_of_bits / CHAR_BIT);
}

TEST_F(CopyBitmaskTest, TestCopyColumnViewVectorContiguous)
{
  cudf::data_type t{cudf::type_id::INT32};
  cudf::size_type num_elements = 1001;
  thrust::host_vector<int> validity_bit(num_elements);
  for (auto &m : validity_bit) { m = this->generate(); }
  auto gold_mask = cudf::test::detail::make_null_mask(validity_bit.begin(), validity_bit.end());

  auto copy_mask = gold_mask;
  cudf::column original{t, num_elements, rmm::device_buffer{num_elements * sizeof(int)}, copy_mask};
  std::vector<cudf::size_type> indices{0,
                                       104,
                                       104,
                                       128,
                                       128,
                                       152,
                                       152,
                                       311,
                                       311,
                                       491,
                                       491,
                                       583,
                                       583,
                                       734,
                                       734,
                                       760,
                                       760,
                                       num_elements};
  std::vector<cudf::column_view> views    = cudf::slice(original, indices);
  rmm::device_buffer concatenated_bitmask = cudf::concatenate_masks(views);
  cleanEndWord(concatenated_bitmask, 0, num_elements);
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(
    concatenated_bitmask.data(), gold_mask.data(), num_elements / CHAR_BIT);
}

TEST_F(CopyBitmaskTest, TestCopyColumnViewVectorDiscontiguous)
{
  cudf::data_type t{cudf::type_id::INT32};
  cudf::size_type num_elements = 1001;
  thrust::host_vector<int> validity_bit(num_elements);
  for (auto &m : validity_bit) { m = this->generate(); }
  auto gold_mask = cudf::test::detail::make_null_mask(validity_bit.begin(), validity_bit.end());
  std::vector<cudf::size_type> split{0, 104, 128, 152, 311, 491, 583, 734, 760, num_elements};

  std::vector<cudf::column> cols;
  std::vector<cudf::column_view> views;
  for (unsigned i = 0; i < split.size() - 1; i++) {
    cols.emplace_back(t,
                      split[i + 1] - split[i],
                      rmm::device_buffer{sizeof(int) * (split[i + 1] - split[i])},
                      cudf::test::detail::make_null_mask(validity_bit.begin() + split[i],
                                                         validity_bit.begin() + split[i + 1]));
    views.push_back(cols.back());
  }
  rmm::device_buffer concatenated_bitmask = cudf::concatenate_masks(views);
  cleanEndWord(concatenated_bitmask, 0, num_elements);
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(
    concatenated_bitmask.data(), gold_mask.data(), num_elements / CHAR_BIT);
}

CUDF_TEST_PROGRAM_MAIN()
