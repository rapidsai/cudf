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

#include <cudf/bitmask/bitmask.hpp>
#include <cudf/bitmask/bitmask_view.hpp>
#include <cudf/types.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

struct BitmaskTest : public ::testing::Test {
  std::unique_ptr<cudf::bitmask> bitmask;
};

TEST_F(BitmaskTest, DefaultConstructor) {
  EXPECT_NO_THROW(bitmask = std::make_unique<cudf::bitmask>());
  EXPECT_EQ(0, bitmask->size());
  EXPECT_EQ(nullptr, bitmask->data());
}

TEST_F(BitmaskTest, SizeConstructorWithDefaults) {
  cudf::size_type size{100};
  EXPECT_NO_THROW(bitmask = std::make_unique<cudf::bitmask>(size));
  EXPECT_EQ(size, bitmask->size());
  EXPECT_NE(nullptr, bitmask->data());
}

// TEST_F(BitmaskTest, CopyConstructor){
//  cudf::size_type size{100};
//  EXPECT_NO_THROW(bitmask = std::make_unique<cudf::bitmask>(size));
//  cudf::bitmask copy{*bitmask};
//}

TEST_F(BitmaskTest, MoveConstructor) {
  cudf::size_type size{100};
  EXPECT_NO_THROW(bitmask = std::make_unique<cudf::bitmask>(size));
  cudf::bitmask copy{std::move(*bitmask)};
}

TEST_F(BitmaskTest, TestViews) {
  cudf::size_type size{100};
  EXPECT_NO_THROW(bitmask = std::make_unique<cudf::bitmask>(size));
  EXPECT_EQ(size, bitmask->size());
  EXPECT_NE(nullptr, bitmask->data());

  cudf::bitmask_view view = bitmask->view();
  EXPECT_EQ(size, view.size());
  EXPECT_EQ(0, view.offset());
  EXPECT_NE(nullptr, view.data());
  EXPECT_EQ(bitmask->data(), view.data());

  // Implicit conversion of a bitmask to a bitmask view
  cudf::bitmask_view converted_view = *bitmask;
  EXPECT_EQ(converted_view.size(), view.size());
  EXPECT_EQ(converted_view.offset(), view.offset());
  EXPECT_EQ(converted_view.data(), view.data());

  cudf::mutable_bitmask_view mutable_view = bitmask->mutable_view();
  EXPECT_EQ(size, mutable_view.size());
  EXPECT_EQ(0, mutable_view.offset());
  EXPECT_NE(nullptr, mutable_view.data());
  EXPECT_EQ(bitmask->data(), mutable_view.data());

  // Implicit conversion of a bitmask to a mutable view
  cudf::mutable_bitmask_view converted_mutable_view = *bitmask;
  EXPECT_EQ(converted_mutable_view.size(), mutable_view.size());
  EXPECT_EQ(converted_mutable_view.offset(), mutable_view.offset());
  EXPECT_EQ(converted_mutable_view.data(), mutable_view.data());

  // Members of mutable and immutable views should be equal
  EXPECT_EQ(mutable_view.size(), view.size());
  EXPECT_EQ(mutable_view.offset(), view.offset());
  EXPECT_EQ(mutable_view.data(), view.data());
}