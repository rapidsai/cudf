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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.cuh>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_list.hpp>
#include <tests/utilities/typed_tests.hpp>

#include <thrust/sequence.h>

#include <gmock/gmock.h>

struct ColumnTest : public cudf::test::BaseFixture {
  static std::size_t data_size() { return 1000; }
  static std::size_t mask_size() { return 100; }
  cudf::size_type num_elements() { return 100; }

  ColumnTest() : data{data_size()}, mask{mask_size()} {
    // This input data is not intended to be realistic for the type or number of
    // elements. Instead, we fill the column with dummy data to verify data
    // movement and conversion
    auto typed_data = static_cast<char*>(data.data());
    auto typed_mask = static_cast<char*>(mask.data());
    thrust::sequence(thrust::device, typed_data, typed_data + data_size());
    thrust::sequence(thrust::device, typed_mask, typed_mask + mask_size());
  }

  rmm::device_buffer data{};
  rmm::device_buffer mask{};
};

template <typename T>
struct TypedColumnTest : public ColumnTest {
  cudf::data_type type() { return cudf::data_type{cudf::exp::type_to_id<T>()}; }
};

/**---------------------------------------------------------------------------*
 * @brief Verifies equality of the properties and data of a `column`'s views.
 *
 * @param col The `column` to verify
 *---------------------------------------------------------------------------**/
void verify_column_views(cudf::column col) {
  cudf::column_view view = col;
  cudf::mutable_column_view mutable_view = col;
  EXPECT_EQ(col.type(), view.type());
  EXPECT_EQ(col.type(), mutable_view.type());
  EXPECT_EQ(col.size(), view.size());
  EXPECT_EQ(col.size(), mutable_view.size());
  // TODO: These won't work until on-demand null count is implemented
  // EXPECT_EQ(col.null_count(), view.null_count());
  // EXPECT_EQ(col.null_count(), mutable_view.null_count());
  EXPECT_EQ(col.nullable(), view.nullable());
  EXPECT_EQ(col.nullable(), mutable_view.nullable());
  EXPECT_EQ(col.num_children(), view.num_children());
  EXPECT_EQ(col.num_children(), mutable_view.num_children());
  EXPECT_EQ(view.head(), mutable_view.head());
  EXPECT_EQ(view.data<char>(), mutable_view.data<char>());
  EXPECT_EQ(view.offset(), mutable_view.offset());
}

TYPED_TEST_CASE(TypedColumnTest, testing::Types<int32_t>);

TYPED_TEST(TypedColumnTest, CopyDataAndMask) {
  cudf::column col{this->type(), this->num_elements(), this->data, this->mask};
  EXPECT_EQ(this->type(), col.type());
  EXPECT_TRUE(col.nullable());
  EXPECT_EQ(this->num_elements(), col.size());
  EXPECT_EQ(0, col.num_children());

  verify_column_views(col);

  // Verify deep copy
  cudf::column_view v = col;
  EXPECT_NE(v.head(), this->data.data());
  EXPECT_NE(v.null_mask(), this->mask.data());
  cudf::test::expect_equal_buffers(v.head(), this->data.data(),
                                   this->data.size());
  cudf::test::expect_equal_buffers(v.null_mask(), this->mask.data(),
                                   this->mask.size());
}

TYPED_TEST(TypedColumnTest, MoveDataAndMask) {
  void* original_data = this->data.data();
  void* original_mask = this->mask.data();
  cudf::column col{this->type(), this->num_elements(), std::move(this->data),
                   std::move(this->mask)};
  EXPECT_EQ(this->type(), col.type());
  EXPECT_TRUE(col.nullable());
  EXPECT_EQ(this->num_elements(), col.size());
  EXPECT_EQ(0, col.num_children());

  verify_column_views(col);

  // Verify shallow copy
  cudf::column_view v = col;
  EXPECT_EQ(v.head(), original_data);
  EXPECT_EQ(v.null_mask(), original_mask);
}
