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
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.cuh>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_list.hpp>
#include <tests/utilities/typed_tests.hpp>

#include <thrust/sequence.h>
#include <random>

#include <gmock/gmock.h>

template <typename T>
struct TypedColumnTest : public cudf::test::BaseFixture {
  static std::size_t data_size() { return 1000; }
  static std::size_t mask_size() { return 100; }
  cudf::data_type type() { return cudf::data_type{cudf::exp::type_to_id<T>()}; }

  TypedColumnTest()
      : data{_num_elements * cudf::size_of(type())},
        mask{cudf::bitmask_allocation_size_bytes(_num_elements)} {
    auto typed_data = static_cast<char*>(data.data());
    auto typed_mask = static_cast<char*>(mask.data());
    thrust::sequence(thrust::device, typed_data, typed_data + data_size());
    thrust::sequence(thrust::device, typed_mask, typed_mask + mask_size());
  }

  cudf::size_type num_elements() { return _num_elements; }

  std::random_device r;
  std::default_random_engine generator{r()};
  std::uniform_int_distribution<cudf::size_type> distribution{200, 1000};
  cudf::size_type _num_elements{distribution(generator)};
  rmm::device_buffer data{};
  rmm::device_buffer mask{};
  rmm::device_buffer all_valid_mask{
      create_null_mask(num_elements(), cudf::mask_state::ALL_VALID)};
  rmm::device_buffer all_null_mask{
      create_null_mask(num_elements(), cudf::mask_state::ALL_NULL)};
};

TYPED_TEST_CASE(TypedColumnTest, cudf::test::Types<int32_t>);

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
  EXPECT_EQ(col.null_count(), view.null_count());
  EXPECT_EQ(col.null_count(), mutable_view.null_count());
  EXPECT_EQ(col.nullable(), view.nullable());
  EXPECT_EQ(col.nullable(), mutable_view.nullable());
  EXPECT_EQ(col.num_children(), view.num_children());
  EXPECT_EQ(col.num_children(), mutable_view.num_children());
  EXPECT_EQ(view.head(), mutable_view.head());
  EXPECT_EQ(view.data<char>(), mutable_view.data<char>());
  EXPECT_EQ(view.offset(), mutable_view.offset());
}

TYPED_TEST(TypedColumnTest, DefaultNullCountNoMask) {
  cudf::column col{this->type(), this->num_elements(), this->data};
  EXPECT_FALSE(col.nullable());
  EXPECT_FALSE(col.has_nulls());
  EXPECT_EQ(0, col.null_count());
}

TYPED_TEST(TypedColumnTest, DefaultNullCountEmptyMask) {
  cudf::column col{this->type(), this->num_elements(), this->data,
                   rmm::device_buffer{}};
  EXPECT_FALSE(col.nullable());
  EXPECT_FALSE(col.has_nulls());
  EXPECT_EQ(0, col.null_count());
}

TYPED_TEST(TypedColumnTest, DefaultNullCountAllValid) {
  cudf::column col{this->type(), this->num_elements(), this->data,
                   this->all_valid_mask};
  EXPECT_TRUE(col.nullable());
  EXPECT_FALSE(col.has_nulls());
  EXPECT_EQ(0, col.null_count());
}

TYPED_TEST(TypedColumnTest, ExplicitNullCountAllValid) {
  cudf::column col{this->type(), this->num_elements(), this->data,
                   this->all_valid_mask, 0};
  EXPECT_TRUE(col.nullable());
  EXPECT_FALSE(col.has_nulls());
  EXPECT_EQ(0, col.null_count());
}

TYPED_TEST(TypedColumnTest, DefaultNullCountAllNull) {
  cudf::column col{this->type(), this->num_elements(), this->data,
                   this->all_null_mask};
  EXPECT_TRUE(col.nullable());
  EXPECT_TRUE(col.has_nulls());
  EXPECT_EQ(this->num_elements(), col.null_count());
}

TYPED_TEST(TypedColumnTest, ExplicitNullCountAllNull) {
  cudf::column col{this->type(), this->num_elements(), this->data,
                   this->all_null_mask, this->num_elements()};
  EXPECT_TRUE(col.nullable());
  EXPECT_TRUE(col.has_nulls());
  EXPECT_EQ(this->num_elements(), col.null_count());
}

TYPED_TEST(TypedColumnTest, SetNullCountNoMask) {
  cudf::column col{this->type(), this->num_elements(), this->data};
  EXPECT_THROW(col.set_null_count(1), cudf::logic_error);
}

TYPED_TEST(TypedColumnTest, SetNullCountEmptyMask) {
  cudf::column col{this->type(), this->num_elements(), this->data,
                   rmm::device_buffer{}};
  EXPECT_THROW(col.set_null_count(1), cudf::logic_error);
}

TYPED_TEST(TypedColumnTest, SetNullCountAllValid) {
  cudf::column col{this->type(), this->num_elements(), this->data,
                   this->all_valid_mask};
  EXPECT_NO_THROW(col.set_null_count(0));
  EXPECT_EQ(0, col.null_count());
}

TYPED_TEST(TypedColumnTest, SetNullCountAllNull) {
  cudf::column col{this->type(), this->num_elements(), this->data,
                   this->all_null_mask};
  EXPECT_NO_THROW(col.set_null_count(this->num_elements()));
  EXPECT_EQ(this->num_elements(), col.null_count());
}

TYPED_TEST(TypedColumnTest, ResetNullCountAllNull) {
  cudf::column col{this->type(), this->num_elements(), this->data,
                   this->all_null_mask};

  EXPECT_EQ(this->num_elements(), col.null_count());
  EXPECT_NO_THROW(col.set_null_count(cudf::UNKNOWN_NULL_COUNT));
  EXPECT_EQ(this->num_elements(), col.null_count());
}

TYPED_TEST(TypedColumnTest, ResetNullCountAllValid) {
  cudf::column col{this->type(), this->num_elements(), this->data,
                   this->all_valid_mask};
  EXPECT_EQ(0, col.null_count());
  EXPECT_NO_THROW(col.set_null_count(cudf::UNKNOWN_NULL_COUNT));
  EXPECT_EQ(0, col.null_count());
}

TYPED_TEST(TypedColumnTest, CopyDataNoMask) {
  cudf::column col{this->type(), this->num_elements(), this->data};
  EXPECT_EQ(this->type(), col.type());
  EXPECT_FALSE(col.nullable());
  EXPECT_EQ(0, col.null_count());
  EXPECT_EQ(this->num_elements(), col.size());
  EXPECT_EQ(0, col.num_children());

  verify_column_views(col);

  // Verify deep copy
  cudf::column_view v = col;
  EXPECT_NE(v.head(), this->data.data());
  cudf::test::expect_equal_buffers(v.head(), this->data.data(),
                                   this->data.size());
}

TYPED_TEST(TypedColumnTest, MoveDataNoMask) {
  void* original_data = this->data.data();
  cudf::column col{this->type(), this->num_elements(), std::move(this->data)};
  EXPECT_EQ(this->type(), col.type());
  EXPECT_FALSE(col.nullable());
  EXPECT_EQ(0, col.null_count());
  EXPECT_EQ(this->num_elements(), col.size());
  EXPECT_EQ(0, col.num_children());

  verify_column_views(col);

  // Verify shallow copy
  cudf::column_view v = col;
  EXPECT_EQ(v.head(), original_data);
}

TYPED_TEST(TypedColumnTest, CopyDataAndMask) {
  cudf::column col{this->type(), this->num_elements(), this->data,
                   this->all_valid_mask};
  EXPECT_EQ(this->type(), col.type());
  EXPECT_TRUE(col.nullable());
  EXPECT_EQ(0, col.null_count());
  EXPECT_EQ(this->num_elements(), col.size());
  EXPECT_EQ(0, col.num_children());

  verify_column_views(col);

  // Verify deep copy
  cudf::column_view v = col;
  EXPECT_NE(v.head(), this->data.data());
  EXPECT_NE(v.null_mask(), this->all_valid_mask.data());
  cudf::test::expect_equal_buffers(v.head(), this->data.data(),
                                   this->data.size());
  cudf::test::expect_equal_buffers(v.null_mask(), this->all_valid_mask.data(),
                                   this->mask.size());
}

TYPED_TEST(TypedColumnTest, MoveDataAndMask) {
  void* original_data = this->data.data();
  void* original_mask = this->all_valid_mask.data();
  cudf::column col{this->type(), this->num_elements(), std::move(this->data),
                   std::move(this->all_valid_mask)};
  EXPECT_EQ(this->type(), col.type());
  EXPECT_TRUE(col.nullable());
  EXPECT_EQ(0, col.null_count());
  EXPECT_EQ(this->num_elements(), col.size());
  EXPECT_EQ(0, col.num_children());

  verify_column_views(col);

  // Verify shallow copy
  cudf::column_view v = col;
  EXPECT_EQ(v.head(), original_data);
  EXPECT_EQ(v.null_mask(), original_mask);
}

TYPED_TEST(TypedColumnTest, CopyConstructorNoMask) {
  cudf::column original{this->type(), this->num_elements(), this->data};
  cudf::column copy{original};
  verify_column_views(copy);
  cudf::test::expect_columns_equal(original, copy);

  // Verify deep copy
  cudf::column_view original_view = original;
  cudf::column_view copy_view = copy;
  EXPECT_NE(original_view.head(), copy_view.head());
}

TYPED_TEST(TypedColumnTest, CopyConstructorWithMask) {
  cudf::column original{this->type(), this->num_elements(), this->data,
                        this->all_valid_mask};
  cudf::column copy{original};
  verify_column_views(copy);
  cudf::test::expect_columns_equal(original, copy);

  // Verify deep copy
  cudf::column_view original_view = original;
  cudf::column_view copy_view = copy;
  EXPECT_NE(original_view.head(), copy_view.head());
  EXPECT_NE(original_view.null_mask(), copy_view.null_mask());
}

TYPED_TEST(TypedColumnTest, MoveConstructorNoMask) {
  cudf::column original{this->type(), this->num_elements(), this->data};

  auto original_data = original.view().head();

  cudf::column moved_to{std::move(original)};

  EXPECT_EQ(0, original.size());
  EXPECT_EQ(cudf::data_type{cudf::type_id::EMPTY}, original.type());

  verify_column_views(moved_to);

  // Verify move
  cudf::column_view moved_to_view = moved_to;
  EXPECT_EQ(original_data, moved_to_view.head());
}

TYPED_TEST(TypedColumnTest, MoveConstructorWithMask) {
  cudf::column original{this->type(), this->num_elements(), this->data,
                        this->all_valid_mask};
  auto original_data = original.view().head();
  auto original_mask = original.view().null_mask();
  cudf::column moved_to{std::move(original)};
  verify_column_views(moved_to);

  EXPECT_EQ(0, original.size());
  EXPECT_EQ(cudf::data_type{cudf::type_id::EMPTY}, original.type());

  // Verify move
  cudf::column_view moved_to_view = moved_to;
  EXPECT_EQ(original_data, moved_to_view.head());
  EXPECT_EQ(original_mask, moved_to_view.null_mask());
}