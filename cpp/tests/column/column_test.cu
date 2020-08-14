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
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_list_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

#include <thrust/sequence.h>
#include <random>

template <typename T>
struct TypedColumnTest : public cudf::test::BaseFixture {
  cudf::data_type type() { return cudf::data_type{cudf::type_to_id<T>()}; }

  TypedColumnTest()
    : data{_num_elements * cudf::size_of(type())},
      mask{cudf::bitmask_allocation_size_bytes(_num_elements)}
  {
    auto typed_data = static_cast<char*>(data.data());
    auto typed_mask = static_cast<char*>(mask.data());
    thrust::sequence(thrust::device, typed_data, typed_data + data.size());
    thrust::sequence(thrust::device, typed_mask, typed_mask + mask.size());
  }

  cudf::size_type num_elements() { return _num_elements; }

  std::random_device r;
  std::default_random_engine generator{r()};
  std::uniform_int_distribution<cudf::size_type> distribution{200, 1000};
  cudf::size_type _num_elements{distribution(generator)};
  rmm::device_buffer data{};
  rmm::device_buffer mask{};
  rmm::device_buffer all_valid_mask{create_null_mask(num_elements(), cudf::mask_state::ALL_VALID)};
  rmm::device_buffer all_null_mask{create_null_mask(num_elements(), cudf::mask_state::ALL_NULL)};
};

TYPED_TEST_CASE(TypedColumnTest, cudf::test::Types<int32_t>);

/**
 * @brief Verifies equality of the properties and data of a `column`'s views.
 *
 * @param col The `column` to verify
 **/
void verify_column_views(cudf::column col)
{
  cudf::column_view view                 = col;
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

TYPED_TEST(TypedColumnTest, DefaultNullCountNoMask)
{
  cudf::column col{this->type(), this->num_elements(), this->data};
  EXPECT_FALSE(col.nullable());
  EXPECT_FALSE(col.has_nulls());
  EXPECT_EQ(0, col.null_count());
}

TYPED_TEST(TypedColumnTest, DefaultNullCountEmptyMask)
{
  cudf::column col{this->type(), this->num_elements(), this->data, rmm::device_buffer{}};
  EXPECT_FALSE(col.nullable());
  EXPECT_FALSE(col.has_nulls());
  EXPECT_EQ(0, col.null_count());
}

TYPED_TEST(TypedColumnTest, DefaultNullCountAllValid)
{
  cudf::column col{this->type(), this->num_elements(), this->data, this->all_valid_mask};
  EXPECT_TRUE(col.nullable());
  EXPECT_FALSE(col.has_nulls());
  EXPECT_EQ(0, col.null_count());
}

TYPED_TEST(TypedColumnTest, ExplicitNullCountAllValid)
{
  cudf::column col{this->type(), this->num_elements(), this->data, this->all_valid_mask, 0};
  EXPECT_TRUE(col.nullable());
  EXPECT_FALSE(col.has_nulls());
  EXPECT_EQ(0, col.null_count());
}

TYPED_TEST(TypedColumnTest, DefaultNullCountAllNull)
{
  cudf::column col{this->type(), this->num_elements(), this->data, this->all_null_mask};
  EXPECT_TRUE(col.nullable());
  EXPECT_TRUE(col.has_nulls());
  EXPECT_EQ(this->num_elements(), col.null_count());
}

TYPED_TEST(TypedColumnTest, ExplicitNullCountAllNull)
{
  cudf::column col{
    this->type(), this->num_elements(), this->data, this->all_null_mask, this->num_elements()};
  EXPECT_TRUE(col.nullable());
  EXPECT_TRUE(col.has_nulls());
  EXPECT_EQ(this->num_elements(), col.null_count());
}

TYPED_TEST(TypedColumnTest, SetNullCountNoMask)
{
  cudf::column col{this->type(), this->num_elements(), this->data};
  EXPECT_THROW(col.set_null_count(1), cudf::logic_error);
}

TYPED_TEST(TypedColumnTest, SetEmptyNullMaskNonZeroNullCount)
{
  cudf::column col{this->type(), this->num_elements(), this->data};
  rmm::device_buffer empty_null_mask{};
  EXPECT_THROW(col.set_null_mask(empty_null_mask, this->num_elements()), cudf::logic_error);
}

TYPED_TEST(TypedColumnTest, SetInvalidSizeNullMaskNonZeroNullCount)
{
  cudf::column col{this->type(), this->num_elements(), this->data};
  auto invalid_size_null_mask =
    create_null_mask(std::min(this->num_elements() - 50, 0), cudf::mask_state::ALL_VALID);
  EXPECT_THROW(col.set_null_mask(invalid_size_null_mask, this->num_elements()), cudf::logic_error);
}

TYPED_TEST(TypedColumnTest, SetNullCountEmptyMask)
{
  cudf::column col{this->type(), this->num_elements(), this->data, rmm::device_buffer{}};
  EXPECT_THROW(col.set_null_count(1), cudf::logic_error);
}

TYPED_TEST(TypedColumnTest, SetNullCountAllValid)
{
  cudf::column col{this->type(), this->num_elements(), this->data, this->all_valid_mask};
  EXPECT_NO_THROW(col.set_null_count(0));
  EXPECT_EQ(0, col.null_count());
}

TYPED_TEST(TypedColumnTest, SetNullCountAllNull)
{
  cudf::column col{this->type(), this->num_elements(), this->data, this->all_null_mask};
  EXPECT_NO_THROW(col.set_null_count(this->num_elements()));
  EXPECT_EQ(this->num_elements(), col.null_count());
}

TYPED_TEST(TypedColumnTest, ResetNullCountAllNull)
{
  cudf::column col{this->type(), this->num_elements(), this->data, this->all_null_mask};

  EXPECT_EQ(this->num_elements(), col.null_count());
  EXPECT_NO_THROW(col.set_null_count(cudf::UNKNOWN_NULL_COUNT));
  EXPECT_EQ(this->num_elements(), col.null_count());
}

TYPED_TEST(TypedColumnTest, ResetNullCountAllValid)
{
  cudf::column col{this->type(), this->num_elements(), this->data, this->all_valid_mask};
  EXPECT_EQ(0, col.null_count());
  EXPECT_NO_THROW(col.set_null_count(cudf::UNKNOWN_NULL_COUNT));
  EXPECT_EQ(0, col.null_count());
}

TYPED_TEST(TypedColumnTest, CopyDataNoMask)
{
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
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(v.head(), this->data.data(), this->data.size());
}

TYPED_TEST(TypedColumnTest, MoveDataNoMask)
{
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

TYPED_TEST(TypedColumnTest, CopyDataAndMask)
{
  cudf::column col{this->type(), this->num_elements(), this->data, this->all_valid_mask};
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
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(v.head(), this->data.data(), this->data.size());
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(v.null_mask(), this->all_valid_mask.data(), this->mask.size());
}

TYPED_TEST(TypedColumnTest, MoveDataAndMask)
{
  void* original_data = this->data.data();
  void* original_mask = this->all_valid_mask.data();
  cudf::column col{
    this->type(), this->num_elements(), std::move(this->data), std::move(this->all_valid_mask)};
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

TYPED_TEST(TypedColumnTest, CopyConstructorNoMask)
{
  cudf::column original{this->type(), this->num_elements(), this->data};
  cudf::column copy{original};
  verify_column_views(copy);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(original, copy);

  // Verify deep copy
  cudf::column_view original_view = original;
  cudf::column_view copy_view     = copy;
  EXPECT_NE(original_view.head(), copy_view.head());
}

TYPED_TEST(TypedColumnTest, CopyConstructorWithMask)
{
  cudf::column original{this->type(), this->num_elements(), this->data, this->all_valid_mask};
  cudf::column copy{original};
  verify_column_views(copy);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(original, copy);

  // Verify deep copy
  cudf::column_view original_view = original;
  cudf::column_view copy_view     = copy;
  EXPECT_NE(original_view.head(), copy_view.head());
  EXPECT_NE(original_view.null_mask(), copy_view.null_mask());
}

TYPED_TEST(TypedColumnTest, MoveConstructorNoMask)
{
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

TYPED_TEST(TypedColumnTest, MoveConstructorWithMask)
{
  cudf::column original{this->type(), this->num_elements(), this->data, this->all_valid_mask};
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

TYPED_TEST(TypedColumnTest, ConstructWithChildren)
{
  std::vector<std::unique_ptr<cudf::column>> children;
  children.emplace_back(std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::INT8}, 42, this->data, this->all_valid_mask));
  children.emplace_back(std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::FLOAT64}, 314, this->data, this->all_valid_mask));
  cudf::column col{this->type(),
                   this->num_elements(),
                   this->data,
                   this->all_valid_mask,
                   cudf::UNKNOWN_NULL_COUNT,
                   std::move(children)};

  verify_column_views(col);
  EXPECT_EQ(2, col.num_children());
  EXPECT_EQ(cudf::data_type{cudf::type_id::INT8}, col.child(0).type());
  EXPECT_EQ(42, col.child(0).size());
  EXPECT_EQ(cudf::data_type{cudf::type_id::FLOAT64}, col.child(1).type());
  EXPECT_EQ(314, col.child(1).size());
}

TYPED_TEST(TypedColumnTest, ReleaseNoChildren)
{
  cudf::column col{this->type(), this->num_elements(), this->data, this->all_valid_mask};
  auto original_data = col.view().head();
  auto original_mask = col.view().null_mask();

  cudf::column::contents contents = col.release();
  EXPECT_EQ(original_data, contents.data->data());
  EXPECT_EQ(original_mask, contents.null_mask->data());
  EXPECT_EQ(0u, contents.children.size());
  EXPECT_EQ(0, col.size());
  EXPECT_EQ(0, col.null_count());
  EXPECT_EQ(cudf::data_type{cudf::type_id::EMPTY}, col.type());
  EXPECT_EQ(0, col.num_children());
}

TYPED_TEST(TypedColumnTest, ReleaseWithChildren)
{
  std::vector<std::unique_ptr<cudf::column>> children;
  children.emplace_back(std::make_unique<cudf::column>(
    this->type(), this->num_elements(), this->data, this->all_valid_mask));
  children.emplace_back(std::make_unique<cudf::column>(
    this->type(), this->num_elements(), this->data, this->all_valid_mask));
  cudf::column col{this->type(),
                   this->num_elements(),
                   this->data,
                   this->all_valid_mask,
                   cudf::UNKNOWN_NULL_COUNT,
                   std::move(children)};

  auto original_data = col.view().head();
  auto original_mask = col.view().null_mask();

  cudf::column::contents contents = col.release();
  EXPECT_EQ(original_data, contents.data->data());
  EXPECT_EQ(original_mask, contents.null_mask->data());
  EXPECT_EQ(2u, contents.children.size());
  EXPECT_EQ(0, col.size());
  EXPECT_EQ(0, col.null_count());
  EXPECT_EQ(cudf::data_type{cudf::type_id::EMPTY}, col.type());
  EXPECT_EQ(0, col.num_children());
}

TYPED_TEST(TypedColumnTest, ColumnViewConstructorWithMask)
{
  cudf::column original{this->type(), this->num_elements(), this->data, this->all_valid_mask};
  cudf::column_view original_view = original;
  cudf::column copy{original_view};
  verify_column_views(copy);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(original, copy);

  // Verify deep copy
  cudf::column_view copy_view = copy;
  EXPECT_NE(original_view.head(), copy_view.head());
  EXPECT_NE(original_view.null_mask(), copy_view.null_mask());
}

template <typename T>
struct ListsColumnTest : public cudf::test::BaseFixture {
};

using NumericTypesNotBool =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

TYPED_TEST_CASE(ListsColumnTest, NumericTypesNotBool);

TYPED_TEST(ListsColumnTest, ListsColumnViewConstructor)
{
  cudf::test::lists_column_wrapper<TypeParam> list{{1, 2}, {3, 4}, {5, 6, 7}, {8, 9}};

  auto result = std::make_unique<cudf::column>(list);

  cudf::test::expect_columns_equal(list, result->view());
}

TYPED_TEST(ListsColumnTest, ListsSlicedColumnViewConstructor)
{
  cudf::test::lists_column_wrapper<TypeParam> list{{1, 2}, {3, 4}, {5, 6, 7}, {8, 9}};
  cudf::test::lists_column_wrapper<TypeParam> expect{{3, 4}, {5, 6, 7}};

  auto sliced = cudf::slice(list, {1, 3}).front();
  auto result = std::make_unique<cudf::column>(sliced);

  cudf::test::expect_columns_equal(expect, result->view());
}

TYPED_TEST(ListsColumnTest, ListsSlicedColumnViewConstructorWithNulls)
{
  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  auto expect_valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? false : true; });

  using LCW = cudf::test::lists_column_wrapper<TypeParam>;

  cudf::test::lists_column_wrapper<TypeParam> list{
    {{{{1, 2}, {3, 4}}, valids}, LCW{}, {{{5, 6, 7}, LCW{}, {8, 9}}, valids}, LCW{}, LCW{}},
    valids};

  cudf::test::lists_column_wrapper<TypeParam> expect{
    {LCW{}, {{{5, 6, 7}, LCW{}, {8, 9}}, valids}, LCW{}, LCW{}}, expect_valids};

  auto sliced = cudf::slice(list, {1, 5}).front();
  auto result = std::make_unique<cudf::column>(sliced);

  cudf::test::expect_columns_equal(expect, result->view());

  // TODO: null mask equality is being checked separately because
  // expect_columns_equal doesn't do the check for lists columns.
  // This is fixed in https://github.com/rapidsai/cudf/pull/5904,
  // so we should remove this check after that's merged:
  cudf::test::expect_columns_equal(
    cudf::mask_to_bools(result->view().null_mask(), 0, 4)->view(),
    cudf::mask_to_bools(static_cast<cudf::column_view>(expect).null_mask(), 0, 4)->view());
}

CUDF_TEST_PROGRAM_MAIN()
