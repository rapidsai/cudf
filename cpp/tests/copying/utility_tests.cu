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

#include <tests/utilities/base_fixture.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/column/column_factories.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/type_lists.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <string>

template <typename T>
struct EmptyLikeTest : public cudf::test::BaseFixture {};

using numeric_types = cudf::test::NumericTypes;

TYPED_TEST_CASE(EmptyLikeTest, numeric_types);

TYPED_TEST(EmptyLikeTest, ColumnNumericTests) {
    cudf::size_type size = 10;
    cudf::mask_state state = cudf::ALL_VALID;
    auto input = make_numeric_column(cudf::data_type{cudf::experimental::type_to_id<TypeParam>()}, size, state);
    auto expected = make_numeric_column(cudf::data_type{cudf::experimental::type_to_id<TypeParam>()}, 0);
    auto got = cudf::experimental::empty_like(input->view());
    cudf::test::expect_column_properties_equal(*expected, *got);
}

struct EmptyLikeStringTest : public EmptyLikeTest <std::string> {};

rmm::device_vector<thrust::pair<const char*,cudf::size_type>> create_test_string () {
    std::vector<const char*> h_test_strings{ "the quick brown fox jumps over the lazy dog",
                                             "thé result does not include the value in the sum in",
                                             "", nullptr, "absent stop words" };
    cudf::size_type memsize = 0;
    for( auto itr=h_test_strings.begin(); itr!=h_test_strings.end(); ++itr )
        memsize += *itr ? (cudf::size_type)strlen(*itr) : 0;
    cudf::size_type count = (cudf::size_type)h_test_strings.size();
    thrust::host_vector<char> h_buffer(memsize);
    thrust::device_vector<char> d_buffer(memsize);
    thrust::host_vector<thrust::pair<const char*,cudf::size_type> > strings(count);
    thrust::host_vector<cudf::size_type> h_offsets(count+1);
    cudf::size_type offset = 0;
    cudf::size_type nulls = 0;
    h_offsets[0] = 0;
    for( cudf::size_type idx=0; idx < count; ++idx )
    {
        const char* str = h_test_strings[idx];
        if( !str )
        {
            strings[idx] = thrust::pair<const char*,cudf::size_type>{nullptr,0};
            nulls++;
        }
        else
        {
            cudf::size_type length = (cudf::size_type)strlen(str);
            memcpy( h_buffer.data() + offset, str, length );
            strings[idx] = thrust::pair<const char*,cudf::size_type>{d_buffer.data().get()+offset,length};
            offset += length;
        }
        h_offsets[idx+1] = offset;
    }

    rmm::device_vector<thrust::pair<const char*,cudf::size_type>> d_strings(strings);
    cudaMemcpy( d_buffer.data().get(), h_buffer.data(), memsize, cudaMemcpyHostToDevice );

    return d_strings;
}

void check_empty_string_columns(cudf::column_view lhs, cudf::column_view rhs)
{
    EXPECT_EQ(lhs.type(), rhs.type());
    EXPECT_EQ(lhs.size(), 0);
    EXPECT_EQ(lhs.null_count(), 0);
    EXPECT_EQ(lhs.nullable(), false);
    EXPECT_EQ(lhs.has_nulls(), false);
    // An empty column is not required to have children
}

TEST_F(EmptyLikeStringTest, ColumnStringTest) {
    rmm::device_vector<thrust::pair<const char*,cudf::size_type>> d_strings = create_test_string();

    auto column = cudf::make_strings_column(d_strings);

    auto got = cudf::experimental::empty_like(column->view());

    check_empty_string_columns(got->view(), column->view());
}

std::unique_ptr<cudf::experimental::table> create_table (cudf::size_type size, cudf::mask_state state){
    auto num_column_1 = make_numeric_column(cudf::data_type{cudf::INT64}, size, state); 
    auto num_column_2 = make_numeric_column(cudf::data_type{cudf::INT32}, size, state);
    auto num_column_3 = make_numeric_column(cudf::data_type{cudf::FLOAT64}, size, state); 
    auto num_column_4 = make_numeric_column(cudf::data_type{cudf::FLOAT32}, size, state);
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(num_column_1));
    columns.push_back(std::move(num_column_2));
    columns.push_back(std::move(num_column_3));
    columns.push_back(std::move(num_column_4));

    return std::make_unique<cudf::experimental::table>(std::move(columns));
}

void expect_tables_prop_equal(cudf::table_view lhs, cudf::table_view rhs)
{
    EXPECT_EQ (lhs.num_columns(), rhs.num_columns());
    for (cudf::size_type index = 0; index < lhs.num_columns(); index++)
        cudf::test::expect_column_properties_equal(lhs.column(index), rhs.column(index));    
}

struct EmptyLikeTableTest : public cudf::test::BaseFixture {};

TEST_F(EmptyLikeTableTest, TableTest) {
    cudf::mask_state state = cudf::ALL_VALID;
    cudf::size_type size = 10;
    auto input = create_table(size, state);
    auto expected = create_table(0, cudf::UNINITIALIZED);
    auto got = cudf::experimental::empty_like(input->view());

    expect_tables_prop_equal(got->view(), expected->view()); 
}

template <typename T>
struct AllocateLikeTest : public cudf::test::BaseFixture {};;

TYPED_TEST_CASE(AllocateLikeTest, numeric_types);

TYPED_TEST(AllocateLikeTest, ColumnNumericTestSameSize) {
    // For same size as input
    cudf::size_type size = 10;
    cudf::mask_state state = cudf::ALL_VALID;
    auto input = make_numeric_column(cudf::data_type{cudf::experimental::type_to_id<TypeParam>()}, size, state);
    auto expected = make_numeric_column(cudf::data_type{cudf::experimental::type_to_id<TypeParam>()}, size, cudf::UNINITIALIZED);
    auto got = cudf::experimental::allocate_like(input->view());
    cudf::test::expect_column_properties_equal(*expected, *got);
}

TYPED_TEST(AllocateLikeTest, ColumnNumericTestSpecifiedSize) {
    // For same size as input
    cudf::size_type size = 10;
    cudf::size_type specified_size = 5;
    cudf::mask_state state = cudf::ALL_VALID;
    auto input = make_numeric_column(cudf::data_type{cudf::experimental::type_to_id<TypeParam>()}, size, state);
    auto expected = make_numeric_column(cudf::data_type{cudf::experimental::type_to_id<TypeParam>()}, specified_size, cudf::UNINITIALIZED);
    auto got = cudf::experimental::allocate_like(input->view(), specified_size);
    cudf::test::expect_column_properties_equal(*expected, *got);
}
