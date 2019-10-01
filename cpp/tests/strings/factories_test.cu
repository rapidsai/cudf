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

#include <cudf/strings/strings_column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <tests/utilities/cudf_test_fixtures.h>

#include <vector>
#include <cstring>


struct FactoriesTest : public GdfTest {};

TEST_F(FactoriesTest, CreateColumnFromArray)
{
    std::vector<const char*> h_test_strings{ "the quick brown fox jumps over the lazy dog",
                                             "the fat cat lays next to the other accénted cat",
                                             "a slow moving turtlé cannot catch the bird",
                                             "which can be composéd together to form a more complete",
                                             "thé result does not include the value in the sum in",
                                             "", nullptr, "absent stop words" };

    cudf::size_type memsize = 0;
    for( auto itr=h_test_strings.begin(); itr!=h_test_strings.end(); ++itr )
        memsize += *itr ? (cudf::size_type)strlen(*itr) : 0;
    cudf::size_type count = (cudf::size_type)h_test_strings.size();
    thrust::host_vector<char> h_buffer(memsize);
    thrust::device_vector<char> d_buffer(memsize);
    thrust::host_vector<thrust::pair<const char*,size_t> > strings(count);
    thrust::host_vector<cudf::size_type> h_offsets(count+1);
    cudf::size_type offset = 0;
    cudf::size_type nulls = 0;
    h_offsets[0] = 0;
    for( cudf::size_type idx=0; idx < count; ++idx )
    {
        const char* str = h_test_strings[idx];
        if( !str )
        {
            strings[idx] = thrust::pair<const char*,size_t>{nullptr,0};
            nulls++;
        }
        else
        {
            cudf::size_type length = (cudf::size_type)strlen(str);
            memcpy( h_buffer.data() + offset, str, length );
            strings[idx] = thrust::pair<const char*,size_t>{d_buffer.data().get()+offset,(size_t)length};
            offset += length;
        }
        h_offsets[idx+1] = offset;
    }
    rmm::device_vector<thrust::pair<const char*,size_t>> d_strings(strings);
    cudaMemcpy( d_buffer.data().get(), h_buffer.data(), memsize, cudaMemcpyHostToDevice );
    auto column = cudf::make_strings_column( d_strings );
    EXPECT_EQ(column->type(), cudf::data_type{cudf::STRING});
    EXPECT_EQ(column->null_count(), nulls);
    if( nulls )
    {
        EXPECT_TRUE(column->nullable());
        EXPECT_TRUE(column->has_nulls());
    }
    EXPECT_EQ(2, column->num_children());

    cudf::strings_column_view strings_view(column->view());
    EXPECT_EQ( strings_view.size(), count);
    EXPECT_EQ( strings_view.offsets().size(), count+1 );
    EXPECT_EQ( strings_view.chars().size(), memsize );

    // check string data
    auto strings_data = cudf::strings::create_offsets(strings_view);
    thrust::host_vector<char> h_chars_data(strings_data.first);
    thrust::host_vector<cudf::size_type> h_offsets_data(strings_data.second);
    EXPECT_EQ( memcmp(h_buffer.data(), h_chars_data.data(), h_buffer.size()), 0 );
    EXPECT_EQ( memcmp(h_offsets.data(), h_offsets_data.data(), h_offsets.size()*sizeof(cudf::size_type)), 0);
}

TEST_F(FactoriesTest, CreateColumnFromOffsets)
{
    std::vector<const char*> h_test_strings{ "the quick brown fox jumps over the lazy dog",
                                         "the fat cat lays next to the other accénted cat",
                                         "a slow moving turtlé cannot catch the bird",
                                         "which can be composéd together to form a more complete",
                                         "thé result does not include the value in the sum in",
                                         "absent stop words" };

    cudf::size_type memsize = 0;
    for( auto itr=h_test_strings.begin(); itr!=h_test_strings.end(); ++itr )
        memsize += *itr ? (cudf::size_type)strlen(*itr) : 0;
    cudf::size_type count = (cudf::size_type)h_test_strings.size();
    thrust::host_vector<char> h_buffer(memsize);
    thrust::host_vector<cudf::size_type> h_offsets(count+1);
    cudf::size_type offset = 0;
    h_offsets[0] = 0;
    for( cudf::size_type idx=0; idx < count; ++idx )
    {
        const char* str = h_test_strings[idx];
        if( str )
        {
            cudf::size_type length = (cudf::size_type)strlen(str);
            memcpy( h_buffer.data() + offset, str, length );
            offset += length;
        }
        h_offsets[idx+1] = offset;
    }
    rmm::device_vector<char> d_buffer(h_buffer);
    rmm::device_vector<cudf::size_type> d_offsets(h_offsets);
    rmm::device_vector<cudf::bitmask_type> d_nulls;
    auto column = cudf::make_strings_column( d_buffer, d_offsets, d_nulls, 0 );
    EXPECT_EQ(column->type(), cudf::data_type{cudf::STRING});
    EXPECT_EQ(column->null_count(), 0);
    EXPECT_EQ(2, column->num_children());

    cudf::strings_column_view strings_view(column->view());
    EXPECT_EQ( strings_view.size(), count);
    EXPECT_EQ( strings_view.offsets().size(), count+1 );
    EXPECT_EQ( strings_view.chars().size(), memsize );

    // check string data
    auto strings_data = cudf::strings::create_offsets(strings_view);
    thrust::host_vector<char> h_chars_data(strings_data.first);
    thrust::host_vector<cudf::size_type> h_offsets_data(strings_data.second);
    EXPECT_EQ( memcmp(h_buffer.data(), h_chars_data.data(), h_buffer.size()), 0 );
    EXPECT_EQ( memcmp(h_offsets.data(), h_offsets_data.data(), h_offsets.size()*sizeof(cudf::size_type)), 0);
}
