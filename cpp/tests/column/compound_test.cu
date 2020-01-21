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
#include <cudf/column/column_device_view.cuh>
#include <cudf/null_mask.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/cudf_gtest.hpp>

#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>
#include <vector>

#include <gmock/gmock.h>

struct CompoundColumnTest : public cudf::test::BaseFixture {};


struct checker_for_level1
{
    cudf::column_device_view d_column;
    bool __device__ operator()(int32_t idx)
    {
        int32_t val1 = d_column.child(0).element<int32_t>(idx);
        int32_t val2 = d_column.child(1).element<int32_t>(idx);
        int32_t val3 = d_column.child(2).element<int32_t>(idx);
        return ((val1+100)==val2) && ((val2+100)==val3);
    }
};

struct checker_for_level2
{
    cudf::column_device_view d_column;
    bool __device__ operator()(int32_t idx)
    {
        bool bcheck = true;
        for( int i=0; i<2 && bcheck; ++i )
        {
            auto child = d_column.child(i);
            int32_t val1 = child.child(0).element<int32_t>(idx);
            int32_t val2 = child.child(1).element<int32_t>(idx);
            int32_t val3 = child.child(2).element<int32_t>(idx);
            bcheck = ((val1+100)==val2) && ((val2+100)==val3);
        }
        return bcheck;
    }
};

TEST_F(CompoundColumnTest, ChildrenLevel1) 
{
  thrust::device_vector<int32_t> data(1000);
  thrust::sequence( thrust::device, data.begin(), data.end(), 1);

  auto null_mask = cudf::create_null_mask(100,cudf::mask_state::UNALLOCATED);
  rmm::device_buffer data1(data.data().get()+100,100*sizeof(int32_t));
  rmm::device_buffer data2(data.data().get()+200,100*sizeof(int32_t));
  rmm::device_buffer data3(data.data().get()+300,100*sizeof(int32_t));
  auto child1 = std::make_unique<cudf::column>(cudf::data_type{cudf::INT32}, 100, data1, null_mask, 0); 
  auto child2 = std::make_unique<cudf::column>(cudf::data_type{cudf::INT32}, 200, data2, null_mask, 0);
  auto child3 = std::make_unique<cudf::column>(cudf::data_type{cudf::INT32}, 300, data3, null_mask, 0);

  std::vector<std::unique_ptr<cudf::column>> children;
  children.emplace_back(std::move(child1));
  children.emplace_back(std::move(child2));
  children.emplace_back(std::move(child3));

  auto parent = std::make_unique<cudf::column>(
        cudf::data_type{cudf::STRING}, 100, rmm::device_buffer{0},
        rmm::device_buffer{0}, 0, std::move(children));

  auto column = cudf::column_device_view::create(parent->view());
  auto d_column = *column;

  EXPECT_FALSE(d_column.nullable());
  EXPECT_FALSE(d_column.has_nulls());
  EXPECT_EQ(0, d_column.null_count());

  bool check = thrust::any_of( thrust::device, thrust::make_counting_iterator<int32_t>(0),
        thrust::make_counting_iterator<int32_t>(100), checker_for_level1{d_column} );
  EXPECT_TRUE(check);
}

TEST_F(CompoundColumnTest, ChildrenLevel2) 
{
  thrust::device_vector<int32_t> data(1000);
  thrust::sequence( thrust::device, data.begin(), data.end(), 1);

  auto null_mask = cudf::create_null_mask(100,cudf::mask_state::UNALLOCATED);
  rmm::device_buffer data11(data.data().get()+100,100*sizeof(int32_t));
  rmm::device_buffer data12(data.data().get()+200,100*sizeof(int32_t));
  rmm::device_buffer data13(data.data().get()+300,100*sizeof(int32_t));
  rmm::device_buffer data21(data.data().get()+400,100*sizeof(int32_t));
  rmm::device_buffer data22(data.data().get()+500,100*sizeof(int32_t));
  rmm::device_buffer data23(data.data().get()+600,100*sizeof(int32_t));
  auto gchild11 = std::make_unique<cudf::column>(cudf::data_type{cudf::INT32}, 100, data11, null_mask, 0); 
  auto gchild12 = std::make_unique<cudf::column>(cudf::data_type{cudf::INT32}, 200, data12, null_mask, 0);
  auto gchild13 = std::make_unique<cudf::column>(cudf::data_type{cudf::INT32}, 300, data13, null_mask, 0);
  auto gchild21 = std::make_unique<cudf::column>(cudf::data_type{cudf::INT32}, 400, data21, null_mask, 0); 
  auto gchild22 = std::make_unique<cudf::column>(cudf::data_type{cudf::INT32}, 500, data22, null_mask, 0);
  auto gchild23 = std::make_unique<cudf::column>(cudf::data_type{cudf::INT32}, 600, data23, null_mask, 0);

  std::vector<std::unique_ptr<cudf::column>> gchildren1;
  gchildren1.emplace_back(std::move(gchild11));
  gchildren1.emplace_back(std::move(gchild12));
  gchildren1.emplace_back(std::move(gchild13));
  std::vector<std::unique_ptr<cudf::column>> gchildren2;
  gchildren2.emplace_back(std::move(gchild21));
  gchildren2.emplace_back(std::move(gchild22));
  gchildren2.emplace_back(std::move(gchild23));

  auto children1 = std::make_unique<cudf::column>(
        cudf::data_type{cudf::STRING}, 100, rmm::device_buffer{0},
        rmm::device_buffer{0}, 0, std::move(gchildren1));
  auto children2 = std::make_unique<cudf::column>(
        cudf::data_type{cudf::STRING}, 100, rmm::device_buffer{0},
        rmm::device_buffer{0}, 0, std::move(gchildren2));

  std::vector<std::unique_ptr<cudf::column>> children;
  children.emplace_back(std::move(children1));
  children.emplace_back(std::move(children2));
  auto parent = std::make_unique<cudf::column>(
        cudf::data_type{cudf::STRING}, 100, rmm::device_buffer{0},
        rmm::device_buffer{0}, 0, std::move(children));

  auto column = cudf::column_device_view::create(parent->view());
  auto d_column = *column;

  EXPECT_FALSE(d_column.nullable());
  EXPECT_FALSE(d_column.has_nulls());
  EXPECT_EQ(0, d_column.null_count());

  bool check = thrust::any_of( thrust::device, thrust::make_counting_iterator<int32_t>(0),
        thrust::make_counting_iterator<int32_t>(100), checker_for_level2{d_column} );
  EXPECT_TRUE(check);
}


