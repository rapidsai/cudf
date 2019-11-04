/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#include <tests/utilities/legacy/cudf_test_fixtures.h>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <tests/utilities/legacy/column_wrapper.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/replace.hpp>

#include <cudf/cudf.h>

#include <gtest/gtest.h>

// This is the main test feature
template <class T>
struct ReplaceTest : public GdfTest
{

  ReplaceTest()
  {    
  }

  ~ReplaceTest()
  {
  }
};

// ugly.  only here to get around strict-aliasing compiler warnings
uint32_t hard_cast_float(float val)
{
   void *v_val = &val;
   return *((uint32_t*)v_val);
}
uint64_t hard_cast_double(double val)
{
   void *v_val = &val;
   return *((uint64_t*)v_val);
}

// Test for normalize_nans_and_nulls
TEST(ReplaceTest, NormalizeNansAndZeros)
{      
   int num_els = 8;   
   
   // floats
   {
      float els[8] = { 32.5f, -0.0f, 111.0f, -NAN, NAN, 1.0f, 0.0f, 54.3f };   

      // copy the data to mutable device column
      auto test_data = cudf::make_numeric_column(cudf::data_type(cudf::FLOAT32), num_els, cudf::ALL_VALID, 0);
      auto test_data_mview = test_data->mutable_view();
      cudaMemcpy(test_data_mview.head(), els, sizeof(float) * num_els, cudaMemcpyHostToDevice);

      // mutable overload
      {                    
         auto mutable_view = test_data->mutable_view(); 
         cudf::normalize_nans_and_zeros(mutable_view);

         // get the data back
         float chk[8] = { 99.0f };
         cudaMemcpy(chk, mutable_view.head(), sizeof(float) * num_els, cudaMemcpyDeviceToHost);      
         
         // can't compare nans and -nans directly since they will always be equal, so we'll compare against
         // bit patterns.       
         uint32_t nan = hard_cast_float(NAN);            
         EXPECT_TRUE(hard_cast_float(chk[3]) == nan);
         EXPECT_TRUE(hard_cast_float(chk[4]) == nan);
         EXPECT_TRUE(chk[1] == 0.0f);
      }

      // returned column overload
      {
         auto out = cudf::normalize_nans_and_zeros(test_data->view());

         // get the data back
         float chk[8] = { 99.0f };
         cudaMemcpy(chk, out->view().head(), sizeof(float) * num_els, cudaMemcpyDeviceToHost);      
         
         // can't compare nans and -nans directly since they will always be equal, so we'll compare against
         // bit patterns.       
         uint32_t nan = hard_cast_float(NAN);            
         EXPECT_TRUE(hard_cast_float(chk[3]) == nan);
         EXPECT_TRUE(hard_cast_float(chk[4]) == nan);
         EXPECT_TRUE(chk[1] == 0.0f);
      }
   }
   
    // doubles
   {
      double els[8] = { 32.5, -0.0, 111.0, -NAN, NAN, 1.0, 0.0, 54.3 }; 

      // copy the data to mutable device column
      auto test_data = cudf::make_numeric_column(cudf::data_type(cudf::FLOAT64), num_els, cudf::ALL_VALID, 0);
      auto test_data_mview = test_data->mutable_view();
      cudaMemcpy(test_data_mview.head(), els, sizeof(double) * num_els, cudaMemcpyHostToDevice);

      // mutable overload
      {                    
         auto mutable_view = test_data->mutable_view(); 
         cudf::normalize_nans_and_zeros(mutable_view);

         // get the data back
         double chk[8] = { 99.0 };
         cudaMemcpy(chk, mutable_view.head(), sizeof(double) * num_els, cudaMemcpyDeviceToHost);      
         
         // can't compare nans and -nans directly since they will always be equal, so we'll compare against
         // bit patterns.       
         uint64_t nan = hard_cast_double(NAN);            
         EXPECT_TRUE(hard_cast_double(chk[3]) == nan);
         EXPECT_TRUE(hard_cast_double(chk[4]) == nan);
         EXPECT_TRUE(chk[1] == 0.0);
      }

      // returned column overload
      {
         auto out = cudf::normalize_nans_and_zeros(test_data->view());

         // get the data back
         double chk[8] = { 99.0 };
         cudaMemcpy(chk, out->view().head(), sizeof(double) * num_els, cudaMemcpyDeviceToHost);      
         
         // can't compare nans and -nans directly since they will always be equal, so we'll compare against
         // bit patterns.       
         uint64_t nan = hard_cast_double(NAN);            
         EXPECT_TRUE(hard_cast_double(chk[3]) == nan);
         EXPECT_TRUE(hard_cast_double(chk[4]) == nan);
         EXPECT_TRUE(chk[1] == 0.0);
      }
   }
}