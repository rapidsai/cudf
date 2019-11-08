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

#include <tests/utilities/cudf_gtest.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy_if_else.cuh>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/legacy/cudf_test_utils.cuh>

#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>

#include <cudf/column/column.hpp>

template <typename T>
struct CopyTest : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(CopyTest, cudf::test::NumericTypes);

// to keep names shorter
#define wrapper cudf::test::fixed_width_column_wrapper
using bool_wrapper = wrapper<cudf::experimental::bool8>;

TYPED_TEST(CopyTest, CopyIfElseTestBoolMask) 
{ 
   using T = TypeParam;   
      
   bool_wrapper  mask_w    { true, true, false, true, true }; 
   wrapper<T>   lhs_w      { 5, 5, 5, 5, 5 };
   wrapper<T>   rhs_w      { 6, 6, 6, 6, 6 };
   wrapper<T>   expected_w { 5, 5, 6, 5, 5 };

   // construct input views
   cudf::column mask(mask_w);
   cudf::column_view mask_v(mask);
   //
   cudf::column lhs(lhs_w);
   cudf::column_view lhs_v = lhs.view();
   //
   cudf::column rhs(rhs_w);
   cudf::column_view rhs_v = rhs.view();
   //
   cudf::column expected(expected_w);
   cudf::column_view expected_v = expected.view();

   // get the result
   auto out = cudf::experimental::copy_if_else(mask_v, lhs_v, rhs_v);
   cudf::column_view out_v = out->view();      

   // compare
   cudf::test::expect_columns_equal(out_v, expected_v);   
}

struct copy_if_test_functor {
   bool __device__ operator()(cudf::size_type i) const
   {
      return i == 2 ? false : true;
   }
};

TYPED_TEST(CopyTest, CopyIfElseTestFilter) 
{ 
   using T = TypeParam;   
         
   wrapper<T>   lhs_w      { 5, 5, 5, 5, 5 };
   wrapper<T>   rhs_w      { 6, 6, 6, 6, 6 };
   wrapper<T>   expected_w { 5, 5, 6, 5, 5 };

   // construct input views   
   //
   cudf::column lhs(lhs_w);
   cudf::column_view lhs_v = lhs.view();
   //
   cudf::column rhs(rhs_w);
   cudf::column_view rhs_v = rhs.view();
   //
   cudf::column expected(expected_w);
   cudf::column_view expected_v = expected.view();
  
   // get the result
   auto out = cudf::experimental::detail::copy_if_else(copy_if_test_functor{}, lhs_v, rhs_v);
   cudf::column_view out_v = out->view();      

   // compare
   cudf::test::expect_columns_equal(out_v, expected_v);   
}
