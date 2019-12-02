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
#include <cudf/column/column_device_view.cuh>

template <typename T>
struct CopyTest : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(CopyTest, cudf::test::FixedWidthTypes);

// to keep names shorter
#define wrapper cudf::test::fixed_width_column_wrapper
using bool_wrapper = wrapper<cudf::experimental::bool8>;

TYPED_TEST(CopyTest, CopyIfElseTestShort) 
{ 
   using T = TypeParam;

   // short one. < 1 warp/bitmask length
   int num_els = 4;

   bool mask[]    = { 1, 0, 0, 0 };
   bool_wrapper mask_w(mask, mask + num_els);

   T lhs[]        = { 5, 5, 5, 5 }; 
   bool lhs_v[]   = { 1, 1, 1, 1 };
   wrapper<T> lhs_w(lhs, lhs + num_els, lhs_v);

   T rhs[]        = { 6, 6, 6, 6 };
   bool rhs_v[]   = { 1, 1, 1, 1 };
   wrapper<T> rhs_w(rhs, rhs + num_els, rhs_v);
   
   T expected[]   = { 5, 6, 6, 6 };
   // bool exp_v[]   = { 1, 1, 1, 1 };
   wrapper<T> expected_w(expected, expected + num_els);

   auto out = cudf::experimental::copy_if_else(lhs_w, rhs_w, mask_w);
   cudf::test::expect_columns_equal(out->view(), expected_w);
}

TYPED_TEST(CopyTest, CopyIfElseTestManyNulls) 
{ 
   using T = TypeParam;

   // bunch of nulls in output, non-aligned # of elements
   int num_els = 7;

   bool mask[]    = { 1, 0, 0, 0, 0, 0, 1 };
   bool_wrapper mask_w(mask, mask + num_els);

   T lhs[]        = { 5, 5, 5, 5, 5, 5, 5 }; 
   bool lhs_v[]   = { 1, 1, 1, 1, 1, 1, 1 };
   wrapper<T> lhs_w(lhs, lhs + num_els, lhs_v);

   T rhs[]        = { 6, 6, 6, 6, 6, 6, 6 };
   bool rhs_v[]   = { 1, 0, 0, 0, 0, 0, 1 };
   wrapper<T> rhs_w(rhs, rhs + num_els, rhs_v);
   
   T expected[]   = { 5, 6, 6, 6, 6, 6, 5 };
   bool exp_v[]   = { 1, 0, 0, 0, 0, 0, 1 };
   wrapper<T> expected_w(expected, expected + num_els, exp_v);

   auto out = cudf::experimental::copy_if_else(lhs_w, rhs_w, mask_w);      
   cudf::test::expect_columns_equal(out->view(), expected_w);   
}

struct copy_if_else_tiny_grid_functor {
   template <typename T, typename Filter>
   std::unique_ptr<cudf::column> operator()(cudf::column_view const& lhs,
                                          cudf::column_view const& rhs,
                                          Filter filter,
                                          rmm::mr::device_memory_resource *mr,
                                          cudaStream_t stream)
   {
      // output
      std::unique_ptr<cudf::column> out = cudf::experimental::allocate_like(lhs, lhs.size(), cudf::experimental::mask_allocation_policy::RETAIN, mr);

      // device views
      auto lhs_dv = cudf::column_device_view::create(lhs);
      auto rhs_dv = cudf::column_device_view::create(rhs);
      auto out_dv = cudf::mutable_column_device_view::create(*out);
             
      // call the kernel with an artificially small grid
      cudf::experimental::detail::copy_if_else_kernel<32, T, Filter, false><<<1, 32, 0, stream>>>(
         *lhs_dv, *rhs_dv, filter, *out_dv, nullptr);

      return out;
   }
};

std::unique_ptr<cudf::column> tiny_grid_launch(cudf::column_view const& lhs, cudf::column_view const& rhs, cudf::column_view const& boolean_mask)
{
   auto bool_mask_device_p = cudf::column_device_view::create(boolean_mask);
   cudf::column_device_view bool_mask_device = *bool_mask_device_p;
   auto filter = [bool_mask_device] __device__ (cudf::size_type i) { return bool_mask_device.element<cudf::experimental::bool8>(i); };
   return cudf::experimental::type_dispatcher(lhs.type(),
                                             copy_if_else_tiny_grid_functor{},
                                             lhs,
                                             rhs,
                                             filter,
                                             rmm::mr::get_default_resource(),
                                             (cudaStream_t)0);
}

TYPED_TEST(CopyTest, CopyIfElseTestTinyGrid) 
{  
   using T = TypeParam;

   // make sure we span at least 2 warps      
   int num_els = 64;

   bool mask[]    = { 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                     0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };   
   bool_wrapper mask_w(mask, mask + num_els);

   T lhs[]        = { 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
                     5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 };   
   wrapper<T> lhs_w(lhs, lhs + num_els);

   T rhs[]        = { 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6 };
   wrapper<T> rhs_w(rhs, rhs + num_els);
   
   T expected[]   = { 5, 6, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
                     6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 };   
   wrapper<T> expected_w(expected, expected + num_els);

   auto out = tiny_grid_launch(lhs_w, rhs_w, mask_w);
     
   cudf::test::expect_columns_equal(out->view(), expected_w);   
}

TYPED_TEST(CopyTest, CopyIfElseTestLong) 
{  
   using T = TypeParam;

   // make sure we span at least 2 warps      
   int num_els = 64;

   bool mask[]    = { 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                     0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };   
   bool_wrapper mask_w(mask, mask + num_els);

   T lhs[]        = { 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
                     5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 };
   bool lhs_v[]   = { 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };   
   wrapper<T> lhs_w(lhs, lhs + num_els, lhs_v);

   T rhs[]        = { 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6 };
   bool rhs_v[]   = { 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };                      
   wrapper<T> rhs_w(rhs, rhs + num_els, rhs_v);
   
   T expected[]   = { 5, 6, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
                     6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 };
   bool exp_v[]   = { 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };   
   wrapper<T> expected_w(expected, expected + num_els, exp_v);

   auto out = cudf::experimental::copy_if_else(lhs_w, rhs_w, mask_w);      
   cudf::test::expect_columns_equal(out->view(), expected_w);   
}

TYPED_TEST(CopyTest, CopyIfElseTestEmptyInputs) 
{ 
   using T = TypeParam;
         
   int num_els = 0;
   
   bool mask[]    = {};
   bool_wrapper mask_w(mask, mask + num_els);

   T lhs[]        = {};
   wrapper<T> lhs_w(lhs, lhs + num_els);

   T rhs[]        = {};
   wrapper<T> rhs_w(rhs, rhs + num_els);
   
   T expected[]   = {};
   wrapper<T> expected_w(expected, expected + num_els);

   auto out = cudf::experimental::copy_if_else(lhs_w, rhs_w, mask_w);      
   cudf::test::expect_columns_equal(out->view(), expected_w);
}

TYPED_TEST(CopyTest, CopyIfElseMixedInputValidity)
{ 
   using T = TypeParam;   
         
   int num_els = 4;

   bool mask[]    = { 1, 0, 1, 1 };
   bool_wrapper mask_w(mask, mask + num_els);

   T lhs[]        = { 5, 5, 5, 5 };   
   wrapper<T> lhs_w(lhs, lhs + num_els);

   T rhs[]        = { 6, 6, 6, 6 };                      
   wrapper<T> rhs_w(rhs, rhs + num_els, mask);      

   T expected[]   = { 5, 6, 5, 5 };                      
   wrapper<T> expected_w(expected, expected + num_els, mask);      

   auto out = cudf::experimental::copy_if_else(lhs_w, rhs_w, mask_w);     
   cudf::test::expect_columns_equal(out->view(), expected_w);             
}

TYPED_TEST(CopyTest, CopyIfElseBadInputLength)
{ 
   using T = TypeParam;   
         
   int num_els = 4;

   // mask length mismatch
   {
      bool mask[]    = { 1, 1, 1, 1 };
      bool_wrapper mask_w(mask, mask + 3);

      T lhs[]        = { 5, 5, 5, 5 };
      wrapper<T> lhs_w(lhs, lhs + num_els, mask);

      T rhs[]        = { 6, 6, 6, 6 };
      wrapper<T> rhs_w(rhs, rhs + num_els, mask);

      EXPECT_THROW(  cudf::experimental::copy_if_else(lhs_w, rhs_w, mask_w),
                     cudf::logic_error);
   }

   // column length mismatch
   {
      bool mask[]    = { 1, 1, 1, 1 };
      bool_wrapper mask_w(mask, mask + num_els);

      T lhs[]        = { 5, 5, 5 };
      wrapper<T> lhs_w(lhs, lhs + 3, mask);

      T rhs[]        = { 6, 6, 6, 6 };
      wrapper<T> rhs_w(rhs, rhs + num_els, mask);

      EXPECT_THROW(  cudf::experimental::copy_if_else(lhs_w, rhs_w, mask_w), 
                     cudf::logic_error);
   }
}

struct CopyTestUntyped : public cudf::test::BaseFixture {};

TEST_F(CopyTestUntyped, CopyIfElseTypeMismatch)
{               
   int num_els = 4;

   bool mask[]    = { 1, 1, 1, 1 };
   bool_wrapper mask_w(mask, mask + num_els);
   cudf::column _mask(mask_w);

   float lhs[]    = { 5, 5, 5, 5 };   
   wrapper<float> lhs_w(lhs, lhs + num_els, mask);
   cudf::column _lhs(lhs_w);

   int rhs[]      = { 6, 6, 6, 6 };                      
   wrapper<int> rhs_w(rhs, rhs + num_els, mask);
   cudf::column _rhs(rhs_w);      

   EXPECT_THROW(  cudf::experimental::copy_if_else(_lhs, _rhs, _mask),
                  cudf::logic_error);
}

struct StringsCopyIfElseTest : public cudf::test::BaseFixture {};

struct filter_test_fn
{
    __host__ __device__ bool operator()(cudf::size_type idx) const
    {
        return static_cast<bool>(idx % 2);
    }
};

TEST_F(StringsCopyIfElseTest, CopyIfElse)
{
    std::vector<const char*> h_strings1{ "eee", "bb", "", "aa", "bbb", "ééé" };
    cudf::test::strings_column_wrapper strings1( h_strings1.begin(), h_strings1.end() );
    auto lhs = cudf::strings_column_view(strings1);
    std::vector<const char*> h_strings2{ "zz",  "", "yyy", "w", "ééé", "ooo" };
    cudf::test::strings_column_wrapper strings2( h_strings2.begin(), h_strings2.end() );
    auto rhs = cudf::strings_column_view(strings2);

    auto results = cudf::strings::detail::copy_if_else(lhs,rhs,filter_test_fn{});

    std::vector<const char*> h_expected;
    for( cudf::size_type idx=0; idx < static_cast<cudf::size_type>(h_strings1.size()); ++idx )
    {
        if( filter_test_fn()(idx) )
            h_expected.push_back( h_strings1[idx] );
        else
            h_expected.push_back( h_strings2[idx] );
    }
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end());
    cudf::test::expect_columns_equal(*results,expected);
}
