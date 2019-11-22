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

#include <thrust/device_ptr.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>

#include <cudf/cudf.h>
#include <rmm/rmm.h>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace {  // anonymous

using namespace cudf;       

template <typename T>
struct normalize_nans_and_zeros_lambda {
   column_device_view in;
   T __device__ operator()(size_type i)
   {
      auto e = in.element<T>(i);
      if (isnan(e)) {
         return std::numeric_limits<T>::quiet_NaN();
      }
      if (T{0.0} == e) {
         return T{0.0};
      }
      return e;
   }
};
   
/* --------------------------------------------------------------------------*/
/**
* @brief Functor called by the `type_dispatcher` in order to invoke and instantiate
*        `normalize_nans_and_zeros` with the appropriate data types.
*/
/* ----------------------------------------------------------------------------*/
struct normalize_nans_and_zeros_kernel_forwarder {
   // floats and doubles. what we really care about.
   template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
   void operator()(  column_device_view in,
                     mutable_column_device_view out,
                     cudaStream_t stream)
   {      
      thrust::transform(rmm::exec_policy(stream)->on(stream),
                        thrust::make_counting_iterator(0),
                        thrust::make_counting_iterator(in.size()),
                        out.head<T>(), normalize_nans_and_zeros_lambda<T>{in});      
   }

   // if we get in here for anything but a float or double, that's a problem.
   template <typename T, std::enable_if_t<not std::is_floating_point<T>::value>* = nullptr>
   void operator()(  column_device_view in,
                     mutable_column_device_view out,
                     cudaStream_t stream)
   {
      CUDF_FAIL("Unexpected non floating-point type.");      
   }   
};

} // end anonymous namespace

namespace cudf {
namespace detail {                                               

void normalize_nans_and_zeros(mutable_column_view in_out,
                              cudaStream_t stream = 0)
{      
   if(in_out.size() == 0){
      return;
   }
   CUDF_EXPECTS(in_out.type() == data_type(FLOAT32) || in_out.type() == data_type(FLOAT64), "Expects float or double input");

   // wrapping the in_out data in a column_view so we can call the same lower level code.
   // that we use for the non in-place version.
   column_view input = in_out;

   // to device. unique_ptr which gets automatically cleaned up when we leave
   auto device_in = column_device_view::create(input);

   // from device. unique_ptr which gets automatically cleaned up when we leave.   
   auto device_out = mutable_column_device_view::create(in_out);

    // invoke the actual kernel.  
   cudf::experimental::type_dispatcher(input.type(), 
                                       normalize_nans_and_zeros_kernel_forwarder{},
                                       *device_in,
                                       *device_out,
                                       stream);
}

}  // namespace detail

/*
 * @brief Makes all NaNs and zeroes positive.
 *
 * Converts floating point values from @p input using the following rules:
 *        Convert  -NaN  -> NaN
 *        Convert  -0.0  -> 0.0
 *
 * @throws cudf::logic_error if column does not have floating point data type.
 * @param[in] column_view representing input data
 * @param[in] device_memory_resource allocator for allocating output data 
 *
 * @returns new column with the modified data
 */
std::unique_ptr<column> normalize_nans_and_zeros( column_view const& input,                                                                                                    
                                                  rmm::mr::device_memory_resource *mr)
{
   // output. copies the input
   std::unique_ptr<column> out = std::make_unique<column>(input, (cudaStream_t)0, mr);   
   // from device. unique_ptr which gets automatically cleaned up when we leave.
   auto out_view = out->mutable_view();

   detail::normalize_nans_and_zeros(out_view, 0);

   return out;
}

/*
 * @brief Makes all Nans and zeroes positive.
 * 
 * Converts floating point values from @p in_out using the following rules:
 *        Convert  -NaN  -> NaN
 *        Convert  -0.0  -> 0.0
 *
 * @throws cudf::logic_error if column does not have floating point data type.
 * @param[in, out] mutable_column_view representing input data. data is processed in-place
 */
void normalize_nans_and_zeros(mutable_column_view& in_out)
{
   detail::normalize_nans_and_zeros(in_out, 0);
}

}  // namespace cudf

