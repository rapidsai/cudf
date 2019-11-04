/*
 * Copyright 2018 BlazingDB, Inc.

 *     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
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
#include <cub/cub.cuh>

#include <cudf/legacy/copying.hpp>
#include <cudf/legacy/replace.hpp>
#include <cudf/cudf.h>
#include <rmm/rmm.h>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <utilities/cudf_utils.h>
#include <utilities/cuda_utils.hpp>
#include <utilities/column_utils.hpp>
#include <bitmask/legacy/legacy_bitmask.hpp>
#include <bitmask/legacy/bit_mask.cuh>

namespace {  // anonymous

using namespace cudf;

static constexpr int BLOCK_SIZE = 256;

/* --------------------------------------------------------------------------*/
/**
 * @brief Kernel that converts inputs from `in` to `out`  using the following
 *        rule:   Convert  -NaN  -> NaN
 *                Convert  -0.0  -> 0.0
 *
 * @param[in] column_device_view representing input data
 * @param[in] mutable_column_device_view representing output data. can be
 *            the same actual underlying buffer that in points to. 
 *
 * @returns
 */
/* ----------------------------------------------------------------------------*/
template <typename T>
__global__
void normalize_nans_and_zeros(column_device_view in, 
                              mutable_column_device_view out)
{
   int tid = threadIdx.x;
   int blkid = blockIdx.x;
   int blksz = blockDim.x;
   int gridsz = gridDim.x;

   int start = tid + blkid * blksz;
   int step = blksz * gridsz;

   // grid-stride
   for (int i=start; i<in.size(); i+=step) {
      if(!in.is_valid(i)){
         continue;
      }

      T el = in.element<T>(i);
      if(std::isnan(el)){
         out.element<T>(i) = std::numeric_limits<T>::quiet_NaN();
      } else if(el == (T)-0.0){
         out.element<T>(i) = (T)0.0;
      } else {
         out.element<T>(i) = el;
      }
   }
}                        

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
      util::cuda::grid_config_1d grid{in.size(), BLOCK_SIZE};
      normalize_nans_and_zeros<T><<<grid.num_blocks, BLOCK_SIZE, 0, stream>>>(in, out);
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

std::unique_ptr<column> normalize_nans_and_zeros( column_view input,                                            
                                                  cudaStream_t stream,
                                                  rmm::mr::device_memory_resource *mr)
{      
   CUDF_EXPECTS(input.head() != nullptr, "Null input data");
   if(input.size() == 0 || input.head() == nullptr){
      return make_numeric_column(input.type(), input.size(), ALL_VALID, stream, mr);
   }   
   CUDF_EXPECTS(input.type() == data_type(FLOAT32) || input.type() == data_type(FLOAT64), "Expects float or double input");

    // to device. unique_ptr which gets automatically cleaned up when we leave
   auto device_in = column_device_view::create(input);
   
   // ultimately, the output.
   auto out = make_numeric_column(input.type(), input.size(), ALL_VALID, stream, mr);
   // from device. unique_ptr which gets automatically cleaned up when we leave.
   auto device_out = mutable_column_device_view::create(*out);

   // invoke the actual kernel.  
  experimental::type_dispatcher(input.type(), 
                                normalize_nans_and_zeros_kernel_forwarder{},
                                *device_in,
                                *device_out,
                                stream);

   return out;                 
}                                                 

void normalize_nans_and_zeros(mutable_column_view in_out,
                              cudaStream_t stream)
{   
   CUDF_EXPECTS(in_out.head() != nullptr, "Null input data");
   if(in_out.size() == 0 || in_out.head() == nullptr){
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

/**
 * @brief Function that converts inputs from `input` using the following rule
 *        rule:   Convert  -NaN  -> NaN
 *                Convert  -0.0  -> 0.0
 *
 * @param[in] column_device_view representing input data
 * @param[in] device_memory_resource allocator for allocating output data 
 *
 * @returns new column
 */
std::unique_ptr<column> normalize_nans_and_zeros( column_view input,                                                                                                    
                                                  rmm::mr::device_memory_resource *mr)
{
   return detail::normalize_nans_and_zeros(input, 0, mr);
}

/**
 * @brief Function that processes values in-place from `in_out` using the following rule
 *        rule:   Convert  -NaN  -> NaN
 *                Convert  -0.0  -> 0.0
 *
 * @param[in, out] mutable_column_view representing input data. data is processed in-place
 *
 * @returns new column
 */
void normalize_nans_and_zeros(mutable_column_view in_out)
{
   return detail::normalize_nans_and_zeros(in_out, 0);
}

}  // namespace cudf

