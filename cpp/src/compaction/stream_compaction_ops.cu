/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Felipe Aramburu <felipe@blazingdb.com>
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

#include <cuda_runtime.h>
#include <vector>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/iterator/counting_iterator.h>

#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>

#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.h"
#include "rmm/thrust_rmm_allocator.h"
#include "utilities/division.hpp"
#include "utilities/type_dispatcher.hpp"

//std lib
#include <map>

// Anonymous namespace
namespace {

struct is_bit_set
{
    __host__ __device__
    bool operator()(const thrust::tuple<gdf_size_type, thrust::device_ptr<gdf_valid_type>> value)
    {
        gdf_size_type position = thrust::get<0>(value);

        return gdf_is_valid(thrust::get<1>(value).get(), position);
    }
}; 

using mask_tuple = thrust::tuple<thrust::counting_iterator<gdf_size_type>, thrust::constant_iterator<gdf_valid_type*>>;
using zipped_mask = thrust::zip_iterator<mask_tuple>;
using bit_set_iterator = thrust::transform_iterator<is_bit_set, zipped_mask>;

template<typename stencil_type>
struct is_stencil_true
{
    __host__ __device__
    bool operator()(const thrust::tuple<stencil_type, bit_set_iterator::value_type> value)
    {
        return thrust::get<1>(value) && (thrust::get<0>(value) != 0);
    }
};

struct bit_mask_pack_op : public thrust::unary_function<int64_t,gdf_valid_type>
{
    static_assert(sizeof(gdf_valid_type) == 1, "Unexpected size of gdf_valid_type");
    __host__ __device__
        gdf_valid_type operator()(const int64_t expanded)
        {
            gdf_valid_type result = 0;
            for(unsigned i = 0; i < GDF_VALID_BITSIZE; i++){
                unsigned char byte = (expanded >> (i * CHAR_BIT));
                result |= (byte & 1) << i;
            }
            return result;
        }
};

//zip the stencil and the valid iterator together
using zipped_stencil_tuple = thrust::tuple<thrust::detail::normal_iterator<thrust::device_ptr<int8_t>>, bit_set_iterator>;
using zipped_stencil_iterator = thrust::zip_iterator<zipped_stencil_tuple>;

struct apply_stencil_functor{
    template <typename col_type>
    __host__
    void operator()(gdf_column* col, gdf_column* output, zipped_stencil_iterator zipped_stencil_iter)
    {
        auto input_start = thrust::detail::make_normal_iterator(thrust::device_pointer_cast((col_type *) col->data));
        auto output_start = thrust::detail::make_normal_iterator(thrust::device_pointer_cast((col_type *) output->data));
        auto output_end = thrust::copy_if(input_start, input_start + col->size, zipped_stencil_iter, output_start, 
                    is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int8_t> >::value_type >());
        output->size = output_end - output_start;
    }
};

} // Anonymous namespace

gdf_error gpu_apply_stencil(gdf_column * col, gdf_column * stencil, gdf_column * output) {
    GDF_REQUIRE(output->size == col->size, GDF_COLUMN_SIZE_MISMATCH);
    GDF_REQUIRE(col->dtype == output->dtype, GDF_DTYPE_MISMATCH);

    cudaStream_t stream;
    CUDA_TRY( cudaStreamCreate(&stream) );

    auto zipped_mask_stencil_iter = thrust::make_zip_iterator(
        thrust::make_tuple(
            thrust::make_counting_iterator<gdf_size_type>(0),
            thrust::make_constant_iterator(stencil->valid)
        )
    );

    auto bit_set_stencil_iter = thrust::make_transform_iterator<is_bit_set, zipped_mask>(
            zipped_mask_stencil_iter,
            is_bit_set()
    );

    //well basically we are zipping up an iterator to the stencil, one to the bit masks, and one which lets us get the bit position based on our index
    auto zipped_stencil_iter = thrust::make_zip_iterator(
        thrust::make_tuple(
                thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int8_t * )stencil->data)),
                bit_set_stencil_iter
        ));

    //NOTE!!!! the output column is getting set to a specific size but we are NOT compacting the allocation,
    //whoever calls that should handle that
    cudf::type_dispatcher(col->dtype, apply_stencil_functor{}, col, output, zipped_stencil_iter);

    if(col->valid != nullptr) {
        gdf_size_type num_values = col->size;

        rmm::device_vector<gdf_valid_type> valid_bit_mask; //we are expanding the bit mask to an int8
        if(num_values % GDF_VALID_BITSIZE != 0){
            valid_bit_mask.resize(gdf::util::div_round_up_safe(num_values, GDF_VALID_BITSIZE) * GDF_VALID_BITSIZE); //align this allocation on GDF_VALID_BITSIZE so we don't have to bounds check
        }else{
            valid_bit_mask.resize(num_values);
        }

        auto  zipped_mask_col_iter = thrust::make_zip_iterator(
            thrust::make_tuple(
                thrust::make_counting_iterator<gdf_size_type>(0),
                thrust::make_constant_iterator(col->valid)
            )
        );

        auto bit_set_col_iter = thrust::make_transform_iterator<is_bit_set, zipped_mask>(
                zipped_mask_col_iter,
                is_bit_set()
        );

        //copy the bitmask to device_vector of int8
        thrust::copy(rmm::exec_policy(stream), bit_set_col_iter, bit_set_col_iter + num_values, valid_bit_mask.begin());

        //remove the values that don't pass the stencil
        thrust::copy_if(rmm::exec_policy(stream), valid_bit_mask.begin(), valid_bit_mask.begin() + num_values, zipped_stencil_iter, valid_bit_mask.begin(),
                is_stencil_true<thrust::detail::normal_iterator<thrust::device_ptr<int8_t>>::value_type>());

        //recompact the values and store them in the output bitmask
        //we can group them into pieces of 8 because we aligned this earlier on when we made the device_vector
        thrust::detail::normal_iterator<thrust::device_ptr<int64_t> > valid_bit_mask_group_8_iter =
                thrust::detail::make_normal_iterator(thrust::device_pointer_cast((int64_t *) valid_bit_mask.data().get()));

        //you may notice that we can write out more bytes than our valid_num_bytes, this only happens when we are not aligned to  GDF_VALID_BITSIZE bytes, becasue the
        //arrow standard requires 64 byte alignment, this is a safe assumption to make
        thrust::transform(rmm::exec_policy(stream), valid_bit_mask_group_8_iter, valid_bit_mask_group_8_iter + ((num_values + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE),
                thrust::detail::make_normal_iterator(thrust::device_pointer_cast(output->valid)),bit_mask_pack_op());
    }

    CUDA_TRY( cudaStreamSynchronize(stream) );
    CUDA_TRY( cudaStreamDestroy(stream) );

    return GDF_SUCCESS;
}