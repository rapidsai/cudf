/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
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

#include "jit/core/launcher.h"
// #include "compiled/binary_ops.hpp"
#include <bitmask/bitmask_ops.hpp>
#include <utilities/error_utils.hpp>
#include <utilities/cudf_utils.h>
#include <cudf/cudf.h>
#include <bitmask/legacy_bitmask.hpp>
#include <string/nvcategory_util.hpp>
#include <cudf/copying.hpp>
#include <nvstrings/NVCategory.h>

#include <utilities/column_utils.hpp>

namespace cudf {
namespace transform {

    /**---------------------------------------------------------------------------*
     * @brief Computes output valid mask for op between a column and a scalar
     * 
     * @param out_null_coun[out] number of nulls in output
     * @param valid_out preallocated output mask
     * @param valid_col input mask of column
     * @param num_values number of values in input mask valid_col
     *---------------------------------------------------------------------------**/
    void col_valid_mask_and(gdf_size_type& out_null_count,
                                        gdf_valid_type* valid_out,
                                        const gdf_valid_type* valid_col,
                                        gdf_size_type num_values)
    {
        if (num_values == 0) {
            out_null_count = 0;
            return;
        }

        if(valid_out == nullptr && valid_col == nullptr){
            // if in col has no mask and scalar is valid, then out col is allowed to have no mask
            out_null_count = 0;
            return;
        }

        CUDF_EXPECTS((valid_out != nullptr), "Output valid mask pointer is null");

    	  gdf_size_type num_bitmask_elements = gdf_num_bitmask_elements(num_values);

        if(valid_col != nullptr){
            CUDA_TRY(cudaMemcpy(valid_out, valid_col, num_bitmask_elements, cudaMemcpyDeviceToDevice));
        }else{
            CUDA_TRY(cudaMemset(valid_out, 0xff, num_bitmask_elements));
        }

        gdf_size_type non_nulls;
    	  auto error = gdf_count_nonzero_mask(valid_out, num_values, &non_nulls);
        CUDF_EXPECTS(error == GDF_SUCCESS, "Unable to count number of valids");
        out_null_count = num_values - non_nulls;
    }

namespace jit {

    void unary_operation(gdf_column* out, gdf_column* in, const std::string& ptx)  {
        Launcher(ptx).setKernelInst("kernel_v", GDF_GENERIC_UNARY, out, in)
                     .launch(out, in);

    }

} // namespace jit
} // namespace transform

void unary_transform(gdf_column* out, gdf_column* in, const std::string& ptx) {
    // Check for null pointers in input
    CUDF_EXPECTS((out != nullptr) && (in != nullptr),
        "Input pointers are null");

    // Check for 0 sized data
    if((out->size == 0) && (in->size == 0)) return;
    CUDF_EXPECTS((out->size == in->size),
        "Column sizes don't match");

    // Check for null data pointer
    CUDF_EXPECTS((out->data != nullptr) &&
                 (in->data != nullptr), 
        "Column data pointers are null");

    // Check for datatype
    CUDF_EXPECTS((out->dtype == in->dtype) &&
                 (out->dtype == GDF_FLOAT32 || out->dtype == GDF_FLOAT64 || 
                  out->dtype == GDF_INT64   || out->dtype == GDF_INT32     ),
        "Invalid/Unsupported datatype");
    
    transform::col_valid_mask_and(out->null_count, out->valid, in->valid, in->size);

    transform::jit::unary_operation(out, in, ptx);
}

gdf_column unary_transform(const gdf_column& input,
                           const std::string& ptx_unary_function){
  // First create a gdf_column and then call the above function
  gdf_column output_col = copy(input);
 
  unary_transform(&output_col, const_cast<gdf_column*>(&input), ptx_unary_function);

  return output_col;
}


} // namespace cudf
