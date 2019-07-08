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
#include "jit/util/operator.h"
#include "compiled/binary_ops.hpp"
#include <bitmask/legacy/bitmask_ops.hpp>
#include <utilities/error_utils.hpp>
#include <utilities/cudf_utils.h>
#include <cudf/cudf.h>
#include <bitmask/legacy/legacy_bitmask.hpp>
#include <string/nvcategory_util.hpp>
#include <cudf/copying.hpp>
#include <nvstrings/NVCategory.h>

namespace cudf {
namespace binops {

    /**---------------------------------------------------------------------------*
     * @brief Computes bitwise AND of two input valid masks
     * 
     * This is just a wrapper on apply_bitmask_to_bitmask that can also handle
     * cases when one or both of the input masks are nullptr, in which case, it
     * copies the mask from the non nullptr input or sets all the output mask to
     * valid respectively
     * 
     * @param out_null_coun[out] number of nulls in output
     * @param valid_out preallocated output mask
     * @param valid_left input mask 1
     * @param valid_right input mask 2
     * @param num_values number of values in each input mask valid_left and valid_right
     *---------------------------------------------------------------------------**/
    void binary_valid_mask_and(gdf_size_type & out_null_count,
                                    gdf_valid_type * valid_out,
                                    gdf_valid_type * valid_left,
                                    gdf_valid_type * valid_right,
                                    gdf_size_type num_values) {
        if (num_values == 0) {
            out_null_count = 0;
            return;
        }

        if (valid_out == nullptr && valid_left == nullptr && valid_right == nullptr) {
            // if both in cols have no mask, then out col is allowed to have no mask
            out_null_count = 0;
            return;
        }
        
        CUDF_EXPECTS((valid_out != nullptr), "Output valid mask pointer is null");

        if ( valid_left != nullptr && valid_right != nullptr ) {
            cudaStream_t stream;
            CUDA_TRY( cudaStreamCreate(&stream) );
            auto error = apply_bitmask_to_bitmask(out_null_count, valid_out, valid_left, valid_right, stream, num_values);
            CUDA_TRY(cudaStreamSynchronize(stream));
            CUDA_TRY(cudaStreamDestroy(stream));
            CUDF_EXPECTS(error == GDF_SUCCESS, "Unable to combine bitmasks");
        }
        
    	gdf_size_type num_bitmask_elements = gdf_num_bitmask_elements( num_values );

        if ( valid_left == nullptr && valid_right != nullptr ) {
            CUDA_TRY( cudaMemcpy(valid_out, valid_right, num_bitmask_elements, cudaMemcpyDeviceToDevice) );
        } 
        else if ( valid_left != nullptr && valid_right == nullptr ) {
            CUDA_TRY( cudaMemcpy(valid_out, valid_left, num_bitmask_elements, cudaMemcpyDeviceToDevice) );
        } 
        else if ( valid_left == nullptr && valid_right == nullptr ) {
            CUDA_TRY( cudaMemset(valid_out, 0xff, num_bitmask_elements) );
        }

        gdf_size_type non_nulls;
    	auto error = gdf_count_nonzero_mask(valid_out, num_values, &non_nulls);
        CUDF_EXPECTS(error == GDF_SUCCESS, "Unable to count number of valids");
        out_null_count = num_values - non_nulls;
    }

    /**---------------------------------------------------------------------------*
     * @brief Computes output valid mask for op between a column and a scalar
     * 
     * @param out_null_coun[out] number of nulls in output
     * @param valid_out preallocated output mask
     * @param valid_col input mask of column
     * @param valid_scalar bool indicating if scalar is valid
     * @param num_values number of values in input mask valid_col
     *---------------------------------------------------------------------------**/
    void scalar_col_valid_mask_and(gdf_size_type & out_null_count,
                                        gdf_valid_type * valid_out,
                                        gdf_valid_type * valid_col,
                                        bool valid_scalar,
                                        gdf_size_type num_values)
    {
        if (num_values == 0) {
            out_null_count = 0;
            return;
        }

        if (valid_out == nullptr && valid_col == nullptr && valid_scalar == true) {
            // if in col has no mask and scalar is valid, then out col is allowed to have no mask
            out_null_count = 0;
            return;
        }

        CUDF_EXPECTS((valid_out != nullptr), "Output valid mask pointer is null");

    	gdf_size_type num_bitmask_elements = gdf_num_bitmask_elements( num_values );

        if ( valid_scalar == false ) {
            CUDA_TRY( cudaMemset(valid_out, 0x00, num_bitmask_elements) );
        } 
        else if ( valid_scalar == true && valid_col != nullptr ) {
            CUDA_TRY( cudaMemcpy(valid_out, valid_col, num_bitmask_elements, cudaMemcpyDeviceToDevice) );
        } 
        else if ( valid_scalar == true && valid_col == nullptr ) {
            CUDA_TRY( cudaMemset(valid_out, 0xff, num_bitmask_elements) );
        }

        gdf_size_type non_nulls;
    	auto error = gdf_count_nonzero_mask(valid_out, num_values, &non_nulls);
        CUDF_EXPECTS(error == GDF_SUCCESS, "Unable to count number of valids");
        out_null_count = num_values - non_nulls;
    }

namespace jit {

    void binary_operation(gdf_column* out, gdf_scalar* lhs, gdf_column* rhs, gdf_binary_operator ope) {

        Launcher().setKernelInst("kernel_v_s", ope, Operator::Type::Reverse, out, rhs, lhs)
                  .launch(out, rhs, lhs);

    }

    void binary_operation(gdf_column* out, gdf_column* lhs, gdf_scalar* rhs, gdf_binary_operator ope) {

        Launcher().setKernelInst("kernel_v_s", ope, Operator::Type::Direct, out, lhs, rhs)
                  .launch(out, lhs, rhs);

    }

    void binary_operation(gdf_column* out, gdf_column* lhs, gdf_column* rhs, gdf_binary_operator ope) {

        Launcher().setKernelInst("kernel_v_v", ope, Operator::Type::Direct, out, lhs, rhs)
                  .launch(out, lhs, rhs);

    }
    
    void binary_operation(gdf_column* out, gdf_column* lhs, gdf_column* rhs, const std::string& ptx)  {

        Launcher(ptx).setKernelInst("kernel_v_v", GDF_GENERIC_OP, Operator::Type::Direct, out, lhs, rhs)
                     .launch(out, lhs, rhs);

    }

} // namespace jit
} // namespace binops

void binary_operation(gdf_column* out, gdf_scalar* lhs, gdf_column* rhs, gdf_binary_operator ope) {
    // Check for null pointers in input
    CUDF_EXPECTS((out != nullptr) && (lhs != nullptr) && (rhs != nullptr),
        "Input pointers are null");

    // Check for 0 sized data
    if((out->size == 0) && (rhs->size == 0)) return;
    CUDF_EXPECTS((out->size == rhs->size), "Column sizes don't match");

    // Check for null data pointer
    CUDF_EXPECTS((out->data != nullptr) && (rhs->data != nullptr),
        "Column data pointers are null");

    // Check for datatype
    CUDF_EXPECTS((out->dtype > GDF_invalid) && (out->dtype < N_GDF_TYPES) &&
                 (lhs->dtype > GDF_invalid) && (lhs->dtype < N_GDF_TYPES) &&
                 (rhs->dtype > GDF_invalid) && (rhs->dtype < N_GDF_TYPES) ,
        "Invalid/Unsupported datatype");

    binops::scalar_col_valid_mask_and(out->null_count, out->valid, rhs->valid, lhs->is_valid, rhs->size);

    binops::jit::binary_operation(out, lhs, rhs, ope);
}

void binary_operation(gdf_column* out, gdf_column* lhs, gdf_scalar* rhs, gdf_binary_operator ope) {
    // Check for null pointers in input
    CUDF_EXPECTS((out != nullptr) && (lhs != nullptr) && (rhs != nullptr),
        "Input pointers are null");

    // Check for 0 sized data
    if ((out->size == 0) && (lhs->size == 0)) return;
    CUDF_EXPECTS((out->size == lhs->size), "Column sizes don't match");

    // Check for null data pointer
    CUDF_EXPECTS((out->data != nullptr) && (lhs->data != nullptr), 
        "Column data pointers are null");

    // Check for datatype
    CUDF_EXPECTS((out->dtype > GDF_invalid) && (out->dtype < N_GDF_TYPES) &&
                 (lhs->dtype > GDF_invalid) && (lhs->dtype < N_GDF_TYPES) &&
                 (rhs->dtype > GDF_invalid) && (rhs->dtype < N_GDF_TYPES) ,
        "Invalid/Unsupported datatype");

    binops::scalar_col_valid_mask_and(out->null_count, out->valid, lhs->valid, rhs->is_valid, lhs->size);

    binops::jit::binary_operation(out, lhs, rhs, ope);
}

void binary_operation(gdf_column* out, gdf_column* lhs, gdf_column* rhs, gdf_binary_operator ope) {
    // Check for null pointers in input
    CUDF_EXPECTS((out != nullptr) && (lhs != nullptr) && (rhs != nullptr),
        "Input pointers are null");

    // Check for 0 sized data
    if((out->size == 0) && (lhs->size == 0) && (rhs->size == 0)) return;
    CUDF_EXPECTS((out->size == lhs->size) && (lhs->size == rhs->size),
        "Column sizes don't match");

    // Check for null data pointer
    CUDF_EXPECTS((out->data != nullptr) &&
                 (lhs->data != nullptr) &&
                 (rhs->data != nullptr) , 
        "Column data pointers are null");

    // Check for datatype
    CUDF_EXPECTS((out->dtype > GDF_invalid) && (out->dtype < N_GDF_TYPES) &&
                 (lhs->dtype > GDF_invalid) && (lhs->dtype < N_GDF_TYPES) &&
                 (rhs->dtype > GDF_invalid) && (rhs->dtype < N_GDF_TYPES) ,
        "Invalid/Unsupported datatype");
    
    binops::binary_valid_mask_and(out->null_count, out->valid, lhs->valid, rhs->valid, rhs->size);

    if (lhs->dtype == GDF_STRING_CATEGORY && rhs->dtype == GDF_STRING_CATEGORY) {
        // if the columns are string types then we need to combine categories
        // before checking for equality because the same category values can mean
        // different things for different columns

        // make temporary columns which will have synced categories
        auto temp_lhs = cudf::allocate_like(*lhs);
        auto temp_rhs = cudf::allocate_like(*rhs);
        gdf_column* input_cols[2]   = {lhs, rhs};
        gdf_column* temp_cols[2]    = {&temp_lhs, &temp_rhs};

        // sync categories
        sync_column_categories(input_cols, temp_cols, 2);

        // now it's ok to directly compare the column data
        auto err = binops::compiled::binary_operation(out, &temp_lhs, &temp_rhs, ope);
        if (err == GDF_UNSUPPORTED_DTYPE || err == GDF_INVALID_API_CALL)
            binops::jit::binary_operation(out, &temp_lhs, &temp_rhs, ope);

        // TODO: Need a better way to deallocate temporary columns
        RMM_TRY(RMM_FREE(temp_lhs.data, 0));
        RMM_TRY(RMM_FREE(temp_rhs.data, 0));
        if (temp_lhs.valid != nullptr)
            RMM_TRY(RMM_FREE(temp_lhs.valid, 0));
        if (temp_rhs.valid != nullptr)
            RMM_TRY(RMM_FREE(temp_rhs.valid, 0));
        NVCategory::destroy(
            reinterpret_cast<NVCategory*>(temp_lhs.dtype_info.category));
        NVCategory::destroy(
            reinterpret_cast<NVCategory*>(temp_rhs.dtype_info.category));
        return;
    }
    auto err = binops::compiled::binary_operation(out, lhs, rhs, ope);
    if (err == GDF_UNSUPPORTED_DTYPE || err == GDF_INVALID_API_CALL)
        binops::jit::binary_operation(out, lhs, rhs, ope);
}

void binary_operation(gdf_column* out, gdf_column* lhs, gdf_column* rhs, const std::string& ptx) {
    // Check for null pointers in input
    CUDF_EXPECTS((out != nullptr) && (lhs != nullptr) && (rhs != nullptr),
        "Input pointers are null");

    // Check for 0 sized data
    if((out->size == 0) && (lhs->size == 0) && (rhs->size == 0)) return;
    CUDF_EXPECTS((out->size == lhs->size) && (lhs->size == rhs->size),
        "Column sizes don't match");

    // Check for null data pointer
    CUDF_EXPECTS((out->data != nullptr) &&
                 (lhs->data != nullptr) &&
                 (rhs->data != nullptr) , 
        "Column data pointers are null");

    // Check for datatype
    CUDF_EXPECTS((out->dtype == lhs->dtype) && (lhs->dtype == rhs->dtype) &&
                 (out->dtype == GDF_FLOAT32 || out->dtype == GDF_FLOAT64 || 
                  out->dtype == GDF_INT64   || out->dtype == GDF_INT32     ),
        "Invalid/Unsupported datatype");
    
    binops::binary_valid_mask_and(out->null_count, out->valid, lhs->valid, rhs->valid, rhs->size);

    binops::jit::binary_operation(out, lhs, rhs, ptx);
}

} // namespace cudf
