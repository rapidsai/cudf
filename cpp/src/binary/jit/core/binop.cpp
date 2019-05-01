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

#include "binary/jit/core/launcher.h"
#include "binary/jit/util/operator.h"
#include <bitmask/bitmask_ops.hpp>
#include "utilities/error_utils.hpp"
#include "utilities/cudf_utils.h"
#include "cudf.h"
#include "bitmask/legacy_bitmask.hpp"

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
     * @return gdf_error 
     *---------------------------------------------------------------------------**/
    gdf_error binary_valid_mask_and(gdf_size_type & out_null_count,
                                    gdf_valid_type * valid_out,
                                    gdf_valid_type * valid_left,
                                    gdf_valid_type * valid_right,
                                    gdf_size_type num_values) {
        if (num_values == 0) {
            out_null_count = 0;
            return GDF_SUCCESS;
        }

        if (valid_out == nullptr && valid_left == nullptr && valid_right == nullptr) {
            // if both in cols have no mask, then out col is allowed to have no mask
            out_null_count = 0;
            return GDF_SUCCESS;
        }
        
        GDF_REQUIRE((valid_out != nullptr), GDF_DATASET_EMPTY)

        if ( valid_left != nullptr && valid_right != nullptr ) {
            cudaStream_t stream;
            CUDA_TRY( cudaStreamCreate(&stream) );
            auto error = apply_bitmask_to_bitmask(out_null_count, valid_out, valid_left, valid_right, stream, num_values);
            CUDA_TRY(cudaStreamSynchronize(stream));
            CUDA_TRY(cudaStreamDestroy(stream));
            return error;
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
        out_null_count = num_values - non_nulls;
        return error;
    }

    /**---------------------------------------------------------------------------*
     * @brief Computes output valid mask for op between a column and a scalar
     * 
     * @param out_null_coun[out] number of nulls in output
     * @param valid_out preallocated output mask
     * @param valid_col input mask of column
     * @param valid_scalar bool indicating if scalar is valid
     * @param num_values number of values in input mask valid_col
     * @return gdf_error 
     *---------------------------------------------------------------------------**/
    gdf_error scalar_col_valid_mask_and(gdf_size_type & out_null_count,
                                        gdf_valid_type * valid_out,
                                        gdf_valid_type * valid_col,
                                        bool valid_scalar,
                                        gdf_size_type num_values)
    {
        if (num_values == 0) {
            out_null_count = 0;
            return GDF_SUCCESS;
        }

        if (valid_out == nullptr && valid_col == nullptr && valid_scalar == true) {
            // if in col has no mask and scalar is valid, then out col is allowed to have no mask
            out_null_count = 0;
            return GDF_SUCCESS;
        }

        GDF_REQUIRE((valid_out != nullptr), GDF_DATASET_EMPTY)

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
        out_null_count = num_values - non_nulls;
        return error;
    }

namespace jit {

    gdf_error binary_operation(gdf_column* out, gdf_scalar* lhs, gdf_column* rhs, gdf_binary_operator ope) {

        // Check for null pointers in input
        GDF_REQUIRE((out != nullptr) && (lhs != nullptr) && (rhs != nullptr), GDF_DATASET_EMPTY)

        // Check for 0 sized data
        GDF_REQUIRE((out->size != 0) && (rhs->size != 0), GDF_SUCCESS)
        GDF_REQUIRE((out->size == rhs->size), GDF_COLUMN_SIZE_MISMATCH)

        // Check for null data pointer
        GDF_REQUIRE((out->data != nullptr) && (rhs->data != nullptr), GDF_DATASET_EMPTY)

        // Check for datatype
        GDF_REQUIRE((out->dtype > GDF_invalid) && (lhs->dtype > GDF_invalid) && (rhs->dtype > GDF_invalid), GDF_UNSUPPORTED_DTYPE)
        GDF_REQUIRE((out->dtype < N_GDF_TYPES) && (lhs->dtype < N_GDF_TYPES) && (rhs->dtype < N_GDF_TYPES), GDF_UNSUPPORTED_DTYPE)

        scalar_col_valid_mask_and(out->null_count, out->valid, rhs->valid, lhs->is_valid, rhs->size);

        Launcher::launch().kernel("kernel_v_s")
                          .instantiate(ope, Operator::Type::Reverse, out, rhs, lhs)
                          .launch(out, rhs, lhs);

        return GDF_SUCCESS;
    }

    gdf_error binary_operation(gdf_column* out, gdf_column* lhs, gdf_scalar* rhs, gdf_binary_operator ope) {

        // Check for null pointers in input
        GDF_REQUIRE((out != nullptr) && (lhs != nullptr) && (rhs != nullptr), GDF_DATASET_EMPTY)

        // Check for 0 sized data
        GDF_REQUIRE((out->size != 0) && (lhs->size != 0), GDF_SUCCESS)
        GDF_REQUIRE((out->size == lhs->size), GDF_COLUMN_SIZE_MISMATCH)

        // Check for null data pointer
        GDF_REQUIRE((out->data != nullptr) && (lhs->data != nullptr), GDF_DATASET_EMPTY)

        // Check for datatype
        GDF_REQUIRE((out->dtype > GDF_invalid) && (lhs->dtype > GDF_invalid) && (rhs->dtype > GDF_invalid), GDF_UNSUPPORTED_DTYPE)
        GDF_REQUIRE((out->dtype < N_GDF_TYPES) && (lhs->dtype < N_GDF_TYPES) && (rhs->dtype < N_GDF_TYPES), GDF_UNSUPPORTED_DTYPE)

        scalar_col_valid_mask_and(out->null_count, out->valid, lhs->valid, rhs->is_valid, lhs->size);

        Launcher::launch().kernel("kernel_v_s")
                          .instantiate(ope, Operator::Type::Direct, out, lhs, rhs)
                          .launch(out, lhs, rhs);

        return GDF_SUCCESS;
    }

    gdf_error binary_operation(gdf_column* out, gdf_column* lhs, gdf_column* rhs, gdf_binary_operator ope) {

        // Check for null pointers in input
        GDF_REQUIRE((out != nullptr) && (lhs != nullptr) && (rhs != nullptr), GDF_DATASET_EMPTY)

        // Check for 0 sized data
        GDF_REQUIRE((out->size != 0) && (lhs->size != 0) && (rhs->size != 0), GDF_SUCCESS)
        GDF_REQUIRE((out->size == lhs->size) && (lhs->size == rhs->size), GDF_COLUMN_SIZE_MISMATCH)

        // Check for null data pointer
        GDF_REQUIRE((out->data != nullptr) && (lhs->data != nullptr) && (rhs->data != nullptr), GDF_DATASET_EMPTY)

        // Check for datatype
        GDF_REQUIRE((out->dtype > GDF_invalid) && (lhs->dtype > GDF_invalid) && (rhs->dtype > GDF_invalid), GDF_UNSUPPORTED_DTYPE)
        GDF_REQUIRE((out->dtype < N_GDF_TYPES) && (lhs->dtype < N_GDF_TYPES) && (rhs->dtype < N_GDF_TYPES), GDF_UNSUPPORTED_DTYPE)
        
        binary_valid_mask_and(out->null_count, out->valid, lhs->valid, rhs->valid, rhs->size);

        Launcher::launch().kernel("kernel_v_v")
                          .instantiate(ope, Operator::Type::Direct, out, lhs, rhs)
                          .launch(out, lhs, rhs);

        return GDF_SUCCESS;
    }

} // namespace jit
} // namespace binops
} // namespace cudf


gdf_error gdf_binary_operation_s_v(gdf_column* out, gdf_scalar* lhs, gdf_column* rhs, gdf_binary_operator ope) {
    return cudf::binops::jit::binary_operation(out, lhs, rhs, ope);
}

gdf_error gdf_binary_operation_v_s(gdf_column* out, gdf_column* lhs, gdf_scalar* rhs, gdf_binary_operator ope) {
    return cudf::binops::jit::binary_operation(out, lhs, rhs, ope);
}

gdf_error gdf_binary_operation_v_v(gdf_column* out, gdf_column* lhs, gdf_column* rhs, gdf_binary_operator ope) {
    return cudf::binops::jit::binary_operation(out, lhs, rhs, ope);
}
