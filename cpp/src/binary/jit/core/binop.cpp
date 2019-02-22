/*
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
#include "bitmask/bitmask_ops.h"
#include "utilities/error_utils.h"
#include "cudf.h"

namespace cudf {
namespace binops {

    
    gdf_error binary_valid_mask_and(gdf_size_type & out_null_count,
                                    gdf_valid_type * valid_out,
                                    gdf_valid_type * valid_left,
                                    gdf_valid_type * valid_right,
                                    gdf_size_type num_values) {
        if (num_values == 0) {
            out_null_count = 0;
            return GDF_SUCCESS;
        }

        GDF_REQUIRE((valid_out != nullptr), GDF_DATASET_EMPTY)

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        if ( valid_left != nullptr && valid_right != nullptr ) {
            return apply_bitmask_to_bitmask(out_null_count, valid_out, valid_left, valid_right, stream, num_values);
        }
        
    	gdf_size_type num_chars_bitmask = ( ( num_values +( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE );

        if ( valid_left == nullptr && valid_right != nullptr ) {
            CUDA_TRY( cudaMemcpy(valid_out, valid_right, num_chars_bitmask, cudaMemcpyDeviceToDevice) );
        } 
        else if ( valid_left != nullptr && valid_right == nullptr ) {
            CUDA_TRY( cudaMemcpy(valid_out, valid_left, num_chars_bitmask, cudaMemcpyDeviceToDevice) );
        } 
        else if ( valid_left == nullptr && valid_right == nullptr ) {
            CUDA_TRY( cudaMemset(valid_out, 0xff, num_chars_bitmask) );
        }

    	return update_null_count(out_null_count, valid_out, stream, num_values);
    }

namespace jit {

    gdf_error binary_operation(gdf_column* out, gdf_scalar* lhs, gdf_column* rhs, gdf_binary_operator ope) {

        // Check for null pointers in input
        GDF_REQUIRE((out != nullptr) && (lhs != nullptr) && (rhs != nullptr), GDF_DATASET_EMPTY)

        // Check for 0 sized data
        GDF_REQUIRE((out->size != 0) && (rhs->size != 0), GDF_SUCCESS)
        GDF_REQUIRE((out->size == rhs->size), GDF_COLUMN_SIZE_MISMATCH)

        // Check for null data pointer
        GDF_REQUIRE((out->data != nullptr) && (lhs->data != nullptr) && (rhs->data != nullptr), GDF_DATASET_EMPTY)

        // Check for datatype
        GDF_REQUIRE((out->dtype > GDF_invalid) && (lhs->dtype > GDF_invalid) && (rhs->dtype > GDF_invalid), GDF_UNSUPPORTED_DTYPE)
        GDF_REQUIRE((out->dtype < N_GDF_TYPES) && (lhs->dtype < N_GDF_TYPES) && (rhs->dtype < N_GDF_TYPES), GDF_UNSUPPORTED_DTYPE)

        binary_valid_mask_and(out->null_count, out->valid, nullptr, rhs->valid, rhs->size);

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
        GDF_REQUIRE((out->data != nullptr) && (lhs->data != nullptr) && (rhs->data != nullptr), GDF_DATASET_EMPTY)

        // Check for datatype
        GDF_REQUIRE((out->dtype > GDF_invalid) && (lhs->dtype > GDF_invalid) && (rhs->dtype > GDF_invalid), GDF_UNSUPPORTED_DTYPE)
        GDF_REQUIRE((out->dtype < N_GDF_TYPES) && (lhs->dtype < N_GDF_TYPES) && (rhs->dtype < N_GDF_TYPES), GDF_UNSUPPORTED_DTYPE)

        binary_valid_mask_and(out->null_count, out->valid, lhs->valid, nullptr, lhs->size);

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
