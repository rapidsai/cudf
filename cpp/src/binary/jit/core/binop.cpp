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
#include "utilities/error_utils.h"
#include "cudf.h"

namespace cudf {
namespace binops {
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
