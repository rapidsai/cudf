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
#include "cudf.h"

namespace gdf {
namespace binops {
namespace jit {

    struct Option {
        Option(bool state, gdf_error value)
         : is_correct{state}, gdf_error_value{value}
        { }

        operator bool() {
            return is_correct;
        }

        gdf_error get_gdf_error() {
            return gdf_error_value;
        }

    private:
        bool is_correct;
        gdf_error gdf_error_value;
    };

    Option verify_scalar(gdf_scalar* scalar) {
        if (scalar == nullptr) {
            return Option(false, GDF_DATASET_EMPTY);
        }
        if (scalar->data == nullptr) {
            return Option(false, GDF_DATASET_EMPTY);
        }
        if ((scalar->dtype <= GDF_invalid) || (N_GDF_TYPES <= scalar->dtype)) {
            return Option(false, GDF_UNSUPPORTED_DTYPE);
        }
        return Option(true, GDF_SUCCESS);
    }

    Option verify_column(gdf_column* vector) {
        if (vector == nullptr) {
            return Option(false, GDF_DATASET_EMPTY);
        }
        if (vector->size == 0) {
            return Option(false, GDF_SUCCESS);
        }
        if (vector->data == nullptr) {
            return Option(false, GDF_DATASET_EMPTY);
        }
        if ((vector->dtype <= GDF_invalid) || (N_GDF_TYPES <= vector->dtype)) {
            return Option(false, GDF_UNSUPPORTED_DTYPE);
        }
        return Option(true, GDF_SUCCESS);
    }

    Option verify_column(gdf_column* out, gdf_column* lhs) {
        auto result = verify_column(out);
        if (!result) {
            return result;
        }
        result = verify_column(lhs);
        if (!result) {
            return result;
        }
        if (out->size < lhs->size) {
            return Option(false, GDF_COLUMN_SIZE_MISMATCH);
        }
        return Option(true, GDF_SUCCESS);
    }

    Option verify_column(gdf_column* out, gdf_column* lhs, gdf_column* rhs) {
        auto result = verify_column(out);
        if (!result) {
            return result;
        }
        result = verify_column(lhs);
        if (!result) {
            return result;
        }
        result = verify_column(rhs);
        if (!result) {
            return result;
        }
        if ((out->size < lhs->size) || (out->size < rhs->size) || (rhs->size != lhs->size)) {
            return Option(false, GDF_COLUMN_SIZE_MISMATCH);
        }
        return Option(true, GDF_SUCCESS);
    }

    gdf_error binary_operation(gdf_column* out, gdf_scalar* lhs, gdf_column* rhs, gdf_binary_operator ope) {
        auto option_scalar = verify_scalar(lhs);
        if (!option_scalar) {
            return option_scalar.get_gdf_error();
        }
        auto option_column = verify_column(out, rhs);
        if (!option_column) {
            return option_column.get_gdf_error();
        }

        Launcher::launch().kernel("kernel_v_s")
                          .instantiate(ope, Operator::Type::Reverse, out, rhs, lhs)
                          .launch(out, rhs, lhs);

        return GDF_SUCCESS;
    }

    gdf_error binary_operation(gdf_column* out, gdf_column* lhs, gdf_scalar* rhs, gdf_binary_operator ope) {
        auto option_scalar = verify_scalar(rhs);
        if (!option_scalar) {
            return option_scalar.get_gdf_error();
        }
        auto option_column = verify_column(out, lhs);
        if (!option_column) {
            return option_column.get_gdf_error();
        }

        Launcher::launch().kernel("kernel_v_s")
                          .instantiate(ope, Operator::Type::Direct, out, lhs, rhs)
                          .launch(out, lhs, rhs);

        return GDF_SUCCESS;
    }

    gdf_error binary_operation(gdf_column* out, gdf_column* lhs, gdf_column* rhs, gdf_binary_operator ope) {
        auto option_column = verify_column(out, lhs, rhs);
        if (!option_column) {
            return option_column.get_gdf_error();
        }

        Launcher::launch().kernel("kernel_v_v")
                          .instantiate(ope, Operator::Type::Direct, out, lhs, rhs)
                          .launch(out, lhs, rhs);

        return GDF_SUCCESS;
    }

} // namespace jit
} // namespace binops
} // namespace gdf


gdf_error gdf_binary_operation_s_v(gdf_column* out, gdf_scalar* lhs, gdf_column* rhs, gdf_binary_operator ope) {
    return gdf::binops::jit::binary_operation(out, lhs, rhs, ope);
}

gdf_error gdf_binary_operation_v_s(gdf_column* out, gdf_column* lhs, gdf_scalar* rhs, gdf_binary_operator ope) {
    return gdf::binops::jit::binary_operation(out, lhs, rhs, ope);
}

gdf_error gdf_binary_operation_v_v(gdf_column* out, gdf_column* lhs, gdf_column* rhs, gdf_binary_operator ope) {
    return gdf::binops::jit::binary_operation(out, lhs, rhs, ope);
}
