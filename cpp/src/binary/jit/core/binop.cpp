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

    Option verify_column(gdf_column* out, gdf_column* vax) {
        auto result = verify_column(out);
        if (!result) {
            return result;
        }
        result = verify_column(vax);
        if (!result) {
            return result;
        }
        if (out->size < vax->size) {
            return Option(false, GDF_COLUMN_SIZE_MISMATCH);
        }
        return Option(true, GDF_SUCCESS);
    }

    Option verify_column(gdf_column* out, gdf_column* vax, gdf_column* vay) {
        auto result = verify_column(out);
        if (!result) {
            return result;
        }
        result = verify_column(vax);
        if (!result) {
            return result;
        }
        result = verify_column(vay);
        if (!result) {
            return result;
        }
        if ((out->size < vax->size) || (out->size < vay->size) || (vay->size != vax->size)) {
            return Option(false, GDF_COLUMN_SIZE_MISMATCH);
        }
        return Option(true, GDF_SUCCESS);
    }

    gdf_error binary_operation(gdf_column* out, gdf_scalar* vax, gdf_column* vay, gdf_binary_operator ope) {
        auto option_scalar = verify_scalar(vax);
        if (!option_scalar) {
            return option_scalar.get_gdf_error();
        }
        auto option_column = verify_column(out, vay);
        if (!option_column) {
            return option_column.get_gdf_error();
        }

        Launcher::launch().kernel("kernel_v_s")
                          .instantiate(ope, Operator::Type::Reverse, out, vay, vax)
                          .launch(out, vay, vax);

        return GDF_SUCCESS;
    }

    gdf_error binary_operation(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_binary_operator ope) {
        auto option_scalar = verify_scalar(vay);
        if (!option_scalar) {
            return option_scalar.get_gdf_error();
        }
        auto option_column = verify_column(out, vax);
        if (!option_column) {
            return option_column.get_gdf_error();
        }

        Launcher::launch().kernel("kernel_v_s")
                          .instantiate(ope, Operator::Type::Direct, out, vax, vay)
                          .launch(out, vax, vay);

        return GDF_SUCCESS;
    }

    gdf_error binary_operation(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_binary_operator ope) {
        auto option_column = verify_column(out, vax, vay);
        if (!option_column) {
            return option_column.get_gdf_error();
        }

        Launcher::launch().kernel("kernel_v_v")
                          .instantiate(ope, Operator::Type::Direct, out, vax, vay)
                          .launch(out, vax, vay);

        return GDF_SUCCESS;
    }

    gdf_error binary_operation(gdf_column* out, gdf_scalar* vax, gdf_column* vay, gdf_scalar* def, gdf_binary_operator ope) {
        auto option_column = verify_column(out, vay);
        if (!option_column) {
            return option_column.get_gdf_error();
        }
        auto option_scalar_vax = verify_scalar(vax);
        if (!option_scalar_vax) {
            return option_scalar_vax.get_gdf_error();
        }
        auto option_scalar_def = verify_scalar(def);
        if (!option_scalar_def) {
            return option_scalar_def.get_gdf_error();
        }

        Launcher::launch().kernel("kernel_v_s_d")
                          .instantiate(ope, Operator::Type::Reverse, out, vay, vax, def)
                          .launch(out, vay, vax, def);

        return GDF_SUCCESS;
    }

    gdf_error binary_operation(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_scalar* def, gdf_binary_operator ope) {
        auto option_column = verify_column(out, vax);
        if (!option_column) {
            return option_column.get_gdf_error();
        }
        auto option_scalar_vax = verify_scalar(vay);
        if (!option_scalar_vax) {
            return option_scalar_vax.get_gdf_error();
        }
        auto option_scalar_def = verify_scalar(def);
        if (!option_scalar_def) {
            return option_scalar_def.get_gdf_error();
        }

        Launcher::launch().kernel("kernel_v_s_d")
                          .instantiate(ope, Operator::Type::Direct, out, vax, vay, def)
                          .launch(out, vax, vay, def);

        return GDF_SUCCESS;
    }

    gdf_error binary_operation(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_scalar* def, gdf_binary_operator ope) {
        auto option_column = verify_column(out, vax, vay);
        if (!option_column) {
            return option_column.get_gdf_error();
        }
        auto option_scalar = verify_scalar(def);
        if (!option_scalar) {
            return option_scalar.get_gdf_error();
        }

        Launcher::launch().kernel("kernel_v_v_d")
                          .instantiate(ope, Operator::Type::Direct, out, vax, vay, def)
                          .launch(out, vax, vay, def);

        return GDF_SUCCESS;
    }

} // namespace jit
} // namespace binops
} // namespace gdf


gdf_error gdf_binary_operation_v_s_v(gdf_column* out, gdf_scalar* vax, gdf_column* vay, gdf_binary_operator ope) {
    return gdf::binops::jit::binary_operation(out, vax, vay, ope);
}

gdf_error gdf_binary_operation_v_v_s(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_binary_operator ope) {
    return gdf::binops::jit::binary_operation(out, vax, vay, ope);
}

gdf_error gdf_binary_operation_v_v_v(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_binary_operator ope) {
    return gdf::binops::jit::binary_operation(out, vax, vay, ope);
}

gdf_error gdf_binary_operation_v_s_v_d(gdf_column* out, gdf_scalar* vax, gdf_column* vay, gdf_scalar* def, gdf_binary_operator ope) {
    return gdf::binops::jit::binary_operation(out, vax, vay, def, ope);
}

gdf_error gdf_binary_operation_v_v_s_d(gdf_column* out, gdf_column* vax, gdf_scalar* vay, gdf_scalar* def, gdf_binary_operator ope) {
    return gdf::binops::jit::binary_operation(out, vax, vay, def, ope);
}

gdf_error gdf_binary_operation_v_v_v_d(gdf_column* out, gdf_column* vax, gdf_column* vay, gdf_scalar* def, gdf_binary_operator ope) {
    return gdf::binops::jit::binary_operation(out, vax, vay, def, ope);
}
