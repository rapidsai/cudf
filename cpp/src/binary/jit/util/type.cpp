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

#include "binary/jit/util/type.h"
#include "utilities/type_dispatcher.hpp"

namespace cudf {
namespace binops {
namespace jit {

    struct type_name {
        template <class T>
        CUDA_HOST_DEVICE_CALLABLE
        std::string operator()() {
#if defined(__clang__) || defined(__GNUC__)
            std::string p = __PRETTY_FUNCTION__;
            std::string search_str = "T = ";
            size_t start_pos = p.find(search_str) + search_str.size();
            std::string wrapper_str = "wrapper<";
            size_t wrapper_pos = p.find(wrapper_str, start_pos);
            if (wrapper_pos != std::string::npos)
                start_pos = wrapper_pos + wrapper_str.size();
            size_t end_pos = p.find_first_of(",;]", start_pos);
            return p.substr(start_pos, end_pos - start_pos);
#else
#   error Only clang and gcc supported
#endif
        }
    };

    std::string getTypeName(gdf_dtype type) {
        return type_dispatcher(type, type_name());
    }
 
    std::string getOperatorName(gdf_binary_operator ope) {
        switch (ope) {
            case GDF_ADD:
                return "Add";
            case GDF_SUB:
                return "Sub";
            case GDF_MUL:
                return "Mul";
            case GDF_DIV:
                return "Div";
            case GDF_TRUE_DIV:
                return "TrueDiv";
            case GDF_FLOOR_DIV:
                return "FloorDiv";
            case GDF_MOD:
                return "Mod";
            case GDF_POW:
                return "Pow";
            case GDF_EQUAL:
                return "Equal";
            case GDF_NOT_EQUAL:
                return "NotEqual";
            case GDF_LESS:
                return "Less";
            case GDF_GREATER:
                return "Greater";
            case GDF_LESS_EQUAL:
                return "LessEqual";
            case GDF_GREATER_EQUAL:
                return "GreaterEqual";
            default:
                return "None";
        }
    }

} // namespace jit
} // namespace binops
} // namespace cudf
