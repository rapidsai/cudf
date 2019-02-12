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
#include "binary/jit/code/code.h"
#include <cstdint>

namespace gdf {
namespace binops {
namespace jit {

    static thread_local jitify::JitCache JitCache;

    std::istream* headersCode(std::string filename, std::iostream& stream) {
        if (filename == "operation.h") {
            stream << code::operation;
            return &stream;
        }
        if (filename == "traits.h") {
            stream << code::traits;
            return &stream;
        }
        if (filename == "gdf_data.h") {
            stream << code::gdf_data;
            return &stream;
        }
        return nullptr;
    }

    Launcher::Launcher()
     : program {JitCache.program(code::kernel, headersName, compilerFlags, headersCode)}
    { }

    Launcher::Launcher(Launcher&& launcher)
     : program {std::move(launcher.program)}
    { }

    Launcher& Launcher::kernel(std::string&& value) {
        kernelName = value;
        return *this;
    }

    gdf_error Launcher::launch(gdf_column* out, gdf_column* vax, gdf_scalar* vay) {
        program.kernel(kernelName.c_str())
               .instantiate(arguments)
               .configure_1d_max_occupancy()
               .launch(out->size,
                       out->data, vax->data, vay->data,
                       out->valid, vax->valid);

        return GDF_SUCCESS;
    }

    gdf_error Launcher::launch(gdf_column* out, gdf_column* vax, gdf_column* vay) {
        program.kernel(kernelName.c_str())
               .instantiate(arguments)
               .configure_1d_max_occupancy()
               .launch(out->size,
                       out->data, vax->data, vay->data,
                       out->valid, vax->valid, vay->valid);

        return GDF_SUCCESS;
    }

} // namespace jit
} // namespace binops
} // namespace gdf
