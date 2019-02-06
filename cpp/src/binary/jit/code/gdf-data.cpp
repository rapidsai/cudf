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

namespace gdf {
namespace binops {
namespace jit {
namespace code {

const char* gdf_data =
R"***(
#pragma once

    union gdf_data {
        int8_t   si08;
        int16_t  si16;
        int32_t  si32;
        int64_t  si64;
        uint8_t  ui08;
        uint16_t ui16;
        uint32_t ui32;
        uint64_t ui64;
        float    fp32;
        double   fp64;

        operator int8_t() const {
            return si08;
        }

        operator int16_t() const {
            return si16;
        }

        operator int32_t() const {
            return si32;
        }

        operator int64_t() const {
            return si64;
        }

        operator uint8_t() const {
            return ui08;
        }

        operator uint16_t() const {
            return ui16;
        }

        operator uint32_t() const {
            return ui32;
        }

        operator uint64_t() const {
            return ui64;
        }

        operator float() const {
            return fp32;
        }

        operator double() const {
            return fp64;
        }
    };

)***";

} // namespace code
} // namespace jit
} // namespace binops
} // namespace gdf
