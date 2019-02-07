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

#include "tests/binary/util/types.h"

namespace gdf {
namespace library {

    const char* getTypeName(gdf_dtype type) {
        switch (type) {
            case GDF_INT8:
                return "GDF_INT8";
            case GDF_INT16:
                return "GDF_INT16";
            case GDF_INT32:
                return "GDF_INT32";
            case GDF_INT64:
                return "GDF_INT64";
            case GDF_UINT8:
                return "GDF_UINT8";
            case GDF_UINT16:
                return "GDF_UINT16";
            case GDF_UINT32:
                return "GDF_UINT32";
            case GDF_UINT64:
                return "GDF_UINT64";
            case GDF_FLOAT32:
                return "GDF_FLOAT32";
            case GDF_FLOAT64:
                return "GDF_FLOAT64";
            case GDF_DATE32:
                return "GDF_DATE32";
            case GDF_DATE64:
                return "GDF_DATE64";
            case GDF_TIMESTAMP:
                return "GDF_TIMESTAMP";
        }
    }

    namespace helper {
        void setScalar(gdf_scalar& scalar, int8_t value) {
            scalar.data.si08 = value;
        }

        void setScalar(gdf_scalar& scalar, int16_t value) {
            scalar.data.si16 = value;
        }

        void setScalar(gdf_scalar& scalar, int32_t value) {
            scalar.data.si32 = value;
        }

        void setScalar(gdf_scalar& scalar, int64_t value) {
            scalar.data.si64 = value;
        }

        void setScalar(gdf_scalar& scalar, uint8_t value) {
            scalar.data.ui08 = value;
        }

        void setScalar(gdf_scalar& scalar, uint16_t value) {
            scalar.data.ui16 = value;
        }

        void setScalar(gdf_scalar& scalar, uint32_t value) {
            scalar.data.ui32 = value;
        }

        void setScalar(gdf_scalar& scalar, uint64_t value) {
            scalar.data.ui64 = value;
        }

        void setScalar(gdf_scalar& scalar, float value) {
            scalar.data.fp32 = value;
        }

        void setScalar(gdf_scalar& scalar, double value) {
            scalar.data.fp64 = value;
        }
    }

    int8_t getScalar(int8_t, gdf_scalar* scalar) {
        return scalar->data.si08;
    }

    int16_t getScalar(int16_t, gdf_scalar* scalar) {
        return scalar->data.si16;
    }

    int32_t getScalar(int32_t, gdf_scalar* scalar) {
        return scalar->data.si32;
    }

    int64_t getScalar(int64_t, gdf_scalar* scalar) {
        return scalar->data.si64;
    }

    uint8_t getScalar(uint8_t, gdf_scalar* scalar) {
        return scalar->data.ui08;
    }

    uint16_t getScalar(uint16_t, gdf_scalar* scalar) {
        return scalar->data.ui16;
    }

    uint32_t getScalar(uint32_t, gdf_scalar* scalar) {
        return scalar->data.ui32;
    }

    uint64_t getScalar(uint64_t, gdf_scalar* scalar) {
        return scalar->data.ui64;
    }

    float getScalar(float, gdf_scalar* scalar) {
        return scalar->data.fp32;
    }

    double getScalar(double, gdf_scalar* scalar) {
        return scalar->data.fp64;
    }

}  // namespace library
}  // namespace gdf
