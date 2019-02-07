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

#ifndef GDF_TESTS_BINARY_OPERATION_UTIL_SCALAR_H
#define GDF_TESTS_BINARY_OPERATION_UTIL_SCALAR_H

#include "tests/binary/util/types.h"
#include "cudf.h"

namespace gdf {
namespace library {

    template <typename Type>
    class Scalar {
    public:
        Scalar& setValue(Type value) {
            mScalar.dtype = gdf::library::GdfDataType<Type>::Value;
            gdf::library::setScalar(mScalar, value);
            mScalar.is_valid = true;
            return *this;
        }

        Scalar& setValid(bool value) {
            mScalar.is_valid = value;
            return *this;
        }

    public:
        Type getValue() {
            return (Type)*this;
        }

        gdf_dtype getType() {
            return mScalar.dtype;
        }

        bool isValid() {
            return mScalar.is_valid;
        }

    public:
        gdf_scalar* scalar() {
            return &mScalar;
        }

    public:
        operator int8_t() const {
            return mScalar.data.si08;
        }

        operator int16_t() const {
            return mScalar.data.si16;
        }

        operator int32_t() const {
            return mScalar.data.si32;
        }

        operator int64_t() const {
            return mScalar.data.si64;
        }

        operator uint8_t() const {
            return mScalar.data.ui08;
        }

        operator uint16_t() const {
            return mScalar.data.ui16;
        }

        operator uint32_t() const {
            return mScalar.data.ui32;
        }

        operator uint64_t() const {
            return mScalar.data.ui64;
        }

        operator float() const {
            return mScalar.data.fp32;
        }

        operator double() const {
            return mScalar.data.fp64;
        }

    private:
        gdf_scalar mScalar;
    };

}  // namespace library
}  // namespace gdf

#endif
