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

#ifndef GDF_TESTS_BINARY_OPERATION_UTIL_VECTOR_H
#define GDF_TESTS_BINARY_OPERATION_UTIL_VECTOR_H

#include "tests/binary/util/types.h"
#include "tests/binary/util/field.h"
#include "cudf.h"

namespace gdf {
namespace library {

    template <typename Type>
    class Vector {
    public:
        using ValidType = int32_t;
        static constexpr int ValidSize = 32;

    private:
        template <typename T>
        class InnerWrapper {
        public:
            InnerWrapper(Field<T>& container)
             : mField (container)
            { }

            T operator[](int index) {
                return mField[index];
            }

        private:
            Field<T>& mField;
        };

    public:
        ~Vector() {
            eraseGpu();
        }

        Vector& clearGpu() {
            eraseGpu();
            return *this;
        }

        Vector& rangeData(Type init, Type final, Type step) {
            assert((Type)0 < step);
            assert(init < final);

            int size = (final - init) / step;
            mData.resize(size);
            for (int k = 0; k < size; ++k) {
                mData[k] = init;
                init += step;
            }
            mData.write();
            updateData();
            return *this;
        }

        Vector& fillData(int size, Type value) {
            mData.resize(size);
            std::fill(mData.begin(), mData.end(), value);
            mData.write();
            updateData();
            return *this;
        }

        Vector& rangeValid(bool value) {
            int size = (mData.size() / ValidSize) + ((mData.size() % ValidSize) ? 1 : 0);
            mValid.resize(size);

            std::generate(mValid.begin(), mValid.end(), [value] { return -(ValidType)value; });
            clearPaddingBits();

            mValid.write();
            updateValid();
            return *this;
        }

        Vector& rangeValid(bool value, int init, int step) {
            int final = mData.size();
            int size = (final / ValidSize) + ((final % ValidSize) ? 1 : 0);
            mValid.resize(size);

            for (int index = 0; index < size; ++index) {
                ValidType val = 0;
                while (((init / ValidSize) == index) && (init < final)) {
                    val |= (1 << (init % ValidSize));
                    init += step;
                }
                if (value) {
                    mValid[index] = val;
                } else {
                    mValid[index] = ~val;
                }
            }
            clearPaddingBits();

            mValid.write();
            updateValid();
            return *this;
        }

        void emplaceVector(int size) {
            int validSize = (size / ValidSize) + ((size % ValidSize) ? 1 : 0);
            mData.resize(size);
            mValid.resize(validSize);
            updateData();
            updateValid();
        }

        void readVector() {
            mData.read();
            mValid.read();
        }

    public:
        int dataSize() {
            return mData.size();
        }

        int validSize() {
            return mValid.size();
        }

        gdf_column* column() {
            return &mColumn;
        }

    public:
        InnerWrapper<Type> data{mData};

        InnerWrapper<ValidType> valid{mValid};

    private:
        void eraseGpu() {
            mData.clear();
            mValid.clear();
        }

        void updateData() {
            mColumn.size = mData.size();
            mColumn.dtype = GdfDataType<Type>::Value;
            mColumn.data = (void*)mData.getGpuData();
        }

        void updateValid() {
            mColumn.valid = (gdf_valid_type*)mValid.getGpuData();
        }

        void clearPaddingBits() {
            int padding = mData.size() % ValidSize;
            if (padding) {
                padding = (1 << padding) - 1;
                mValid.back() &= padding;
            }
        }

    private:
        gdf_column mColumn;

    private:
        Field<Type> mData;
        Field<ValidType> mValid;
    };

} // namespace library
} // namespace gdf

#endif
