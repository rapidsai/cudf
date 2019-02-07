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

#ifndef GDF_TESTS_BINARY_OPERATION_UTIL_FIELD_H
#define GDF_TESTS_BINARY_OPERATION_UTIL_FIELD_H

#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

namespace gdf {
namespace library {

    template <typename Type>
    class Field {
    public:
        ~Field() {
            destroy();
        }

    public:
        void clear() {
            mCpuData.clear();
            destroy();
        }

        void resize(int size) {
            int sizeBytes = size * sizeof(Type);
            if (sizeBytes != mSizeAllocBytes) {
                mCpuData.resize(size);
                destroy();
                create(size);
            }
        }

    public:
        auto getGpuData() -> Type* {
            return mGpuData;
        }

    public:
        auto begin() -> typename std::vector<Type>::iterator {
            return mCpuData.begin();
        }

        auto end() -> typename std::vector<Type>::iterator {
            return mCpuData.end();
        }

        auto back() -> typename std::vector<Type>::reference {
            return mCpuData.back();
        }

    public:
        auto size() -> std::size_t {
            return mCpuData.size();
        }

        auto operator[](int index) -> Type& {
            assert(index < mCpuData.size());
            return mCpuData[index];
        }

    public:
        void write() {
            if (mSizeAllocBytes) {
                cudaMemcpy(mGpuData, mCpuData.data(), mSizeAllocBytes, cudaMemcpyHostToDevice);
            }
        }

        void read() {
            if (mSizeAllocBytes) {
                cudaMemcpy(mCpuData.data(), mGpuData, mSizeAllocBytes, cudaMemcpyDeviceToHost);
            }
        }

    protected:
        void create(int size) {
            mSizeAllocBytes = size * sizeof(Type);
            cudaMalloc((void**)&(mGpuData), mSizeAllocBytes);
        }

        void destroy() {
            if (mSizeAllocBytes) {
                mSizeAllocBytes = 0;
                cudaFree(mGpuData);
            }
        }

    private:
        int mSizeAllocBytes {0};
        Type* mGpuData {nullptr};
        std::vector<Type> mCpuData;
    };

} // namespace library
} // namespace gdf

#endif
