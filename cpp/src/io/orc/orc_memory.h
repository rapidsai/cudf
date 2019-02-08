/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#ifndef __ORC_BUFFERS_HEADER__
#define __ORC_BUFFERS_HEADER__

#include "orc_read.h"
#include "orc_debug.h"
#include "orc_util.hpp"
#include "kernel_orc.cuh"
#include <vector>

namespace cudf {
namespace orc {

namespace OrcMem {
    struct OrcMemArgument {
        int blockSize;               //< byte size of the block size
        //int initBlockCount;        //< initial count of the blocks to be allocated
        //int additionalBlockCount;  //< additional count of the blocks to be allocated
    };

    // 
    void InitializeOrcMem(OrcMemArgument& arg);

    int GetBlockSize();

    // return null if there is no free block, thread safe API
    orc_byte* GetAvailableManagedBlock();

    // free the block to the pool, thread safe API
    void FreeManagedBlock(const orc_byte* block);
}

/** ---------------------------------------------------------------------------*
* @brief a utility class just to hold OrcBuffer and free the buffer when the class is destroyed
* ---------------------------------------------------------------------------**/
class OrcBufferHolder {
public:
    OrcBufferHolder() {};
    ~OrcBufferHolder() { ReleaseAll(); };

public:
    void AddBuffer(OrcBuffer& buf) {
        buffers.push_back(buf); 
    };

    void AddBuffers(std::vector<OrcBuffer>& bufs)
    {
        buffers.reserve(buffers.size() + bufs.size());
        for (OrcBuffer& buf : bufs) {
            buffers.push_back(buf);
        }
    };

    void ReleaseAll() {
        for (OrcBuffer& buf : buffers) {
            OrcMem::FreeManagedBlock(buf.buffer);
        }
        buffers.clear();
    };

protected:
    std::vector<OrcBuffer> buffers;
};


class OrcBufferInfoHolder : public OrcBufferHolder {
public:
    OrcBufferInfoHolder() 
        : offset(0)
    {
        current.buffer = NULL;
        current.bufferSize = 0;
    };
    ~OrcBufferInfoHolder() {  };

public:
    // allocate and construct OrcBufferArray backed by cuda managed memory from std::vector<OrcBuffer>
    void ConstructOrcBufferArray(OrcBufferArray& array, std::vector<OrcBuffer>& bufs);

    // request a managed buffer with requested size
    CudaOrcError_t RequestBuffer(orc_byte** dest, size_t size);

    int AvailableMemory() {
        return current.bufferSize - offset;
    }

protected:
    OrcBuffer   current;
    size_t      offset;
};

// decode chunks by CPU
class OrcChunkDecoder{
public:
    OrcChunkDecoder() {};
    ~OrcChunkDecoder() {};

    void SetCompressionKind(OrcCompressionKind kind) { compKind = kind; };

    // return true if input is a single chunk and it is uncompressed, or the compression kind is uncompressed.
    bool IsSingleUncompressed(const orc_byte* input, size_t input_size);

    // decode as input chunks into OrcBuffer vector form
    CudaOrcError_t Decode(std::vector<OrcBuffer>& bufs, OrcBufferInfoHolder* holder,
        const orc_byte* cpu_input, const orc_byte* gpu_input, size_t input_size);

private:
    OrcCompressionKind compKind;

};

}   // namespace orc
}   // namespace cudf

#endif // __ORC_BUFFERS_HEADER__
