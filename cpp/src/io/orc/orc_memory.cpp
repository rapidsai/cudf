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

#include "orc_memory.h"
#include "orc_debug.h"
#include "orc_util.hpp"

#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include "orc_gzip_stream.h"

namespace OrcMem
{
    static size_t blockSize = 2 * 1024 * 1024;

    void InitializeOrcMem(OrcMemArgument& arg)
    {
        // future option
    };

    int GetBlockSize()
    {
        return blockSize;
    };

    // return null if there is no free block, thread safe API
    orc_byte* GetAvailableManagedBlock()
    {
        // for now, allocate on demand
        orc_byte* ret = NULL;
        CudaFuncCall( cudaMallocManaged(reinterpret_cast<void**>(&ret), blockSize) );

        return ret;
    }

    // free the block to the pool, thread safe API
    void FreeManagedBlock(const orc_byte* block)
    {
        CudaFuncCall(cudaFree(const_cast<orc_byte*>(block)));
    }
}

void OrcBufferInfoHolder::ConstructOrcBufferArray(OrcBufferArray& array, std::vector<OrcBuffer>& bufs)
{
    int array_buf_size = bufs.size() * sizeof(OrcBuffer);
    orc_byte* array_buf;

    // request managed memory
    RequestBuffer(&array_buf, array_buf_size);

    // copy bufs into managed memory
    memcpy(array_buf, bufs.data(), bufs.size() * array_buf_size);

    array.numBuffers = bufs.size();
    array.buffers = reinterpret_cast<OrcBuffer*>(array_buf);
}

CudaOrcError_t OrcBufferInfoHolder::RequestBuffer(orc_byte** dest, size_t size)
{
    //    size = ((size + 15) & ~0xf ); // align to 16 bytes
    size = ((size + 7) & ~0x7 ); // align to 8 bytes

    if (current.buffer == NULL || offset + size >= current.bufferSize)
    {   // allocate a new buffer if there is no buffer or the available buffer size is less than requested
        int constant_block_size = OrcMem::GetBlockSize();

        if (size >= constant_block_size)
        {   // newly allocate a new block with requested size if it is bigger than constant_block_size.
            CudaFuncCall(cudaMallocManaged(reinterpret_cast<void**>(dest), size));

            OrcBuffer buf;
            buf.buffer = *dest;
            buf.bufferSize = size;
            AddBuffer(buf);

            return CudaOrcError_t::GDF_ORC_SUCCESS;
        }

        // allocate a new block with constant size if current buffer is not allocated or available memory is less than requested.
        current.buffer = OrcMem::GetAvailableManagedBlock();
        current.bufferSize = OrcMem::GetBlockSize();
        AddBuffer(current);
        offset = 0;
    }

    *dest = reinterpret_cast<orc_byte*>(current.buffer + offset);
    offset += size;
    return CudaOrcError_t::GDF_ORC_SUCCESS;
}


bool
OrcChunkDecoder::IsSingleUncompressed(const orc_byte* input, size_t input_size)
{
    const chunk_header *ch = reinterpret_cast<const chunk_header*>(input);

    // return true if isOriginal == true && input_size - 3 == ch->getSize()
    if (ch->isOriginal()) {
        if (input_size - 3 == ch->getSize())return true;
    }

    return false;
}


CudaOrcError_t
OrcChunkDecoder::Decode(std::vector<OrcBuffer>& bufs, OrcBufferInfoHolder* holder, 
    const orc_byte* cpu_input, const orc_byte* gpu_input, size_t input_size)
{
    OrcBuffer theBuf;

    size_t offset = 0;
    while (input_size > offset)
    {
        if (input_size - offset < 3) {
            D_MSG("*** invalid chunk size or stream size. chunk header error");
            return GDF_ORC_INVALID_FILE_FORMAT;
        }

        const chunk_header *ch = reinterpret_cast<const chunk_header*>(cpu_input + offset);
        offset += 3;
        if (input_size - offset < ch->getSize()) {
            D_MSG("*** invalid chunk size or stream size. chunk size error");
            return GDF_ORC_INVALID_FILE_FORMAT;
        }

        if (ch->isOriginal()) {
            theBuf.bufferSize = ch->getSize();
            theBuf.buffer = const_cast<orc_byte*>(gpu_input) + offset;

            bufs.push_back(theBuf);
        }
        else {
            // do decode by cpu
            switch (compKind) {
            case OrcCompressionKind::ZLIB:
            {
                // decode using OrcZlibInputStream class
                google::protobuf::io::ArrayInputStream astream(reinterpret_cast<const void*>(cpu_input + offset), ch->getSize());
                google::protobuf::io::OrcZlibInputStream gstream(&astream, 
                    google::protobuf::io::OrcZlibInputStream::ZLIB, 4 * 256 * 1024);

                int block_size = OrcMem::GetBlockSize();
                bool is_stream_end = false;

                do {
#if 0               // this won't work, since gstream.Next will return internal CPU buffer address.
                    theBuf.buffer = OrcMem::GetAvailableManagedBlock();
                    if (!theBuf.buffer)return GDF_ORC_OUT_OF_MEMORY;

                    int buf_size = block_size;
                    is_stream_end = gstream.Next(const_cast<const void**>( reinterpret_cast<void**>(&theBuf.buffer) ), &buf_size);
                    theBuf.bufferSize = buf_size;
                    bufs.push_back(theBuf);
#else
                    // this is workaround
                    // ToDo: more efficient implementation.
                    int buf_size = block_size;
                    const void* buf = NULL;
                    is_stream_end = gstream.Next(const_cast<const void**>(&buf), &buf_size);

                    if (buf_size > block_size) {
                        D_EXIT("buf_size > block_size");
                    }

                    holder->RequestBuffer(&theBuf.buffer, buf_size);

                    memcpy(theBuf.buffer, buf, buf_size);
                    theBuf.bufferSize = buf_size;
                    bufs.push_back(theBuf);
#endif
                } while (!is_stream_end);

                break;
            }
            default:
            case OrcCompressionKind::NONE:
            case OrcCompressionKind::SNAPPY:
            case OrcCompressionKind::LZO:
            case OrcCompressionKind::LZ4:
            case OrcCompressionKind::ZSTD:
                D_MSG("*** the compression kind is not supported yet.");
                return GDF_ORC_UNSUPPORTED_COMPRESSION_TYPE;
                break;
            }
        }
        
        offset += ch->getSize();
    }

    return GDF_ORC_SUCCESS;
}



#if 0
// this is tuning option
/** ---------------------------------------------------------------------------*
* @brief memory manager for temporary cuda managed memories.
* ---------------------------------------------------------------------------**/
class OrcMemoryPool {
public:
    OrcMemoryPool() {};
    ~OrcMemoryPool() { ReleasePools(); };

    // set byte size of block
    void SetBlockSize(size_t block_size_) { block_size = block_size_; };

    // allocate memory pools
    bool AddAllocatedPool(size_t count);

    // release memory pools allocated by AllocatePool().
    void ReleasePools();

public:
    // return null if there is no free block, thread safe API
    orc_byte* GetAvailableBlock();

    // free the block to the pool, thread safe API
    void FreeBlock(orc_byte* block);


protected:
    size_t block_size;
    std::vector<orc_byte*> memoryPools;

    std::vector<orc_byte*> freePools;

};

#endif
