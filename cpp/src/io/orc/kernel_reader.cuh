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

#ifndef __ORC_KERNEL_READER_H__
#define __ORC_KERNEL_READER_H__

#include "kernel_private_common.cuh"

namespace cudf {
namespace orc {

template <class  T>
class byte_reader {
public:
    __device__ byte_reader(const T* input, size_t size_)
        : top(input), size(size_), local_offset(0)
    {};

    __device__ ~byte_reader() {};

    __device__ T getLocal(size_t offset) {
        const T *val = top + offset + local_offset;

        return *val;
    };

    __device__ bool end() {
        return (local_offset >= size);
    };

    __device__ void add_offset(int count) {
        local_offset += count;
    };

    __device__ int get_read_count() {
        return local_offset;
    };


protected:
    const T* top;
    size_t size;

    size_t local_offset;
};

// -----------------------------------------------------------------
// @brief a reader class just wrapping byte_reader
// -----------------------------------------------------------------
class data_reader_single_buffer : public byte_reader<orc_byte> {
public:
    __device__ data_reader_single_buffer(const KernelParamBase* kernParam)
        : byte_reader<orc_byte>(kernParam->input, kernParam->input_size)
    {};
    __device__ ~data_reader_single_buffer() {};

};

// -----------------------------------------------------------------
// @brief a reader class to support OrcBufferArray for CPU decoded buffer array
// -----------------------------------------------------------------
class data_reader_multi_buffer {
public:
    __device__ data_reader_multi_buffer(const KernelParamBase* kernParam)
        :   buffer_array(& kernParam->bufferArray),
        orcBufferID(0),
        the_read_count(0),
        local_offset(0),
        is_end(false)
    {
        set_current(0);
    };
    __device__ ~data_reader_multi_buffer() {};

public:
    __device__ orc_byte getLocal(size_t offset) {
        if (is_local_range(offset)) {
            const orc_byte *val = current.buffer + local_offset + offset;
            return *val;
        }
        else {  // if out of local buffer's range
            int id;
            size_t new_offset;
            get_over_range_position(offset, id, new_offset);

            return buffer_array->buffers[id].buffer[new_offset];
        }
    };

    __device__ bool end() {
        return is_end;
    };

    __device__ void add_offset(int count) {
        the_read_count += count;

        if (is_local_range(count))
        {
            local_offset += count;
        }
        else {
            int now_id = orcBufferID;
            get_over_range_position(count, orcBufferID, local_offset);
            if (now_id != orcBufferID) {
                set_current(orcBufferID);
            }
        }
    };

    __device__ int get_read_count() {
        return the_read_count;
    };

protected:
    __device__ bool is_local_range(size_t offset) {
        return (local_offset + offset < current.bufferSize) ? true : false;
    }

    __device__ bool get_over_range_position(size_t offset, int& id, size_t& new_offset)
    {
        int next_offset = local_offset + offset;
        int buf_id = orcBufferID;

        do {
            next_offset -= buffer_array->buffers[buf_id].bufferSize;
            buf_id++;
            if (buf_id >= buffer_array->numBuffers) {

                is_end = true;
                return false;
            }

        } while (next_offset > buffer_array->buffers[buf_id].bufferSize);

        id = buf_id;
        new_offset = next_offset;

        return true;
    }

    __device__ void set_current(int id)
    {
        if (id >= buffer_array->numBuffers)
        {
            is_end = true;
        }
        current.buffer      = buffer_array->buffers[id].buffer;
        current.bufferSize  = buffer_array->buffers[id].bufferSize;
    }


protected:
    const OrcBufferArray* buffer_array;

    OrcBuffer           current;        // the copy of current OrcBuffer
    size_t              local_offset;   // the local offset from current.buffer
    int                 orcBufferID;    // the current buffer ID for OrcBuffer in OrcBufferArray.
    size_t              the_read_count; // the accumulated read byte count 

    bool                is_end;
};

// -----------------------------------------------------------------
// @brief a reader class to support gpu decoded buffers
// not implemented yet
// -----------------------------------------------------------------
class data_reader_decoding {
public:
    __device__ data_reader_decoding(const KernelParamBase* kernParam)

    {};
    __device__ ~data_reader_decoding() {};

public:

    // this method must be called at same time by all threads.
    __device__ void add_offset(int count) {
        
        // if count is over the current range, this class discards old buffer and decode next buffer.
    };

protected:
    __device__ bool decode_next_range() {
        return false;
    };

};


// -----------------------------------------------------------------
// @brief a hub class for data input classes:
//  data_reader_single_buffer, data_reader_multi_buffer, data_reader_decoding
// -----------------------------------------------------------------
template <class  T_reader_input>
class stream_reader {
public:
    __device__ stream_reader(const KernelParamBase* kernParam)
        : reader(kernParam)
    {};

    __device__ ~stream_reader() {};

    __device__ orc_byte getLocal(size_t offset) {
        return reader.getLocal(offset);
    };

    __device__ bool end() {
        return reader.end();
    };

    __device__ void add_offset(int count) {
        reader.add_offset(count);
    };

    __device__ int get_read_count() {
        return reader.get_read_count();
    };

protected:
    T_reader_input        reader;
};

}   // namespace orc
}   // namespace cudf

#endif // __ORC_KERNEL_READER_H__
