/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
 *     Copyright 2018 William Malpica <william@blazingdb.com>
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
#ifndef GDF_ARROW_UTIL_RLE_DECODER_H
#define GDF_ARROW_UTIL_RLE_DECODER_H

#include "bit-stream.h"
#include "cu_decoder.cuh"
#include <arrow/util/bit-stream-utils.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>

namespace parquet {
class ColumnDescriptor;
}

namespace gdf {
namespace arrow {
namespace internal {

/// Decoder class for RLE encoded data.
class RleDecoder {
public:
    /// Create a decoder object. buffer/buffer_len is the decoded data.
    /// bit_width is the width of each value (before encoding).
    RleDecoder(const uint8_t *buffer, int buffer_len, int bit_width)
      : bit_reader_(buffer, buffer_len), bit_width_(bit_width),
        current_value_(0), repeat_count_(0), literal_count_(0) {
        DCHECK_GE(bit_width_, 0);
        DCHECK_LE(bit_width_, 64);
    }

    RleDecoder() : bit_width_(-1) {}

    void Reset(const uint8_t *buffer, int buffer_len, int bit_width) {
        DCHECK_GE(bit_width, 0);
        DCHECK_LE(bit_width, 64);
        bit_reader_.Reset(buffer, buffer_len);
        bit_width_     = bit_width;
        current_value_ = 0;
        repeat_count_  = 0;
        literal_count_ = 0;
    }

    /// Gets the next value.  Returns false if there are no more.
    template <typename T>
    bool Get(T *val);

    /// Gets a batch of values.  Returns the number of decoded elements.
    template <typename T>
    int GetBatch(T *values, int batch_size);

    /// Like GetBatch but the values are then decoded using the provided
    /// dictionary
    template <typename T>
    int GetBatchWithDict(const T *dictionary,
                         int      num_dictionary_values,
                         T *      values,
                         int      batch_size);

    /// Like GetBatchWithDict but add spacing for null entries
    template <typename T>
    int GetBatchWithDictSpaced(const T *      dictionary,
                               int            num_dictionary_values,
                               T *            values,
                               int            batch_size,
                               int            null_count,
                               const uint8_t *valid_bits,
                               int64_t        valid_bits_offset);

protected:
    BitReader bit_reader_;
    /// Number of bits needed to encode the value. Must be between 0 and 64.
    int      bit_width_;
    uint64_t current_value_;
    uint32_t repeat_count_;
    uint32_t literal_count_;

private:
    /// Fills literal_count_ and repeat_count_ with next values. Returns false if
    /// there
    /// are no more.
    template <typename T>
    bool NextCounts();
};

template <typename T>
inline bool RleDecoder::Get(T *val) {
    return GetBatch(val, 1) == 1;
}

template <typename T>
inline int RleDecoder::GetBatch(T *values, int batch_size) {
    DCHECK_GE(bit_width_, 0);
    int values_read = 0;

    std::vector<uint32_t>                       rleRuns;
    std::vector<uint64_t>                       rleValues;
    int                                         numRle;
    int                                         numBitpacked;
    std::vector<std::pair<uint32_t, uint32_t> > bitpackset;
    std::vector<int> unpack32InputOffsets, unpack32InputRunLengths,
      unpack32OutputOffsets;
    std::vector<int> remainderInputOffsets, remainderBitOffsets,
      remainderSetSize, remainderOutputOffsets;

    while (values_read < batch_size) {
        if (repeat_count_ > 0) {
            int repeat_batch = std::min(batch_size - values_read,
                                        static_cast<int>(repeat_count_));
            rleRuns.push_back(repeat_batch);
            rleValues.push_back(current_value_);

            repeat_count_ -= repeat_batch;
            values_read += repeat_batch;
        } else if (literal_count_ > 0) {
            int literal_batch = std::min(batch_size - values_read,
                                         static_cast<int>(literal_count_));
            rleRuns.push_back(literal_batch);
            rleValues.push_back(0);

            bit_reader_.SetGpuBatchMetadata(bit_width_,
                                            values + values_read,
                                            literal_batch,
                                            values_read,
                                            unpack32InputOffsets,
                                            unpack32InputRunLengths,
                                            unpack32OutputOffsets,
                                            remainderInputOffsets,
                                            remainderBitOffsets,
                                            remainderSetSize,
                                            remainderOutputOffsets);

            literal_count_ -= literal_batch;
            values_read += literal_batch;
        } else {
            if (!NextCounts<T>()) return values_read;
        }
    }
    gdf::arrow::internal::decode_def_levels(this->bit_reader_.get_buffer(),
                                            this->bit_reader_.get_buffer_len(),
                                            rleRuns,
                                            rleValues,
                                            unpack32InputOffsets,
                                            unpack32InputRunLengths,
                                            unpack32OutputOffsets,
                                            remainderInputOffsets,
                                            remainderBitOffsets,
                                            remainderSetSize,
                                            remainderOutputOffsets,
                                            bit_width_,
                                            values,
                                            batch_size);

    return values_read;
}

template <typename T>
inline int RleDecoder::GetBatchWithDict(const T *dictionary,
                                        int      num_dictionary_values,
                                        T *      values,
                                        int      batch_size) {
    DCHECK_GE(bit_width_, 0);
    int values_read = 0;

    std::vector<uint32_t> rleRuns;
    std::vector<uint64_t> rleValues;
    int                   numRle;
    int                   numBitpacked;
    std::vector<int>      unpack32InputOffsets, unpack32InputRunLengths,
      unpack32OutputOffsets;
    std::vector<int> remainderInputOffsets, remainderBitOffsets,
      remainderSetSize, remainderOutputOffsets;

    while (values_read < batch_size) {
        if (repeat_count_ > 0) {
            int repeat_batch = std::min(batch_size - values_read,
                                        static_cast<int>(repeat_count_));
            rleRuns.push_back(repeat_batch);
            rleValues.push_back(current_value_);
            numRle++;

            repeat_count_ -= repeat_batch;
            values_read += repeat_batch;
        } else if (literal_count_ > 0) {
            int literal_batch = std::min(batch_size - values_read,
                                         static_cast<int>(literal_count_));

            const int buffer_size =
              1024;  //@todo, check this buffer size for optimization
            int indices[buffer_size];
            literal_batch = std::min(literal_batch, buffer_size);
            rleRuns.push_back(literal_batch);
            rleValues.push_back(0);
            numBitpacked++;
            bit_reader_.SetGpuBatchMetadata(bit_width_,
                                            &indices[0],
                                            literal_batch,
                                            values_read,
                                            unpack32InputOffsets,
                                            unpack32InputRunLengths,
                                            unpack32OutputOffsets,
                                            remainderInputOffsets,
                                            remainderBitOffsets,
                                            remainderSetSize,
                                            remainderOutputOffsets);
            literal_count_ -= literal_batch;
            values_read += literal_batch;
        } else {
            if (!NextCounts<T>()) return values_read;
        }
    }
    int actual_read =
      gdf::arrow::internal::decode_using_gpu(dictionary,
                                             num_dictionary_values,
                                             values,
                                             this->bit_reader_.get_buffer(),
                                             this->bit_reader_.get_buffer_len(),
                                             rleRuns,
                                             rleValues,
                                             unpack32InputOffsets,
                                             unpack32InputRunLengths,
                                             unpack32OutputOffsets,
                                             remainderInputOffsets,
                                             remainderBitOffsets,
                                             remainderSetSize,
                                             remainderOutputOffsets,
                                             bit_width_,
                                             batch_size);

    return values_read;
}

template <typename T>
inline int RleDecoder::GetBatchWithDictSpaced(const T *dictionary,
                                              int      num_dictionary_values,
                                              T *      values,
                                              int      batch_size,
                                              int      null_count,
                                              const uint8_t *valid_bits,
                                              int64_t valid_bits_offset) {
    DCHECK_GE(bit_width_, 0);

    int values_read =
      GetBatchWithDict(dictionary, num_dictionary_values, values, batch_size);

    return values_read;
}

template <typename T>
inline bool RleDecoder::NextCounts() {
    // Read the next run's indicator int, it could be a literal or repeated run.
    // The int is encoded as a vlq-encoded value.
    int32_t indicator_value = 0;
    bool    result          = bit_reader_.GetVlqInt(&indicator_value);
    if (!result) return false;

    // lsb indicates if it is a literal run or repeated run
    bool is_literal = indicator_value & 1;
    if (is_literal) {
        literal_count_ = (indicator_value >> 1) * 8;
    } else {
        repeat_count_ = indicator_value >> 1;
        bool result   = bit_reader_.GetAligned<T>(
          static_cast<int>(::arrow::BitUtil::CeilDiv(bit_width_, 8)),
          reinterpret_cast<T *>(&current_value_));
        DCHECK(result);
    }
    return true;
}

}  // namespace internal
}  // namespace arrow
}  // namespace gdf
#endif
