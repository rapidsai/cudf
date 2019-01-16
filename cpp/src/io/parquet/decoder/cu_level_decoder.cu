//
// Created by aocsa on 8/25/18.
//

#include "arrow/util/rle-encoding.h"
#include <arrow/util/bit-util.h>
#include "../decoder/arrow/rle_decoder.h"
#include "../decoder/arrow/bit-stream.h"

#include "cu_level_decoder.h"

namespace gdf {
namespace parquet {
namespace decoder {

CUDALevelDecoder::CUDALevelDecoder()
    : num_values_remaining_(0), rle_decoder_(nullptr), bit_packed_decoder_(nullptr)
{
}

CUDALevelDecoder::~CUDALevelDecoder() {}

int CUDALevelDecoder::SetData(::parquet::Encoding::type encoding,
    int16_t max_level, int num_buffered_values,
    const uint8_t* data)
{
    int32_t num_bytes = 0;
    encoding_ = encoding;
    num_values_remaining_ = num_buffered_values;
    bit_width_ = ::arrow::BitUtil::Log2(max_level + 1);
    switch (encoding) {
    case ::parquet::Encoding::RLE: {
        num_bytes = *reinterpret_cast<const int32_t*>(data);
        const uint8_t* decoder_data = data + sizeof(int32_t);
        if (rle_decoder_ == nullptr) {
            rle_decoder_.reset(
                new gdf::arrow::internal::RleDecoder(decoder_data, num_bytes, bit_width_));
        } else {
            rle_decoder_->Reset(decoder_data, num_bytes, bit_width_);
        }
        return sizeof(int32_t) + num_bytes;
    }
    case ::parquet::Encoding::BIT_PACKED: {
        num_bytes = static_cast<int32_t>(
            ::arrow::BitUtil::CeilDiv(num_buffered_values * bit_width_, 8));
        if (!bit_packed_decoder_) {
            bit_packed_decoder_.reset(new gdf::arrow::internal::BitReader(data, num_bytes));
        } else {
            bit_packed_decoder_->Reset(data, num_bytes);
        }
        return num_bytes;
    }
    default:
        throw ::parquet::ParquetException("Unknown encoding type for levels.");
    }
}

int CUDALevelDecoder::Decode(int batch_size, int16_t* d_levels)
{
    int num_decoded = 0;
    int num_values = std::min(num_values_remaining_, batch_size);
    if (encoding_ == ::parquet::Encoding::RLE) {
        num_decoded = rle_decoder_->GetBatch(d_levels, num_values);
    } else {
        // num_decoded = bit_packed_decoder_->GetBatch(bit_width_, d_levels, num_values);
        int literal_batch = num_values;
        int values_read = 0;
        std::vector<uint32_t> rleRuns;
        std::vector<uint64_t> rleValues;
        int numRle;
        int numBitpacked;
        std::vector<int> unpack32InputOffsets, unpack32InputRunLengths, unpack32OutputOffsets;
        std::vector<int> remainderInputOffsets, remainderBitOffsets, remainderSetSize,
                remainderOutputOffsets;

        bit_packed_decoder_->SetGpuBatchMetadata(
                1, d_levels, literal_batch, values_read, unpack32InputOffsets, unpack32InputRunLengths,
                unpack32OutputOffsets, remainderInputOffsets, remainderBitOffsets,
                remainderSetSize, remainderOutputOffsets);

        num_decoded = gdf::arrow::internal::unpack_using_gpu<int16_t> (
                bit_packed_decoder_->get_buffer(), bit_packed_decoder_->get_buffer_len(),
                unpack32InputOffsets,
				unpack32InputRunLengths,
                unpack32OutputOffsets,
                remainderInputOffsets, remainderBitOffsets, remainderSetSize,
                remainderOutputOffsets, bit_width_, d_levels, literal_batch);
    }
    num_values_remaining_ -= num_decoded;
    return num_decoded;
}

} // namespace decoder
} // namespace parquet
} // namespace gdf
