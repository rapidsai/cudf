//
// Created by aocsa on 8/25/18.
//

#ifndef _GDF_PARQUET_CUDALEVELDECODER_H
#define _GDF_PARQUET_CUDALEVELDECODER_H

#include "arrow/bit-stream.h"
#include "arrow/rle_decoder.h"
#include "parquet/types.h"
#include <parquet/column_reader.h>

namespace gdf {
namespace parquet {
namespace decoder {

class CUDALevelDecoder {
public:
    CUDALevelDecoder();

    ~CUDALevelDecoder();

    // Initialize the LevelDecoder state with new data
    // and return the number of bytes consumed
    int SetData(::parquet::Encoding::type encoding,
                int16_t                   max_level,
                int                       num_buffered_values,
                const uint8_t *           data);

    // Decodes a batch of levels into an array and returns the number of levels
    // decoded
    int Decode(int batch_size, int16_t *levels);

private:
    int                                               bit_width_;
    int                                               num_values_remaining_;
    ::parquet::Encoding::type                         encoding_;
    std::unique_ptr<gdf::arrow::internal::RleDecoder> rle_decoder_;
    std::unique_ptr<gdf::arrow::internal::BitReader>  bit_packed_decoder_;
};
}  // namespace decoder
}  // namespace parquet
}  // namespace gdf

#endif  //_GDF_PARQUET_CUDALEVELDECODER_H
