/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
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

#ifndef _GDF_PARQUET_COLUMN_READER_H
#define _GDF_PARQUET_COLUMN_READER_H

#include "decoder/cu_level_decoder.h"
#include <cudf.h>
#include <parquet/column_reader.h>

namespace gdf {
namespace parquet {

template <class DataType>
class ColumnReader : public ::parquet::ColumnReader {
public:
    using T = typename DataType::c_type;

    ColumnReader(const ::parquet::ColumnDescriptor *     schema,
                 std::unique_ptr< ::parquet::PageReader> pager,
                 ::arrow::MemoryPool *pool = ::arrow::default_memory_pool())
      : ::parquet::ColumnReader(schema, std::move(pager), pool),
        current_decoder_(nullptr) {}

    bool HasNext();

    std::int64_t ReadBatch(std::int64_t  batch_size,
                           std::int16_t *def_levels,
                           std::int16_t *rep_levels,
                           T *           values,
                           std::int64_t *values_read);

    std::int64_t ReadBatchSpaced(std::int64_t  batch_size,
                                 std::int16_t *definition_levels,
                                 std::int16_t *repetition_levels,
                                 T *           values,
                                 std::uint8_t *valid_bits,
                                 std::int64_t  valid_bits_offset,
                                 std::int64_t *levels_read,
                                 std::int64_t *values_read,
                                 std::int64_t *nulls_count);

    /// \brief Append data from column to gdf column
    /// \param[in,out] column with data appended
    /// \param[in] offset of `column` to start append data
    std::size_t ToGdfColumn(const gdf_column &   column,
                            const std::ptrdiff_t offset = 0);

    /// \brief Append data from column to gdf column
    /// \param[in,out] column with data appended
    /// \param[in] offset of `column` to start append data
    /// \param[out] d_definition_levels array with the definition level
    //              of each item appended to `column` from parquet file
    std::size_t ToGdfColumn(const gdf_column &   column,
                            const std::ptrdiff_t offset,
                            std::int16_t *       d_definition_levels);

    std::size_t ToGdfColumn(const gdf_column &   column,
                            const std::ptrdiff_t offset,
                            std::uint8_t &       first_valid_byte,
                            std::uint8_t &       last_valid_byte);

    int64_t ReadDefinitionLevels(int64_t batch_size, int16_t *levels) {
        if (descr_->max_definition_level() == 0) { return 0; }
        return def_level_decoder_.Decode(static_cast<int>(batch_size), levels);
    }

private:
    bool ReadNewPage() final;

    using DecoderType = ::parquet::Decoder<DataType>;

    std::unordered_map<int, std::shared_ptr<DecoderType> > decoders_;
    DecoderType *                                          current_decoder_;
    gdf::parquet::decoder::CUDALevelDecoder                def_level_decoder_;
};

using BoolReader   = ColumnReader< ::parquet::BooleanType>;
using Int32Reader  = ColumnReader< ::parquet::Int32Type>;
using Int64Reader  = ColumnReader< ::parquet::Int64Type>;
using FloatReader  = ColumnReader< ::parquet::FloatType>;
using DoubleReader = ColumnReader< ::parquet::DoubleType>;

}  // namespace parquet
}  // namespace gdf

#endif
