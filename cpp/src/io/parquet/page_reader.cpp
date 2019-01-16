/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
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

#include "page_reader.h"

#include <thrift/protocol/TCompactProtocol.h>
#include <thrift/transport/TBufferTransports.h>

#include <arrow/util/compression.h>

namespace gdf {
namespace parquet {
namespace internal {

namespace {
template <class T>
inline void
DeserializeThriftMsg(const std::uint8_t *buf,
                     std::uint32_t *     len,
                     T *                 deserialized_msg) {
    std::shared_ptr<apache::thrift::transport::TMemoryBuffer> tmem_transport(
      new apache::thrift::transport::TMemoryBuffer(
        const_cast<std::uint8_t *>(buf), *len));
    apache::thrift::protocol::TCompactProtocolFactoryT<
      apache::thrift::transport::TMemoryBuffer>
                                                         tproto_factory;
    std::shared_ptr<apache::thrift::protocol::TProtocol> tproto =
      tproto_factory.getProtocol(tmem_transport);

    try {
        deserialized_msg->read(tproto.get());
    } catch (std::exception &e) {
        std::stringstream ss;
        ss << "Couldn't deserialize thrift: " << e.what() << "\n";
        throw ::parquet::ParquetException(ss.str());
    }

    std::uint32_t bytes_left = tmem_transport->available_read();

    *len = *len - bytes_left;
}

static inline ::parquet::Encoding::type
FromThrift(::parquet::format::Encoding::type type) {
    return static_cast<::parquet::Encoding::type>(type);
}
}  // namespace

PageReader::PageReader(std::unique_ptr<::parquet::InputStream> stream,
                       std::int64_t                            total_num_rows,
                       ::parquet::Compression::type            codec,
                       arrow::MemoryPool *                     pool)
  : stream_(std::move(stream)),
    decompression_buffer_(::parquet::AllocateBuffer(pool, 0)),
    seen_num_rows_(0), total_num_rows_(total_num_rows) {
    max_page_header_size_ = kDefaultMaxPageHeaderSize;
    decompressor_         = GetCodecFromArrow(codec);
}

std::shared_ptr<::parquet::Page>
PageReader::NextPage() {
    while (seen_num_rows_ < total_num_rows_) {
        std::int64_t        bytes_read      = 0;
        std::int64_t        bytes_available = 0;
        std::uint32_t       header_size     = 0;
        const std::uint8_t *buffer;
        std::uint32_t       allowed_page_size = kDefaultPageHeaderSize;

        for (;;) {
            buffer = stream_->Peek(allowed_page_size, &bytes_available);
            if (bytes_available == 0) {
                return std::shared_ptr<::parquet::Page>(nullptr);
            }

            header_size = static_cast<std::uint32_t>(bytes_available);
            try {
                DeserializeThriftMsg(
                  buffer, &header_size, &current_page_header_);
                break;
            } catch (std::exception &e) {
                std::stringstream ss;
                ss << e.what();
                allowed_page_size *= 2;
                if (allowed_page_size > max_page_header_size_) {
                    ss << "Deserializing page header failed.\n";
                    throw ::parquet::ParquetException(ss.str());
                }
            }
        }
        stream_->Advance(header_size);

        int compressed_len   = current_page_header_.compressed_page_size;
        int uncompressed_len = current_page_header_.uncompressed_page_size;

        buffer = stream_->Read(compressed_len, &bytes_read);
        if (bytes_read != compressed_len) {
            ::parquet::ParquetException::EofException();
        }

        if (decompressor_ != nullptr) {
            if (uncompressed_len
                > static_cast<int>(decompression_buffer_->size())) {
                PARQUET_THROW_NOT_OK(
                  decompression_buffer_->Resize(uncompressed_len, false));
            }
            PARQUET_THROW_NOT_OK(
              decompressor_->Decompress(compressed_len,
                                        buffer,
                                        uncompressed_len,
                                        decompression_buffer_->mutable_data()));
            buffer = decompression_buffer_->data();
        }

        auto page_buffer =
          std::make_shared<::parquet::Buffer>(buffer, uncompressed_len);

        if (current_page_header_.type
            == ::parquet::format::PageType::DICTIONARY_PAGE) {
            const ::parquet::format::DictionaryPageHeader &dict_header =
              current_page_header_.dictionary_page_header;

            bool is_sorted =
              dict_header.__isset.is_sorted ? dict_header.is_sorted : false;

            return std::make_shared<::parquet::DictionaryPage>(
              page_buffer,
              dict_header.num_values,
              FromThrift(dict_header.encoding),
              is_sorted);
        } else if (current_page_header_.type
                   == ::parquet::format::PageType::DATA_PAGE) {
            const ::parquet::format::DataPageHeader &header =
              current_page_header_.data_page_header;

            ::parquet::EncodedStatistics page_statistics;
            if (header.__isset.statistics) {
                const ::parquet::format::Statistics &stats = header.statistics;
                if (stats.__isset.max) { page_statistics.set_max(stats.max); }
                if (stats.__isset.min) { page_statistics.set_min(stats.min); }
                if (stats.__isset.null_count) {
                    page_statistics.set_null_count(stats.null_count);
                }
                if (stats.__isset.distinct_count) {
                    page_statistics.set_distinct_count(stats.distinct_count);
                }
            }

            seen_num_rows_ += header.num_values;

            return std::make_shared<::parquet::DataPage>(
              page_buffer,
              header.num_values,
              FromThrift(header.encoding),
              FromThrift(header.definition_level_encoding),
              FromThrift(header.repetition_level_encoding),
              page_statistics);
        } else if (current_page_header_.type
                   == ::parquet::format::PageType::DATA_PAGE_V2) {
            const ::parquet::format::DataPageHeaderV2 &header =
              current_page_header_.data_page_header_v2;
            bool is_compressed =
              header.__isset.is_compressed ? header.is_compressed : false;

            seen_num_rows_ += header.num_values;

            return std::make_shared<::parquet::DataPageV2>(
              page_buffer,
              header.num_values,
              header.num_nulls,
              header.num_rows,
              FromThrift(header.encoding),
              header.definition_levels_byte_length,
              header.repetition_levels_byte_length,
              is_compressed);
        } else {
            continue;
        }
    }
    return std::shared_ptr<::parquet::Page>(nullptr);
}

void
PageReader::set_max_page_header_size(std::uint32_t size) {
    max_page_header_size_ = size;
}

}  // namespace internal
}  // namespace parquet
}  // namespace gdf
