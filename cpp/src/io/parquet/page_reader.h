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

#include "parquet/column_reader.h"
#include "parquet/parquet_types.h"

namespace gdf {
namespace parquet {
namespace internal {

class PageReader : public ::parquet::PageReader {
public:
    PageReader(std::unique_ptr<::parquet::InputStream> stream,
               std::int64_t                            total_num_rows,
               ::parquet::Compression::type            codec,
               arrow::MemoryPool *                     pool);

    std::shared_ptr<::parquet::Page>
    NextPage() final;

    void
    set_max_page_header_size(std::uint32_t size) override;

private:
    static const std::uint32_t kDefaultMaxPageHeaderSize = 16 * 1024 * 1024;
    static const std::uint32_t kDefaultPageHeaderSize    = 16 * 1024;

    std::unique_ptr<::parquet::InputStream> stream_;

    ::parquet::format::PageHeader    current_page_header_;
    std::shared_ptr<::parquet::Page> current_page_;

    std::unique_ptr<arrow::util::Codec>     decompressor_;
    std::shared_ptr<arrow::ResizableBuffer> decompression_buffer_;

    std::uint32_t max_page_header_size_;

    std::int64_t seen_num_rows_;

    std::int64_t total_num_rows_;
};

}  // namespace internal
}  // namespace parquet
}  // namespace gdf
