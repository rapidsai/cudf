/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reader_impl.hpp"

namespace cudf::io::parquet::detail {

reader::reader() = default;

reader::reader(std::vector<std::unique_ptr<datasource>>&& sources,
               parquet_reader_options const& options,
               rmm::cuda_stream_view stream,
               rmm::device_async_resource_ref mr)
  : _impl(std::make_unique<reader_impl>(std::move(sources), options, stream, mr))
{
}

reader::~reader() = default;

table_with_metadata reader::read() { return _impl->read(); }

chunked_reader::chunked_reader(std::size_t chunk_read_limit,
                               std::size_t pass_read_limit,
                               std::vector<std::unique_ptr<datasource>>&& sources,
                               parquet_reader_options const& options,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  _impl = std::make_unique<reader_impl>(
    chunk_read_limit, pass_read_limit, std::move(sources), options, stream, mr);
}

chunked_reader::~chunked_reader() = default;

bool chunked_reader::has_next() const { return _impl->has_next(); }

table_with_metadata chunked_reader::read_chunk() const { return _impl->read_chunk(); }

}  // namespace cudf::io::parquet::detail
