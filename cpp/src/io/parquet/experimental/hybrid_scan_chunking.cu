/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "hybrid_scan_helpers.hpp"
#include "hybrid_scan_impl.hpp"
#include "io/comp/gpuinflate.hpp"
#include "io/comp/io_uncomp.hpp"
#include "io/comp/nvcomp_adapter.hpp"
#include "io/parquet/compact_protocol_reader.hpp"
#include "io/parquet/reader_impl_chunking.hpp"
#include "io/utilities/time_utils.cuh"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/config_utils.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/logical.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <thrust/unique.h>

#include <numeric>

namespace cudf::experimental::io::parquet::detail {

namespace nvcomp = cudf::io::detail::nvcomp;

using compression_result    = cudf::io::detail::compression_result;
using compression_status    = cudf::io::detail::compression_status;
using compression_type      = cudf::io::compression_type;
using ColumnChunkDesc       = cudf::io::parquet::detail::ColumnChunkDesc;
using CompactProtocolReader = cudf::io::parquet::detail::CompactProtocolReader;
using Compression           = cudf::io::parquet::Compression;
using level_type            = cudf::io::parquet::detail::level_type;
using LogicalType           = cudf::io::parquet::LogicalType;
using PageInfo              = cudf::io::parquet::detail::PageInfo;
using Type                  = cudf::io::parquet::Type;

void impl::create_global_chunk_info(cudf::io::parquet_reader_options const& options) {}

void impl::compute_input_passes() {}

void impl::compute_output_chunks_for_subpass() {}

void impl::handle_chunking(std::vector<rmm::device_buffer> column_chunk_buffers,
                           cudf::io::parquet_reader_options const& options)
{
}

void impl::setup_next_pass(std::vector<rmm::device_buffer> column_chunk_buffers,
                           cudf::io::parquet_reader_options const& options)
{
}

void impl::setup_next_subpass(cudf::io::parquet_reader_options const& options) {}

}  // namespace cudf::experimental::io::parquet::detail
