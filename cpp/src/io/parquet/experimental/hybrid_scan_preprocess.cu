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
#include "io/parquet/parquet_gpu.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/batched_memset.hpp>
#include <cudf/detail/utilities/functional.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/unique.h>

#include <bitset>
#include <limits>
#include <numeric>

namespace cudf::experimental::io::parquet::detail {

using chunk_page_info        = cudf::io::parquet::detail::chunk_page_info;
using ColumnChunkDesc        = cudf::io::parquet::detail::ColumnChunkDesc;
using Compression            = cudf::io::parquet::Compression;
using decode_error           = cudf::io::parquet::detail::decode_error;
using Encoding               = cudf::io::parquet::Encoding;
using kernel_error           = cudf::io::parquet::kernel_error;
using level_type             = cudf::io::parquet::detail::level_type;
using LogicalType            = cudf::io::parquet::LogicalType;
using PageInfo               = cudf::io::parquet::detail::PageInfo;
using PageNestingDecodeInfo  = cudf::io::parquet::detail::PageNestingDecodeInfo;
using PageNestingInfo        = cudf::io::parquet::detail::PageNestingInfo;
using pass_intermediate_data = cudf::io::parquet::detail::pass_intermediate_data;
using SchemaElement          = cudf::io::parquet::SchemaElement;
using string_index_pair      = cudf::io::parquet::detail::string_index_pair;
using Type                   = cudf::io::parquet::Type;

void impl::prepare_row_groups(cudf::host_span<std::vector<size_type> const> row_group_indices,
                              cudf::io::parquet_reader_options const& options)
{
}

void impl::allocate_level_decode_space() {}

void impl::build_string_dict_indices() {}

bool impl::setup_column_chunks() { return {}; }

void impl::preprocess_subpass_pages(size_t chunk_read_limit) {}

cudf::detail::host_vector<size_t> impl::calculate_page_string_offsets()
{
  return cudf::detail::make_host_vector<size_t>(0, _stream);
}

void impl::update_row_mask(cudf::column_view in_row_mask,
                           cudf::mutable_column_view out_row_mask,
                           rmm::cuda_stream_view stream)
{
}

}  // namespace cudf::experimental::io::parquet::detail
