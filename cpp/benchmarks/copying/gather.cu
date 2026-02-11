/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/wide_gather.cuh>

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/reverse.h>
#include <thrust/shuffle.h>

#include <nvbench/nvbench.cuh>

std::unique_ptr<cudf::table> gather_bitwise_nonnull(cudf::table_view const& table,
                                                    cudf::size_type const* indices,
                                                    cudf::size_type size,
                                                    rmm::cuda_stream_view stream)
{
  // -  assert that the type is a linear bitwise type
  // - use the new bitwise gather kernel
  // -
  std::vector<void*> h_src;
  std::vector<void*> h_dst;
  std::vector<std::unique_ptr<cudf::column>> dst_columns;
  cudf::size_type num_b32 = 0;
  cudf::size_type num_b64 = 0;

  for (auto& col : table) {
    CUDF_EXPECTS(col.type() == cudf::data_type{cudf::type_id::INT32} ||
                   col.type() == cudf::data_type{cudf::type_id::INT64},
                 "gather_bitwise_nonnull only supports INT32 and INT64 columns");

    if (col.type() == cudf::data_type{cudf::type_id::INT32}) {
      h_src.push_back((void*)col.data<int32_t>());
      num_b32++;
    } else if (col.type() == cudf::data_type{cudf::type_id::INT64}) {
      h_src.push_back((void*)col.data<int64_t>());
      num_b64++;
    }

    auto out =
      cudf::make_fixed_width_column(col.type(), size, cudf::mask_state::UNALLOCATED, stream);
    h_dst.push_back( (void *) out->view().head<void>());
    dst_columns.push_back(std::move(out));
  }

  rmm::device_uvector<void*> d_src(h_src.size(), stream);
  rmm::device_uvector<void*> d_dst(h_dst.size(), stream);

  // copy source pointers to device
  cudf::detail::cuda_memcpy_async_impl(d_src.data(),
                                       (void const*)h_src.data(),
                                       h_src.size() * sizeof(void*),
                                       cudf::detail::host_memory_kind::PAGEABLE,
                                       stream);

  cudf::detail::cuda_memcpy_async_impl(d_dst.data(),
                                       (void const*)h_dst.data(),
                                       h_dst.size() * sizeof(void*),
                                       cudf::detail::host_memory_kind::PAGEABLE,
                                       stream);

  cudf::detail::byte_gather_params params{.indices  = indices,
                                          .src      = (const void* const*)d_src.data(),
                                          .dst      = (void* const*)d_dst.data(),
                                          .size     = size,
                                          .num_b8   = 0,
                                          .num_b16  = 0,
                                          .num_b32  = num_b32,
                                          .num_b64  = num_b64,
                                          .num_b128 = 0};

  int32_t grid_size = 0;
  int32_t block_size = 0;

  cudaOccupancyMaxPotentialBlockSize(
    &grid_size, &block_size, &cudf::detail::wide_byte_gather_kernel_generic);

  cudf::detail::wide_byte_gather_kernel_generic<<<grid_size, block_size, 0, stream.value()>>>(
    params);

  return std::make_unique<cudf::table>(std::move(dst_columns));
}

static void bench_gather(nvbench::state& state)
{
  auto const num_rows        = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols        = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const coalesce        = static_cast<bool>(state.get_int64("coalesce"));
  auto const use_wide_gather = static_cast<bool>(state.get_int64("use_wide_gather"));

  // Gather indices
  auto gather_map_table =
    create_sequence_table({cudf::type_to_id<cudf::size_type>()}, row_count{num_rows});
  auto gather_map = gather_map_table->get_column(0).mutable_view();

  if (coalesce) {
    thrust::reverse(
      thrust::device, gather_map.begin<cudf::size_type>(), gather_map.end<cudf::size_type>());
  } else {
    thrust::shuffle(thrust::device,
                    gather_map.begin<cudf::size_type>(),
                    gather_map.end<cudf::size_type>(),
                    thrust::default_random_engine());
  }

  // Every element is valid
  auto source_table = create_sequence_table(cycle_dtypes({cudf::type_to_id<int64_t>()}, num_cols),
                                            row_count{num_rows});

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_global_memory_reads<int8_t>(source_table->alloc_size());
  state.add_global_memory_writes<int8_t>(source_table->alloc_size());

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    if (use_wide_gather) {
      gather_bitwise_nonnull(*source_table, gather_map.data<cudf::size_type>(), num_rows, stream);
    } else {
      cudf::gather(*source_table, gather_map);
    }
  });
}

NVBENCH_BENCH(bench_gather)
  .set_name("gather")
  .add_int64_axis("num_rows", {64, 512, 4096, 32768, 262144, 2097152})
  .add_int64_axis("num_cols", {1, 8, 32, 128, 128 *4, 128 * 16, 128 * 32})
  .add_int64_axis("coalesce", {true, false})
  .add_int64_axis("use_wide_gather", {true, false});
