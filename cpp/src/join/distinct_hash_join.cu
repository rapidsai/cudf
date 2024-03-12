/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include "join_common_utils.cuh"
#include "join_common_utils.hpp"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/distinct_hash_join.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/join.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cooperative_groups.h>
#include <cub/block/block_scan.cuh>
#include <cuco/static_set.cuh>
#include <thrust/fill.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

namespace cudf {
namespace detail {
namespace {

static auto constexpr DISTINCT_JOIN_BLOCK_SIZE = 256;

template <cudf::has_nested HasNested>
auto prepare_device_equal(
  std::shared_ptr<cudf::experimental::row::equality::preprocessed_table> build,
  std::shared_ptr<cudf::experimental::row::equality::preprocessed_table> probe,
  bool has_nulls,
  cudf::null_equality compare_nulls)
{
  auto const two_table_equal =
    cudf::experimental::row::equality::two_table_comparator(build, probe);
  return comparator_adapter{two_table_equal.equal_to<HasNested == cudf::has_nested::YES>(
    nullate::DYNAMIC{has_nulls}, compare_nulls)};
}

/**
 * @brief Device functor to create a pair of {hash_value, row_index} for a given row.
 *
 * @tparam Hasher The type of internal hasher to compute row hash.
 */
template <typename Hasher, typename T>
class build_keys_fn {
 public:
  CUDF_HOST_DEVICE build_keys_fn(Hasher const& hash) : _hash{hash} {}

  __device__ __forceinline__ auto operator()(size_type i) const noexcept
  {
    return cuco::pair{_hash(i), T{i}};
  }

 private:
  Hasher _hash;
};

/**
 * @brief Device output transform functor to construct `size_type` with `cuco::pair<hash_value_type,
 * lhs_index_type>`
 */
struct output_fn {
  __device__ constexpr cudf::size_type operator()(
    cuco::pair<hash_value_type, lhs_index_type> const& x) const
  {
    return static_cast<cudf::size_type>(x.second);
  }
};

template <typename Tile>
__device__ void flush_buffer(Tile const& tile,
                             cudf::size_type tile_count,
                             cuco::pair<cudf::size_type, cudf::size_type>* buffer,
                             cudf::size_type* counter,
                             cudf::size_type* build_indices,
                             cudf::size_type* probe_indices)
{
  cudf::size_type offset;
  auto const lane_id = tile.thread_rank();
  if (0 == lane_id) { offset = atomicAdd(counter, tile_count); }
  offset = tile.shfl(offset, 0);

  for (cudf::size_type i = lane_id; i < tile_count; i += tile.size()) {
    auto const& [build_idx, probe_idx] = buffer[i];
    *(build_indices + offset + i)      = build_idx;
    *(probe_indices + offset + i)      = probe_idx;
  }
}

__device__ void flush_buffer(cooperative_groups::thread_block const& block,
                             cudf::size_type buffer_size,
                             cuco::pair<cudf::size_type, cudf::size_type>* buffer,
                             cudf::size_type* counter,
                             cudf::size_type* build_indices,
                             cudf::size_type* probe_indices)
{
  auto i = block.thread_rank();
  __shared__ cudf::size_type offset;

  if (i == 0) { offset = atomicAdd(counter, buffer_size); }
  block.sync();

  while (i < buffer_size) {
    auto const& [build_idx, probe_idx] = buffer[i];
    *(build_indices + offset + i)      = build_idx;
    *(probe_indices + offset + i)      = probe_idx;

    i += block.size();
  }
}

// TODO: custom kernel to be replaced by cuco::static_set::retrieve
template <typename Iter, typename HashTable>
CUDF_KERNEL void distinct_join_probe_kernel(Iter iter,
                                            cudf::size_type n,
                                            HashTable hash_table,
                                            cudf::size_type* counter,
                                            cudf::size_type* build_indices,
                                            cudf::size_type* probe_indices)
{
  namespace cg = cooperative_groups;

  auto constexpr tile_size   = HashTable::cg_size;
  auto constexpr window_size = HashTable::window_size;

  auto idx          = cudf::detail::grid_1d::global_thread_id() / tile_size;
  auto const stride = cudf::detail::grid_1d::grid_stride() / tile_size;
  auto const block  = cg::this_thread_block();

  // CG-based probing algorithm
  if constexpr (tile_size != 1) {
    auto const tile = cg::tiled_partition<tile_size>(block);

    auto constexpr flushing_tile_size = cudf::detail::warp_size / window_size;
    // random choice to tune
    auto constexpr flushing_buffer_size = 2 * flushing_tile_size;
    auto constexpr num_flushing_tiles   = DISTINCT_JOIN_BLOCK_SIZE / flushing_tile_size;
    auto constexpr max_matches          = flushing_tile_size / tile_size;

    auto const flushing_tile    = cg::tiled_partition<flushing_tile_size>(block);
    auto const flushing_tile_id = block.thread_rank() / flushing_tile_size;

    __shared__ cuco::pair<cudf::size_type, cudf::size_type>
      flushing_tile_buffer[num_flushing_tiles][flushing_tile_size];
    // per flushing-tile counter to track number of filled elements
    __shared__ cudf::size_type flushing_counter[num_flushing_tiles];

    if (flushing_tile.thread_rank() == 0) { flushing_counter[flushing_tile_id] = 0; }
    flushing_tile.sync();  // sync still needed since cg.any doesn't imply a memory barrier

    while (flushing_tile.any(idx < n)) {
      bool active_flag = idx < n;
      auto const active_flushing_tile =
        cg::binary_partition<flushing_tile_size>(flushing_tile, active_flag);
      if (active_flag) {
        auto const found = hash_table.find(tile, *(iter + idx));
        if (tile.thread_rank() == 0 and found != hash_table.end()) {
          auto const offset = atomicAdd_block(&flushing_counter[flushing_tile_id], 1);
          flushing_tile_buffer[flushing_tile_id][offset] = cuco::pair{
            static_cast<cudf::size_type>(found->second), static_cast<cudf::size_type>(idx)};
        }
      }

      flushing_tile.sync();
      if (flushing_counter[flushing_tile_id] + max_matches > flushing_buffer_size) {
        flush_buffer(flushing_tile,
                     flushing_counter[flushing_tile_id],
                     flushing_tile_buffer[flushing_tile_id],
                     counter,
                     build_indices,
                     probe_indices);
        flushing_tile.sync();
        if (flushing_tile.thread_rank() == 0) { flushing_counter[flushing_tile_id] = 0; }
        flushing_tile.sync();
      }

      idx += stride;
    }  // while

    if (flushing_counter[flushing_tile_id] > 0) {
      flush_buffer(flushing_tile,
                   flushing_counter[flushing_tile_id],
                   flushing_tile_buffer[flushing_tile_id],
                   counter,
                   build_indices,
                   probe_indices);
    }
  }
  // Scalar probing for CG size 1
  else {
    using block_scan = cub::BlockScan<cudf::size_type, DISTINCT_JOIN_BLOCK_SIZE>;
    __shared__ typename block_scan::TempStorage block_scan_temp_storage;

    auto constexpr buffer_capacity = 2 * DISTINCT_JOIN_BLOCK_SIZE;
    __shared__ cuco::pair<cudf::size_type, cudf::size_type> buffer[buffer_capacity];
    cudf::size_type buffer_size = 0;

    while (idx - block.thread_rank() < n) {  // the whole thread block falls into the same iteration
      auto const found     = idx < n ? hash_table.find(*(iter + idx)) : hash_table.end();
      auto const has_match = found != hash_table.end();

      // Use a whole-block scan to calculate the output location
      cudf::size_type offset;
      cudf::size_type block_count;
      block_scan(block_scan_temp_storage)
        .ExclusiveSum(static_cast<cudf::size_type>(has_match), offset, block_count);

      if (buffer_size + block_count > buffer_capacity) {
        flush_buffer(block, buffer_size, buffer, counter, build_indices, probe_indices);
        block.sync();
        buffer_size = 0;
      }

      if (has_match) {
        buffer[buffer_size + offset] = cuco::pair{static_cast<cudf::size_type>(found->second),
                                                  static_cast<cudf::size_type>(idx)};
      }
      buffer_size += block_count;
      block.sync();

      idx += stride;
    }  // while

    if (buffer_size > 0) {
      flush_buffer(block, buffer_size, buffer, counter, build_indices, probe_indices);
    }
  }
}
}  // namespace

template <cudf::has_nested HasNested>
distinct_hash_join<HasNested>::distinct_hash_join(cudf::table_view const& build,
                                                  cudf::table_view const& probe,
                                                  bool has_nulls,
                                                  cudf::null_equality compare_nulls,
                                                  rmm::cuda_stream_view stream)
  : _has_nulls{has_nulls},
    _nulls_equal{compare_nulls},
    _build{build},
    _probe{probe},
    _preprocessed_build{
      cudf::experimental::row::equality::preprocessed_table::create(_build, stream)},
    _preprocessed_probe{
      cudf::experimental::row::equality::preprocessed_table::create(_probe, stream)},
    _hash_table{build.num_rows(),
                CUCO_DESIRED_LOAD_FACTOR,
                cuco::empty_key{cuco::pair{std::numeric_limits<hash_value_type>::max(),
                                           lhs_index_type{JoinNoneValue}}},
                prepare_device_equal<HasNested>(
                  _preprocessed_build, _preprocessed_probe, has_nulls, compare_nulls),
                {},
                cuco::thread_scope_device,
                cuco_storage_type{},
                cudf::detail::cuco_allocator{stream},
                stream.value()}
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(0 != this->_build.num_columns(), "Hash join build table is empty");

  if (this->_build.num_rows() == 0) { return; }

  auto const row_hasher = experimental::row::hash::row_hasher{this->_preprocessed_build};
  auto const d_hasher   = row_hasher.device_hasher(nullate::DYNAMIC{this->_has_nulls});

  auto const iter = cudf::detail::make_counting_transform_iterator(
    0, build_keys_fn<decltype(d_hasher), lhs_index_type>{d_hasher});

  size_type const build_table_num_rows{build.num_rows()};
  if (this->_nulls_equal == cudf::null_equality::EQUAL or (not cudf::nullable(this->_build))) {
    this->_hash_table.insert_async(iter, iter + build_table_num_rows, stream.value());
  } else {
    auto stencil = thrust::counting_iterator<size_type>{0};
    auto const row_bitmask =
      cudf::detail::bitmask_and(this->_build, stream, rmm::mr::get_current_device_resource()).first;
    auto const pred =
      cudf::detail::row_is_valid{reinterpret_cast<bitmask_type const*>(row_bitmask.data())};

    // insert valid rows
    this->_hash_table.insert_if_async(
      iter, iter + build_table_num_rows, stencil, pred, stream.value());
  }
}

template <cudf::has_nested HasNested>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
distinct_hash_join<HasNested>::inner_join(rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr) const
{
  cudf::scoped_range range{"distinct_hash_join::inner_join"};

  size_type const probe_table_num_rows{this->_probe.num_rows()};

  // If output size is zero, return immediately
  if (probe_table_num_rows == 0) {
    return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  auto build_indices =
    std::make_unique<rmm::device_uvector<size_type>>(probe_table_num_rows, stream, mr);
  auto probe_indices =
    std::make_unique<rmm::device_uvector<size_type>>(probe_table_num_rows, stream, mr);

  auto const probe_row_hasher =
    cudf::experimental::row::hash::row_hasher{this->_preprocessed_probe};
  auto const d_probe_hasher = probe_row_hasher.device_hasher(nullate::DYNAMIC{this->_has_nulls});
  auto const iter           = cudf::detail::make_counting_transform_iterator(
    0, build_keys_fn<decltype(d_probe_hasher), rhs_index_type>{d_probe_hasher});
  auto counter = rmm::device_scalar<cudf::size_type>{stream};
  counter.set_value_to_zero_async(stream);

  cudf::detail::grid_1d grid{probe_table_num_rows, DISTINCT_JOIN_BLOCK_SIZE};
  distinct_join_probe_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
    iter,
    probe_table_num_rows,
    this->_hash_table.ref(cuco::find),
    counter.data(),
    build_indices->data(),
    probe_indices->data());

  auto const actual_size = counter.value(stream);
  build_indices->resize(actual_size, stream);
  probe_indices->resize(actual_size, stream);

  return {std::move(build_indices), std::move(probe_indices)};
}

template <cudf::has_nested HasNested>
std::unique_ptr<rmm::device_uvector<size_type>> distinct_hash_join<HasNested>::left_join(
  rmm::cuda_stream_view stream, rmm::mr::device_memory_resource* mr) const
{
  cudf::scoped_range range{"distinct_hash_join::left_join"};

  size_type const probe_table_num_rows{this->_probe.num_rows()};

  // If output size is zero, return empty
  if (probe_table_num_rows == 0) {
    return std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr);
  }

  auto build_indices =
    std::make_unique<rmm::device_uvector<size_type>>(probe_table_num_rows, stream, mr);

  // If build table is empty, return probe table
  if (this->_build.num_rows() == 0) {
    thrust::fill(
      rmm::exec_policy_nosync(stream), build_indices->begin(), build_indices->end(), JoinNoneValue);
  } else {
    auto const probe_row_hasher =
      cudf::experimental::row::hash::row_hasher{this->_preprocessed_probe};
    auto const d_probe_hasher = probe_row_hasher.device_hasher(nullate::DYNAMIC{this->_has_nulls});
    auto const iter           = cudf::detail::make_counting_transform_iterator(
      0, build_keys_fn<decltype(d_probe_hasher), rhs_index_type>{d_probe_hasher});

    auto const output_begin =
      thrust::make_transform_output_iterator(build_indices->begin(), output_fn{});
    // TODO conditional find for nulls once `cuco::static_set::find_if` is added
    this->_hash_table.find_async(iter, iter + probe_table_num_rows, output_begin, stream.value());
  }

  return build_indices;
}
}  // namespace detail

template <>
distinct_hash_join<cudf::has_nested::YES>::~distinct_hash_join() = default;

template <>
distinct_hash_join<cudf::has_nested::NO>::~distinct_hash_join() = default;

template <>
distinct_hash_join<cudf::has_nested::YES>::distinct_hash_join(cudf::table_view const& build,
                                                              cudf::table_view const& probe,
                                                              nullable_join has_nulls,
                                                              null_equality compare_nulls,
                                                              rmm::cuda_stream_view stream)
  : _impl{std::make_unique<impl_type>(
      build, probe, has_nulls == nullable_join::YES, compare_nulls, stream)}
{
}

template <>
distinct_hash_join<cudf::has_nested::NO>::distinct_hash_join(cudf::table_view const& build,
                                                             cudf::table_view const& probe,
                                                             nullable_join has_nulls,
                                                             null_equality compare_nulls,
                                                             rmm::cuda_stream_view stream)
  : _impl{std::make_unique<impl_type>(
      build, probe, has_nulls == nullable_join::YES, compare_nulls, stream)}
{
}

template <>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
distinct_hash_join<cudf::has_nested::YES>::inner_join(rmm::cuda_stream_view stream,
                                                      rmm::mr::device_memory_resource* mr) const
{
  return _impl->inner_join(stream, mr);
}

template <>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
distinct_hash_join<cudf::has_nested::NO>::inner_join(rmm::cuda_stream_view stream,
                                                     rmm::mr::device_memory_resource* mr) const
{
  return _impl->inner_join(stream, mr);
}

template <>
std::unique_ptr<rmm::device_uvector<size_type>>
distinct_hash_join<cudf::has_nested::YES>::left_join(rmm::cuda_stream_view stream,
                                                     rmm::mr::device_memory_resource* mr) const
{
  return _impl->left_join(stream, mr);
}

template <>
std::unique_ptr<rmm::device_uvector<size_type>> distinct_hash_join<cudf::has_nested::NO>::left_join(
  rmm::cuda_stream_view stream, rmm::mr::device_memory_resource* mr) const
{
  return _impl->left_join(stream, mr);
}
}  // namespace cudf
