/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common_utils.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/join/distinct_filtered_join.cuh>
#include <cudf/detail/join/filtered_join.cuh>
#include <cudf/detail/join/mark_join.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>
#include <rmm/resource_ref.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuco/bucket_storage.cuh>
#include <cuco/detail/equal_wrapper.cuh>
#include <cuco/detail/open_addressing/functors.cuh>
#include <cuco/detail/open_addressing/kernels.cuh>
#include <cuco/extent.cuh>
#include <cuco/operator.hpp>
#include <cuco/static_multiset_ref.cuh>
#include <cuco/static_set_ref.cuh>
#include <cuda/atomic>
#include <cuda/iterator>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

#include <memory>

namespace cg = cooperative_groups;

namespace cudf {
namespace detail {
namespace {

/**
 * @brief Build a row bitmask for the input table.
 *
 * The output bitmask will have invalid bits corresponding to the input rows having nulls (at
 * any nested level) and vice versa.
 *
 * @param input The input table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A pair of pointer to the output bitmask and the buffer containing the bitmask
 */
std::pair<rmm::device_buffer, bitmask_type const*> build_row_bitmask(table_view const& input,
                                                                     rmm::cuda_stream_view stream)
{
  auto const nullable_columns = get_nullable_columns(input);
  CUDF_EXPECTS(nullable_columns.size() > 0,
               "The input table has nulls thus it should have nullable columns.");

  // If there are more than one nullable column, we compute `bitmask_and` of their null masks.
  // Otherwise, we have only one nullable column and can use its null mask directly.
  if (nullable_columns.size() > 1) {
    auto row_bitmask =
      cudf::detail::bitmask_and(
        table_view{nullable_columns}, stream, cudf::get_current_device_resource_ref())
        .first;
    auto const row_bitmask_ptr = static_cast<bitmask_type const*>(row_bitmask.data());
    return std::pair(std::move(row_bitmask), row_bitmask_ptr);
  }

  return std::pair(rmm::device_buffer{0, stream}, nullable_columns.front().null_mask());
}

struct gather_mask {
  join_kind kind;
  device_span<bool const> flagged;
  __device__ bool operator()(size_type idx) const noexcept
  {
    return flagged[idx] == (kind == join_kind::LEFT_SEMI_JOIN);
  }
};

static constexpr int32_t mark_block_size = 1024;

using slot_type = filtered_join::key;

template <int32_t block_size,
          int32_t cg_size,
          typename ProbingScheme,
          typename StorageRef,
          typename Comparator,
          typename ProbeKeyType>
__global__ __launch_bounds__(block_size) void mark_probe_kernel(
  StorageRef storage,
  ProbingScheme probing_scheme,
  Comparator comparator,
  slot_type empty_sentinel,
  ProbeKeyType const* __restrict__ probe_rows,
  cudf::size_type num_rows,
  cudf::size_type* global_mark_counter)
{
  auto const grid  = cg::this_grid();
  auto const block = cg::this_thread_block();
  auto const tile  = cg::tiled_partition<32>(block);

  cudf::size_type mark_counter = 0;
  __shared__ cuda::atomic<cudf::size_type, cuda::thread_scope_block> cta_mark_counter;
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> global_counter{*global_mark_counter};
  cg::invoke_one(block, [&]() { cta_mark_counter.store(0, cuda::memory_order_relaxed); });
  block.sync();

  auto const loop_bound = ((num_rows + 31) / 32) * 32;

  for (cudf::size_type i = grid.thread_rank(); i < loop_bound; i += grid.num_threads()) {
    bool is_active = (i < num_rows);

    if (is_active) {
      ProbeKeyType query = probe_rows[i];
      auto probing_iter =
        probing_scheme.template make_iterator<StorageRef::bucket_size>(query, storage.extent());

      bool found_empty = false;
      while (!found_empty) {
        auto bucket_idx      = *probing_iter;
        auto* mutable_slot_p = &storage.data()[bucket_idx];
        auto entry_value     = *mutable_slot_p;

        if (cuco::detail::bitwise_compare(entry_value, empty_sentinel)) {
          found_empty = true;
        } else {
          auto const probe_hash = mark_utils::unset_mark(query.first);
          auto const entry_hash = mark_utils::unset_mark(entry_value.first);
          if (probe_hash == entry_hash && comparator(query, entry_value)) {
            auto expected = entry_value.first;
            if (!mark_utils::is_marked(expected)) {
              auto desired = mark_utils::set_mark(expected);
              cuda::atomic_ref<hash_value_type, cuda::thread_scope_device> key_ref{
                mutable_slot_p->first};
              if (key_ref.compare_exchange_strong(expected, desired, cuda::memory_order_relaxed)) {
                ++mark_counter;
              }
            }
          }
        }
        ++probing_iter;
      }
    }
  }

  auto warp_sum = cg::reduce(tile, mark_counter, cg::plus<cudf::size_type>{});
  cg::invoke_one(tile, [&]() { cta_mark_counter.fetch_add(warp_sum, cuda::memory_order_relaxed); });
  block.sync();

  cg::invoke_one(block, [&]() {
    global_counter.fetch_add(cta_mark_counter.load(cuda::memory_order_relaxed),
                             cuda::memory_order_relaxed);
  });
}

template <int32_t block_size, bool is_anti_join, typename StorageRef>
__global__ __launch_bounds__(block_size) void mark_scan_kernel(StorageRef storage,
                                                               slot_type empty_sentinel,
                                                               cudf::size_type* __restrict__ output,
                                                               cudf::size_type* global_offset,
                                                               cudf::size_type num_buckets)
{
  auto const grid  = cg::this_grid();
  auto const block = cg::this_thread_block();
  auto const tile  = cg::tiled_partition<32>(block);

  constexpr int buffer_capacity_factor = 4;
  constexpr int warp_buffer_capacity   = 32 * buffer_capacity_factor;
  constexpr int buffer_capacity        = block_size * buffer_capacity_factor;
  int const warp_buffer_offset         = warp_buffer_capacity * tile.meta_group_rank();
  uint32_t build_buffer_offset         = 0;

  __shared__ alignas(buffer_capacity) cudf::size_type build_buffer[buffer_capacity];
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> global_off{*global_offset};

  auto const loop_bound = ((num_buckets + 31) / 32) * 32;

  for (cudf::size_type i = grid.thread_rank(); i < loop_bound; i += grid.num_threads()) {
    bool do_fill = false;
    cudf::size_type row_idx{};

    if (i < num_buckets) {
      auto entry_value = storage.data()[i];
      bool is_filled   = !cuco::detail::bitwise_compare(entry_value, empty_sentinel);
      if (is_filled) {
        bool marked = mark_utils::is_marked(entry_value.first);
        if constexpr (is_anti_join) {
          do_fill = !marked;
        } else {
          do_fill = marked;
        }
        if (do_fill) { row_idx = static_cast<cudf::size_type>(entry_value.second); }
      }
    }

    bool work_todo = tile.any(do_fill);
    while (work_todo) {
      uint32_t offset = 0;
      if (do_fill) {
        auto active_group = cg::coalesced_threads();
        offset            = build_buffer_offset + active_group.thread_rank();
        if (offset < static_cast<uint32_t>(warp_buffer_capacity)) {
          build_buffer[offset + warp_buffer_offset] = row_idx;
          do_fill                                   = false;
        }
      }
      offset              = cg::reduce(tile, offset, cg::greater<uint32_t>{});
      build_buffer_offset = offset + 1;
      if (work_todo = (offset >= static_cast<uint32_t>(warp_buffer_capacity))) {
        build_buffer_offset       = 0;
        cudf::size_type flush_off = cg::invoke_one_broadcast(tile, [&]() {
          return global_off.fetch_add(warp_buffer_capacity, cuda::memory_order_relaxed);
        });
#pragma unroll
        for (int k = tile.thread_rank(); k < warp_buffer_capacity; k += 32) {
          output[flush_off + k] = build_buffer[k + warp_buffer_offset];
        }
      }
    }
  }

  if (build_buffer_offset > 0) {
    cudf::size_type flush_off = cg::invoke_one_broadcast(tile, [&]() {
      return global_off.fetch_add(build_buffer_offset, cuda::memory_order_relaxed);
    });
    for (uint32_t k = tile.thread_rank(); k < build_buffer_offset; k += tile.num_threads()) {
      output[flush_off + k] = build_buffer[k + warp_buffer_offset];
    }
  }
}

template <int32_t block_size, typename StorageRef>
__global__ __launch_bounds__(block_size) void clear_marks_kernel(StorageRef storage,
                                                                 slot_type empty_sentinel,
                                                                 cudf::size_type num_buckets)
{
  for (cudf::size_type i = blockIdx.x * blockDim.x + threadIdx.x; i < num_buckets;
       i += blockDim.x * gridDim.x) {
    auto entry_value = storage.data()[i];
    if (!cuco::detail::bitwise_compare(entry_value, empty_sentinel)) {
      if (mark_utils::is_marked(entry_value.first)) {
        cuda::atomic_ref<hash_value_type, cuda::thread_scope_device> key_ref{
          storage.data()[i].first};
        key_ref.store(mark_utils::unset_mark(entry_value.first), cuda::memory_order_relaxed);
      }
    }
  }
}

}  // namespace

auto filtered_join::compute_bucket_storage_size(cudf::table_view tbl, double load_factor)
{
  auto const size_with_primitive_probe = static_cast<std::size_t>(
    cuco::make_valid_extent<primitive_probing_scheme, storage_type, std::size_t>(tbl.num_rows(),
                                                                                 load_factor));
  auto const size_with_nested_probe = static_cast<std::size_t>(
    cuco::make_valid_extent<nested_probing_scheme, storage_type, std::size_t>(tbl.num_rows(),
                                                                              load_factor));
  auto const size_with_simple_probe = static_cast<std::size_t>(
    cuco::make_valid_extent<simple_probing_scheme, storage_type, std::size_t>(tbl.num_rows(),
                                                                              load_factor));
  return std::max({size_with_primitive_probe, size_with_nested_probe, size_with_simple_probe});
}

template <int32_t CGSize, typename Ref>
void filtered_join::insert_build_table(Ref const& insert_ref, rmm::cuda_stream_view stream)
{
  cudf::scoped_range range{"filtered_join::insert_build_table"};
  auto insert = [&]<typename Iterator>(Iterator build_iter) {
    auto const grid_size = cuco::detail::grid_size(_build.num_rows(), CGSize);

    if (cudf::has_nested_nulls(_build) && _nulls_equal == null_equality::UNEQUAL) {
      auto const bitmask_buffer_and_ptr = build_row_bitmask(_build, stream);
      auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;
      cuco::detail::open_addressing_ns::insert_if_n<CGSize, cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          build_iter,
          _build.num_rows(),
          thrust::counting_iterator<size_type>{0},
          row_is_valid{row_bitmask_ptr},
          insert_ref);
    } else {
      cuco::detail::open_addressing_ns::insert_if_n<CGSize, cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          build_iter,
          _build.num_rows(),
          cuda::constant_iterator<bool>{true},
          cuda::std::identity{},
          insert_ref);
    }
  };

  // Any mismatch in nullate between probe and build row operators results in UB. Ideally, nullate
  // should be determined by the logical OR of probe nulls and build nulls. However, since we do not
  // know if the probe has nulls apriori, we set nullate::DYNAMIC{true} (in the case of primitive
  // row operators) and nullate::YES (in the case of non-primitive row operators) to ensure both
  // build and probe row operators use consistent null handling.
  if (is_primitive_row_op_compatible(_build)) {
    auto const d_build_hasher = primitive_row_hasher{nullate::DYNAMIC{true}, _preprocessed_build};
    auto const build_iter     = cudf::detail::make_counting_transform_iterator(
      size_type{0}, key_pair_fn<lhs_index_type, primitive_row_hasher>{d_build_hasher});

    insert(build_iter);
  } else {
    auto const d_build_hasher =
      cudf::detail::row::hash::row_hasher{_preprocessed_build}.device_hasher(nullate::YES{});
    auto const build_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0}, key_pair_fn<lhs_index_type, row_hasher>{d_build_hasher});

    insert(build_iter);
  }
}

template <int32_t CGSize, typename Ref>
std::unique_ptr<rmm::device_uvector<cudf::size_type>> distinct_filtered_join::query_build_table(
  cudf::table_view const& probe,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> preprocessed_probe,
  join_kind kind,
  Ref query_ref,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  cudf::scoped_range range{"distinct_filtered_join::query_build_table"};
  auto const probe_has_nulls = has_nested_nulls(probe);

  auto query_set = [this,
                    probe,
                    probe_has_nulls,
                    query_ref,
                    stream]<typename InputProbeIterator, typename OutputContainsIterator>(
                     InputProbeIterator probe_iter, OutputContainsIterator contains_iter) {
    auto const grid_size = cuco::detail::grid_size(probe.num_rows(), CGSize);
    if (probe_has_nulls && _nulls_equal == null_equality::UNEQUAL) {
      auto const bitmask_buffer_and_ptr = build_row_bitmask(probe, stream);
      auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;
      cuco::detail::open_addressing_ns::contains_if_n<CGSize, cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          probe_iter,
          probe.num_rows(),
          thrust::counting_iterator<size_type>{0},
          row_is_valid{row_bitmask_ptr},
          contains_iter,
          query_ref);
    } else {
      cuco::detail::open_addressing_ns::contains_if_n<CGSize, cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          probe_iter,
          probe.num_rows(),
          cuda::constant_iterator<bool>{true},
          cuda::std::identity{},
          contains_iter,
          query_ref);
    }
  };

  auto contains_map = rmm::device_uvector<bool>(probe.num_rows(), stream);
  if (is_primitive_row_op_compatible(_build)) {
    auto const d_probe_hasher = primitive_row_hasher{nullate::DYNAMIC{true}, preprocessed_probe};
    auto const probe_iter     = cudf::detail::make_counting_transform_iterator(
      size_type{0}, key_pair_fn<rhs_index_type, primitive_row_hasher>{d_probe_hasher});

    query_set(probe_iter, contains_map.begin());
  } else {
    auto const d_probe_hasher =
      cudf::detail::row::hash::row_hasher{preprocessed_probe}.device_hasher(nullate::YES{});
    auto const probe_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0}, key_pair_fn<rhs_index_type, row_hasher>{d_probe_hasher});

    query_set(probe_iter, contains_map.begin());
  }
  rmm::device_uvector<size_type> gather_map(probe.num_rows(), stream, mr);
  auto gather_map_end = thrust::copy_if(rmm::exec_policy_nosync(stream),
                                        thrust::counting_iterator<size_type>(0),
                                        thrust::counting_iterator<size_type>(probe.num_rows()),
                                        gather_map.begin(),
                                        gather_mask{kind, contains_map});
  gather_map.resize(cuda::std::distance(gather_map.begin(), gather_map_end), stream);
  return std::make_unique<rmm::device_uvector<size_type>>(std::move(gather_map));
}

void mark_join::clear_marks(rmm::cuda_stream_view stream)
{
  auto const storage_ref = _bucket_storage.ref();
  auto const num_buckets = static_cast<cudf::size_type>(storage_ref.num_buckets());
  if (num_buckets == 0) return;

  auto const grid_size = (num_buckets + mark_block_size - 1) / mark_block_size;
  clear_marks_kernel<mark_block_size><<<grid_size, mark_block_size, 0, stream.value()>>>(
    storage_ref, static_cast<slot_type>(mark_empty_sentinel_key), num_buckets);
  _num_marks.store(0, std::memory_order_relaxed);
}

template <int32_t CGSize, typename ProbingScheme, typename Comparator>
std::unique_ptr<rmm::device_uvector<cudf::size_type>> mark_join::mark_probe_and_scan(
  cudf::table_view const& probe,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> preprocessed_probe,
  join_kind kind,
  ProbingScheme probing_scheme,
  Comparator comparator,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  cudf::scoped_range range{"mark_join::mark_probe_and_scan"};

  using probe_key_type = cuco::pair<hash_value_type, rhs_index_type>;

  auto materialize_probe_rows = [&](auto const& probe_iter) {
    rmm::device_uvector<probe_key_type> probe_rows(probe.num_rows(), stream);
    thrust::copy(rmm::exec_policy_nosync(stream),
                 probe_iter,
                 probe_iter + probe.num_rows(),
                 probe_rows.begin());
    return probe_rows;
  };

  rmm::device_uvector<probe_key_type> probe_rows(0, stream);
  if (is_primitive_row_op_compatible(_build)) {
    auto const d_probe_hasher = primitive_row_hasher{nullate::DYNAMIC{true}, preprocessed_probe};
    auto const probe_iter     = cudf::detail::make_counting_transform_iterator(
      size_type{0}, masked_key_pair_fn<rhs_index_type, primitive_row_hasher>{d_probe_hasher});
    probe_rows = materialize_probe_rows(probe_iter);
  } else {
    auto const d_probe_hasher =
      cudf::detail::row::hash::row_hasher{preprocessed_probe}.device_hasher(nullate::YES{});
    auto const probe_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0}, masked_key_pair_fn<rhs_index_type, row_hasher>{d_probe_hasher});
    probe_rows = materialize_probe_rows(probe_iter);
  }

  auto const storage_ref = _bucket_storage.ref();
  auto const num_buckets = static_cast<cudf::size_type>(storage_ref.num_buckets());

  rmm::device_scalar<cudf::size_type> d_mark_counter(0, stream);

  {
    int grid_size = 0;
    CUDF_CUDA_TRY(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size,
                                                    mark_probe_kernel<mark_block_size,
                                                                      CGSize,
                                                                      ProbingScheme,
                                                                      decltype(storage_ref),
                                                                      Comparator,
                                                                      probe_key_type>,
                                                    mark_block_size,
                                                    0));
    int num_sms = 0;
    CUDF_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    grid_size *= num_sms;

    mark_probe_kernel<mark_block_size, CGSize><<<grid_size, mark_block_size, 0, stream.value()>>>(
      storage_ref,
      probing_scheme,
      comparator,
      static_cast<slot_type>(mark_empty_sentinel_key),
      probe_rows.data(),
      probe.num_rows(),
      d_mark_counter.data());
  }

  auto const marked_count = d_mark_counter.value(stream);
  _num_marks.store(marked_count, std::memory_order_relaxed);

  auto const result_count =
    (kind == join_kind::LEFT_SEMI_JOIN) ? marked_count : (_build.num_rows() - marked_count);

  auto result = rmm::device_uvector<size_type>(result_count, stream, mr);
  if (result_count == 0) {
    return std::make_unique<rmm::device_uvector<size_type>>(std::move(result));
  }

  rmm::device_scalar<cudf::size_type> d_scan_offset(0, stream);

  {
    int grid_size = 0;
    if (kind == join_kind::LEFT_SEMI_JOIN) {
      CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &grid_size,
        mark_scan_kernel<mark_block_size, false, decltype(storage_ref)>,
        mark_block_size,
        0));
    } else {
      CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &grid_size,
        mark_scan_kernel<mark_block_size, true, decltype(storage_ref)>,
        mark_block_size,
        0));
    }
    int num_sms = 0;
    CUDF_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    grid_size *= num_sms;

    if (kind == join_kind::LEFT_SEMI_JOIN) {
      mark_scan_kernel<mark_block_size, false><<<grid_size, mark_block_size, 0, stream.value()>>>(
        storage_ref,
        static_cast<slot_type>(mark_empty_sentinel_key),
        result.data(),
        d_scan_offset.data(),
        num_buckets);
    } else {
      mark_scan_kernel<mark_block_size, true><<<grid_size, mark_block_size, 0, stream.value()>>>(
        storage_ref,
        static_cast<slot_type>(mark_empty_sentinel_key),
        result.data(),
        d_scan_offset.data(),
        num_buckets);
    }
  }

  return std::make_unique<rmm::device_uvector<size_type>>(std::move(result));
}

filtered_join::filtered_join(cudf::table_view const& build,
                             cudf::null_equality compare_nulls,
                             double load_factor,
                             rmm::cuda_stream_view stream)
  : _build_props{build_properties{cudf::has_nested_columns(build)}},
    _nulls_equal{compare_nulls},
    _build{build},
    _preprocessed_build{cudf::detail::row::equality::preprocessed_table::create(_build, stream)},
    _bucket_storage{cuco::extent<std::size_t>{compute_bucket_storage_size(build, load_factor)},
                    rmm::mr::polymorphic_allocator<char>{},
                    stream.value()}
{
  if (_build.num_rows() == 0) return;
  _bucket_storage.initialize(empty_sentinel_key, stream);
}

distinct_filtered_join::distinct_filtered_join(cudf::table_view const& build,
                                               cudf::null_equality compare_nulls,
                                               double load_factor,
                                               rmm::cuda_stream_view stream)
  : filtered_join(build, compare_nulls, load_factor, stream)
{
  cudf::scoped_range range{"distinct_filtered_join::distinct_filtered_join"};
  if (_build.num_rows() == 0) return;
  // Any mismatch in nullate between probe and build row operators results in UB. Ideally, nullate
  // should be determined by the logical OR of probe nulls and build nulls. However, since we do not
  // know if the probe has nulls apriori, we set nullate::DYNAMIC{true} (in the case of primitive
  // row operators) and nullate::YES (in the case of non-primitive row operators) to ensure both
  // build and probe row operators use consistent null handling.
  if (is_primitive_row_op_compatible(build)) {
    auto const d_build_comparator = primitive_row_comparator{
      nullate::DYNAMIC{true}, _preprocessed_build, _preprocessed_build, compare_nulls};
    cuco::static_set_ref set_ref{empty_sentinel_key,
                                 insertion_adapter{d_build_comparator},
                                 primitive_probing_scheme{},
                                 cuco::thread_scope_device,
                                 _bucket_storage.ref()};
    auto insert_ref = set_ref.rebind_operators(cuco::insert);
    insert_build_table<primitive_probing_scheme::cg_size>(insert_ref, stream);
  } else if (_build_props.has_nested_columns) {
    auto const d_build_comparator =
      cudf::detail::row::equality::self_comparator{_preprocessed_build}.equal_to<true>(
        nullate::YES{},
        compare_nulls,
        cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
    cuco::static_set_ref set_ref{empty_sentinel_key,
                                 insertion_adapter{d_build_comparator},
                                 nested_probing_scheme{},
                                 cuco::thread_scope_device,
                                 _bucket_storage.ref()};
    auto insert_ref = set_ref.rebind_operators(cuco::insert);
    insert_build_table<nested_probing_scheme::cg_size>(insert_ref, stream);
  } else {
    auto const d_build_comparator =
      cudf::detail::row::equality::self_comparator{_preprocessed_build}.equal_to<false>(
        nullate::YES{},
        compare_nulls,
        cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
    cuco::static_set_ref set_ref{empty_sentinel_key,
                                 insertion_adapter{d_build_comparator},
                                 simple_probing_scheme{},
                                 cuco::thread_scope_device,
                                 _bucket_storage.ref()};
    auto insert_ref = set_ref.rebind_operators(cuco::insert);
    insert_build_table<simple_probing_scheme::cg_size>(insert_ref, stream);
  }
}

mark_join::mark_join(cudf::table_view const& build,
                     cudf::null_equality compare_nulls,
                     double load_factor,
                     rmm::cuda_stream_view stream)
  : filtered_join(build, compare_nulls, load_factor, stream)
{
  cudf::scoped_range range{"mark_join::mark_join"};
  if (_build.num_rows() == 0) return;
  _bucket_storage.initialize(mark_empty_sentinel_key, stream);
  // Any mismatch in nullate between probe and build row operators results in UB. Ideally, nullate
  // should be determined by the logical OR of probe nulls and build nulls. However, since we do not
  // know if the probe has nulls apriori, we set nullate::DYNAMIC{true} (in the case of primitive
  // row operators) and nullate::YES (in the case of non-primitive row operators) to ensure both
  // build and probe row operators use consistent null handling.
  auto insert_masked = [&]<typename Iterator, int32_t CGSize>(Iterator build_iter,
                                                              auto const& insert_ref) {
    auto const grid_size = cuco::detail::grid_size(_build.num_rows(), CGSize);
    cuco::detail::open_addressing_ns::insert_if_n<CGSize, cuco::detail::default_block_size()>
      <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
        build_iter,
        _build.num_rows(),
        cuda::constant_iterator<bool>{true},
        cuda::std::identity{},
        insert_ref);
  };

  if (is_primitive_row_op_compatible(build)) {
    auto const d_build_comparator = primitive_row_comparator{
      nullate::DYNAMIC{true}, _preprocessed_build, _preprocessed_build, compare_nulls};
    auto const d_build_hasher = primitive_row_hasher{nullate::DYNAMIC{true}, _preprocessed_build};
    auto const build_iter     = cudf::detail::make_counting_transform_iterator(
      size_type{0}, masked_key_pair_fn<lhs_index_type, primitive_row_hasher>{d_build_hasher});
    cuco::static_multiset_ref set_ref{
      mark_empty_sentinel_key,
      insertion_adapter<decltype(d_build_comparator), set_as_build_table::LEFT>{d_build_comparator},
      mark_aware_simple_probing_scheme{},
      cuco::thread_scope_device,
      _bucket_storage.ref()};
    auto insert_ref = set_ref.rebind_operators(cuco::insert);
    insert_masked
      .template operator()<decltype(build_iter), mark_aware_simple_probing_scheme::cg_size>(
        build_iter, insert_ref);
  } else if (_build_props.has_nested_columns) {
    auto const d_build_comparator =
      cudf::detail::row::equality::self_comparator{_preprocessed_build}.equal_to<true>(
        nullate::YES{},
        compare_nulls,
        cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
    auto const d_build_hasher =
      cudf::detail::row::hash::row_hasher{_preprocessed_build}.device_hasher(nullate::YES{});
    auto const build_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0}, masked_key_pair_fn<lhs_index_type, row_hasher>{d_build_hasher});
    cuco::static_multiset_ref set_ref{
      mark_empty_sentinel_key,
      insertion_adapter<decltype(d_build_comparator), set_as_build_table::LEFT>{d_build_comparator},
      mark_aware_simple_probing_scheme{},
      cuco::thread_scope_device,
      _bucket_storage.ref()};
    auto insert_ref = set_ref.rebind_operators(cuco::insert);
    insert_masked
      .template operator()<decltype(build_iter), mark_aware_simple_probing_scheme::cg_size>(
        build_iter, insert_ref);
  } else {
    auto const d_build_comparator =
      cudf::detail::row::equality::self_comparator{_preprocessed_build}.equal_to<false>(
        nullate::YES{},
        compare_nulls,
        cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
    auto const d_build_hasher =
      cudf::detail::row::hash::row_hasher{_preprocessed_build}.device_hasher(nullate::YES{});
    auto const build_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0}, masked_key_pair_fn<lhs_index_type, row_hasher>{d_build_hasher});
    cuco::static_multiset_ref set_ref{
      mark_empty_sentinel_key,
      insertion_adapter<decltype(d_build_comparator), set_as_build_table::LEFT>{d_build_comparator},
      mark_aware_simple_probing_scheme{},
      cuco::thread_scope_device,
      _bucket_storage.ref()};
    auto insert_ref = set_ref.rebind_operators(cuco::insert);
    insert_masked
      .template operator()<decltype(build_iter), mark_aware_simple_probing_scheme::cg_size>(
        build_iter, insert_ref);
  }
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> distinct_filtered_join::semi_anti_join(
  cudf::table_view const& probe,
  join_kind kind,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  cudf::scoped_range range{"distinct_filtered_join::semi_anti_join"};

  auto const preprocessed_probe = [&probe, stream] {
    cudf::scoped_range range{"distinct_filtered_join::semi_anti_join::preprocessed_probe"};
    return cudf::detail::row::equality::preprocessed_table::create(probe, stream);
  }();

  if (is_primitive_row_op_compatible(_build)) {
    auto const d_build_probe_comparator = primitive_row_comparator{
      nullate::DYNAMIC{true}, _preprocessed_build, preprocessed_probe, _nulls_equal};

    cuco::static_set_ref set_ref{empty_sentinel_key,
                                 comparator_adapter{d_build_probe_comparator},
                                 primitive_probing_scheme{},
                                 cuco::thread_scope_device,
                                 _bucket_storage.ref()};
    auto query_ref = set_ref.rebind_operators(cuco::op::contains);
    return query_build_table<primitive_probing_scheme::cg_size>(
      probe, preprocessed_probe, kind, query_ref, stream, mr);
  } else {
    auto const d_build_probe_comparator =
      cudf::detail::row::equality::two_table_comparator{_preprocessed_build, preprocessed_probe};

    if (_build_props.has_nested_columns) {
      auto d_build_probe_nan_comparator = d_build_probe_comparator.equal_to<true>(
        nullate::YES{},
        _nulls_equal,
        cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
      cuco::static_set_ref set_ref{empty_sentinel_key,
                                   comparator_adapter{d_build_probe_nan_comparator},
                                   nested_probing_scheme{},
                                   cuco::thread_scope_device,
                                   _bucket_storage.ref()};
      auto query_ref = set_ref.rebind_operators(cuco::op::contains);
      return query_build_table<nested_probing_scheme::cg_size>(
        probe, preprocessed_probe, kind, query_ref, stream, mr);
    } else {
      auto d_build_probe_nan_comparator = d_build_probe_comparator.equal_to<false>(
        nullate::YES{},
        _nulls_equal,
        cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
      cuco::static_set_ref set_ref{empty_sentinel_key,
                                   comparator_adapter{d_build_probe_nan_comparator},
                                   simple_probing_scheme{},
                                   cuco::thread_scope_device,
                                   _bucket_storage.ref()};
      auto query_ref = set_ref.rebind_operators(cuco::op::contains);
      return query_build_table<simple_probing_scheme::cg_size>(
        probe, preprocessed_probe, kind, query_ref, stream, mr);
    }
  }
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> mark_join::semi_anti_join(
  cudf::table_view const& probe,
  join_kind kind,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  cudf::scoped_range range{"mark_join::semi_anti_join"};

  clear_marks(stream);

  auto const preprocessed_probe = [&probe, stream] {
    cudf::scoped_range range{"mark_join::semi_anti_join::preprocessed_probe"};
    return cudf::detail::row::equality::preprocessed_table::create(probe, stream);
  }();

  if (is_primitive_row_op_compatible(_build)) {
    auto const d_build_probe_comparator = primitive_row_comparator{
      nullate::DYNAMIC{true}, _preprocessed_build, preprocessed_probe, _nulls_equal};

    return mark_probe_and_scan<mark_aware_simple_probing_scheme::cg_size>(
      probe,
      preprocessed_probe,
      kind,
      mark_aware_simple_probing_scheme{},
      mark_aware_comparator_adapter{d_build_probe_comparator},
      stream,
      mr);
  } else {
    auto const d_build_probe_comparator =
      cudf::detail::row::equality::two_table_comparator{_preprocessed_build, preprocessed_probe};

    if (_build_props.has_nested_columns) {
      auto d_build_probe_nan_comparator = d_build_probe_comparator.equal_to<true>(
        nullate::YES{},
        _nulls_equal,
        cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
      return mark_probe_and_scan<mark_aware_simple_probing_scheme::cg_size>(
        probe,
        preprocessed_probe,
        kind,
        mark_aware_simple_probing_scheme{},
        mark_aware_comparator_adapter{d_build_probe_nan_comparator},
        stream,
        mr);
    } else {
      auto d_build_probe_nan_comparator = d_build_probe_comparator.equal_to<false>(
        nullate::YES{},
        _nulls_equal,
        cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
      return mark_probe_and_scan<mark_aware_simple_probing_scheme::cg_size>(
        probe,
        preprocessed_probe,
        kind,
        mark_aware_simple_probing_scheme{},
        mark_aware_comparator_adapter{d_build_probe_nan_comparator},
        stream,
        mr);
    }
  }
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> distinct_filtered_join::semi_join(
  cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  // Early return for empty build or probe table
  if (_build.num_rows() == 0 || probe.num_rows() == 0) {
    return std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
  }

  return semi_anti_join(probe, join_kind::LEFT_SEMI_JOIN, stream, mr);
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> mark_join::semi_join(
  cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  // Early return for empty build or probe table
  if (_build.num_rows() == 0 || probe.num_rows() == 0) {
    return std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
  }

  return semi_anti_join(probe, join_kind::LEFT_SEMI_JOIN, stream, mr);
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> distinct_filtered_join::anti_join(
  cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  // Early return for empty probe table
  if (probe.num_rows() == 0) {
    return std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
  }
  if (_build.num_rows() == 0) {
    auto result =
      std::make_unique<rmm::device_uvector<cudf::size_type>>(probe.num_rows(), stream, mr);
    thrust::sequence(rmm::exec_policy_nosync(stream), result->begin(), result->end());
    return result;
  }

  return semi_anti_join(probe, join_kind::LEFT_ANTI_JOIN, stream, mr);
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> mark_join::anti_join(
  cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  // Early return for empty build table
  if (_build.num_rows() == 0) {
    return std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
  }
  // Early return for empty probe table - all build rows have no match
  if (probe.num_rows() == 0) {
    auto result =
      std::make_unique<rmm::device_uvector<cudf::size_type>>(_build.num_rows(), stream, mr);
    thrust::sequence(rmm::exec_policy_nosync(stream), result->begin(), result->end());
    return result;
  }

  return semi_anti_join(probe, join_kind::LEFT_ANTI_JOIN, stream, mr);
}

}  // namespace detail

filtered_join::~filtered_join() = default;

filtered_join::filtered_join(cudf::table_view const& build,
                             null_equality compare_nulls,
                             set_as_build_table reuse_tbl,
                             double load_factor,
                             rmm::cuda_stream_view stream)
{
  _reuse_tbl = reuse_tbl;
  if (reuse_tbl == set_as_build_table::RIGHT) {
    _impl = std::make_unique<cudf::detail::distinct_filtered_join>(
      build, compare_nulls, load_factor, stream);
  } else {
    _impl = std::make_unique<cudf::detail::mark_join>(build, compare_nulls, load_factor, stream);
  }
}

filtered_join::filtered_join(cudf::table_view const& build,
                             null_equality compare_nulls,
                             set_as_build_table reuse_tbl,
                             rmm::cuda_stream_view stream)
  : filtered_join(build, compare_nulls, reuse_tbl, cudf::detail::CUCO_DESIRED_LOAD_FACTOR, stream)
{
}

std::unique_ptr<rmm::device_uvector<size_type>> filtered_join::semi_join(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  return _impl->semi_join(probe, stream, mr);
}

std::unique_ptr<rmm::device_uvector<size_type>> filtered_join::anti_join(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  return _impl->anti_join(probe, stream, mr);
}

}  // namespace cudf
