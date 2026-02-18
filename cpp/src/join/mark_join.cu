/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common_utils.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/join/filtered_join.cuh>
#include <cudf/detail/join/mark_join.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
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
#include <cuco/detail/open_addressing/functors.cuh>
#include <cuco/detail/open_addressing/kernels.cuh>
#include <cuco/extent.cuh>
#include <cuco/static_multiset_ref.cuh>
#include <cuda/atomic>
#include <cuda/iterator>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

#include <memory>

namespace cg = cooperative_groups;

namespace cudf {
namespace detail {
namespace {

std::pair<rmm::device_buffer, bitmask_type const*> build_row_bitmask(table_view const& input,
                                                                     rmm::cuda_stream_view stream)
{
  auto const nullable_columns = get_nullable_columns(input);
  CUDF_EXPECTS(nullable_columns.size() > 0,
               "The input table has nulls thus it should have nullable columns.");

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

struct row_is_null {
  bitmask_type const* _row_bitmask;
  __device__ bool operator()(size_type i) const noexcept
  {
    return !cudf::bit_is_set(_row_bitmask, i);
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
  cudf::size_type* global_mark_counter,
  bitmask_type const* probe_row_bitmask)
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
    bool is_active =
      (i < num_rows) && (probe_row_bitmask == nullptr || cudf::bit_is_set(probe_row_bitmask, i));

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

  auto const probe_has_nulls = has_nested_nulls(probe);
  rmm::device_buffer probe_bitmask_buffer(0, stream);
  bitmask_type const* probe_bitmask_ptr = nullptr;
  if (probe_has_nulls && _nulls_equal == null_equality::UNEQUAL) {
    auto bitmask_buffer_and_ptr = build_row_bitmask(probe, stream);
    probe_bitmask_buffer        = std::move(bitmask_buffer_and_ptr.first);
    probe_bitmask_ptr           = bitmask_buffer_and_ptr.second;
  }

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
      d_mark_counter.data(),
      probe_bitmask_ptr);
  }

  auto const marked_count = d_mark_counter.value(stream);
  _num_marks.store(marked_count, std::memory_order_relaxed);

  auto const null_build_rows = static_cast<size_type>(_build.num_rows()) - _build_num_valid_rows;
  auto const is_anti         = (kind == join_kind::LEFT_ANTI_JOIN);
  auto const unmatched_valid = _build_num_valid_rows - marked_count;
  auto const null_contribution =
    (is_anti && _nulls_equal == null_equality::UNEQUAL) ? null_build_rows : size_type{0};
  auto const result_count =
    (kind == join_kind::LEFT_SEMI_JOIN) ? marked_count : (unmatched_valid + null_contribution);

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

  if (null_contribution > 0) {
    auto const bitmask_buffer_and_ptr = build_row_bitmask(_build, stream);
    auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;
    thrust::copy_if(rmm::exec_policy_nosync(stream),
                    thrust::counting_iterator<size_type>(0),
                    thrust::counting_iterator<size_type>(_build.num_rows()),
                    result.begin() + unmatched_valid,
                    row_is_null{row_bitmask_ptr});
  }

  return std::make_unique<rmm::device_uvector<size_type>>(std::move(result));
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

  if (cudf::has_nested_nulls(_build) && _nulls_equal == null_equality::UNEQUAL) {
    auto const nullable_columns = get_nullable_columns(_build);
    auto const row_bitmask =
      cudf::detail::bitmask_and(
        table_view{nullable_columns}, stream, cudf::get_current_device_resource_ref())
        .first;
    _build_num_valid_rows = cudf::detail::count_set_bits(
      static_cast<bitmask_type const*>(row_bitmask.data()), 0, _build.num_rows(), stream);
  } else {
    _build_num_valid_rows = _build.num_rows();
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

std::unique_ptr<rmm::device_uvector<cudf::size_type>> mark_join::semi_join(
  cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  // Early return for empty build or probe table
  if (_build.num_rows() == 0 || probe.num_rows() == 0) {
    return std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
  }

  return semi_anti_join(probe, join_kind::LEFT_SEMI_JOIN, stream, mr);
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
}  // namespace cudf
