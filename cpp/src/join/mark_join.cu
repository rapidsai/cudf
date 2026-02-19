/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common_utils.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/join/mark_join.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/join/mark_join.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuco/detail/open_addressing/kernels.cuh>
#include <cuco/static_multiset_ref.cuh>
#include <cuda/atomic>
#include <cuda/iterator>
#include <thrust/copy.h>
#include <thrust/sequence.h>

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

using slot_type = mark_key_type;

static_assert(sizeof(slot_type) == sizeof(uint64_t));

__device__ inline bool slot_is_empty(slot_type const& slot, slot_type const& sentinel)
{
  return *reinterpret_cast<uint64_t const*>(&slot) == *reinterpret_cast<uint64_t const*>(&sentinel);
}

using storage_ref_type = cuco::bucket_storage_ref<slot_type, 1, cuco::extent<std::size_t>>;
using probe_key_type   = cuco::pair<hash_value_type, rhs_index_type>;

template <int32_t block_size, typename Comparator>
__global__ __launch_bounds__(block_size) void mark_probe_kernel(
  storage_ref_type storage,
  masked_probing_scheme probing_scheme,
  Comparator comparator,
  slot_type empty_sentinel,
  probe_key_type const* __restrict__ probe_rows,
  cudf::size_type num_rows,
  cudf::size_type* global_mark_counter,
  bitmask_type const* probe_row_bitmask)
{
  auto const block = cg::this_thread_block();
  auto const warp  = cg::tiled_partition<cudf::detail::warp_size>(block);

  cudf::size_type mark_counter = 0;
  __shared__ cuda::atomic<cudf::size_type, cuda::thread_scope_block> block_mark_counter;
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> global_counter{*global_mark_counter};
  cg::invoke_one(block, [&]() { block_mark_counter.store(0, cuda::memory_order_relaxed); });
  block.sync();

  auto const tid    = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();

  for (thread_index_type i = tid; i < num_rows; i += stride) {
    bool const is_active = (probe_row_bitmask == nullptr || cudf::bit_is_set(probe_row_bitmask, i));

    if (is_active) {
      auto const query  = probe_rows[i];
      auto probing_iter = probing_scheme.template make_iterator<storage_ref_type::bucket_size>(
        query, storage.extent());

      bool found_empty = false;
      while (!found_empty) {
        auto const bucket_idx  = *probing_iter;
        auto* mutable_slot_p   = &storage.data()[bucket_idx];
        auto const entry_value = *mutable_slot_p;

        if (slot_is_empty(entry_value, empty_sentinel)) {
          found_empty = true;
        } else {
          auto const probe_hash = unset_mark(query.first);
          auto const entry_hash = unset_mark(entry_value.first);
          if (probe_hash == entry_hash && comparator(query, entry_value)) {
            auto expected = entry_value.first;
            if (!is_marked(expected)) {
              auto const desired = set_mark(expected);
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

  auto const warp_sum = cg::reduce(warp, mark_counter, cg::plus<cudf::size_type>{});
  cg::invoke_one(warp,
                 [&]() { block_mark_counter.fetch_add(warp_sum, cuda::memory_order_relaxed); });
  block.sync();

  cg::invoke_one(block, [&]() {
    global_counter.fetch_add(block_mark_counter.load(cuda::memory_order_relaxed),
                             cuda::memory_order_relaxed);
  });
}

template <int32_t block_size, bool is_anti_join>
__global__ __launch_bounds__(block_size) void mark_retrieve_kernel(
  storage_ref_type storage,
  slot_type empty_sentinel,
  cudf::size_type* __restrict__ output,
  cudf::size_type* global_offset,
  cudf::size_type num_buckets)
{
  auto const block = cg::this_thread_block();
  auto const warp  = cg::tiled_partition<cudf::detail::warp_size>(block);

  constexpr int buffer_capacity_factor = 4;
  constexpr int warp_buffer_capacity   = cudf::detail::warp_size * buffer_capacity_factor;
  constexpr int buffer_capacity        = block_size * buffer_capacity_factor;
  int const warp_buffer_offset         = warp_buffer_capacity * warp.meta_group_rank();
  uint32_t build_buffer_offset         = 0;

  __shared__ alignas(buffer_capacity) cudf::size_type build_buffer[buffer_capacity];
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> global_off{*global_offset};

  auto const tid    = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();
  auto const loop_bound =
    cudf::util::round_up_unsafe(static_cast<thread_index_type>(num_buckets), stride);

  for (thread_index_type i = tid; i < loop_bound; i += stride) {
    bool has_match = false;
    cudf::size_type row_idx{};

    if (i < num_buckets) {
      auto const entry_value = storage.data()[i];
      bool const is_filled   = !slot_is_empty(entry_value, empty_sentinel);
      if (is_filled) {
        bool const marked = is_marked(entry_value.first);
        if constexpr (is_anti_join) {
          has_match = !marked;
        } else {
          has_match = marked;
        }
        if (has_match) { row_idx = static_cast<cudf::size_type>(entry_value.second); }
      }
    }

    bool pending_writes = warp.any(has_match);
    while (pending_writes) {
      uint32_t offset = 0;
      if (has_match) {
        auto active_group = cg::coalesced_threads();
        offset            = build_buffer_offset + active_group.thread_rank();
        if (offset < static_cast<uint32_t>(warp_buffer_capacity)) {
          build_buffer[offset + warp_buffer_offset] = row_idx;
          has_match                                 = false;
        }
      }
      offset              = cg::reduce(warp, offset, cg::greater<uint32_t>{});
      build_buffer_offset = offset + 1;
      if (pending_writes = (offset >= static_cast<uint32_t>(warp_buffer_capacity))) {
        build_buffer_offset      = 0;
        auto const output_offset = cg::invoke_one_broadcast(warp, [&]() {
          return global_off.fetch_add(warp_buffer_capacity, cuda::memory_order_relaxed);
        });
#pragma unroll
        for (int k = warp.thread_rank(); k < warp_buffer_capacity; k += cudf::detail::warp_size) {
          output[output_offset + k] = build_buffer[k + warp_buffer_offset];
        }
      }
    }
  }

  if (build_buffer_offset > 0) {
    auto const output_offset = cg::invoke_one_broadcast(warp, [&]() {
      return global_off.fetch_add(build_buffer_offset, cuda::memory_order_relaxed);
    });
    for (uint32_t k = warp.thread_rank(); k < build_buffer_offset; k += warp.num_threads()) {
      output[output_offset + k] = build_buffer[k + warp_buffer_offset];
    }
  }
}

template <int32_t block_size>
__global__ __launch_bounds__(block_size) void clear_marks_kernel(storage_ref_type storage,
                                                                 slot_type empty_sentinel,
                                                                 cudf::size_type num_buckets)
{
  auto const tid    = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();
  for (thread_index_type i = tid; i < num_buckets; i += stride) {
    auto const entry_value = storage.data()[i];
    if (!slot_is_empty(entry_value, empty_sentinel)) {
      if (is_marked(entry_value.first)) {
        cuda::atomic_ref<hash_value_type, cuda::thread_scope_device> key_ref{
          storage.data()[i].first};
        key_ref.store(unset_mark(entry_value.first), cuda::memory_order_relaxed);
      }
    }
  }
}

static std::size_t compute_mark_join_capacity(cudf::table_view tbl, double load_factor)
{
  return static_cast<std::size_t>(
    cuco::make_valid_extent<masked_probing_scheme, mark_storage_type, std::size_t>(tbl.num_rows(),
                                                                                   load_factor));
}

}  // namespace

void mark_join::clear_marks(rmm::cuda_stream_view stream)
{
  auto const storage_ref = _bucket_storage.ref();
  auto const num_buckets = static_cast<cudf::size_type>(storage_ref.num_buckets());
  if (num_buckets == 0) return;

  auto const grid_size = cudf::util::div_rounding_up_unsafe(num_buckets, mark_block_size);
  clear_marks_kernel<mark_block_size><<<grid_size, mark_block_size, 0, stream.value()>>>(
    storage_ref, static_cast<slot_type>(masked_empty_sentinel), num_buckets);
}

template <typename Comparator>
std::unique_ptr<rmm::device_uvector<cudf::size_type>> mark_join::mark_probe_and_retrieve(
  cudf::table_view const& probe,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> preprocessed_probe,
  join_kind kind,
  Comparator comparator,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

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
      size_type{0}, masked_key_fn<rhs_index_type, primitive_row_hasher>{d_probe_hasher});
    probe_rows = materialize_probe_rows(probe_iter);
  } else {
    auto const d_probe_hasher =
      cudf::detail::row::hash::row_hasher{preprocessed_probe}.device_hasher(nullate::YES{});
    auto const probe_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0}, masked_key_fn<rhs_index_type, row_hasher>{d_probe_hasher});
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
    CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &grid_size, mark_probe_kernel<mark_block_size, Comparator>, mark_block_size, 0));
    int num_sms = 0;
    CUDF_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    grid_size *= num_sms;

    mark_probe_kernel<mark_block_size><<<grid_size, mark_block_size, 0, stream.value()>>>(
      storage_ref,
      masked_probing_scheme{},
      comparator,
      static_cast<slot_type>(masked_empty_sentinel),
      probe_rows.data(),
      probe.num_rows(),
      d_mark_counter.data(),
      probe_bitmask_ptr);
  }

  auto const marked_count = d_mark_counter.value(stream);

  auto const null_build_rows = static_cast<size_type>(_build.num_rows()) - num_build_inserted();
  auto const is_anti         = (kind == join_kind::LEFT_ANTI_JOIN);
  auto const unmatched_valid = num_build_inserted() - marked_count;
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
        &grid_size, mark_retrieve_kernel<mark_block_size, false>, mark_block_size, 0));
    } else {
      CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &grid_size, mark_retrieve_kernel<mark_block_size, true>, mark_block_size, 0));
    }
    int num_sms = 0;
    CUDF_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    grid_size *= num_sms;

    if (kind == join_kind::LEFT_SEMI_JOIN) {
      mark_retrieve_kernel<mark_block_size, false>
        <<<grid_size, mark_block_size, 0, stream.value()>>>(
          storage_ref,
          static_cast<slot_type>(masked_empty_sentinel),
          result.data(),
          d_scan_offset.data(),
          num_buckets);
    } else {
      mark_retrieve_kernel<mark_block_size, true>
        <<<grid_size, mark_block_size, 0, stream.value()>>>(
          storage_ref,
          static_cast<slot_type>(masked_empty_sentinel),
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
  : _has_nested_columns{cudf::has_nested_columns(build)},
    _build{build},
    _nulls_equal{compare_nulls},
    _preprocessed_build{cudf::detail::row::equality::preprocessed_table::create(build, stream)},
    _bucket_storage{cuco::extent<std::size_t>{compute_mark_join_capacity(build, load_factor)},
                    rmm::mr::polymorphic_allocator<char>{},
                    stream.value()}
{
  cudf::scoped_range range{"mark_join::mark_join"};
  if (_build.num_rows() == 0) return;
  _bucket_storage.initialize(masked_empty_sentinel, stream);

  // Any mismatch in nullate between probe and build row operators results in UB. Ideally, nullate
  // should be determined by the logical OR of probe nulls and build nulls. However, since we do not
  // know if the probe has nulls apriori, we set nullate::DYNAMIC{true} (in the case of primitive
  // row operators) and nullate::YES (in the case of non-primitive row operators) to ensure both
  // build and probe row operators use consistent null handling.
  auto do_insert = [&](auto const& build_iter, auto const& insert_ref) {
    auto const grid_size =
      cuco::detail::grid_size(_build.num_rows(), masked_probing_scheme::cg_size);
    if (cudf::has_nested_nulls(_build) && _nulls_equal == null_equality::UNEQUAL) {
      auto const bitmask_buffer_and_ptr = build_row_bitmask(_build, stream);
      auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;
      cuco::detail::open_addressing_ns::insert_if_n<masked_probing_scheme::cg_size,
                                                    cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          build_iter,
          _build.num_rows(),
          thrust::counting_iterator<size_type>{0},
          row_is_valid{row_bitmask_ptr},
          insert_ref);
    } else {
      cuco::detail::open_addressing_ns::insert_if_n<masked_probing_scheme::cg_size,
                                                    cuco::detail::default_block_size()>
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
      size_type{0}, masked_key_fn<lhs_index_type, primitive_row_hasher>{d_build_hasher});
    cuco::static_multiset_ref set_ref{masked_empty_sentinel,
                                      insertion_adapter{d_build_comparator},
                                      masked_probing_scheme{},
                                      cuco::thread_scope_device,
                                      _bucket_storage.ref()};
    do_insert(build_iter, set_ref.rebind_operators(cuco::insert));
  } else if (_has_nested_columns) {
    auto const d_build_comparator =
      cudf::detail::row::equality::self_comparator{_preprocessed_build}.equal_to<true>(
        nullate::YES{},
        compare_nulls,
        cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
    auto const d_build_hasher =
      cudf::detail::row::hash::row_hasher{_preprocessed_build}.device_hasher(nullate::YES{});
    auto const build_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0}, masked_key_fn<lhs_index_type, row_hasher>{d_build_hasher});
    cuco::static_multiset_ref set_ref{masked_empty_sentinel,
                                      insertion_adapter{d_build_comparator},
                                      masked_probing_scheme{},
                                      cuco::thread_scope_device,
                                      _bucket_storage.ref()};
    do_insert(build_iter, set_ref.rebind_operators(cuco::insert));
  } else {
    auto const d_build_comparator =
      cudf::detail::row::equality::self_comparator{_preprocessed_build}.equal_to<false>(
        nullate::YES{},
        compare_nulls,
        cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
    auto const d_build_hasher =
      cudf::detail::row::hash::row_hasher{_preprocessed_build}.device_hasher(nullate::YES{});
    auto const build_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0}, masked_key_fn<lhs_index_type, row_hasher>{d_build_hasher});
    cuco::static_multiset_ref set_ref{masked_empty_sentinel,
                                      insertion_adapter{d_build_comparator},
                                      masked_probing_scheme{},
                                      cuco::thread_scope_device,
                                      _bucket_storage.ref()};
    do_insert(build_iter, set_ref.rebind_operators(cuco::insert));
  }

  if (cudf::has_nested_nulls(_build) && _nulls_equal == null_equality::UNEQUAL) {
    auto const nullable_columns = get_nullable_columns(_build);
    auto const row_bitmask =
      cudf::detail::bitmask_and(
        table_view{nullable_columns}, stream, cudf::get_current_device_resource_ref())
        .first;
    _num_build_inserted = cudf::detail::count_set_bits(
      static_cast<bitmask_type const*>(row_bitmask.data()), 0, _build.num_rows(), stream);
  } else {
    _num_build_inserted = _build.num_rows();
  }
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> mark_join::semi_anti_join(
  cudf::table_view const& probe,
  join_kind kind,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  clear_marks(stream);

  auto const preprocessed_probe = [&probe, stream] {
    cudf::scoped_range range{"mark_join::semi_anti_join::preprocessed_probe"};
    return cudf::detail::row::equality::preprocessed_table::create(probe, stream);
  }();

  if (is_primitive_row_op_compatible(_build)) {
    auto const d_build_probe_comparator = primitive_row_comparator{
      nullate::DYNAMIC{true}, _preprocessed_build, preprocessed_probe, _nulls_equal};

    return mark_probe_and_retrieve(
      probe, preprocessed_probe, kind, masked_comparator_fn{d_build_probe_comparator}, stream, mr);
  } else {
    auto const d_build_probe_comparator =
      cudf::detail::row::equality::two_table_comparator{_preprocessed_build, preprocessed_probe};

    if (_has_nested_columns) {
      auto d_build_probe_nan_comparator = d_build_probe_comparator.equal_to<true>(
        nullate::YES{},
        _nulls_equal,
        cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
      return mark_probe_and_retrieve(probe,
                                     preprocessed_probe,
                                     kind,
                                     masked_comparator_fn{d_build_probe_nan_comparator},
                                     stream,
                                     mr);
    } else {
      auto d_build_probe_nan_comparator = d_build_probe_comparator.equal_to<false>(
        nullate::YES{},
        _nulls_equal,
        cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
      return mark_probe_and_retrieve(probe,
                                     preprocessed_probe,
                                     kind,
                                     masked_comparator_fn{d_build_probe_nan_comparator},
                                     stream,
                                     mr);
    }
  }
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> mark_join::semi_join(
  cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  if (_build.num_rows() == 0 || probe.num_rows() == 0) {
    return std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
  }
  return semi_anti_join(probe, join_kind::LEFT_SEMI_JOIN, stream, mr);
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> mark_join::anti_join(
  cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  if (_build.num_rows() == 0) {
    return std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
  }
  if (probe.num_rows() == 0) {
    auto result =
      std::make_unique<rmm::device_uvector<cudf::size_type>>(_build.num_rows(), stream, mr);
    thrust::sequence(rmm::exec_policy_nosync(stream), result->begin(), result->end());
    return result;
  }
  return semi_anti_join(probe, join_kind::LEFT_ANTI_JOIN, stream, mr);
}

}  // namespace detail

// Public API

mark_join::~mark_join() = default;

mark_join::mark_join(cudf::table_view const& build,
                     cudf::null_equality compare_nulls,
                     rmm::cuda_stream_view stream)
  : _impl{std::make_unique<cudf::detail::mark_join>(
      build, compare_nulls, cudf::detail::CUCO_DESIRED_LOAD_FACTOR, stream)}
{
}

mark_join::mark_join(cudf::table_view const& build,
                     cudf::null_equality compare_nulls,
                     double load_factor,
                     rmm::cuda_stream_view stream)
  : _impl{std::make_unique<cudf::detail::mark_join>(build, compare_nulls, load_factor, stream)}
{
}

std::unique_ptr<rmm::device_uvector<size_type>> mark_join::semi_join(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"mark_join::semi_join"};
  return _impl->semi_join(probe, stream, mr);
}

std::unique_ptr<rmm::device_uvector<size_type>> mark_join::anti_join(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"mark_join::anti_join"};
  return _impl->anti_join(probe, stream, mr);
}

}  // namespace cudf
