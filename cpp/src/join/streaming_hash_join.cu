/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join/hash_join/hash_csr.cuh"
#include "join/join_common_utils.cuh"
#include "join/segmented_dynamic_csr.cuh"

#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/join/join.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/row_operator/primitive_equality.cuh>
#include <cudf/detail/row_operator/primitive_hashing.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/streaming_hash_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/pair.cuh>
#include <cuco/static_multiset.cuh>
#include <cuda/iterator>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/stream>

#include <algorithm>
#include <atomic>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cudf::detail {
namespace {

using slot_type = cuco::pair<hash_value_type, size_type>;

template <typename Equality, typename Factory>
auto make_device_comparators(
  std::span<std::shared_ptr<row::equality::preprocessed_table> const> preprocessed_right,
  Factory factory,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref temp_mr)
{
  using allocator_type  = cudf::detail::rmm_host_allocator<Equality>;
  auto host_comparators = std::vector<Equality, allocator_type>{
    allocator_type{cudf::get_pinned_memory_resource(), stream}};
  host_comparators.reserve(preprocessed_right.size());
  std::transform(preprocessed_right.begin(),
                 preprocessed_right.end(),
                 std::back_inserter(host_comparators),
                 factory);
  auto d_comparators = cudf::detail::make_device_uvector_async(
    cudf::host_span<Equality const>{
      host_comparators.data(), host_comparators.size(), /*is_device_accessible=*/true},
    stream,
    temp_mr);
  stream.sync();  // wait for host_comparators to finish copying before it goes out of scope
  return d_comparators;
}

template <bool has_nested>
auto make_device_row_comparators(
  std::shared_ptr<row::equality::preprocessed_table> const& preprocessed_left,
  std::span<std::shared_ptr<row::equality::preprocessed_table> const> preprocessed_right,
  nullate::DYNAMIC has_nulls,
  null_equality compare_nulls,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref temp_mr)
{
  using equality_type =
    row::equality::device_row_comparator<has_nested,
                                         nullate::DYNAMIC,
                                         row::equality::nan_equal_physical_equality_comparator>;

  return make_device_comparators<equality_type>(
    preprocessed_right,
    [&](auto const& right) {
      auto const comparator = row::equality::two_table_comparator{preprocessed_left, right};
      return comparator.equal_to<has_nested>(has_nulls, compare_nulls).comparator;
    },
    stream,
    temp_mr);
}

auto make_device_primitive_row_comparators(
  std::shared_ptr<row::equality::preprocessed_table> const& preprocessed_left,
  std::span<std::shared_ptr<row::equality::preprocessed_table> const> preprocessed_right,
  nullate::DYNAMIC has_nulls,
  null_equality compare_nulls,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref temp_mr)
{
  using equality_type = row::primitive::row_equality_comparator;

  return make_device_comparators<equality_type>(
    preprocessed_right,
    [&](auto const& right) {
      return equality_type{has_nulls, preprocessed_left, right, compare_nulls};
    },
    stream,
    temp_mr);
}

size_type checked_batch_count(size_type batches)
{
  CUDF_EXPECTS(
    batches > 0, "streaming_hash_join requires max_num_batches > 0.", std::invalid_argument);
  return batches;
}

bool use_segmented_dynamic_csr()
{
  // This branch intentionally consumes only its own bit. The integration branch owns the final
  // process-wide selector validation and other implementation bits.
  auto const* const value = std::getenv("CUDF_STREAMING_HASH_JOIN_IMPL_MASK");
  return value != nullptr && (std::strtoul(value, nullptr, 0) & 0x2UL) != 0;
}

/**
 * @brief Whether this process selected the base-CSR + batch-delta experiment.
 *
 * The integration branch owns the public selector.  Keeping the same bit here makes this branch
 * independently runnable and, importantly, leaves the production streaming implementation as the
 * default while the experiment is incomplete.
 */
bool use_base_csr_batch_deltas()
{
  auto const value = std::getenv("CUDF_STREAMING_HASH_JOIN_IMPL_MASK");
  if (value == nullptr) { return false; }
  char* end{};
  auto const mask = std::strtoul(value, &end, 0);
  return end != value && (*end == '\0') && (mask & 0x1UL) != 0;
}

/**
 * @brief Capacity for the segmented key map.
 *
 * The append store needs one row of payload capacity per declared build row, whereas the
 * open-addressed key map benefits substantially from a lower load factor.  Keep these two
 * resources independent: at 0.5 load the map is a power-of-two sized probe table and all probe
 * arithmetic is a mask rather than an integer divide.
 */
size_type segmented_dynamic_key_map_capacity(size_type total_rows)
{
  auto const requested = std::max<std::uint64_t>(
    1, static_cast<std::uint64_t>(total_rows) * 2);  // ceil(total_rows / 0.5)
  auto const capacity = std::bit_ceil(requested);
  CUDF_EXPECTS(capacity <= static_cast<std::uint64_t>(std::numeric_limits<size_type>::max()),
               "streaming_hash_join segmented key-map capacity overflowed",
               std::overflow_error);
  return static_cast<size_type>(capacity);
}

/**
 * @brief Describes how a slot's first 32-bit word is divided between hash and batch ID.
 *
 * The high `batch_bits` store the batch ID and the remaining low bits store the truncated row hash.
 * CUCO's probing hash functions use only the low hash bits, ensuring that equal rows from different
 * batches share the same probe sequence.
 */
struct batch_hash_layout {
  static constexpr int32_t num_hash_bits = std::numeric_limits<hash_value_type>::digits;
  static_assert(num_hash_bits == 32, "streaming_hash_join requires a 32-bit row hash");

  explicit batch_hash_layout(size_type max_num_batches)
    : batch_bits{std::bit_width(
        static_cast<hash_value_type>(checked_batch_count(max_num_batches) - 1))},
      batch_shift{num_hash_bits - batch_bits},
      hash_mask{std::numeric_limits<hash_value_type>::max() >> batch_bits}
  {
  }

  [[nodiscard]] CUDF_HOST_DEVICE constexpr hash_value_type masked_hash(
    hash_value_type hash) const noexcept
  {
    return hash & hash_mask;
  }

  [[nodiscard]] CUDF_HOST_DEVICE constexpr hash_value_type pack(hash_value_type hash,
                                                                size_type batch_id) const noexcept
  {
    if (batch_bits == 0) { return hash; }
    return masked_hash(hash) | (static_cast<hash_value_type>(batch_id) << batch_shift);
  }

  [[nodiscard]] CUDF_HOST_DEVICE constexpr size_type batch_id(
    hash_value_type packed_hash) const noexcept
  {
    if (batch_bits == 0) { return 0; }
    return static_cast<size_type>(packed_hash >> batch_shift);
  }

  int32_t batch_bits;
  int32_t batch_shift;
  hash_value_type hash_mask;
};

struct always_not_equal {
  __device__ constexpr bool operator()(slot_type const&, slot_type const&) const noexcept
  {
    return false;
  }
};

struct masked_hasher1 {
  hash_value_type hash_mask{std::numeric_limits<hash_value_type>::max()};

  __device__ constexpr hash_value_type operator()(slot_type const& key) const noexcept
  {
    return key.first & hash_mask;
  }
};

struct masked_hasher2 {
  masked_hasher2(hash_value_type mask = std::numeric_limits<hash_value_type>::max(),
                 hash_value_type seed = cudf::DEFAULT_HASH_SEED)
    : hash_mask{mask}, hash{seed}
  {
  }

  __device__ constexpr hash_value_type operator()(slot_type const& key) const noexcept
  {
    return hash(key.first & hash_mask);
  }

  hash_value_type hash_mask;
  cuco::xxhash_32<hash_value_type> hash;
};

using probing_scheme  = cuco::double_hashing<DEFAULT_JOIN_CG_SIZE, masked_hasher1, masked_hasher2>;
using hash_table_type = cuco::static_multiset<slot_type,
                                              cuco::extent<std::size_t>,
                                              cuda::thread_scope_device,
                                              always_not_equal,
                                              probing_scheme,
                                              rmm::mr::polymorphic_allocator<char>,
                                              cuco::storage<2>>;

template <typename Hasher>
struct build_pair_fn {
  Hasher hash;
  batch_hash_layout layout;
  size_type batch_id;

  __device__ slot_type operator()(size_type row_index) const noexcept
  {
    return slot_type{layout.pack(hash(row_index), batch_id), row_index};
  }
};

template <typename Hasher>
struct probe_pair_fn {
  Hasher hash;
  batch_hash_layout layout;

  __device__ slot_type operator()(size_type row_index) const noexcept
  {
    return slot_type{layout.masked_hash(hash(row_index)), row_index};
  }
};

template <typename RowEqual>
struct n_table_pair_equal {
  RowEqual const* comparators;
  batch_hash_layout layout;

  __device__ bool operator()(slot_type const& probe, slot_type const& build) const noexcept
  {
    auto const batch_id = layout.batch_id(build.first);
    return layout.masked_hash(probe.first) == layout.masked_hash(build.first) &&
           comparators[batch_id](probe.second, build.second);
  }
};

struct extract_index_fn {
  __device__ constexpr size_type operator()(slot_type const& value) const noexcept
  {
    return value.second;
  }
};

struct decode_slot_fn {
  batch_hash_layout layout;

  __device__ cuda::std::tuple<size_type, size_type> operator()(slot_type const& slot) const noexcept
  {
    return {layout.batch_id(slot.first), slot.second};
  }
};

// The experimental representation stores a stable, lossless build-row reference separately from
// the 32-bit row hash.  Keeping batch identity out of the hash preserves all hash entropy while
// avoiding a binary search through batch boundaries in equality and retrieval.
using streaming_row_ref = cuda::std::uint64_t;

__device__ constexpr streaming_row_ref make_streaming_row_ref(size_type batch, size_type row)
{
  return (static_cast<streaming_row_ref>(static_cast<cuda::std::uint32_t>(batch)) << 32) |
         static_cast<cuda::std::uint32_t>(row);
}

__device__ constexpr size_type streaming_ref_batch(streaming_row_ref ref)
{
  return static_cast<size_type>(ref >> 32);
}

__device__ constexpr size_type streaming_ref_row(streaming_row_ref ref)
{
  return static_cast<size_type>(ref);
}

constexpr auto streaming_empty_hash   = std::numeric_limits<hash_value_type>::max();
constexpr auto streaming_pending_hash = streaming_empty_hash - 1;

struct streaming_key_slot {
  hash_value_type hash;
  streaming_row_ref representative;
};

/** @brief Collision-aware, power-of-two open-addressed map for the base+deltas experiment. */
struct streaming_hash_table_ref {
  streaming_key_slot* entries;
  cuda::std::uint32_t capacity;

  [[nodiscard]] __device__ cuda::std::uint32_t mask() const { return capacity - 1; }

  template <typename Equal>
  __device__ cuda::std::uint32_t insert(hash_value_type hash,
                                        streaming_row_ref representative,
                                        Equal equal) const
  {
    auto const normalized = hash >= streaming_pending_hash ? streaming_pending_hash - 1 : hash;
    for (cuda::std::uint32_t step = 0; step < capacity; ++step) {
      auto const slot = (static_cast<cuda::std::uint32_t>(normalized) + step) & mask();
      auto key = cuda::atomic_ref<hash_value_type, cuda::thread_scope_device>{entries[slot].hash};
      while (true) {
        auto current = key.load(cuda::memory_order_acquire);
        if (current == streaming_pending_hash) { continue; }
        if (current == streaming_empty_hash) {
          auto expected = streaming_empty_hash;
          if (key.compare_exchange_strong(
                expected, streaming_pending_hash, cuda::memory_order_acq_rel)) {
            cuda::atomic_ref<streaming_row_ref, cuda::thread_scope_device>{
              entries[slot].representative}
              .store(representative, cuda::memory_order_release);
            key.store(normalized, cuda::memory_order_release);
            return slot;
          }
          continue;
        }
        auto const stored =
          cuda::atomic_ref<streaming_row_ref, cuda::thread_scope_device>{
            entries[slot].representative}
            .load(cuda::memory_order_acquire);
        if (current == normalized && equal(representative, stored)) { return slot; }
        break;
      }
    }
    return capacity;
  }

  template <typename Equal>
  __device__ cuda::std::uint32_t find(hash_value_type hash, size_type probe, Equal equal) const
  {
    auto const normalized = hash >= streaming_pending_hash ? streaming_pending_hash - 1 : hash;
    for (cuda::std::uint32_t step = 0; step < capacity; ++step) {
      auto const slot = (static_cast<cuda::std::uint32_t>(normalized) + step) & mask();
      auto key = cuda::atomic_ref<hash_value_type, cuda::thread_scope_device>{entries[slot].hash};
      auto current = key.load(cuda::memory_order_acquire);
      while (current == streaming_pending_hash) {
        current = key.load(cuda::memory_order_acquire);
      }
      if (current == streaming_empty_hash) { return capacity; }
      auto const stored =
        cuda::atomic_ref<streaming_row_ref, cuda::thread_scope_device>{entries[slot].representative}
          .load(cuda::memory_order_acquire);
      if (current == normalized && equal(probe, stored)) { return slot; }
    }
    return capacity;
  }
};

struct streaming_csr_ref {
  size_type const* cumulative_ends;
  streaming_row_ref const* values;

  [[nodiscard]] __device__ size_type begin(size_type slot) const
  {
    return slot == 0 ? size_type{0} : cumulative_ends[slot - 1];
  }
  [[nodiscard]] __device__ size_type size(size_type slot) const
  {
    return cumulative_ends[slot] - begin(slot);
  }
};

struct streaming_csr_generations_ref {
  streaming_csr_ref base;
  streaming_csr_ref const* generations;
  size_type num_generations;

  [[nodiscard]] __device__ size_type size(size_type slot) const
  {
    auto count = base.size(slot);
    for (size_type generation = 0; generation < num_generations; ++generation) {
      count += generations[generation].size(slot);
    }
    return count;
  }

  [[nodiscard]] __device__ streaming_row_ref at(size_type slot, size_type ordinal) const
  {
    auto const base_size = base.size(slot);
    if (ordinal < base_size) { return base.values[base.begin(slot) + ordinal]; }
    ordinal -= base_size;
    for (size_type generation = 0; generation < num_generations; ++generation) {
      auto const generation_size = generations[generation].size(slot);
      if (ordinal < generation_size) {
        return generations[generation].values[generations[generation].begin(slot) + ordinal];
      }
      ordinal -= generation_size;
    }
    return streaming_row_ref{0};  // Unreachable when ordinal < size(slot).
  }
};

struct streaming_delta {
  std::unique_ptr<rmm::device_uvector<cuda::std::uint32_t>> slots;
  std::unique_ptr<rmm::device_uvector<streaming_row_ref>> rows;
  std::unique_ptr<rmm::device_uvector<size_type>> ends;
  std::unique_ptr<rmm::device_uvector<streaming_row_ref>> values;
};

constexpr auto streaming_empty_slot = cuda::std::uint32_t{-1};

inline cuda::std::uint32_t streaming_hash_csr_capacity(size_type rows, double load_factor)
{
  auto const requested = static_cast<cuda::std::uint64_t>(
    std::ceil(static_cast<double>(std::max(rows, size_type{1})) / load_factor));
  auto const capacity = cuda::std::bit_ceil(requested);
  CUDF_EXPECTS(capacity <= std::numeric_limits<cuda::std::uint32_t>::max(),
               "streaming HashCSR table capacity is not representable",
               std::overflow_error);
  return static_cast<cuda::std::uint32_t>(capacity);
}

template <typename Equal>
struct streaming_build_equal {
  Equal const* comparators;

  __device__ bool operator()(streaming_row_ref candidate, streaming_row_ref existing) const noexcept
  {
    auto const existing_batch = streaming_ref_batch(existing);
    return comparators[existing_batch](streaming_ref_row(candidate), streaming_ref_row(existing));
  }
};

template <typename Equal>
struct streaming_probe_equal {
  Equal const* comparators;

  __device__ bool operator()(size_type probe, streaming_row_ref existing) const noexcept
  {
    return comparators[streaming_ref_batch(existing)](probe, streaming_ref_row(existing));
  }
};

template <typename Equal, typename Hasher>
CUDF_KERNEL void streaming_delta_build_kernel(size_type rows,
                                              bitmask_type const* valid_rows,
                                              size_type batch_id,
                                              cuda::std::uint32_t* slots,
                                              streaming_row_ref* row_refs,
                                              streaming_hash_table_ref map,
                                              Equal equal,
                                              Hasher hasher)
{
  for (auto row = grid_1d::global_thread_id(); row < rows; row += grid_1d::grid_stride()) {
    auto const index   = static_cast<size_type>(row);
    auto const row_ref = make_streaming_row_ref(batch_id, index);
    row_refs[index]    = row_ref;
    if (valid_rows != nullptr && !cudf::bit_is_set(valid_rows, index)) {
      slots[index] = streaming_empty_slot;
      continue;
    }
    auto const slot = map.insert(hasher(index), row_ref, equal);
    slots[index]    = slot == map.capacity ? streaming_empty_slot : slot;
  }
}

CUDF_KERNEL void streaming_delta_count_kernel(size_type rows,
                                              cuda::std::uint32_t const* slots,
                                              size_type* counts)
{
  for (auto row = grid_1d::global_thread_id(); row < rows; row += grid_1d::grid_stride()) {
    auto const slot = slots[static_cast<size_type>(row)];
    if (slot != streaming_empty_slot) {
      cuda::atomic_ref<size_type, cuda::thread_scope_device>{counts[slot]}.fetch_add(
        size_type{1}, cuda::memory_order_relaxed);
    }
  }
}

CUDF_KERNEL void streaming_delta_fill_kernel(size_type rows,
                                             cuda::std::uint32_t const* slots,
                                             streaming_row_ref const* row_refs,
                                             size_type const* cumulative_ends,
                                             size_type* cursors,
                                             streaming_row_ref* values)
{
  for (auto row = grid_1d::global_thread_id(); row < rows; row += grid_1d::grid_stride()) {
    auto const index = static_cast<size_type>(row);
    auto const slot  = slots[index];
    if (slot == streaming_empty_slot) { continue; }
    auto const begin = slot == 0 ? size_type{0} : cumulative_ends[slot - 1];
    auto const rank =
      cuda::atomic_ref<size_type, cuda::thread_scope_device>{cursors[slot]}.fetch_add(
        size_type{1}, cuda::memory_order_relaxed);
    values[begin + rank] = row_refs[index];
  }
}

/**
 * @brief Expand sparse run ends into the dense slot-indexed CSR end array.
 *
 * Sorting places invalid-row sentinels after all real map slots. A binary search over the
 * run-length encoded slots therefore gives the count for each map slot without either atomics or
 * a full-capacity count-buffer reset.
 */
CUDF_KERNEL void streaming_delta_sparse_ends_kernel(size_type slots,
                                                    cuda::std::uint32_t const* unique_slots,
                                                    size_type const* run_ends,
                                                    int const* num_runs,
                                                    size_type* cumulative_ends)
{
  for (auto slot = grid_1d::global_thread_id(); slot < slots; slot += grid_1d::grid_stride()) {
    auto lower = size_type{0};
    auto upper = static_cast<size_type>(*num_runs);
    while (lower < upper) {
      auto const middle = lower + (upper - lower) / 2;
      if (unique_slots[middle] <= static_cast<cuda::std::uint32_t>(slot)) {
        lower = middle + 1;
      } else {
        upper = middle;
      }
    }
    cumulative_ends[slot] = lower == 0 ? size_type{0} : run_ends[lower - 1];
  }
}

CUDF_KERNEL void streaming_delta_copy_kernel(size_type rows,
                                             streaming_row_ref const* source,
                                             streaming_row_ref* destination)
{
  for (auto row = grid_1d::global_thread_id(); row < rows; row += grid_1d::grid_stride()) {
    destination[static_cast<size_type>(row)] = source[static_cast<size_type>(row)];
  }
}

CUDF_KERNEL void streaming_csr_merge_count_kernel(size_type slots,
                                                  streaming_csr_generations_ref source,
                                                  size_type* counts)
{
  for (auto slot = grid_1d::global_thread_id(); slot < slots; slot += grid_1d::grid_stride()) {
    counts[static_cast<size_type>(slot)] = source.size(static_cast<size_type>(slot));
  }
}

CUDF_KERNEL void streaming_csr_merge_fill_kernel(size_type slots,
                                                 streaming_csr_generations_ref source,
                                                 size_type const* cumulative_ends,
                                                 streaming_row_ref* values)
{
  for (auto slot = grid_1d::global_thread_id(); slot < slots; slot += grid_1d::grid_stride()) {
    auto const index = static_cast<size_type>(slot);
    auto const begin = index == 0 ? size_type{0} : cumulative_ends[index - 1];
    for (auto ordinal = size_type{0}; ordinal < source.size(index); ++ordinal) {
      values[begin + ordinal] = source.at(index, ordinal);
    }
  }
}

template <typename Equal, typename Hasher>
CUDF_KERNEL void streaming_probe_count_kernel(size_type rows,
                                              bitmask_type const* valid_rows,
                                              cuda::std::uint32_t* slots,
                                              size_type* counts,
                                              streaming_hash_table_ref map,
                                              streaming_csr_generations_ref csr,
                                              Equal equal,
                                              Hasher hasher)
{
  for (auto row = grid_1d::global_thread_id(); row < rows; row += grid_1d::grid_stride()) {
    auto const index = static_cast<size_type>(row);
    auto slot        = map.capacity;
    if (valid_rows == nullptr || cudf::bit_is_set(valid_rows, index)) {
      slot = map.find(hasher(index), index, equal);
    }
    slots[index]  = slot == map.capacity ? streaming_empty_slot : slot;
    counts[index] = slot == map.capacity ? size_type{0} : csr.size(slot);
  }
}

CUDF_KERNEL void streaming_probe_fill_kernel(size_type rows,
                                             cuda::std::uint32_t const* slots,
                                             size_type const* offsets,
                                             streaming_csr_generations_ref csr,
                                             size_type* left_indices,
                                             size_type* batch_indices,
                                             size_type* row_indices)
{
  for (auto row = grid_1d::global_thread_id(); row < rows; row += grid_1d::grid_stride()) {
    auto const left = static_cast<size_type>(row);
    auto const slot = slots[left];
    if (slot == streaming_empty_slot) { continue; }
    auto const output_begin = left == 0 ? size_type{0} : offsets[left - 1];
    for (auto i = size_type{0}; i < csr.size(slot); ++i) {
      auto const row_ref              = csr.at(slot, i);
      auto const batch                = streaming_ref_batch(row_ref);
      left_indices[output_begin + i]  = left;
      batch_indices[output_begin + i] = batch;
      row_indices[output_begin + i]   = streaming_ref_row(row_ref);
    }
  }
}

template <typename RowEqual>
struct dynamic_build_ref_equal {
  RowEqual const* comparators;

  __device__ bool operator()(cuda::std::uint64_t candidate,
                             cuda::std::uint64_t representative) const noexcept
  {
    auto const candidate_row        = static_cast<size_type>(candidate);
    auto const representative_row   = static_cast<size_type>(representative);
    auto const representative_batch = static_cast<size_type>(representative >> 32);
    return comparators[representative_batch](candidate_row, representative_row);
  }
};

template <typename RowEqual>
struct dynamic_probe_ref_equal {
  RowEqual const* comparators;
  __device__ bool operator()(cuda::std::uint64_t probe,
                             cuda::std::uint64_t representative) const noexcept
  {
    return comparators[static_cast<size_type>(representative >> 32)](
      static_cast<size_type>(probe), static_cast<size_type>(representative));
  }
};

}  // namespace

class streaming_hash_join_impl {
 public:
  // Empty copy of the right-side schema, owned so validation never depends on caller lifetimes.
  std::unique_ptr<table> right_schema;
  std::vector<size_type> right_key_indices;
  size_type total_right_rows;
  size_type max_num_batches;
  bool has_nulls;
  null_equality compare_nulls;
  double load_factor;
  std::atomic<size_type> inserted_rows{0};
  std::atomic<size_type> inserted_batches{0};
  bool has_nested_keys{false};
  batch_hash_layout layout;
  bool const use_base_deltas;

  // Declared before the hash table so the resource outlives allocations that refer to it.
  cudf::memory_resources mr;
  hash_table_type hash_table;
  // The segmented key map is the result authority when this implementation is selected. It
  // resolves equal hashes with full row equality, so it does not need the legacy multiset as a
  // shadow index.
  std::optional<segmented_dynamic_csr> dynamic_csr;
  // One representative build-row reference per distinct key slot. The dynamic CSR pages are
  // indexed by these slots rather than by raw hashes, so equal-hash unequal keys stay separate.
  std::optional<rmm::device_uvector<dynamic_csr_key_slot>> dynamic_key_map;

  std::vector<std::shared_ptr<row::equality::preprocessed_table>> preprocessed_right;
  // Views of the inserted partitions, indexed by batch ID so callers can resolve the batch IDs
  // reported by `inner_join()` without tracking concurrent insert ordering themselves.
  std::vector<table_view> right_partitions;

  // Option 1 state. `key_entries` is a persistent distinct-key map (one representative global
  // row per equality class). Every insert appends only a sparse (slot, global-row) delta. The
  // CSR is materialized exactly once, on the first probe after the build is sealed.
  cuda::std::uint32_t const base_delta_capacity;
  mutable rmm::device_uvector<streaming_key_slot> base_delta_entries;
  std::vector<std::unique_ptr<streaming_delta>> base_delta_batches;
  // Batches inserted after the canonical base is sealed.  They stay as independently
  // materialized CSRs until the compaction policy publishes their replacement base.
  mutable std::vector<streaming_delta*> base_delta_generations;
  mutable size_type base_delta_generation_rows{0};
  mutable std::unique_ptr<rmm::device_uvector<size_type>> canonical_ends;
  mutable std::unique_ptr<rmm::device_uvector<streaming_row_ref>> canonical_values;
  // Sealing is part of the streaming lifecycle and may happen repeatedly after new batches are
  // inserted. Keep its fixed-capacity scratch in the join state so a re-seal does not allocate
  // and free O(table_capacity) temporaries before every scan/fill pass.
  mutable rmm::device_uvector<size_type> base_delta_counts;
  mutable rmm::device_uvector<size_type> base_delta_cursors;
  mutable std::unique_ptr<rmm::device_buffer> base_delta_scan_temp;
  mutable std::mutex base_delta_mutex;
  mutable bool base_delta_sealed{false};

  streaming_hash_join_impl(table_view const& schema,
                           std::span<size_type const> key_indices,
                           size_type total_rows,
                           size_type maximum_batches,
                           nullable_join nullable,
                           null_equality nulls_equal,
                           double load_factor,
                           cuda::stream_ref stream,
                           cudf::memory_resources resource)
    // Deep copy a zero-row slice rather than `empty_like`, which drops children such as a
    // dictionary's keys column and would make `have_same_types` reject every partition.
    : right_schema{std::make_unique<table>(
        cudf::slice(schema, {0, 0}).front(), stream, resource.get_output_mr())},
      right_key_indices{key_indices.begin(), key_indices.end()},
      total_right_rows{total_rows},
      max_num_batches{checked_batch_count(maximum_batches)},
      has_nulls{nullable == nullable_join::YES},
      compare_nulls{nulls_equal},
      load_factor{checked_load_factor(load_factor)},
      layout{max_num_batches},
      use_base_deltas{use_base_csr_batch_deltas()},
      mr{resource},
      hash_table{
        cuco::extent{[&] {
          CUDF_EXPECTS(total_rows >= 0,
                       "streaming_hash_join requires total_right_rows >= 0.",
                       std::invalid_argument);
          return static_cast<std::size_t>(total_rows);
        }()},
        checked_load_factor(load_factor),
        cuco::empty_key{slot_type{std::numeric_limits<hash_value_type>::max(), cudf::JoinNoMatch}},
        always_not_equal{},
        probing_scheme{masked_hasher1{layout.hash_mask}, masked_hasher2{layout.hash_mask}},
        {},
        {},
        rmm::mr::polymorphic_allocator<char>{mr.get_output_mr()},
        stream.get()},
      preprocessed_right(static_cast<std::size_t>(max_num_batches)),
      right_partitions(static_cast<std::size_t>(max_num_batches)),
      base_delta_capacity{
        streaming_hash_csr_capacity(total_rows, checked_load_factor(load_factor))},
      base_delta_entries(base_delta_capacity, stream, mr.get_output_mr()),
      base_delta_batches(static_cast<std::size_t>(max_num_batches)),
      base_delta_counts(base_delta_capacity, stream, mr.get_temporary_mr()),
      base_delta_cursors(base_delta_capacity, stream, mr.get_temporary_mr())
  {
    CUDF_EXPECTS(right_schema->num_columns() > 0,
                 "streaming_hash_join requires at least one right-side column.",
                 std::invalid_argument);
    CUDF_EXPECTS(!right_key_indices.empty(),
                 "streaming_hash_join requires at least one right-side key column.",
                 std::invalid_argument);
    auto const schema_size = right_schema->num_columns();
    for (auto const index : right_key_indices) {
      CUDF_EXPECTS(index >= 0 && index < schema_size,
                   "streaming_hash_join key index is out of range for the provided schema.",
                   std::invalid_argument);
    }
    // The schema exemplar carries nesting, so this is settled once here rather than on first
    // insert, which is what previously forced the insert path to lock.
    has_nested_keys =
      cudf::detail::has_nested_columns(right_schema->view().select(right_key_indices));
    if (use_base_deltas) {
      CUDF_CUDA_TRY(cudaMemsetAsync(base_delta_entries.data(),
                                    0xff,
                                    base_delta_entries.size() * sizeof(streaming_key_slot),
                                    stream.get()));
      std::size_t scan_temp_bytes{};
      CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(nullptr,
                                                  scan_temp_bytes,
                                                  base_delta_counts.data(),
                                                  base_delta_counts.data(),
                                                  base_delta_capacity,
                                                  stream.get()));
      base_delta_scan_temp =
        std::make_unique<rmm::device_buffer>(scan_temp_bytes, stream, mr.get_temporary_mr());
    }
    if (use_segmented_dynamic_csr() && total_right_rows > 0) {
      auto const map_capacity = segmented_dynamic_key_map_capacity(total_right_rows);
      dynamic_csr.emplace(total_right_rows, map_capacity, stream, mr.get_output_mr());
      dynamic_key_map.emplace(static_cast<std::size_t>(map_capacity), stream, mr.get_output_mr());
      CUDF_CUDA_TRY(cudaMemsetAsync(dynamic_key_map->data(),
                                    0xff,
                                    dynamic_key_map->size() * sizeof(dynamic_csr_key_slot),
                                    stream.get()));
    }
  }

  void insert(table_view const& right_partition, cuda::stream_ref stream)
  {
    std::unique_lock base_delta_lock{base_delta_mutex, std::defer_lock};
    auto append_base_delta_generation = false;
    if (use_base_deltas) {
      base_delta_lock.lock();
      // A sealed canonical CSR is immutable. Later batches are independently materialized and
      // concatenated at probe time until the compaction policy replaces the base.
      append_base_delta_generation = base_delta_sealed;
    }
    CUDF_EXPECTS(right_partition.num_columns() == right_schema->num_columns(),
                 "streaming_hash_join: inserted partition column count does not match schema.",
                 std::invalid_argument);
    CUDF_EXPECTS(cudf::have_same_types(right_schema->view(), right_partition),
                 "streaming_hash_join: inserted partition schema does not match the schema passed "
                 "to the constructor.",
                 cudf::data_type_error);
    auto const keys = right_partition.select(right_key_indices);

    auto preprocessed = row::equality::preprocessed_table::create(keys, stream, mr.get_output_mr());
    auto const batch_rows = keys.num_rows();
    auto row_bitmask      = [&]() -> std::optional<cuda::device_buffer<std::byte>> {
      if (batch_rows > 0 && compare_nulls == null_equality::UNEQUAL && nullable(keys)) {
        return cudf::detail::bitmask_and(keys, stream, mr.get_temporary_mr()).first;
      }
      return std::nullopt;
    }();

    auto old_inserted_rows = inserted_rows.load(std::memory_order_relaxed);
    do {
      CUDF_EXPECTS(batch_rows <= total_right_rows - old_inserted_rows,
                   "streaming_hash_join: cumulative inserted rows would exceed total_right_rows.",
                   std::invalid_argument);
    } while (!inserted_rows.compare_exchange_weak(
      old_inserted_rows, old_inserted_rows + batch_rows, std::memory_order_relaxed));

    auto batch_id = inserted_batches.load(std::memory_order_relaxed);
    do {
      if (batch_id >= max_num_batches) {
        inserted_rows.fetch_sub(batch_rows, std::memory_order_relaxed);
        CUDF_FAIL("streaming_hash_join: number of inserted batches would exceed max_num_batches.",
                  std::invalid_argument);
      }
    } while (
      !inserted_batches.compare_exchange_weak(batch_id, batch_id + 1, std::memory_order_relaxed));

    preprocessed_right[batch_id] = std::move(preprocessed);
    right_partitions[batch_id]   = right_partition;

    if (batch_rows > 0) {
      if (use_base_deltas) {
        // Publish both boundaries before launching. The equality object decodes every persisted
        // representative through these boundaries, so full hash collisions are resolved by row
        // equality across *all* prior batches rather than by a batch ID packed into the hash.
        auto delta   = std::make_unique<streaming_delta>();
        delta->slots = std::make_unique<rmm::device_uvector<cuda::std::uint32_t>>(
          batch_rows, stream, mr.get_output_mr());
        delta->rows = std::make_unique<rmm::device_uvector<streaming_row_ref>>(
          batch_rows, stream, mr.get_output_mr());
        auto const map   = streaming_hash_table_ref{base_delta_entries.data(), base_delta_capacity};
        auto build_delta = [&](auto const& comparators, auto const& row_hasher) {
          auto const equal  = streaming_build_equal{comparators.data()};
          auto const config = grid_1d{batch_rows, 256};
          streaming_delta_build_kernel<<<config.num_blocks,
                                         config.num_threads_per_block,
                                         0,
                                         stream.get()>>>(
            batch_rows,
            row_bitmask ? static_cast<bitmask_type const*>(row_bitmask->data()) : nullptr,
            batch_id,
            delta->slots->data(),
            delta->rows->data(),
            map,
            equal,
            row_hasher);
          CUDF_CUDA_TRY(cudaGetLastError());
        };
        auto const rights =
          std::span{preprocessed_right}.first(static_cast<std::size_t>(batch_id + 1));
        if (cudf::detail::is_primitive_row_op_compatible(keys)) {
          auto const comparators =
            make_device_primitive_row_comparators(preprocessed_right[batch_id],
                                                  rights,
                                                  nullate::DYNAMIC{has_nulls},
                                                  compare_nulls,
                                                  stream,
                                                  mr.get_temporary_mr());
          build_delta(
            comparators,
            row::primitive::row_hasher{nullate::DYNAMIC{has_nulls}, preprocessed_right[batch_id]});
        } else if (has_nested_keys) {
          // Nested comparators cannot share the primitive specialisation, but use the same
          // persistent map and sparse-delta kernel.
          auto const nested_comparators =
            make_device_row_comparators<true>(preprocessed_right[batch_id],
                                              rights,
                                              nullate::DYNAMIC{has_nulls},
                                              compare_nulls,
                                              stream,
                                              mr.get_temporary_mr());
          build_delta(nested_comparators,
                      row::hash::row_hasher{preprocessed_right[batch_id]}.device_hasher(
                        nullate::DYNAMIC{has_nulls}));
        } else {
          auto const comparators = make_device_row_comparators<false>(preprocessed_right[batch_id],
                                                                      rights,
                                                                      nullate::DYNAMIC{has_nulls},
                                                                      compare_nulls,
                                                                      stream,
                                                                      mr.get_temporary_mr());
          build_delta(comparators,
                      row::hash::row_hasher{preprocessed_right[batch_id]}.device_hasher(
                        nullate::DYNAMIC{has_nulls}));
        }
        if (append_base_delta_generation) {
          base_delta_generation_rows += batch_rows;
          base_delta_batches[batch_id] = std::move(delta);
          base_delta_generations.push_back(base_delta_batches[batch_id].get());
        } else {
          base_delta_batches[batch_id] = std::move(delta);
        }
        return;
      }
      auto const nulls = nullate::DYNAMIC{has_nulls};
      auto insert_rows = [&](auto const& row_hasher, auto const& dynamic_equal) {
        auto const input_begin = cudf::detail::make_counting_transform_iterator(
          0, build_pair_fn{row_hasher, layout, batch_id});

        if (compare_nulls == null_equality::EQUAL || !nullable(keys)) {
          if (dynamic_csr) {
            dynamic_csr->append(batch_rows,
                                nullptr,
                                batch_id,
                                old_inserted_rows,
                                row_hasher,
                                {dynamic_key_map->data(),
                                 static_cast<cuda::std::uint32_t>(dynamic_csr->num_buckets()),
                                 static_cast<cuda::std::uint32_t>(dynamic_csr->num_buckets() - 1)},
                                dynamic_equal,
                                stream);
            // The temporary allocator defers the comparator-vector release on this stream, so
            // the append kernel observes it without a host-side synchronization.
          } else {
            hash_table.insert_async(input_begin, input_begin + batch_rows, stream.get());
          }
        } else {
          if (dynamic_csr) {
            dynamic_csr->append(batch_rows,
                                reinterpret_cast<bitmask_type const*>(row_bitmask->data()),
                                batch_id,
                                old_inserted_rows,
                                row_hasher,
                                {dynamic_key_map->data(),
                                 static_cast<cuda::std::uint32_t>(dynamic_csr->num_buckets()),
                                 static_cast<cuda::std::uint32_t>(dynamic_csr->num_buckets() - 1)},
                                dynamic_equal,
                                stream);
          } else {
            hash_table.insert_if_async(
              input_begin,
              input_begin + batch_rows,
              cuda::counting_iterator<size_type>{0},
              row_is_valid{reinterpret_cast<bitmask_type const*>(row_bitmask->data())},
              stream.get());
          }
        }
      };

      if (cudf::detail::is_primitive_row_op_compatible(keys)) {
        auto comparators = make_device_primitive_row_comparators(
          preprocessed_right[batch_id],
          std::span{preprocessed_right}.first(static_cast<std::size_t>(batch_id + 1)),
          nulls,
          compare_nulls,
          stream,
          mr.get_temporary_mr());
        insert_rows(row::primitive::row_hasher{nulls, preprocessed_right[batch_id]},
                    dynamic_build_ref_equal{comparators.data()});
      } else {
        if (has_nested_keys) {
          auto comparators = make_device_row_comparators<true>(
            preprocessed_right[batch_id],
            std::span{preprocessed_right}.first(static_cast<std::size_t>(batch_id + 1)),
            nulls,
            compare_nulls,
            stream,
            mr.get_temporary_mr());
          insert_rows(row::hash::row_hasher{preprocessed_right[batch_id]}.device_hasher(nulls),
                      dynamic_build_ref_equal{comparators.data()});
        } else {
          auto comparators = make_device_row_comparators<false>(
            preprocessed_right[batch_id],
            std::span{preprocessed_right}.first(static_cast<std::size_t>(batch_id + 1)),
            nulls,
            compare_nulls,
            stream,
            mr.get_temporary_mr());
          insert_rows(row::hash::row_hasher{preprocessed_right[batch_id]}.device_hasher(nulls),
                      dynamic_build_ref_equal{comparators.data()});
        }
      }
    }
  }

  void materialize_delta_generation(streaming_delta& delta, cuda::stream_ref stream) const
  {
    if (delta.ends) { return; }
    auto const rows   = static_cast<size_type>(delta.slots->size());
    auto sorted_slots = rmm::device_uvector<cuda::std::uint32_t>{
      static_cast<std::size_t>(rows), stream, mr.get_temporary_mr()};
    auto sorted_rows = rmm::device_uvector<streaming_row_ref>{
      static_cast<std::size_t>(rows), stream, mr.get_temporary_mr()};
    std::size_t sort_temp_bytes{};
    CUDF_CUDA_TRY(cub::DeviceRadixSort::SortPairs(nullptr,
                                                  sort_temp_bytes,
                                                  delta.slots->data(),
                                                  sorted_slots.data(),
                                                  delta.rows->data(),
                                                  sorted_rows.data(),
                                                  rows,
                                                  0,
                                                  sizeof(cuda::std::uint32_t) * CHAR_BIT,
                                                  stream.get()));
    auto sort_temp = rmm::device_buffer{sort_temp_bytes, stream, mr.get_temporary_mr()};
    CUDF_CUDA_TRY(cub::DeviceRadixSort::SortPairs(sort_temp.data(),
                                                  sort_temp_bytes,
                                                  delta.slots->data(),
                                                  sorted_slots.data(),
                                                  delta.rows->data(),
                                                  sorted_rows.data(),
                                                  rows,
                                                  0,
                                                  sizeof(cuda::std::uint32_t) * CHAR_BIT,
                                                  stream.get()));
    auto unique_slots = rmm::device_uvector<cuda::std::uint32_t>{
      static_cast<std::size_t>(rows), stream, mr.get_temporary_mr()};
    auto run_lengths =
      rmm::device_uvector<size_type>{static_cast<std::size_t>(rows), stream, mr.get_temporary_mr()};
    auto num_runs = rmm::device_uvector<int>{1, stream, mr.get_temporary_mr()};
    std::size_t encode_temp_bytes{};
    CUDF_CUDA_TRY(cub::DeviceRunLengthEncode::Encode(nullptr,
                                                     encode_temp_bytes,
                                                     sorted_slots.data(),
                                                     unique_slots.data(),
                                                     run_lengths.data(),
                                                     num_runs.data(),
                                                     rows,
                                                     stream.get()));
    auto encode_temp = rmm::device_buffer{encode_temp_bytes, stream, mr.get_temporary_mr()};
    CUDF_CUDA_TRY(cub::DeviceRunLengthEncode::Encode(encode_temp.data(),
                                                     encode_temp_bytes,
                                                     sorted_slots.data(),
                                                     unique_slots.data(),
                                                     run_lengths.data(),
                                                     num_runs.data(),
                                                     rows,
                                                     stream.get()));
    auto run_ends =
      rmm::device_uvector<size_type>{static_cast<std::size_t>(rows), stream, mr.get_temporary_mr()};
    std::size_t scan_temp_bytes{};
    CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(
      nullptr, scan_temp_bytes, run_lengths.data(), run_ends.data(), rows, stream.get()));
    auto scan_temp = rmm::device_buffer{scan_temp_bytes, stream, mr.get_temporary_mr()};
    CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(
      scan_temp.data(), scan_temp_bytes, run_lengths.data(), run_ends.data(), rows, stream.get()));
    delta.ends = std::make_unique<rmm::device_uvector<size_type>>(
      base_delta_capacity, stream, cudf::get_current_device_resource_ref());
    auto const config = grid_1d{static_cast<size_type>(base_delta_capacity), 256};
    streaming_delta_sparse_ends_kernel<<<config.num_blocks,
                                         config.num_threads_per_block,
                                         0,
                                         stream.get()>>>(
      static_cast<size_type>(base_delta_capacity),
      unique_slots.data(),
      run_ends.data(),
      num_runs.data(),
      delta.ends->data());
    CUDF_CUDA_TRY(cudaGetLastError());
    size_type total{};
    cudf::detail::cuda_memcpy_async(
      cudf::host_span<size_type>{&total, 1},
      cudf::device_span<size_type const>{delta.ends->data() + base_delta_capacity - 1, 1},
      stream);
    stream.sync();
    delta.values = std::make_unique<rmm::device_uvector<streaming_row_ref>>(
      total, stream, cudf::get_current_device_resource_ref());
    if (total > 0) {
      auto const copy_config = grid_1d{total, 256};
      streaming_delta_copy_kernel<<<copy_config.num_blocks,
                                    copy_config.num_threads_per_block,
                                    0,
                                    stream.get()>>>(
        total, sorted_rows.data(), delta.values->data());
      CUDF_CUDA_TRY(cudaGetLastError());
    }
  }

  void materialize_delta_generations(cuda::stream_ref stream) const
  {
    for (auto* generation : base_delta_generations) {
      materialize_delta_generation(*generation, stream);
    }
  }

  void compact_delta_generations(cuda::stream_ref stream) const
  {
    if (base_delta_generations.empty()) { return; }
    std::vector<streaming_csr_ref> host_generations;
    host_generations.reserve(base_delta_generations.size());
    for (auto const* generation : base_delta_generations) {
      host_generations.push_back({generation->ends->data(), generation->values->data()});
    }
    auto device_generations = rmm::device_uvector<streaming_csr_ref>{
      host_generations.size(), stream, mr.get_temporary_mr()};
    cudf::detail::cuda_memcpy_async(
      cudf::device_span<streaming_csr_ref>{device_generations.data(), host_generations.size()},
      cudf::host_span<streaming_csr_ref const>{host_generations.data(), host_generations.size()},
      stream);
    auto const source =
      streaming_csr_generations_ref{{canonical_ends->data(), canonical_values->data()},
                                    device_generations.data(),
                                    static_cast<size_type>(host_generations.size())};
    auto const config = grid_1d{static_cast<size_type>(base_delta_capacity), 256};
    streaming_csr_merge_count_kernel<<<config.num_blocks,
                                       config.num_threads_per_block,
                                       0,
                                       stream.get()>>>(
      static_cast<size_type>(base_delta_capacity), source, base_delta_counts.data());
    CUDF_CUDA_TRY(cudaGetLastError());
    auto merged_ends = std::make_unique<rmm::device_uvector<size_type>>(
      base_delta_capacity, stream, cudf::get_current_device_resource_ref());
    auto merged_scan_bytes = base_delta_scan_temp->size();
    CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(base_delta_scan_temp->data(),
                                                merged_scan_bytes,
                                                base_delta_counts.data(),
                                                merged_ends->data(),
                                                base_delta_capacity,
                                                stream.get()));
    size_type total{};
    cudf::detail::cuda_memcpy_async(
      cudf::host_span<size_type>{&total, 1},
      cudf::device_span<size_type const>{merged_ends->data() + base_delta_capacity - 1, 1},
      stream);
    stream.sync();
    auto merged_values = std::make_unique<rmm::device_uvector<streaming_row_ref>>(
      total, stream, cudf::get_current_device_resource_ref());
    streaming_csr_merge_fill_kernel<<<config.num_blocks,
                                      config.num_threads_per_block,
                                      0,
                                      stream.get()>>>(static_cast<size_type>(base_delta_capacity),
                                                      source,
                                                      merged_ends->data(),
                                                      merged_values->data());
    CUDF_CUDA_TRY(cudaGetLastError());
    // Do not free the old base or generation values until the merge fill has consumed them.
    stream.sync();
    canonical_ends   = std::move(merged_ends);
    canonical_values = std::move(merged_values);
    base_delta_generations.clear();
    base_delta_generation_rows = 0;
  }

  void seal_base_deltas(cuda::stream_ref stream) const
  {
    std::lock_guard lock{base_delta_mutex};
    if (base_delta_sealed) { return; }
    auto const num_batches = inserted_batches.load(std::memory_order_relaxed);
    CUDF_CUDA_TRY(cudaMemsetAsync(
      base_delta_counts.data(), 0, base_delta_counts.size() * sizeof(size_type), stream.get()));
    for (size_type batch = 0; batch < num_batches; ++batch) {
      auto const& delta = base_delta_batches[batch];
      if (!delta || delta->slots->size() == 0) { continue; }
      auto const config = grid_1d{static_cast<size_type>(delta->slots->size()), 256};
      streaming_delta_count_kernel<<<config.num_blocks,
                                     config.num_threads_per_block,
                                     0,
                                     stream.get()>>>(static_cast<size_type>(delta->slots->size()),
                                                     delta->slots->data(),
                                                     base_delta_counts.data());
      CUDF_CUDA_TRY(cudaGetLastError());
    }
    canonical_ends = std::make_unique<rmm::device_uvector<size_type>>(
      base_delta_capacity, stream, cudf::get_current_device_resource_ref());
    auto canonical_scan_bytes = base_delta_scan_temp->size();
    CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(base_delta_scan_temp->data(),
                                                canonical_scan_bytes,
                                                base_delta_counts.data(),
                                                canonical_ends->data(),
                                                base_delta_capacity,
                                                stream.get()));
    size_type total{};
    cudf::detail::cuda_memcpy_async(
      cudf::host_span<size_type>{&total, 1},
      cudf::device_span<size_type const>{canonical_ends->data() + base_delta_capacity - 1, 1},
      stream);
    stream.sync();
    canonical_values = std::make_unique<rmm::device_uvector<streaming_row_ref>>(
      total, stream, cudf::get_current_device_resource_ref());
    CUDF_CUDA_TRY(cudaMemsetAsync(
      base_delta_cursors.data(), 0, base_delta_cursors.size() * sizeof(size_type), stream.get()));
    for (size_type batch = 0; batch < num_batches; ++batch) {
      auto const& delta = base_delta_batches[batch];
      if (!delta || delta->slots->size() == 0) { continue; }
      auto const config = grid_1d{static_cast<size_type>(delta->slots->size()), 256};
      streaming_delta_fill_kernel<<<config.num_blocks,
                                    config.num_threads_per_block,
                                    0,
                                    stream.get()>>>(static_cast<size_type>(delta->slots->size()),
                                                    delta->slots->data(),
                                                    delta->rows->data(),
                                                    canonical_ends->data(),
                                                    base_delta_cursors.data(),
                                                    canonical_values->data());
      CUDF_CUDA_TRY(cudaGetLastError());
    }
    base_delta_sealed = true;
  }

  template <typename Equal, typename Hasher>
  auto probe_base_deltas_impl(table_view const& left,
                              bitmask_type const* valid_rows,
                              Equal const& comparators,
                              Hasher const& row_hasher,
                              streaming_csr_generations_ref csr,
                              std::optional<std::size_t> output_size,
                              cuda::stream_ref stream,
                              cudf::memory_resources resources) const
  {
    if (left.num_rows() == 0) {
      CUDF_EXPECTS(!output_size || *output_size == 0,
                   "The provided join output size does not match an empty left table",
                   std::invalid_argument);
      auto empty_left =
        std::make_unique<rmm::device_uvector<size_type>>(0, stream, resources.get_output_mr());
      auto empty_batches =
        std::make_unique<rmm::device_uvector<size_type>>(0, stream, resources.get_output_mr());
      auto empty_rows =
        std::make_unique<rmm::device_uvector<size_type>>(0, stream, resources.get_output_mr());
      return std::pair{std::move(empty_left),
                       std::pair{std::move(empty_batches), std::move(empty_rows)}};
    }
    auto const map = streaming_hash_table_ref{base_delta_entries.data(), base_delta_capacity};
    auto slots     = rmm::device_uvector<cuda::std::uint32_t>{
      static_cast<std::size_t>(left.num_rows()), stream, resources.get_temporary_mr()};
    auto offsets = rmm::device_uvector<size_type>{
      static_cast<std::size_t>(left.num_rows()), stream, resources.get_temporary_mr()};
    auto const equal  = streaming_probe_equal{comparators.data()};
    auto const config = grid_1d{left.num_rows(), 256};
    streaming_probe_count_kernel<<<config.num_blocks,
                                   config.num_threads_per_block,
                                   0,
                                   stream.get()>>>(
      left.num_rows(), valid_rows, slots.data(), offsets.data(), map, csr, equal, row_hasher);
    CUDF_CUDA_TRY(cudaGetLastError());
    std::size_t temp_bytes{};
    CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(
      nullptr, temp_bytes, offsets.data(), offsets.data(), left.num_rows(), stream.get()));
    auto temp = rmm::device_buffer{temp_bytes, stream, resources.get_temporary_mr()};
    CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(
      temp.data(), temp_bytes, offsets.data(), offsets.data(), left.num_rows(), stream.get()));
    // An internal caller that already owns the exact cardinality need not pay a D2H copy and
    // host synchronization merely to rediscover it.  Preserve the validation path for the
    // public/no-size contract.
    size_type join_size{};
    if (output_size) {
      CUDF_EXPECTS(*output_size <= static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
                   "The provided join output size is too large",
                   std::overflow_error);
      join_size = static_cast<size_type>(*output_size);
    } else {
      cudf::detail::cuda_memcpy_async(
        cudf::host_span<size_type>{&join_size, 1},
        cudf::device_span<size_type const>{offsets.data() + left.num_rows() - 1, 1},
        stream);
      stream.sync();
    }
    auto left_indices = std::make_unique<rmm::device_uvector<size_type>>(
      join_size, stream, resources.get_output_mr());
    auto batch_indices = std::make_unique<rmm::device_uvector<size_type>>(
      join_size, stream, resources.get_output_mr());
    auto row_indices = std::make_unique<rmm::device_uvector<size_type>>(
      join_size, stream, resources.get_output_mr());
    if (join_size > 0) {
      streaming_probe_fill_kernel<<<config.num_blocks,
                                    config.num_threads_per_block,
                                    0,
                                    stream.get()>>>(left.num_rows(),
                                                    slots.data(),
                                                    offsets.data(),
                                                    csr,
                                                    left_indices->data(),
                                                    batch_indices->data(),
                                                    row_indices->data());
      CUDF_CUDA_TRY(cudaGetLastError());
    }
    return std::pair{std::move(left_indices),
                     std::pair{std::move(batch_indices), std::move(row_indices)}};
  }

  auto probe_base_deltas(table_view const& left,
                         std::optional<std::size_t> output_size,
                         cuda::stream_ref stream,
                         cudf::memory_resources resources) const
  {
    seal_base_deltas(stream);
    bool compact_generations = false;
    {
      std::lock_guard lock{base_delta_mutex};
      materialize_delta_generations(stream);
      auto const base_rows = static_cast<size_type>(canonical_values->size());
      compact_generations  = base_delta_generations.size() >= 8 ||
                            (base_delta_generation_rows > 0 &&
                             (base_rows == 0 || base_delta_generation_rows >= (base_rows + 3) / 4));
      if (compact_generations) { compact_delta_generations(stream); }
    }
    std::vector<streaming_csr_ref> generation_refs;
    {
      std::lock_guard lock{base_delta_mutex};
      generation_refs.reserve(base_delta_generations.size());
      for (auto const* generation : base_delta_generations) {
        generation_refs.push_back({generation->ends->data(), generation->values->data()});
      }
    }
    auto device_generation_refs = rmm::device_uvector<streaming_csr_ref>{
      generation_refs.size(), stream, resources.get_temporary_mr()};
    if (!generation_refs.empty()) {
      cudf::detail::cuda_memcpy_async(
        cudf::device_span<streaming_csr_ref>{device_generation_refs.data(), generation_refs.size()},
        cudf::host_span<streaming_csr_ref const>{generation_refs.data(), generation_refs.size()},
        stream);
    }
    auto const csr =
      streaming_csr_generations_ref{{canonical_ends->data(), canonical_values->data()},
                                    device_generation_refs.data(),
                                    static_cast<size_type>(generation_refs.size())};
    auto preprocessed_left =
      row::equality::preprocessed_table::create(left, stream, resources.get_temporary_mr());
    auto const rights = std::span{preprocessed_right}.first(
      static_cast<std::size_t>(inserted_batches.load(std::memory_order_relaxed)));
    auto const valid_rows = [&]() -> std::optional<rmm::device_buffer> {
      if (compare_nulls == null_equality::UNEQUAL && nullable(left)) {
        return cudf::detail::bitmask_and(left, stream, resources.get_temporary_mr()).first;
      }
      return std::nullopt;
    }();
    auto const mask = valid_rows ? static_cast<bitmask_type const*>(valid_rows->data()) : nullptr;
    if (cudf::detail::is_primitive_row_op_compatible(left)) {
      auto const comparators = make_device_primitive_row_comparators(preprocessed_left,
                                                                     rights,
                                                                     nullate::DYNAMIC{has_nulls},
                                                                     compare_nulls,
                                                                     stream,
                                                                     resources.get_temporary_mr());
      return probe_base_deltas_impl(
        left,
        mask,
        comparators,
        row::primitive::row_hasher{nullate::DYNAMIC{has_nulls}, preprocessed_left},
        csr,
        output_size,
        stream,
        resources);
    }
    if (has_nested_keys) {
      auto const comparators = make_device_row_comparators<true>(preprocessed_left,
                                                                 rights,
                                                                 nullate::DYNAMIC{has_nulls},
                                                                 compare_nulls,
                                                                 stream,
                                                                 resources.get_temporary_mr());
      return probe_base_deltas_impl(
        left,
        mask,
        comparators,
        row::hash::row_hasher{preprocessed_left}.device_hasher(nullate::DYNAMIC{has_nulls}),
        csr,
        output_size,
        stream,
        resources);
    }
    auto const comparators = make_device_row_comparators<false>(preprocessed_left,
                                                                rights,
                                                                nullate::DYNAMIC{has_nulls},
                                                                compare_nulls,
                                                                stream,
                                                                resources.get_temporary_mr());
    return probe_base_deltas_impl(
      left,
      mask,
      comparators,
      row::hash::row_hasher{preprocessed_left}.device_hasher(nullate::DYNAMIC{has_nulls}),
      csr,
      output_size,
      stream,
      resources);
  }

  template <bool has_nested, bool use_primitive>
  auto probe(table_view const& left,
             std::optional<std::size_t> output_size,
             cuda::stream_ref stream,
             cudf::memory_resources resources) const
  {
    static_assert(!(has_nested && use_primitive),
                  "primitive row comparators are never used for nested keys");
    auto preprocessed_left =
      row::equality::preprocessed_table::create(left, stream, resources.get_temporary_mr());
    auto const num_batches = inserted_batches.load(std::memory_order_relaxed);
    auto const right_comparators =
      std::span{preprocessed_right}.first(static_cast<std::size_t>(num_batches));
    auto comparators = [&] {
      if constexpr (use_primitive) {
        return make_device_primitive_row_comparators(preprocessed_left,
                                                     right_comparators,
                                                     nullate::DYNAMIC{has_nulls},
                                                     compare_nulls,
                                                     stream,
                                                     resources.get_temporary_mr());
      } else {
        return make_device_row_comparators<has_nested>(preprocessed_left,
                                                       right_comparators,
                                                       nullate::DYNAMIC{has_nulls},
                                                       compare_nulls,
                                                       stream,
                                                       resources.get_temporary_mr());
      }
    }();
    auto const equality   = n_table_pair_equal{comparators.data(), layout};
    auto const row_hasher = [&] {
      if constexpr (use_primitive) {
        return row::primitive::row_hasher{nullate::DYNAMIC{has_nulls}, preprocessed_left};
      } else {
        return row::hash::row_hasher{preprocessed_left}.device_hasher(nullate::DYNAMIC{has_nulls});
      }
    }();
    auto const input_begin =
      cudf::detail::make_counting_transform_iterator(0, probe_pair_fn{row_hasher, layout});

    if (dynamic_csr) {
      dynamic_csr->throw_if_key_map_exhausted(stream);
      auto probe_slots = rmm::device_uvector<size_type>{
        static_cast<std::size_t>(left.num_rows()), stream, resources.get_temporary_mr()};
      auto match_counts = rmm::device_uvector<size_type>{
        // `sizes_to_offsets` consumes a zero sentinel after the per-probe counts, matching the
        // HashCSR retrieval path. Without it, its returned exclusive-scan total omits the final
        // probe row and a one-row probe always produces an empty output.
        static_cast<std::size_t>(left.num_rows()) + 1,
        stream,
        resources.get_temporary_mr()};
      launch_segmented_dynamic_csr_probe_count_kernel(
        left.num_rows(),
        probe_slots.data(),
        match_counts.data(),
        dynamic_csr->ref(),
        {const_cast<dynamic_csr_key_slot*>(dynamic_key_map->data()),
         static_cast<cuda::std::uint32_t>(dynamic_csr->num_buckets()),
         static_cast<cuda::std::uint32_t>(dynamic_csr->num_buckets() - 1)},
        row_hasher,
        dynamic_probe_ref_equal{comparators.data()},
        stream);
      auto offsets = cudf::detail::make_zeroed_device_uvector_async<cuda::std::int64_t>(
        static_cast<std::size_t>(left.num_rows()) + 1, stream, resources.get_temporary_mr());
      auto const dynamic_size = cudf::detail::sizes_to_offsets(match_counts.begin(),
                                                               match_counts.end(),
                                                               offsets.begin(),
                                                               0,
                                                               stream,
                                                               resources.get_temporary_mr());
      CUDF_EXPECTS(
        dynamic_size >= 0, "streaming dynamic join output size overflowed", std::overflow_error);
      auto const result_size = static_cast<std::size_t>(dynamic_size);
      auto left_indices      = std::make_unique<rmm::device_uvector<size_type>>(
        result_size, stream, resources.get_output_mr());
      auto batch_indices = std::make_unique<rmm::device_uvector<size_type>>(
        result_size, stream, resources.get_output_mr());
      auto row_indices = std::make_unique<rmm::device_uvector<size_type>>(
        result_size, stream, resources.get_output_mr());
      auto row_refs =
        rmm::device_uvector<cuda::std::uint64_t>{result_size, stream, resources.get_temporary_mr()};
      if (result_size > 0) {
        constexpr auto threads = 256;
        auto const blocks      = static_cast<unsigned int>((result_size + threads - 1) / threads);
        segmented_dynamic_csr_retrieve_kernel<<<blocks, threads, 0, stream.get()>>>(
          static_cast<size_type>(result_size),
          left.num_rows(),
          probe_slots.data(),
          offsets.data(),
          dynamic_csr->ref(),
          row_refs.data(),
          left_indices->data());
        unpack_segmented_dynamic_refs_kernel<<<blocks, threads, 0, stream.get()>>>(
          static_cast<size_type>(result_size),
          row_refs.data(),
          batch_indices->data(),
          row_indices->data());
        CUDF_CUDA_TRY(cudaGetLastError());
      }
      return std::pair{std::move(left_indices),
                       std::pair{std::move(batch_indices), std::move(row_indices)}};
    }

    // The dynamic implementation resolves a single persistent key-map slot per probe row and
    // derives multiplicity from that slot's append-only segment chain. The existing multiset path
    // remains below while the experimental selector is integrated through the public dispatcher.
    // `probe_slots` and `match_counts` deliberately have the same shape as HashCSR's count pass,
    // so offsets/retrieval can share the normal join-output layout.

    auto const join_size = output_size ? *output_size
                                       : hash_table.count(input_begin,
                                                          input_begin + left.num_rows(),
                                                          equality,
                                                          hash_table.hash_function(),
                                                          stream.get());

    auto const output_mr = resources.get_output_mr();
    auto left_indices =
      std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, output_mr);
    auto batch_indices =
      std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, output_mr);
    auto row_indices =
      std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, output_mr);

    if (join_size > 0) {
      cudf::prefetch::detail::prefetch(*left_indices, stream);
      cudf::prefetch::detail::prefetch(*batch_indices, stream);
      cudf::prefetch::detail::prefetch(*row_indices, stream);
      auto const probe_output =
        cuda::transform_output_iterator{left_indices->begin(), extract_index_fn{}};
      auto const build_output = cuda::transform_output_iterator{
        cuda::zip_iterator(batch_indices->begin(), row_indices->begin()), decode_slot_fn{layout}};
      hash_table.retrieve(input_begin,
                          input_begin + left.num_rows(),
                          equality,
                          hash_table.hash_function(),
                          probe_output,
                          build_output,
                          stream.get());
    }

    return std::pair{std::move(left_indices),
                     std::pair{std::move(batch_indices), std::move(row_indices)}};
  }

  [[nodiscard]] table_view get_partition(size_type batch_id) const
  {
    CUDF_EXPECTS(batch_id >= 0 and batch_id < inserted_batches.load(std::memory_order_relaxed),
                 "streaming_hash_join: no partition has been inserted with this batch ID.",
                 std::out_of_range);
    return right_partitions[batch_id];
  }

  auto inner_join(table_view const& left,
                  std::optional<std::size_t> output_size,
                  cuda::stream_ref stream,
                  cudf::memory_resources resources) const
  {
    CUDF_EXPECTS(inserted_batches.load(std::memory_order_relaxed) > 0,
                 "streaming_hash_join: inner_join called before any insert().",
                 std::logic_error);
    validate_hash_join_probe(right_schema->view().select(right_key_indices), left, has_nulls);

    if (left.num_rows() == 0 || inserted_rows.load(std::memory_order_relaxed) == 0) {
      return std::pair{
        std::make_unique<rmm::device_uvector<size_type>>(0, stream, resources.get_output_mr()),
        std::pair{
          std::make_unique<rmm::device_uvector<size_type>>(0, stream, resources.get_output_mr()),
          std::make_unique<rmm::device_uvector<size_type>>(0, stream, resources.get_output_mr())}};
    }
    if (use_base_deltas) { return probe_base_deltas(left, output_size, stream, resources); }
    if (cudf::detail::is_primitive_row_op_compatible(left)) {
      return probe<false, true>(left, output_size, stream, resources);
    }
    return has_nested_keys ? probe<true, false>(left, output_size, stream, resources)
                           : probe<false, false>(left, output_size, stream, resources);
  }
};

}  // namespace cudf::detail

namespace cudf {

streaming_hash_join::streaming_hash_join(table_view const& right_schema,
                                         std::span<size_type const> right_key_indices,
                                         size_type total_right_rows,
                                         size_type max_num_batches,
                                         nullable_join has_nulls,
                                         null_equality compare_nulls,
                                         double load_factor,
                                         cuda::stream_ref stream,
                                         cudf::memory_resources mr)
  : _impl{std::make_unique<cudf::detail::streaming_hash_join_impl>(right_schema,
                                                                   right_key_indices,
                                                                   total_right_rows,
                                                                   max_num_batches,
                                                                   has_nulls,
                                                                   compare_nulls,
                                                                   load_factor,
                                                                   stream,
                                                                   mr)}
{
}

streaming_hash_join::~streaming_hash_join()                                         = default;
streaming_hash_join::streaming_hash_join(streaming_hash_join&&) noexcept            = default;
streaming_hash_join& streaming_hash_join::operator=(streaming_hash_join&&) noexcept = default;

void streaming_hash_join::insert(table_view const& right_partition, cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  _impl->insert(right_partition, stream);
}

table_view streaming_hash_join::get_partition(size_type batch_id) const
{
  return _impl->get_partition(batch_id);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                    std::unique_ptr<rmm::device_uvector<size_type>>>>
streaming_hash_join::inner_join(table_view const& left,
                                std::optional<std::size_t> output_size,
                                cuda::stream_ref stream,
                                cudf::memory_resources mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->inner_join(left, output_size, stream, mr);
}

}  // namespace cudf
