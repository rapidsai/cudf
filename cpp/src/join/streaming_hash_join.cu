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
#include <cudf/detail/utilities/integer_utils.hpp>
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

#include <cooperative_groups.h>
#include <cub/device/device_scan.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/pair.cuh>
#include <cuco/static_multiset.cuh>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/stream>
#include <thrust/scan.h>

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
 * @brief Experimental base+deltas compaction policy, configurable for benchmark sweeps.
 *
 * The defaults are the measured streaming compromise: defer base rewrites until active delta
 * rows reach twice the base, or 16 generations are live. These are not public join options; they
 * let the selector-`0x1` prototype measure policy sensitivity without rebuilding between trials.
 */
struct base_delta_compaction_policy {
  size_type row_percent;
  size_type generation_limit;
};

base_delta_compaction_policy get_base_delta_compaction_policy()
{
  auto const parse = [](
                       char const* name, size_type fallback, size_type minimum, size_type maximum) {
    auto const value = std::getenv(name);
    if (value == nullptr) { return fallback; }
    char* end{};
    auto const parsed = std::strtoul(value, &end, 0);
    if (end == value || *end != '\0' || parsed < static_cast<unsigned long>(minimum) ||
        parsed > static_cast<unsigned long>(maximum)) {
      return fallback;
    }
    return static_cast<size_type>(parsed);
  };
  return {parse("CUDF_STREAMING_HASH_JOIN_BASE_DELTA_COMPACTION_ROW_PERCENT", 200, 1, 1600),
          parse("CUDF_STREAMING_HASH_JOIN_BASE_DELTA_COMPACTION_GENERATION_LIMIT", 16, 1, 32)};
}

/**
 * @brief Internal benchmark control for the experimental cooperative map lookup.
 *
 * A value of 0 keeps the scalar lookup.  Values 2 and 4 assign that many CUDA lanes to one
 * probe row and load a consecutive linear-probe window cooperatively.  This is deliberately an
 * implementation-only knob: selector semantics remain unchanged while the profile determines
 * whether either variant is worth retaining as the default.
 */
int get_base_delta_probe_group_size()
{
  auto const value = std::getenv("CUDF_STREAMING_HASH_JOIN_BASE_DELTA_PROBE_GROUP_SIZE");
  // The 4-lane window wins the sustained probe and interleaved measurements.  Keep the scalar
  // path explicitly selectable for shallow-probe comparisons and future workload-specific tuning.
  if (value == nullptr) { return 4; }
  char* end{};
  auto const parsed = std::strtol(value, &end, 0);
  return end != value && *end == '\0' && (parsed == 2 || parsed == 4) ? static_cast<int>(parsed)
                                                                      : 0;
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
  streaming_row_ref representative;
  hash_value_type hash;
  cuda::std::uint32_t csr_id;
};

/** @brief Collision-aware, power-of-two open-addressed map for the base+deltas experiment. */
struct streaming_hash_table_ref {
  streaming_key_slot* entries;
  cuda::std::uint32_t map_capacity;
  cuda::std::uint32_t csr_capacity;

  [[nodiscard]] __device__ cuda::std::uint32_t mask() const { return map_capacity - 1; }

  template <typename Equal>
  __device__ cuda::std::uint32_t insert(hash_value_type hash,
                                        streaming_row_ref representative,
                                        cuda::std::uint32_t candidate_csr_id,
                                        Equal equal) const
  {
    auto const normalized = hash >= streaming_pending_hash ? streaming_pending_hash - 1 : hash;
    for (cuda::std::uint32_t step = 0; step < map_capacity; ++step) {
      auto const slot = (static_cast<cuda::std::uint32_t>(normalized) + step) & mask();
      auto key = cuda::atomic_ref<hash_value_type, cuda::thread_scope_device>{entries[slot].hash};
      while (true) {
        auto current = key.load(cuda::memory_order_acquire);
        if (current == streaming_pending_hash) { continue; }
        if (current == streaming_empty_hash) {
          auto expected = streaming_empty_hash;
          if (key.compare_exchange_strong(
                expected, streaming_pending_hash, cuda::memory_order_acq_rel)) {
            if (candidate_csr_id >= csr_capacity) {
              key.store(streaming_empty_hash, cuda::memory_order_release);
              return cuda::std::uint32_t{-1};
            }
            cuda::atomic_ref<streaming_row_ref, cuda::thread_scope_device>{
              entries[slot].representative}
              .store(representative, cuda::memory_order_release);
            cuda::atomic_ref<cuda::std::uint32_t, cuda::thread_scope_device>{entries[slot].csr_id}
              .store(candidate_csr_id, cuda::memory_order_release);
            key.store(normalized, cuda::memory_order_release);
            return candidate_csr_id;
          }
          continue;
        }
        if (current == normalized) {
          auto const stored =
            cuda::atomic_ref<streaming_row_ref, cuda::thread_scope_device>{
              entries[slot].representative}
              .load(cuda::memory_order_acquire);
          if (equal(representative, stored)) {
            return cuda::atomic_ref<cuda::std::uint32_t, cuda::thread_scope_device>{
              entries[slot].csr_id}
              .load(cuda::memory_order_acquire);
          }
        }
        break;
      }
    }
    return cuda::std::uint32_t{-1};
  }

  template <typename Equal>
  __device__ cuda::std::uint32_t find(hash_value_type hash, size_type probe, Equal equal) const
  {
    auto const normalized = hash >= streaming_pending_hash ? streaming_pending_hash - 1 : hash;
    for (cuda::std::uint32_t step = 0; step < map_capacity; ++step) {
      auto const slot = (static_cast<cuda::std::uint32_t>(normalized) + step) & mask();
      auto key = cuda::atomic_ref<hash_value_type, cuda::thread_scope_device>{entries[slot].hash};
      auto current = key.load(cuda::memory_order_acquire);
      while (current == streaming_pending_hash) {
        current = key.load(cuda::memory_order_acquire);
      }
      if (current == streaming_empty_hash) { return cuda::std::uint32_t{-1}; }
      if (current == normalized) {
        auto const stored =
          cuda::atomic_ref<streaming_row_ref, cuda::thread_scope_device>{
            entries[slot].representative}
            .load(cuda::memory_order_acquire);
        if (equal(probe, stored)) {
          return cuda::atomic_ref<cuda::std::uint32_t, cuda::thread_scope_device>{
            entries[slot].csr_id}
            .load(cuda::memory_order_acquire);
        }
      }
    }
    return cuda::std::uint32_t{-1};
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

/**
 * @brief One sparse run from a post-seal delta generation.
 *
 * `next` links runs for the same dense CSR ID.  The append is intentionally unordered: the
 * `generation` field restores base-then-batch order when a probe materializes a match.
 */
constexpr auto streaming_sparse_page_rows = size_type{32};

struct streaming_sparse_page {
  streaming_row_ref values[streaming_sparse_page_rows];
  size_type count;
  cuda::std::uint32_t next;
};

struct streaming_sparse_node {
  size_type begin;
  size_type end;
  size_type generation;
  streaming_sparse_node* next;
  streaming_row_ref const* values;
  streaming_sparse_page const* pages;
  cuda::std::uint32_t first_page;
  cuda::std::uint32_t tail_page;
  size_type page_size;
  int page_lock;
};

struct streaming_sparse_directory_entry {
  cuda::std::uint32_t slot;
  cuda::std::uint32_t node;
};

constexpr auto streaming_max_active_generations = size_type{8};

struct streaming_singleton_generation_descriptor {
  size_type candidate_begin;
  size_type candidate_end;
  size_type batch_id;
  size_type generation;
};

struct streaming_csr_generations_ref {
  streaming_csr_ref base;
  streaming_sparse_node* const* heads;
  streaming_singleton_generation_descriptor const* singleton_generations;
  size_type num_singleton_generations;

  [[nodiscard]] __device__ streaming_singleton_generation_descriptor const* singleton_for(
    size_type slot) const
  {
    for (auto index = size_type{0}; index < num_singleton_generations; ++index) {
      auto const& generation = singleton_generations[index];
      if (slot >= generation.candidate_begin && slot < generation.candidate_end) {
        return &generation;
      }
    }
    return nullptr;
  }

  [[nodiscard]] __device__ static size_type node_size(streaming_sparse_node const* node)
  {
    return node->pages == nullptr ? node->end - node->begin : node->page_size;
  }

  [[nodiscard]] __device__ static streaming_row_ref node_at(streaming_sparse_node const* node,
                                                            size_type ordinal)
  {
    if (node->pages == nullptr) { return node->values[node->begin + ordinal]; }
    auto page_index = node->first_page;
    while (ordinal >= node->pages[page_index].count) {
      ordinal -= node->pages[page_index].count;
      page_index = node->pages[page_index].next;
    }
    return node->pages[page_index].values[ordinal];
  }

  [[nodiscard]] __device__ size_type size(size_type slot) const
  {
    auto count = base.size(slot);
    if (singleton_for(slot) != nullptr) { ++count; }
    if (heads == nullptr) { return count; }
    for (auto node = heads[slot]; node != nullptr; node = node->next) {
      count += node_size(node);
    }
    return count;
  }

  [[nodiscard]] __device__ streaming_row_ref at(size_type slot, size_type ordinal) const
  {
    auto const base_size = base.size(slot);
    if (ordinal < base_size) { return base.values[base.begin(slot) + ordinal]; }
    ordinal -= base_size;
    auto const singleton = singleton_for(slot);
    if ((heads == nullptr || heads[slot] == nullptr) && singleton != nullptr) {
      return make_streaming_row_ref(singleton->batch_id, slot - singleton->candidate_begin);
    }
    if (heads == nullptr) { return streaming_row_ref{0}; }
    // The compaction policy materializes at most eight active generations before replacing the
    // base.  Gather and insertion-sort their one run per CSR ID, rather than binary-searching
    // every generation's sparse run table for every probe output.
    streaming_sparse_node const* ordered[streaming_max_active_generations];
    auto count                = size_type{0};
    auto too_many_generations = false;
    for (auto node = heads[slot]; node != nullptr; node = node->next) {
      // Inserts can accumulate more than the compaction threshold before their next probe.  The
      // usual <=8 case stays on the stack-resident ordered fast path; the rare larger case
      // below preserves correctness until that probe compacts the overlay.
      if (count == streaming_max_active_generations) {
        too_many_generations = true;
        break;
      }
      auto position = count;
      while (position > 0 && ordered[position - 1]->generation > node->generation) {
        ordered[position] = ordered[position - 1];
        --position;
      }
      ordered[position] = node;
      ++count;
    }
    if (!too_many_generations && singleton == nullptr) {
      for (auto index = size_type{0}; index < count; ++index) {
        auto const size = node_size(ordered[index]);
        if (ordinal < size) { return node_at(ordered[index], ordinal); }
        ordinal -= size;
      }
    } else {
      // The host policy will compact this overlay immediately after the current materialization,
      // but an application may have accumulated more than eight batches without probing.  Select
      // the next oldest node without a fixed-size local allocation in that exceptional case.
      auto previous_generation                   = size_type{0};
      auto have_previous                         = false;
      auto singleton_consumed                    = false;
      streaming_sparse_node const* previous_node = nullptr;
      while (true) {
        streaming_sparse_node const* next = nullptr;
        for (auto node = heads[slot]; node != nullptr; node = node->next) {
          if (node != previous_node &&
              (!have_previous || node->generation >= previous_generation) &&
              (next == nullptr || node->generation < next->generation)) {
            next = node;
          }
        }
        auto const singleton_is_next =
          singleton != nullptr && !singleton_consumed &&
          (!have_previous || singleton->generation >= previous_generation) &&
          (next == nullptr || singleton->generation <= next->generation);
        if (next == nullptr && !singleton_is_next) { break; }
        if (singleton_is_next) {
          if (ordinal == 0) {
            return make_streaming_row_ref(singleton->batch_id, slot - singleton->candidate_begin);
          }
          --ordinal;
          previous_generation = singleton->generation;
          singleton_consumed  = true;
        } else {
          auto const size = node_size(next);
          if (ordinal < size) { return node_at(next, ordinal); }
          ordinal -= size;
          previous_generation = next->generation;
          previous_node       = next;
        }
        have_previous = true;
      }
    }
    return streaming_row_ref{0};  // Unreachable when ordinal < size(slot).
  }

  /** Copy one CSR row while resolving every sparse source range only once. */
  __device__ void copy_to(size_type slot, streaming_row_ref* destination) const
  {
    auto output = size_type{0};
    for (auto index = base.begin(slot); index < base.cumulative_ends[slot]; ++index) {
      destination[output++] = base.values[index];
    }

    auto const singleton = singleton_for(slot);
    if ((heads == nullptr || heads[slot] == nullptr) && singleton != nullptr) {
      destination[output++] =
        make_streaming_row_ref(singleton->batch_id, slot - singleton->candidate_begin);
      return;
    }
    if (heads == nullptr) { return; }

    streaming_sparse_node const* ordered[streaming_max_active_generations];
    auto count                = size_type{0};
    auto too_many_generations = false;
    for (auto node = heads[slot]; node != nullptr; node = node->next) {
      if (count == streaming_max_active_generations) {
        too_many_generations = true;
        break;
      }
      auto position = count;
      while (position > 0 && ordered[position - 1]->generation > node->generation) {
        ordered[position] = ordered[position - 1];
        --position;
      }
      ordered[position] = node;
      ++count;
    }
    if (!too_many_generations && singleton == nullptr) {
      for (auto ordered_index = size_type{0}; ordered_index < count; ++ordered_index) {
        auto const node = ordered[ordered_index];
        for (auto ordinal = size_type{0}; ordinal < node_size(node); ++ordinal) {
          destination[output++] = node_at(node, ordinal);
        }
      }
      return;
    }

    // Queued inserts may briefly exceed the fixed local fast path. Preserve chronological order
    // without allocating device memory; compaction immediately replaces this overlay afterwards.
    auto previous_generation                   = size_type{0};
    auto have_previous                         = false;
    auto singleton_consumed                    = false;
    streaming_sparse_node const* previous_node = nullptr;
    while (true) {
      streaming_sparse_node const* next = nullptr;
      for (auto node = heads[slot]; node != nullptr; node = node->next) {
        if (node != previous_node && (!have_previous || node->generation >= previous_generation) &&
            (next == nullptr || node->generation < next->generation)) {
          next = node;
        }
      }
      auto const singleton_is_next =
        singleton != nullptr && !singleton_consumed &&
        (!have_previous || singleton->generation >= previous_generation) &&
        (next == nullptr || singleton->generation <= next->generation);
      if (next == nullptr && !singleton_is_next) { return; }
      if (singleton_is_next) {
        destination[output++] =
          make_streaming_row_ref(singleton->batch_id, slot - singleton->candidate_begin);
        previous_generation = singleton->generation;
        singleton_consumed  = true;
      } else {
        for (auto ordinal = size_type{0}; ordinal < node_size(next); ++ordinal) {
          destination[output++] = node_at(next, ordinal);
        }
        previous_generation = next->generation;
        previous_node       = next;
      }
      have_previous = true;
    }
  }
};

struct streaming_delta {
  size_type batch_id;
  size_type candidate_begin;
  size_type candidate_end;
  size_type generation;
  std::unique_ptr<rmm::device_buffer> valid_rows;
  std::unique_ptr<rmm::device_uvector<int>> overlaps_prior;
  std::unique_ptr<rmm::device_uvector<cuda::std::uint32_t>> slots;
  std::unique_ptr<rmm::device_uvector<streaming_sparse_node>> nodes;
  std::unique_ptr<rmm::device_uvector<streaming_sparse_directory_entry>> directory;
  std::unique_ptr<rmm::device_uvector<streaming_sparse_page>> pages;
  std::unique_ptr<rmm::device_uvector<cuda::std::uint32_t>> directory_next_page;
};

/**
 * @brief Orders mutations of the base+deltas lifecycle across caller streams.
 *
 * The join state serializes host-side lifecycle decisions with `base_delta_mutex`, but its
 * persistent map, CSR ends, and values are consumed asynchronously.  This event carries that
 * ownership ordering to a later caller stream without a host-side synchronization.
 */
class streaming_lifecycle_event {
 public:
  streaming_lifecycle_event()
  {
    CUDF_CUDA_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  }

  ~streaming_lifecycle_event() { cudaEventDestroy(event); }

  streaming_lifecycle_event(streaming_lifecycle_event const&)            = delete;
  streaming_lifecycle_event& operator=(streaming_lifecycle_event const&) = delete;

  void wait(cuda::stream_ref stream) const
  {
    if (recorded) { CUDF_CUDA_TRY(cudaStreamWaitEvent(stream.get(), event)); }
  }

  void record(cuda::stream_ref stream)
  {
    CUDF_CUDA_TRY(cudaEventRecord(event, stream.get()));
    recorded = true;
  }

 private:
  cudaEvent_t event{};
  bool recorded{false};  // Access is serialized by `base_delta_mutex`.
};

constexpr auto streaming_empty_slot       = cuda::std::uint32_t{-1};
constexpr auto streaming_probe_block_size = size_type{256};
constexpr auto streaming_probe_warps_per_block =
  streaming_probe_block_size / cudf::detail::warp_size;
constexpr auto streaming_probe_outputs_per_lane = size_type{32};

inline cuda::std::uint32_t streaming_hash_csr_capacity(size_type rows, double load_factor)
{
  CUDF_EXPECTS(
    rows >= 0, "streaming_hash_join requires total_right_rows >= 0.", std::invalid_argument);
  auto const requested = static_cast<cuda::std::uint64_t>(
    std::ceil(static_cast<double>(std::max(rows, size_type{1})) / load_factor));
  auto const capacity = cuda::std::bit_ceil(requested);
  CUDF_EXPECTS(capacity <= std::numeric_limits<cuda::std::uint32_t>::max(),
               "streaming HashCSR table capacity is not representable",
               std::overflow_error);
  return static_cast<cuda::std::uint32_t>(capacity);
}

inline cuda::std::uint32_t streaming_hash_csr_id_capacity(size_type rows)
{
  CUDF_EXPECTS(
    rows >= 0, "streaming_hash_join requires total_right_rows >= 0.", std::invalid_argument);
  CUDF_EXPECTS(
    static_cast<cuda::std::uint64_t>(rows) <= std::numeric_limits<cuda::std::uint32_t>::max(),
    "streaming HashCSR CSR capacity is not representable",
    std::overflow_error);
  return static_cast<cuda::std::uint32_t>(std::max(rows, size_type{1}));
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
                                              size_type global_row_begin,
                                              cuda::std::uint32_t* slots,
                                              int* overlaps_prior,
                                              size_type* preseal_counts,
                                              streaming_hash_table_ref map,
                                              Equal equal,
                                              Hasher hasher)
{
  for (auto row = grid_1d::global_thread_id(); row < rows; row += grid_1d::grid_stride()) {
    auto const index   = static_cast<size_type>(row);
    auto const row_ref = make_streaming_row_ref(batch_id, index);
    if (valid_rows != nullptr && !cudf::bit_is_set(valid_rows, index)) {
      slots[index] = streaming_empty_slot;
      continue;
    }
    auto const csr_id = map.insert(
      hasher(index), row_ref, static_cast<cuda::std::uint32_t>(global_row_begin + index), equal);
    slots[index] = csr_id;
    if (overlaps_prior != nullptr && csr_id != streaming_empty_slot &&
        csr_id != static_cast<cuda::std::uint32_t>(global_row_begin + index)) {
      cuda::atomic_ref<int, cuda::thread_scope_device>{*overlaps_prior}.store(
        1, cuda::memory_order_relaxed);
    }
    if (preseal_counts != nullptr && csr_id != streaming_empty_slot) {
      // Aggregate equal IDs in a warp before touching the canonical count array. This turns a
      // duplicate-heavy build from one contended atomic per row into one atomic per subgroup.
      auto const active  = __activemask();
      auto const matches = __match_any_sync(active, csr_id);
      auto const leader  = __ffs(static_cast<int>(matches)) - 1;
      if (static_cast<int>(threadIdx.x & (cudf::detail::warp_size - 1)) == leader) {
        cuda::atomic_ref<size_type, cuda::thread_scope_device>{preseal_counts[csr_id]}.fetch_add(
          static_cast<size_type>(__popc(matches)), cuda::memory_order_relaxed);
      }
    }
  }
}

struct streaming_delta_fill_descriptor {
  cuda::std::uint32_t const* slots;
  size_type rows;
  size_type begin;
  size_type batch_id;
};

struct streaming_delta_build_descriptor {
  bitmask_type const* valid_rows;
  cuda::std::uint32_t* slots;
  size_type rows;
  size_type begin;
  size_type batch_id;
};

template <typename Equal>
struct streaming_multibatch_build_equal {
  Equal const* comparators;
  size_type num_batches;

  __device__ bool operator()(streaming_row_ref candidate, streaming_row_ref existing) const noexcept
  {
    auto const candidate_batch = streaming_ref_batch(candidate);
    auto const existing_batch  = streaming_ref_batch(existing);
    return comparators[candidate_batch * num_batches + existing_batch](streaming_ref_row(candidate),
                                                                       streaming_ref_row(existing));
  }
};

template <typename Equal, typename Hasher>
CUDF_KERNEL void streaming_preseal_delta_build_kernel(
  size_type rows,
  streaming_delta_build_descriptor const* descriptors,
  size_type num_descriptors,
  Hasher const* hashers,
  streaming_hash_table_ref map,
  Equal equal,
  size_type* preseal_counts)
{
  for (auto row = grid_1d::global_thread_id(); row < rows; row += grid_1d::grid_stride()) {
    auto const index = static_cast<size_type>(row);
    auto first       = size_type{0};
    auto last        = num_descriptors;
    while (first + 1 < last) {
      auto const middle = first + (last - first) / 2;
      if (descriptors[middle].begin <= index) {
        first = middle;
      } else {
        last = middle;
      }
    }
    auto const descriptor = descriptors[first];
    auto const local_row  = index - descriptor.begin;
    if (descriptor.valid_rows != nullptr && !cudf::bit_is_set(descriptor.valid_rows, local_row)) {
      descriptor.slots[local_row] = streaming_empty_slot;
      continue;
    }
    auto const row_ref          = make_streaming_row_ref(descriptor.batch_id, local_row);
    auto const csr_id           = map.insert(hashers[descriptor.batch_id](local_row),
                                   row_ref,
                                   static_cast<cuda::std::uint32_t>(index),
                                   equal);
    descriptor.slots[local_row] = csr_id;
    if (csr_id != streaming_empty_slot) {
      auto const active  = __activemask();
      auto const matches = __match_any_sync(active, csr_id);
      auto const leader  = __ffs(static_cast<int>(matches)) - 1;
      if (static_cast<int>(threadIdx.x & (cudf::detail::warp_size - 1)) == leader) {
        cuda::atomic_ref<size_type, cuda::thread_scope_device>{preseal_counts[csr_id]}.fetch_add(
          static_cast<size_type>(__popc(matches)), cuda::memory_order_relaxed);
      }
    }
  }
}

CUDF_KERNEL void streaming_delta_fill_batches_kernel(
  size_type rows,
  streaming_delta_fill_descriptor const* descriptors,
  size_type num_descriptors,
  size_type const* cumulative_ends,
  size_type* cursors,
  streaming_row_ref* values)
{
  for (auto row = grid_1d::global_thread_id(); row < rows; row += grid_1d::grid_stride()) {
    auto const index = static_cast<size_type>(row);
    // Initial sealing consumes up to the configured batch count. Find the owning descriptor with
    // a bounded binary search rather than walking all prior batch boundaries per input row.
    auto first = size_type{0};
    auto last  = num_descriptors;
    while (first + 1 < last) {
      auto const middle = first + (last - first) / 2;
      if (descriptors[middle].begin <= index) {
        first = middle;
      } else {
        last = middle;
      }
    }
    auto const descriptor_index = first;
    auto const descriptor       = descriptors[descriptor_index];
    auto const local_row        = index - descriptor.begin;
    auto const slot             = descriptor.slots[local_row];
    if (slot == streaming_empty_slot) { continue; }
    auto const begin = slot == 0 ? size_type{0} : cumulative_ends[slot - 1];
    auto const rank =
      cuda::atomic_ref<size_type, cuda::thread_scope_device>{cursors[slot]}.fetch_add(
        size_type{1}, cuda::memory_order_relaxed);
    values[begin + rank] = make_streaming_row_ref(descriptor.batch_id, local_row);
  }
}

CUDF_KERNEL void streaming_delta_append_pages_kernel(size_type rows,
                                                     size_type batch_id,
                                                     size_type generation,
                                                     size_type candidate_begin,
                                                     cuda::std::uint32_t const* slots,
                                                     cuda::std::uint32_t directory_mask,
                                                     streaming_sparse_directory_entry* directory,
                                                     streaming_sparse_node* nodes,
                                                     streaming_sparse_page* pages,
                                                     cuda::std::uint32_t* next_page,
                                                     streaming_sparse_node** heads)
{
  for (auto row = grid_1d::global_thread_id(); row < rows; row += grid_1d::grid_stride()) {
    auto const index = static_cast<size_type>(row);
    auto const slot  = slots[index];
    // A row that claims its own candidate CSR ID is represented directly by the generation
    // range. Rows resolving to an earlier ID, including a same-batch duplicate, need pages.
    if (slot == streaming_empty_slot ||
        slot == static_cast<cuda::std::uint32_t>(candidate_begin + index)) {
      continue;
    }
    auto const active     = __activemask();
    auto const group      = __match_any_sync(active, slot);
    auto const lane       = threadIdx.x & (cudf::detail::warp_size - 1);
    auto const leader     = __ffs(static_cast<int>(group)) - 1;
    auto const population = static_cast<size_type>(__popc(group));
    auto const rank       = static_cast<size_type>(
      __popc(group & ((cuda::std::uint32_t{1} << lane) - cuda::std::uint32_t{1})));
    auto page       = streaming_empty_slot;
    auto page_begin = size_type{0};

    // A single lane per equal-slot warp subgroup owns directory lookup and page reservation.
    // Keeping the lock only around the reservation permits the group to write its contiguous
    // range in parallel, removing duplicate-heavy per-row lock traffic.
    if (lane == leader) {
      auto entry_index = (slot * cuda::std::uint32_t{0x9e3779b9}) & directory_mask;
      while (true) {
        auto key =
          cuda::atomic_ref<cuda::std::uint32_t, cuda::thread_scope_device>{
            directory[entry_index].slot}
            .load(cuda::memory_order_acquire);
        if (key == streaming_empty_slot) {
          auto expected = streaming_empty_slot;
          if (cuda::atomic_ref<cuda::std::uint32_t, cuda::thread_scope_device>{
                directory[entry_index].slot}
                .compare_exchange_strong(
                  expected, streaming_pending_hash, cuda::memory_order_acq_rel)) {
            auto& node      = nodes[entry_index];
            node.begin      = 0;
            node.end        = 0;
            node.values     = nullptr;
            node.pages      = pages;
            node.first_page = streaming_empty_slot;
            node.tail_page  = streaming_empty_slot;
            node.page_size  = 0;
            node.page_lock  = 0;
            node.generation = generation;
            node.next =
              cuda::atomic_ref<streaming_sparse_node*, cuda::thread_scope_device>{heads[slot]}
                .exchange(&node, cuda::memory_order_acq_rel);
            directory[entry_index].node = entry_index;
            cuda::atomic_ref<cuda::std::uint32_t, cuda::thread_scope_device>{
              directory[entry_index].slot}
              .store(slot, cuda::memory_order_release);
            key = slot;
          }
        }
        if (key == streaming_pending_hash) { continue; }
        if (key == slot) {
          auto& node = nodes[directory[entry_index].node];
          while (cuda::atomic_ref<int, cuda::thread_scope_device>{node.page_lock}.exchange(
                   1, cuda::memory_order_acquire) != 0) {}
          // A subgroup may straddle the current page boundary. Start it on a fresh page so every
          // lane receives one contiguous reservation; readers honor page counts, including any
          // intentionally unused tail capacity in the preceding page.
          if (node.tail_page == streaming_empty_slot ||
              pages[node.tail_page].count + population > streaming_sparse_page_rows) {
            page = cuda::atomic_ref<cuda::std::uint32_t, cuda::thread_scope_device>{*next_page}
                     .fetch_add(1, cuda::memory_order_relaxed);
            pages[page].count = 0;
            pages[page].next  = streaming_empty_slot;
            if (node.tail_page == streaming_empty_slot) {
              node.first_page = page;
            } else {
              pages[node.tail_page].next = page;
            }
            node.tail_page = page;
          } else {
            page = node.tail_page;
          }
          page_begin = pages[page].count;
          pages[page].count += population;
          node.page_size += population;
          cuda::atomic_ref<int, cuda::thread_scope_device>{node.page_lock}.store(
            0, cuda::memory_order_release);
          break;
        }
        entry_index = (entry_index + 1) & directory_mask;
      }
    }
    page                                  = __shfl_sync(group, page, leader);
    page_begin                            = __shfl_sync(group, page_begin, leader);
    pages[page].values[page_begin + rank] = make_streaming_row_ref(batch_id, index);
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
    source.copy_to(index, values + begin);
  }
}

template <typename Equal, typename Hasher>
CUDF_KERNEL void streaming_probe_count_kernel(size_type rows,
                                              bitmask_type const* valid_rows,
                                              cuda::std::uint32_t* slots,
                                              size_type* counts,
                                              int* all_matches_at_most_one,
                                              int* all_matches_exactly_one,
                                              streaming_hash_table_ref map,
                                              streaming_csr_generations_ref csr,
                                              Equal equal,
                                              Hasher hasher)
{
  for (auto row = grid_1d::global_thread_id(); row < rows; row += grid_1d::grid_stride()) {
    auto const index = static_cast<size_type>(row);
    auto slot        = streaming_empty_slot;
    if (valid_rows == nullptr || cudf::bit_is_set(valid_rows, index)) {
      slot = map.find(hasher(index), index, equal);
    }
    slots[index]     = slot;
    auto const count = slot == streaming_empty_slot ? size_type{0} : csr.size(slot);
    counts[index]    = count;
    // A warp reports each negative result once. The flags select fill schedules only; the
    // count/scan path remains authoritative for ordinary unhinted probes.
    auto const active   = __activemask();
    auto const multiple = __ballot_sync(active, count > size_type{1});
    auto const failed   = __ballot_sync(active, count != size_type{1});
    if ((threadIdx.x & (cudf::detail::warp_size - 1)) == 0 && multiple != 0) {
      cuda::atomic_ref<int, cuda::thread_scope_device>{*all_matches_at_most_one}.store(
        0, cuda::memory_order_relaxed);
    }
    if ((threadIdx.x & (cudf::detail::warp_size - 1)) == 0 && failed != 0) {
      cuda::atomic_ref<int, cuda::thread_scope_device>{*all_matches_exactly_one}.store(
        0, cuda::memory_order_relaxed);
    }
  }
}

/**
 * @brief Probe a power-of-two window of the open-addressed map with one CUDA tile per input row.
 *
 * The lowest empty slot terminates a linear probe sequence, so matching-hash slots after it must
 * not be considered.  The tile first evaluates every slot in order, then selects the earliest
 * full-row-equality match preceding that terminator.  This preserves scalar collision and pending
 * publication semantics while coalescing map metadata reads for long probe sequences.
 */
template <int GroupSize, typename Equal, typename Hasher>
CUDF_KERNEL void streaming_probe_count_cooperative_kernel(size_type rows,
                                                          bitmask_type const* valid_rows,
                                                          cuda::std::uint32_t* slots,
                                                          size_type* counts,
                                                          int* all_matches_at_most_one,
                                                          int* all_matches_exactly_one,
                                                          streaming_hash_table_ref map,
                                                          streaming_csr_generations_ref csr,
                                                          Equal equal,
                                                          Hasher hasher)
{
  static_assert(GroupSize == 2 || GroupSize == 4);
  auto const block  = cooperative_groups::this_thread_block();
  auto const tile   = cooperative_groups::tiled_partition<GroupSize>(block);
  auto const rank   = static_cast<int>(tile.thread_rank());
  auto const row    = (static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x) / GroupSize;
  auto const stride = (static_cast<size_type>(gridDim.x) * blockDim.x) / GroupSize;

  for (auto index = row; index < rows; index += stride) {
    auto resolved = streaming_empty_slot;
    if (valid_rows == nullptr || cudf::bit_is_set(valid_rows, index)) {
      auto const hash       = hasher(index);
      auto const normalized = hash >= streaming_pending_hash ? streaming_pending_hash - 1 : hash;
      for (cuda::std::uint32_t base = 0; base < map.map_capacity; base += GroupSize) {
        auto const slot = (static_cast<cuda::std::uint32_t>(normalized) + base + rank) & map.mask();
        auto key =
          cuda::atomic_ref<hash_value_type, cuda::thread_scope_device>{map.entries[slot].hash};
        auto current = key.load(cuda::memory_order_acquire);
        while (current == streaming_pending_hash) {
          current = key.load(cuda::memory_order_acquire);
        }

        auto const empty_mask = tile.ballot(current == streaming_empty_hash);
        auto const first_empty =
          empty_mask == 0 ? GroupSize : __ffs(static_cast<int>(empty_mask)) - 1;
        auto matches                  = false;
        cuda::std::uint32_t candidate = streaming_empty_slot;
        if (rank < first_empty && current == normalized) {
          auto const stored =
            cuda::atomic_ref<streaming_row_ref, cuda::thread_scope_device>{
              map.entries[slot].representative}
              .load(cuda::memory_order_acquire);
          if (equal(index, stored)) {
            matches = true;
            candidate =
              cuda::atomic_ref<cuda::std::uint32_t, cuda::thread_scope_device>{
                map.entries[slot].csr_id}
                .load(cuda::memory_order_acquire);
          }
        }
        auto const match_mask = tile.ballot(matches);
        if (match_mask != 0) {
          auto const winner = __ffs(static_cast<int>(match_mask)) - 1;
          resolved          = tile.shfl(candidate, winner);
          break;
        }
        if (empty_mask != 0) { break; }
      }
    }
    if (rank == 0) {
      slots[index]     = resolved;
      auto const count = resolved == streaming_empty_slot ? size_type{0} : csr.size(resolved);
      counts[index]    = count;
      if (count > size_type{1}) {
        cuda::atomic_ref<int, cuda::thread_scope_device>{*all_matches_at_most_one}.store(
          0, cuda::memory_order_relaxed);
      }
      if (count != size_type{1}) {
        cuda::atomic_ref<int, cuda::thread_scope_device>{*all_matches_exactly_one}.store(
          0, cuda::memory_order_relaxed);
      }
    }
  }
}

/**
 * @brief Materialize streaming CSR matches with an output-range-per-warp schedule.
 *
 * A probe-row schedule gives a single thread ownership of all matches for a duplicate-heavy
 * key.  Instead, each warp owns a contiguous range of output rows. This keeps writes coalesced
 * and distributes a hot key across the full grid while retaining base-before-generation order
 * through the CSR ordinal.
 */
CUDF_KERNEL void streaming_probe_fill_kernel(size_type output_size,
                                             size_type rows,
                                             size_type outputs_per_warp,
                                             cuda::std::uint32_t const* slots,
                                             size_type const* offsets,
                                             int const* all_matches_at_most_one,
                                             int const* all_matches_exactly_one,
                                             streaming_csr_generations_ref csr,
                                             size_type* left_indices,
                                             size_type* batch_indices,
                                             size_type* row_indices)
{
  if (*all_matches_at_most_one != 0) {
    auto const first_probe = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride      = static_cast<size_type>(gridDim.x) * blockDim.x;
    for (auto probe = first_probe; probe < rows; probe += stride) {
      auto const slot = slots[probe];
      if (slot == streaming_empty_slot) { continue; }
      auto const output     = *all_matches_exactly_one != 0 ? probe : offsets[probe] - 1;
      auto const row_ref    = csr.at(slot, 0);
      left_indices[output]  = probe;
      batch_indices[output] = streaming_ref_batch(row_ref);
      row_indices[output]   = streaming_ref_row(row_ref);
    }
    return;
  }

  auto const warp = cooperative_groups::tiled_partition<cudf::detail::warp_size>(
    cooperative_groups::this_thread_block());
  auto const lane_id       = static_cast<size_type>(warp.thread_rank());
  auto const warp_in_block = static_cast<size_type>(threadIdx.x) / cudf::detail::warp_size;
  auto const global_warp =
    static_cast<size_type>(blockIdx.x) * streaming_probe_warps_per_block + warp_in_block;
  auto const range_begin = outputs_per_warp * global_warp;
  if (range_begin >= output_size) { return; }
  auto const range_end = cuda::std::min<size_type>(range_begin + outputs_per_warp, output_size);

  size_type endpoint_probe{};
  if (lane_id < 2) {
    auto const endpoint = lane_id == 0 ? range_begin : range_end - 1;
    endpoint_probe =
      static_cast<size_type>(cuda::std::upper_bound(offsets, offsets + rows, endpoint) - offsets);
  }
  auto const first_probe = warp.shfl(endpoint_probe, 0);
  auto const last_probe  = warp.shfl(endpoint_probe, 1);

  for (size_type item = 0; item < streaming_probe_outputs_per_lane; ++item) {
    auto const output =
      range_begin + lane_id + item * static_cast<size_type>(cudf::detail::warp_size);
    if (output >= range_end) { continue; }
    auto const probe =
      first_probe == last_probe
        ? first_probe
        : static_cast<size_type>(
            cuda::std::upper_bound(offsets + first_probe, offsets + last_probe + 1, output) -
            offsets);
    auto const output_begin = probe == 0 ? size_type{0} : offsets[probe - 1];
    auto const row_ref      = csr.at(slots[probe], output - output_begin);
    left_indices[output]    = probe;
    batch_indices[output]   = streaming_ref_batch(row_ref);
    row_indices[output]     = streaming_ref_row(row_ref);
  }
}

void launch_streaming_probe_fill_kernel(size_type output_size,
                                        size_type rows,
                                        cuda::std::uint32_t const* slots,
                                        size_type const* offsets,
                                        int const* all_matches_at_most_one,
                                        int const* all_matches_exactly_one,
                                        streaming_csr_generations_ref csr,
                                        size_type* left_indices,
                                        size_type* batch_indices,
                                        size_type* row_indices,
                                        cuda::stream_ref stream)
{
  if (output_size == 0) { return; }
  auto const min_blocks            = size_type{2} * cudf::detail::num_multiprocessors();
  constexpr auto outputs_per_block = streaming_probe_warps_per_block *
                                     static_cast<size_type>(cudf::detail::warp_size) *
                                     streaming_probe_outputs_per_lane;
  auto const requested_blocks = cudf::util::div_rounding_up_safe(output_size, outputs_per_block);
  auto const blocks =
    static_cast<unsigned int>(cuda::std::max<size_type>(requested_blocks, min_blocks));
  auto const warps            = static_cast<size_type>(blocks) * streaming_probe_warps_per_block;
  auto const outputs_per_warp = cudf::util::div_rounding_up_safe(output_size, warps);
  streaming_probe_fill_kernel<<<blocks, streaming_probe_block_size, 0, stream.get()>>>(
    output_size,
    rows,
    outputs_per_warp,
    slots,
    offsets,
    all_matches_at_most_one,
    all_matches_exactly_one,
    csr,
    left_indices,
    batch_indices,
    row_indices);
  CUDF_CUDA_TRY(cudaGetLastError());
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
  std::optional<hash_table_type> hash_table;
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
  cuda::std::uint32_t const base_delta_csr_capacity;
  mutable rmm::device_uvector<streaming_key_slot> base_delta_entries;
  std::vector<std::unique_ptr<streaming_delta>> base_delta_batches;
  // Batches inserted after the canonical base is sealed.  They stay as independently
  // materialized CSRs until the compaction policy publishes their replacement base.
  mutable std::vector<streaming_delta*> base_delta_generations;
  mutable size_type base_delta_generation_rows{0};
  mutable std::unique_ptr<rmm::device_uvector<size_type>> canonical_ends;
  mutable std::unique_ptr<rmm::device_uvector<streaming_row_ref>> canonical_values;
  // Heads are persistent across active post-seal generations.  A generation contributes a node
  // only for CSR IDs that it actually touched, avoiding an O(CSR-capacity) end-array expansion
  // for every small streaming batch.
  mutable std::unique_ptr<rmm::device_uvector<streaming_sparse_node*>> base_delta_sparse_heads;
  mutable std::unique_ptr<rmm::device_uvector<streaming_singleton_generation_descriptor>>
    base_delta_singleton_generations;
  mutable std::vector<streaming_singleton_generation_descriptor>
    base_delta_singleton_generations_host;
  mutable std::vector<int> base_delta_overlap_host;
  // Sealing is part of the streaming lifecycle and may happen repeatedly after new batches are
  // inserted. Keep its fixed-capacity scratch in the join state so a re-seal does not allocate
  // and free O(table_capacity) temporaries before every scan/fill pass.
  mutable rmm::device_uvector<size_type> base_delta_counts;
  mutable rmm::device_uvector<size_type> base_delta_cursors;
  // One descriptor per non-empty initial batch lets seal fill the canonical CSR in one launch.
  // Both vectors are owned by the join state so the host descriptors remain valid until their
  // asynchronous H2D copy has completed.
  mutable rmm::device_uvector<streaming_delta_fill_descriptor> base_delta_fill_descriptors;
  mutable std::vector<streaming_delta_fill_descriptor> base_delta_fill_descriptors_host;
  mutable rmm::device_uvector<streaming_delta_build_descriptor> base_delta_build_descriptors;
  mutable std::vector<streaming_delta_build_descriptor> base_delta_build_descriptors_host;
  mutable std::unique_ptr<rmm::device_buffer> base_delta_scan_temp;
  mutable std::mutex base_delta_mutex;
  mutable streaming_lifecycle_event base_delta_lifecycle_event;
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
      preprocessed_right(static_cast<std::size_t>(max_num_batches)),
      right_partitions(static_cast<std::size_t>(max_num_batches)),
      base_delta_capacity{
        streaming_hash_csr_capacity(total_rows, checked_load_factor(load_factor))},
      base_delta_csr_capacity{streaming_hash_csr_id_capacity(total_rows)},
      base_delta_entries(base_delta_capacity, stream, mr.get_output_mr()),
      base_delta_batches(static_cast<std::size_t>(max_num_batches)),
      base_delta_counts(base_delta_csr_capacity, stream, mr.get_temporary_mr()),
      base_delta_cursors(base_delta_csr_capacity, stream, mr.get_temporary_mr()),
      base_delta_fill_descriptors(
        static_cast<std::size_t>(max_num_batches), stream, mr.get_temporary_mr()),
      base_delta_build_descriptors(
        static_cast<std::size_t>(max_num_batches), stream, mr.get_temporary_mr())
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
    if (!use_base_deltas && !use_segmented_dynamic_csr()) {
      hash_table.emplace(
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
        cuco::thread_scope_device,
        cuco::storage<2>{},
        rmm::mr::polymorphic_allocator<char>{mr.get_output_mr()},
        stream.get());
    }
    if (use_base_deltas) {
      CUDF_CUDA_TRY(cudaMemsetAsync(base_delta_entries.data(),
                                    0xff,
                                    base_delta_entries.size() * sizeof(streaming_key_slot),
                                    stream.get()));
      // Pre-seal build kernels accumulate slot counts directly. The buffer is repurposed by
      // compaction only after the canonical CSR has been published.
      CUDF_CUDA_TRY(cudaMemsetAsync(
        base_delta_counts.data(), 0, base_delta_counts.size() * sizeof(size_type), stream.get()));
      std::size_t scan_temp_bytes{};
      CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(nullptr,
                                                  scan_temp_bytes,
                                                  base_delta_counts.data(),
                                                  base_delta_counts.data(),
                                                  base_delta_csr_capacity,
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
    cudf::scoped_range range{"streaming_hash_join::base_deltas::insert"};
    std::unique_lock base_delta_lock{base_delta_mutex, std::defer_lock};
    auto append_base_delta_generation = false;
    if (use_base_deltas) {
      base_delta_lock.lock();
      base_delta_lifecycle_event.wait(stream);
      // A sealed canonical CSR is immutable. Later batches are independently materialized and
      // concatenated at probe time until the compaction policy replaces the base.
      append_base_delta_generation = base_delta_sealed;
      if (append_base_delta_generation) { ensure_post_seal_delta_workspace(stream); }
    }
    CUDF_EXPECTS(right_partition.num_columns() == right_schema->num_columns(),
                 "streaming_hash_join: inserted partition column count does not match schema.",
                 std::invalid_argument);
    CUDF_EXPECTS(cudf::have_same_types(right_schema->view(), right_partition),
                 "streaming_hash_join: inserted partition schema does not match the schema passed "
                 "to the constructor.",
                 cudf::data_type_error);
    auto const keys = right_partition.select(right_key_indices);
    // Queue every pre-seal batch.  The descriptor-driven flush builds the persistent map once,
    // including nested and non-primitive equality specializations, immediately before the first
    // seal.  Post-seal batches still materialize independently as sparse generations.
    auto const defer_preseal_build = use_base_deltas && !append_base_delta_generation;

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
        auto delta             = std::make_unique<streaming_delta>();
        delta->batch_id        = batch_id;
        delta->candidate_begin = old_inserted_rows;
        delta->candidate_end   = old_inserted_rows + batch_rows;
        delta->generation      = batch_id;
        if (append_base_delta_generation) {
          delta->overlaps_prior =
            std::make_unique<rmm::device_uvector<int>>(1, stream, mr.get_temporary_mr());
          CUDF_CUDA_TRY(
            cudaMemsetAsync(delta->overlaps_prior->data(), 0, sizeof(int), stream.get()));
        }
        if (defer_preseal_build && row_bitmask) {
          delta->valid_rows = std::make_unique<rmm::device_buffer>(std::move(*row_bitmask));
        }
        delta->slots = std::make_unique<rmm::device_uvector<cuda::std::uint32_t>>(
          batch_rows, stream, mr.get_output_mr());
        if (defer_preseal_build) {
          // The first probe seals all queued build batches in one descriptor-driven map kernel.
          // Retain the batch's device views and validity mask until that kernel consumes them.
          base_delta_batches[batch_id] = std::move(delta);
          base_delta_lifecycle_event.record(stream);
          return;
        }
        auto const map = streaming_hash_table_ref{
          base_delta_entries.data(), base_delta_capacity, base_delta_csr_capacity};
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
            old_inserted_rows,
            delta->slots->data(),
            delta->overlaps_prior ? delta->overlaps_prior->data() : nullptr,
            append_base_delta_generation ? nullptr : base_delta_counts.data(),
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
        base_delta_lifecycle_event.record(stream);
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
            hash_table->insert_async(input_begin, input_begin + batch_rows, stream.get());
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
            hash_table->insert_if_async(
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

  /**
   * @brief Materialize storage used exclusively by post-seal sparse generations.
   *
   * One-shot build-and-probe joins never need this state. Deferring it removes several
   * capacity-sized allocations and clears from the normal initial-seal path while retaining the
   * existing grow-only workspace policy once streaming appends begin.
   */
  void ensure_post_seal_delta_workspace(cuda::stream_ref stream) const
  {
    if (base_delta_sparse_heads) { return; }
    base_delta_sparse_heads = std::make_unique<rmm::device_uvector<streaming_sparse_node*>>(
      base_delta_csr_capacity, stream, mr.get_temporary_mr());
    base_delta_singleton_generations =
      std::make_unique<rmm::device_uvector<streaming_singleton_generation_descriptor>>(
        static_cast<std::size_t>(max_num_batches), stream, mr.get_temporary_mr());
    base_delta_singleton_generations_host.reserve(static_cast<std::size_t>(max_num_batches));
    base_delta_overlap_host.reserve(static_cast<std::size_t>(max_num_batches));
    CUDF_CUDA_TRY(cudaMemsetAsync(base_delta_sparse_heads->data(),
                                  0,
                                  base_delta_sparse_heads->size() * sizeof(streaming_sparse_node*),
                                  stream.get()));
  }

  void refresh_singleton_generations(cuda::stream_ref stream) const
  {
    base_delta_singleton_generations_host.clear();
    for (auto const* generation : base_delta_generations) {
      base_delta_singleton_generations_host.push_back({generation->candidate_begin,
                                                       generation->candidate_end,
                                                       generation->batch_id,
                                                       generation->generation});
    }
    if (!base_delta_singleton_generations_host.empty()) {
      cudf::detail::cuda_memcpy_async(
        cudf::device_span<streaming_singleton_generation_descriptor>{
          base_delta_singleton_generations->data(), base_delta_singleton_generations_host.size()},
        cudf::host_span<streaming_singleton_generation_descriptor const>{
          base_delta_singleton_generations_host},
        stream);
    }
  }

  void queue_overlap_copy(cuda::stream_ref stream) const
  {
    base_delta_overlap_host.assign(base_delta_generations.size(), 0);
    if (base_delta_overlap_host.empty()) { return; }
    for (auto index = std::size_t{0}; index < base_delta_generations.size(); ++index) {
      auto const* generation = base_delta_generations[index];
      cudf::detail::cuda_memcpy_async(
        cudf::host_span<int>{base_delta_overlap_host.data() + index, 1},
        cudf::device_span<int const>{generation->overlaps_prior->data(), 1},
        stream);
    }
  }

  void materialize_delta_generation(streaming_delta& delta, cuda::stream_ref stream) const
  {
    cudf::scoped_range range{"streaming_hash_join::base_deltas::delta_materialization"};
    if (delta.nodes) { return; }
    ensure_post_seal_delta_workspace(stream);
    auto const rows               = static_cast<size_type>(delta.slots->size());
    auto const directory_capacity = streaming_hash_csr_capacity(rows, 0.5);
    delta.directory = std::make_unique<rmm::device_uvector<streaming_sparse_directory_entry>>(
      directory_capacity, stream, cudf::get_current_device_resource_ref());
    delta.nodes = std::make_unique<rmm::device_uvector<streaming_sparse_node>>(
      directory_capacity, stream, cudf::get_current_device_resource_ref());
    // At most one 32-row page is needed for each input row (the all-distinct upper bound).
    // Duplicate-heavy batches reserve dramatically fewer pages through `next_page`.
    delta.pages = std::make_unique<rmm::device_uvector<streaming_sparse_page>>(
      static_cast<std::size_t>(rows), stream, cudf::get_current_device_resource_ref());
    delta.directory_next_page =
      std::make_unique<rmm::device_uvector<cuda::std::uint32_t>>(1, stream, mr.get_temporary_mr());
    CUDF_CUDA_TRY(
      cudaMemsetAsync(delta.directory->data(),
                      0xff,
                      delta.directory->size() * sizeof(streaming_sparse_directory_entry),
                      stream.get()));
    CUDF_CUDA_TRY(cudaMemsetAsync(
      delta.directory_next_page->data(), 0, sizeof(cuda::std::uint32_t), stream.get()));
    auto const page_config = grid_1d{rows, 256};
    streaming_delta_append_pages_kernel<<<page_config.num_blocks,
                                          page_config.num_threads_per_block,
                                          0,
                                          stream.get()>>>(rows,
                                                          delta.batch_id,
                                                          delta.generation,
                                                          delta.candidate_begin,
                                                          delta.slots->data(),
                                                          directory_capacity - 1,
                                                          delta.directory->data(),
                                                          delta.nodes->data(),
                                                          delta.pages->data(),
                                                          delta.directory_next_page->data(),
                                                          base_delta_sparse_heads->data());
    CUDF_CUDA_TRY(cudaGetLastError());
    delta.slots->set_stream(stream);
    delta.slots.reset();
  }

  void materialize_delta_generations(cuda::stream_ref stream) const
  {
    for (auto* generation : base_delta_generations) {
      materialize_delta_generation(*generation, stream);
    }
    refresh_singleton_generations(stream);
  }

  void compact_delta_generations(cuda::stream_ref stream) const
  {
    cudf::scoped_range range{"streaming_hash_join::base_deltas::compaction"};
    if (base_delta_generations.empty()) { return; }
    auto const source = streaming_csr_generations_ref{
      {canonical_ends->data(), canonical_values->data()},
      base_delta_sparse_heads->data(),
      base_delta_singleton_generations->data(),
      static_cast<size_type>(base_delta_singleton_generations_host.size())};
    auto const config = grid_1d{static_cast<size_type>(base_delta_csr_capacity), 256};
    streaming_csr_merge_count_kernel<<<config.num_blocks,
                                       config.num_threads_per_block,
                                       0,
                                       stream.get()>>>(
      static_cast<size_type>(base_delta_csr_capacity), source, base_delta_counts.data());
    CUDF_CUDA_TRY(cudaGetLastError());
    auto merged_ends = std::make_unique<rmm::device_uvector<size_type>>(
      base_delta_csr_capacity, stream, cudf::get_current_device_resource_ref());
    // `merged_ends.back()` is the logical length. Reserve the maximum possible physical length
    // up front so publishing a compacted CSR does not require a D2H cardinality copy.
    auto merged_values = std::make_unique<rmm::device_uvector<streaming_row_ref>>(
      static_cast<std::size_t>(inserted_rows.load(std::memory_order_relaxed)),
      stream,
      cudf::get_current_device_resource_ref());
    auto merged_scan_bytes = base_delta_scan_temp->size();
    CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(base_delta_scan_temp->data(),
                                                merged_scan_bytes,
                                                base_delta_counts.data(),
                                                merged_ends->data(),
                                                base_delta_csr_capacity,
                                                stream.get()));
    streaming_csr_merge_fill_kernel<<<config.num_blocks,
                                      config.num_threads_per_block,
                                      0,
                                      stream.get()>>>(
      static_cast<size_type>(base_delta_csr_capacity),
      source,
      merged_ends->data(),
      merged_values->data());
    CUDF_CUDA_TRY(cudaGetLastError());
    // Destruction of the replaced base is stream-ordered. The fill above has consumed it before
    // RMM may reclaim its allocation, so no host synchronization is needed here.
    canonical_ends->set_stream(stream);
    canonical_values->set_stream(stream);
    canonical_ends   = std::move(merged_ends);
    canonical_values = std::move(merged_values);
    // The merge kernels above have consumed every linked node.  Clear the persistent overlay
    // before retiring generations so future probes see only the replacement canonical CSR.
    CUDF_CUDA_TRY(cudaMemsetAsync(base_delta_sparse_heads->data(),
                                  0,
                                  base_delta_sparse_heads->size() * sizeof(streaming_sparse_node*),
                                  stream.get()));
    for (auto* generation : base_delta_generations) {
      generation->nodes->set_stream(stream);
      if (generation->directory) { generation->directory->set_stream(stream); }
      if (generation->pages) { generation->pages->set_stream(stream); }
      if (generation->directory_next_page) { generation->directory_next_page->set_stream(stream); }
      if (generation->overlaps_prior) { generation->overlaps_prior->set_stream(stream); }
      generation->nodes.reset();
      generation->directory.reset();
      generation->pages.reset();
      generation->directory_next_page.reset();
      generation->overlaps_prior.reset();
    }
    base_delta_generations.clear();
    base_delta_singleton_generations_host.clear();
    base_delta_overlap_host.clear();
    base_delta_generation_rows = 0;
  }

  template <typename Equal, typename Hasher>
  void flush_preseal_base_deltas_impl(size_type num_batches,
                                      std::vector<Equal> const& host_comparators,
                                      std::vector<Hasher> const& host_hashers,
                                      cuda::stream_ref stream) const
  {
    base_delta_build_descriptors_host.clear();
    auto input_begin = size_type{0};
    for (size_type batch = 0; batch < num_batches; ++batch) {
      auto const& delta = base_delta_batches[batch];
      if (!delta || delta->slots->size() == 0) { continue; }
      base_delta_build_descriptors_host.push_back(
        {delta->valid_rows ? static_cast<bitmask_type const*>(delta->valid_rows->data()) : nullptr,
         delta->slots->data(),
         static_cast<size_type>(delta->slots->size()),
         input_begin,
         batch});
      input_begin += static_cast<size_type>(delta->slots->size());
    }
    if (input_begin == 0) { return; }
    cudf::detail::cuda_memcpy_async(
      cudf::device_span<streaming_delta_build_descriptor>{base_delta_build_descriptors.data(),
                                                          base_delta_build_descriptors_host.size()},
      cudf::host_span<streaming_delta_build_descriptor const>{base_delta_build_descriptors_host},
      stream);
    auto d_comparators =
      cudf::detail::make_device_uvector_async(host_comparators, stream, mr.get_temporary_mr());
    auto d_hashers =
      cudf::detail::make_device_uvector_async(host_hashers, stream, mr.get_temporary_mr());
    auto const map = streaming_hash_table_ref{
      base_delta_entries.data(), base_delta_capacity, base_delta_csr_capacity};
    auto const config = grid_1d{input_begin, 256};
    streaming_preseal_delta_build_kernel<<<config.num_blocks,
                                           config.num_threads_per_block,
                                           0,
                                           stream.get()>>>(
      input_begin,
      base_delta_build_descriptors.data(),
      static_cast<size_type>(base_delta_build_descriptors_host.size()),
      d_hashers.data(),
      map,
      streaming_multibatch_build_equal<Equal>{d_comparators.data(), num_batches},
      base_delta_counts.data());
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  void flush_preseal_base_deltas(size_type num_batches, cuda::stream_ref stream) const
  {
    cudf::scoped_range range{"streaming_hash_join::base_deltas::preseal_flush"};
    if (num_batches == 0) { return; }
    auto const rights = std::span{preprocessed_right}.first(static_cast<std::size_t>(num_batches));
    auto const keys   = right_schema->view().select(right_key_indices);
    auto const nulls  = nullate::DYNAMIC{has_nulls};
    if (cudf::detail::is_primitive_row_op_compatible(keys)) {
      using equal_type = row::primitive::row_equality_comparator;
      std::vector<equal_type> comparators;
      comparators.reserve(static_cast<std::size_t>(num_batches) * num_batches);
      for (size_type candidate = 0; candidate < num_batches; ++candidate) {
        for (auto const& existing : rights) {
          comparators.emplace_back(nulls, preprocessed_right[candidate], existing, compare_nulls);
        }
      }
      std::vector<row::primitive::row_hasher<>> hashers;
      hashers.reserve(num_batches);
      for (auto const& right : rights) {
        hashers.emplace_back(nulls, right);
      }
      flush_preseal_base_deltas_impl(num_batches, comparators, hashers, stream);
    } else if (has_nested_keys) {
      using equal_type =
        row::equality::device_row_comparator<true,
                                             nullate::DYNAMIC,
                                             row::equality::nan_equal_physical_equality_comparator>;
      std::vector<equal_type> comparators;
      comparators.reserve(static_cast<std::size_t>(num_batches) * num_batches);
      for (size_type candidate = 0; candidate < num_batches; ++candidate) {
        for (auto const& existing : rights) {
          auto const comparator =
            row::equality::two_table_comparator{preprocessed_right[candidate], existing};
          comparators.push_back(comparator.equal_to<true>(nulls, compare_nulls).comparator);
        }
      }
      using hasher_type = decltype(row::hash::row_hasher{preprocessed_right.front()}.device_hasher(
        nullate::DYNAMIC{false}));
      std::vector<hasher_type> hashers;
      hashers.reserve(static_cast<std::size_t>(num_batches));
      for (auto const& right : rights) {
        hashers.push_back(row::hash::row_hasher{right}.device_hasher(nulls));
      }
      flush_preseal_base_deltas_impl(num_batches, comparators, hashers, stream);
    } else {
      using equal_type =
        row::equality::device_row_comparator<false,
                                             nullate::DYNAMIC,
                                             row::equality::nan_equal_physical_equality_comparator>;
      std::vector<equal_type> comparators;
      comparators.reserve(static_cast<std::size_t>(num_batches) * num_batches);
      for (size_type candidate = 0; candidate < num_batches; ++candidate) {
        for (auto const& existing : rights) {
          auto const comparator =
            row::equality::two_table_comparator{preprocessed_right[candidate], existing};
          comparators.push_back(comparator.equal_to<false>(nulls, compare_nulls).comparator);
        }
      }
      using hasher_type = decltype(row::hash::row_hasher{preprocessed_right.front()}.device_hasher(
        nullate::DYNAMIC{false}));
      std::vector<hasher_type> hashers;
      hashers.reserve(static_cast<std::size_t>(num_batches));
      for (auto const& right : rights) {
        hashers.push_back(row::hash::row_hasher{right}.device_hasher(nulls));
      }
      flush_preseal_base_deltas_impl(num_batches, comparators, hashers, stream);
    }
  }

  void seal_base_deltas(cuda::stream_ref stream) const
  {
    cudf::scoped_range range{"streaming_hash_join::base_deltas::seal"};
    std::lock_guard lock{base_delta_mutex};
    base_delta_lifecycle_event.wait(stream);
    if (base_delta_sealed) { return; }
    auto const num_batches = inserted_batches.load(std::memory_order_relaxed);
    flush_preseal_base_deltas(num_batches, stream);
    canonical_ends = std::make_unique<rmm::device_uvector<size_type>>(
      base_delta_csr_capacity, stream, cudf::get_current_device_resource_ref());
    {
      cudf::scoped_range scan_range{"streaming_hash_join::base_deltas::seal_scan"};
      auto canonical_scan_bytes = base_delta_scan_temp->size();
      CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(base_delta_scan_temp->data(),
                                                  canonical_scan_bytes,
                                                  base_delta_counts.data(),
                                                  canonical_ends->data(),
                                                  base_delta_csr_capacity,
                                                  stream.get()));
    }
    canonical_values = std::make_unique<rmm::device_uvector<streaming_row_ref>>(
      static_cast<std::size_t>(inserted_rows.load(std::memory_order_relaxed)),
      stream,
      cudf::get_current_device_resource_ref());
    CUDF_CUDA_TRY(cudaMemsetAsync(
      base_delta_cursors.data(), 0, base_delta_cursors.size() * sizeof(size_type), stream.get()));
    base_delta_fill_descriptors_host.clear();
    auto input_begin = size_type{0};
    for (size_type batch = 0; batch < num_batches; ++batch) {
      auto const& delta = base_delta_batches[batch];
      if (!delta || delta->slots->size() == 0) { continue; }
      base_delta_fill_descriptors_host.push_back(
        {delta->slots->data(), static_cast<size_type>(delta->slots->size()), input_begin, batch});
      input_begin += static_cast<size_type>(delta->slots->size());
    }
    if (!base_delta_fill_descriptors_host.empty()) {
      cudf::detail::cuda_memcpy_async(
        cudf::device_span<streaming_delta_fill_descriptor>{base_delta_fill_descriptors.data(),
                                                           base_delta_fill_descriptors_host.size()},
        cudf::host_span<streaming_delta_fill_descriptor const>{base_delta_fill_descriptors_host},
        stream);
      cudf::scoped_range fill_range{"streaming_hash_join::base_deltas::seal_fill"};
      auto const config = grid_1d{input_begin, 256};
      streaming_delta_fill_batches_kernel<<<config.num_blocks,
                                            config.num_threads_per_block,
                                            0,
                                            stream.get()>>>(
        input_begin,
        base_delta_fill_descriptors.data(),
        static_cast<size_type>(base_delta_fill_descriptors_host.size()),
        canonical_ends->data(),
        base_delta_cursors.data(),
        canonical_values->data());
      CUDF_CUDA_TRY(cudaGetLastError());
    }
    // Count/fill work for every pre-seal batch is now enqueued.  Retire the sparse inputs on
    // this stream; the canonical CSR is the only representation needed after publication.
    for (size_type batch = 0; batch < num_batches; ++batch) {
      auto const& delta = base_delta_batches[batch];
      if (!delta || !delta->slots) { continue; }
      delta->slots->set_stream(stream);
      delta->slots.reset();
    }
    base_delta_sealed = true;
    base_delta_lifecycle_event.record(stream);
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
    cudf::scoped_range range{"streaming_hash_join::base_deltas::probe"};
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
    auto const map = streaming_hash_table_ref{
      base_delta_entries.data(), base_delta_capacity, base_delta_csr_capacity};
    auto slots = rmm::device_uvector<cuda::std::uint32_t>{
      static_cast<std::size_t>(left.num_rows()), stream, resources.get_temporary_mr()};
    auto offsets = rmm::device_uvector<size_type>{
      static_cast<std::size_t>(left.num_rows()), stream, resources.get_temporary_mr()};
    auto all_matches_at_most_one =
      rmm::device_uvector<int>{1, stream, resources.get_temporary_mr()};
    auto all_matches_exactly_one =
      rmm::device_uvector<int>{1, stream, resources.get_temporary_mr()};
    // Any non-zero byte pattern is true; memset avoids a host-to-device staging dependency.
    CUDF_CUDA_TRY(cudaMemsetAsync(all_matches_at_most_one.data(), 1, sizeof(int), stream.get()));
    CUDF_CUDA_TRY(cudaMemsetAsync(all_matches_exactly_one.data(), 1, sizeof(int), stream.get()));
    auto const equal      = streaming_probe_equal{comparators.data()};
    auto const group_size = get_base_delta_probe_group_size();
    if (group_size == 0) {
      auto const config = grid_1d{left.num_rows(), 256};
      streaming_probe_count_kernel<<<config.num_blocks,
                                     config.num_threads_per_block,
                                     0,
                                     stream.get()>>>(left.num_rows(),
                                                     valid_rows,
                                                     slots.data(),
                                                     offsets.data(),
                                                     all_matches_at_most_one.data(),
                                                     all_matches_exactly_one.data(),
                                                     map,
                                                     csr,
                                                     equal,
                                                     row_hasher);
    } else {
      CUDF_EXPECTS(left.num_rows() <= std::numeric_limits<size_type>::max() / group_size,
                   "streaming_hash_join cooperative probe launch size overflowed",
                   std::overflow_error);
      auto const config = grid_1d{left.num_rows() * group_size, 256};
      if (group_size == 2) {
        streaming_probe_count_cooperative_kernel<2>
          <<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
            left.num_rows(),
            valid_rows,
            slots.data(),
            offsets.data(),
            all_matches_at_most_one.data(),
            all_matches_exactly_one.data(),
            map,
            csr,
            equal,
            row_hasher);
      } else {
        streaming_probe_count_cooperative_kernel<4>
          <<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
            left.num_rows(),
            valid_rows,
            slots.data(),
            offsets.data(),
            all_matches_at_most_one.data(),
            all_matches_exactly_one.data(),
            map,
            csr,
            equal,
            row_hasher);
      }
    }
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
      launch_streaming_probe_fill_kernel(join_size,
                                         left.num_rows(),
                                         slots.data(),
                                         offsets.data(),
                                         all_matches_at_most_one.data(),
                                         all_matches_exactly_one.data(),
                                         csr,
                                         left_indices->data(),
                                         batch_indices->data(),
                                         row_indices->data(),
                                         stream);
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
    // An ordinary probe already waits for its cardinality copy below.  Queue the overlap copies
    // before that wait and use their result to avoid rebuilding an otherwise disjoint overlay.
    // Hinted probes deliberately retain the previous policy: they do not synchronize on the
    // host, so consulting the device overlap flags there would add a new synchronization point.
    bool compact_before_probe = false;
    {
      std::lock_guard lock{base_delta_mutex};
      base_delta_lifecycle_event.wait(stream);
      materialize_delta_generations(stream);
      if (output_size) {
        auto const base_rows = static_cast<size_type>(canonical_values->size());
        auto const policy    = get_base_delta_compaction_policy();
        compact_before_probe =
          base_delta_generations.size() >= static_cast<std::size_t>(policy.generation_limit) ||
          (base_delta_generation_rows > 0 &&
           (base_rows == 0 || static_cast<cuda::std::uint64_t>(base_delta_generation_rows) * 100 >=
                                static_cast<cuda::std::uint64_t>(base_rows) * policy.row_percent));
        if (compact_before_probe) { compact_delta_generations(stream); }
      } else if (left.num_rows() > 0) {
        queue_overlap_copy(stream);
      }
    }
    auto const csr = streaming_csr_generations_ref{
      {canonical_ends->data(), canonical_values->data()},
      base_delta_sparse_heads ? base_delta_sparse_heads->data() : nullptr,
      base_delta_singleton_generations ? base_delta_singleton_generations->data() : nullptr,
      static_cast<size_type>(base_delta_singleton_generations_host.size())};
    // The sparse head array is the probe snapshot.  Record its completion after count/fill have
    // been queued, so a later stream cannot publish another generation into the overlay while
    // this stream is traversing it.
    auto record_probe_completion = [&] {
      std::lock_guard lock{base_delta_mutex};
      if (!output_size && left.num_rows() > 0) {
        // `probe_base_deltas_impl` synchronizes after its cardinality D2H copy, which also makes
        // the overlap copies queued above visible here.  Do not suppress the generation cap: it
        // bounds descriptor/page memory even when every generation is disjoint.
        auto const any_overlap = std::any_of(base_delta_overlap_host.begin(),
                                             base_delta_overlap_host.end(),
                                             [](auto const overlap) { return overlap != 0; });
        auto const base_rows   = static_cast<size_type>(canonical_values->size());
        auto const policy      = get_base_delta_compaction_policy();
        auto const compact_generations =
          base_delta_generations.size() >= static_cast<std::size_t>(policy.generation_limit) ||
          (any_overlap && base_delta_generation_rows > 0 &&
           (base_rows == 0 || static_cast<cuda::std::uint64_t>(base_delta_generation_rows) * 100 >=
                                static_cast<cuda::std::uint64_t>(base_rows) * policy.row_percent));
        if (compact_generations) { compact_delta_generations(stream); }
      }
      base_delta_lifecycle_event.record(stream);
    };
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
      auto result            = probe_base_deltas_impl(
        left,
        mask,
        comparators,
        row::primitive::row_hasher{nullate::DYNAMIC{has_nulls}, preprocessed_left},
        csr,
        output_size,
        stream,
        resources);
      record_probe_completion();
      return result;
    }
    if (has_nested_keys) {
      auto const comparators = make_device_row_comparators<true>(preprocessed_left,
                                                                 rights,
                                                                 nullate::DYNAMIC{has_nulls},
                                                                 compare_nulls,
                                                                 stream,
                                                                 resources.get_temporary_mr());
      auto result            = probe_base_deltas_impl(
        left,
        mask,
        comparators,
        row::hash::row_hasher{preprocessed_left}.device_hasher(nullate::DYNAMIC{has_nulls}),
        csr,
        output_size,
        stream,
        resources);
      record_probe_completion();
      return result;
    }
    auto const comparators = make_device_row_comparators<false>(preprocessed_left,
                                                                rights,
                                                                nullate::DYNAMIC{has_nulls},
                                                                compare_nulls,
                                                                stream,
                                                                resources.get_temporary_mr());
    auto result            = probe_base_deltas_impl(
      left,
      mask,
      comparators,
      row::hash::row_hasher{preprocessed_left}.device_hasher(nullate::DYNAMIC{has_nulls}),
      csr,
      output_size,
      stream,
      resources);
    record_probe_completion();
    return result;
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
      // `sizes_to_offsets` consumes a zero sentinel after the per-probe counts, matching the
      // HashCSR retrieval path. The count kernel writes only [0, left.num_rows()), so the
      // terminal element must be initialized explicitly.
      auto match_counts = cudf::detail::make_zeroed_device_uvector_async<size_type>(
        static_cast<std::size_t>(left.num_rows()) + 1, stream, resources.get_temporary_mr());
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
      std::size_t result_size{};
      if (output_size) {
        CUDF_EXPECTS(
          *output_size <= static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
          "The provided join output size is too large",
          std::overflow_error);
        // The caller owns the exact cardinality, so construct offsets without the D2H transfer
        // and host synchronization required to discover the scan total.
        thrust::exclusive_scan(rmm::exec_policy_nosync(stream, resources.get_temporary_mr()),
                               match_counts.begin(),
                               match_counts.end(),
                               offsets.begin(),
                               size_type{0});
        result_size = *output_size;
      } else {
        auto const dynamic_size = cudf::detail::sizes_to_offsets(match_counts.begin(),
                                                                 match_counts.end(),
                                                                 offsets.begin(),
                                                                 0,
                                                                 stream,
                                                                 resources.get_temporary_mr());
        CUDF_EXPECTS(
          dynamic_size >= 0, "streaming dynamic join output size overflowed", std::overflow_error);
        result_size = static_cast<std::size_t>(dynamic_size);
      }
      auto left_indices = std::make_unique<rmm::device_uvector<size_type>>(
        result_size, stream, resources.get_output_mr());
      auto batch_indices = std::make_unique<rmm::device_uvector<size_type>>(
        result_size, stream, resources.get_output_mr());
      auto row_indices = std::make_unique<rmm::device_uvector<size_type>>(
        result_size, stream, resources.get_output_mr());
      if (result_size > 0) {
        launch_segmented_dynamic_csr_retrieve_kernel(static_cast<size_type>(result_size),
                                                     left.num_rows(),
                                                     probe_slots.data(),
                                                     offsets.data(),
                                                     dynamic_csr->ref(),
                                                     left_indices->data(),
                                                     batch_indices->data(),
                                                     row_indices->data(),
                                                     stream);
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
                                       : hash_table->count(input_begin,
                                                           input_begin + left.num_rows(),
                                                           equality,
                                                           hash_table->hash_function(),
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
      hash_table->retrieve(input_begin,
                           input_begin + left.num_rows(),
                           equality,
                           hash_table->hash_function(),
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
