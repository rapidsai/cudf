/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common_utils.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/hashing.hpp>
#include <cudf/join/key_remapping.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cooperative_groups.h>
#include <cuco/static_set.cuh>
#include <cuda/std/atomic>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/reduce.h>
#include <thrust/replace.h>
#include <thrust/sort.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace cudf {
namespace detail {
namespace {

// Import necessary types
using cudf::hash_value_type;
using cudf::detail::row::lhs_index_type;
using cudf::detail::row::rhs_index_type;

// Always assume having nulls as the probe table is not known when building the hash table.
bool constexpr has_nulls = true;

/// Load factor for hash table sizing
double constexpr LOAD_FACTOR = 0.5;

/**
 * @brief Hasher that extracts pre-computed hash from key pair.
 */
struct key_hasher {
  template <typename T>
  __device__ constexpr hash_value_type operator()(
    cuco::pair<hash_value_type, T> const& key) const noexcept
  {
    return key.first;
  }
};

/**
 * @brief Device functor to determine if a row has no top-level nulls.
 */
class row_is_valid {
 public:
  row_is_valid(cudf::bitmask_type const* row_bitmask) : _row_bitmask{row_bitmask} {}

  __device__ bool operator()(cudf::size_type const& i) const noexcept
  {
    return cudf::bit_is_set(_row_bitmask, i);
  }

 private:
  cudf::bitmask_type const* _row_bitmask;
};

/**
 * @brief Device functor to create a pair of {hash_value, row_index}.
 */
template <typename T, typename Hasher>
class make_key_pair {
 public:
  CUDF_HOST_DEVICE constexpr make_key_pair(Hasher const& hash) : _hash{hash} {}

  __device__ __forceinline__ auto operator()(cudf::size_type i) const noexcept
  {
    return cuco::pair{_hash(i), T{i}};
  }

 private:
  Hasher _hash;
};

/**
 * @brief Device output transform functor to extract row index from hash table result.
 */
struct extract_index {
  __device__ constexpr cudf::size_type operator()(
    cuco::pair<hash_value_type, rhs_index_type> const& x) const
  {
    return static_cast<cudf::size_type>(x.second);
  }
};

/**
 * @brief Comparator adapter for probe-time comparison (two-table).
 */
template <typename Equal, bool CastToSizeType = false>
struct probe_comparator {
  probe_comparator(Equal const& d_equal) : _d_equal{d_equal} {}

  __device__ constexpr auto operator()(
    cuco::pair<hash_value_type, lhs_index_type> const& lhs,
    cuco::pair<hash_value_type, rhs_index_type> const& rhs) const noexcept
  {
    if (lhs.first != rhs.first) { return false; }
    if constexpr (CastToSizeType) {
      return _d_equal(static_cast<cudf::size_type>(lhs.second),
                      static_cast<cudf::size_type>(rhs.second));
    } else {
      return _d_equal(lhs.second, rhs.second);
    }
  }

 private:
  Equal _d_equal;
};

/**
 * @brief Comparator adapter for build-time self-comparison (deduplication).
 */
template <typename RowEqual>
struct build_comparator {
  build_comparator(RowEqual const& d_equal) : _d_equal{d_equal} {}

  __device__ constexpr auto operator()(
    cuco::pair<hash_value_type, rhs_index_type> const& lhs,
    cuco::pair<hash_value_type, rhs_index_type> const& rhs) const noexcept
  {
    if (lhs.first != rhs.first) { return false; }
    return _d_equal(static_cast<cudf::size_type>(lhs.second),
                    static_cast<cudf::size_type>(rhs.second));
  }

 private:
  RowEqual _d_equal;
};

// ============================================================================
// Hash-based metrics computation helpers
// ============================================================================

/// Block size for key remapping kernel
CUDF_HOST_DEVICE auto constexpr KEY_REMAP_BLOCK_SIZE = 128;

/**
 * @brief Kernel for inserting keys with counting using block-scoped atomics.
 */
template <typename SetRef, typename KeyIter>
CUDF_KERNEL void insert_and_count_kernel(cudf::size_type num_rows,
                                         SetRef set_ref,
                                         KeyIter key_iter,
                                         cudf::size_type* counts_ptr,
                                         cudf::size_type* global_distinct_count,
                                         cudf::bitmask_type const* bitmask_ptr)
{
  auto const block = cooperative_groups::this_thread_block();

  __shared__ cudf::size_type block_insert_count;
  if (block.thread_rank() == 0) { block_insert_count = 0; }
  block.sync();

  auto const stride = static_cast<cudf::size_type>(blockDim.x * gridDim.x);
  for (auto idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
       idx < num_rows;
       idx += stride) {
    bool const is_valid = (bitmask_ptr == nullptr) || cudf::bit_is_set(bitmask_ptr, idx);

    if (is_valid) {
      auto const key              = key_iter[idx];
      auto const [iter, inserted] = set_ref.insert_and_find(key);
      auto const stored_idx       = static_cast<cudf::size_type>(iter->second);

      atomicAdd(&counts_ptr[stored_idx], 1);

      if (inserted) {
        cuda::atomic_ref<cudf::size_type, cuda::thread_scope_block> ref{block_insert_count};
        ref.fetch_add(1, cuda::std::memory_order_relaxed);
      }
    }
  }

  block.sync();

  if (block.thread_rank() == 0 && block_insert_count > 0) {
    atomicAdd(global_distinct_count, block_insert_count);
  }
}

// ============================================================================
// Common helpers
// ============================================================================

/**
 * @brief Functor for inserting keys without tracking metrics.
 */
template <typename SetRef, typename KeyIter>
struct insert_only_fn {
  mutable SetRef set_ref;
  KeyIter key_iter;
  cudf::bitmask_type const* bitmask_ptr;

  __device__ void operator()(cudf::size_type idx) const
  {
    bool const is_valid = (bitmask_ptr == nullptr) || cudf::bit_is_set(bitmask_ptr, idx);
    if (is_valid) { set_ref.insert(key_iter[idx]); }
  }
};

/**
 * @brief Abstract interface for key remap hash table implementations.
 */
class key_remap_table_interface {
 public:
  virtual ~key_remap_table_interface() = default;

  virtual std::unique_ptr<rmm::device_uvector<cudf::size_type>> probe(
    cudf::table_view const& probe_keys,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const = 0;

  virtual bool has_metrics() const                        = 0;
  virtual cudf::size_type get_distinct_count() const      = 0;
  virtual cudf::size_type get_max_duplicate_count() const = 0;
};

/**
 * @brief Hash table implementation for key remapping.
 */
template <typename Comparator>
class key_remap_table : public key_remap_table_interface {
  using probing_scheme_type = cuco::linear_probing<1, key_hasher>;
  using cuco_storage_type   = cuco::storage<1>;
  using hash_table_type     = cuco::static_set<cuco::pair<hash_value_type, rhs_index_type>,
                                               cuco::extent<std::size_t>,
                                               cuda::thread_scope_device,
                                               Comparator,
                                               probing_scheme_type,
                                               rmm::mr::polymorphic_allocator<char>,
                                               cuco_storage_type>;

 public:
  key_remap_table()                                  = delete;
  ~key_remap_table() override                        = default;
  key_remap_table(key_remap_table const&)            = delete;
  key_remap_table(key_remap_table&&)                 = default;
  key_remap_table& operator=(key_remap_table const&) = delete;
  key_remap_table& operator=(key_remap_table&&)      = default;

  template <typename RowHasher>
  key_remap_table(
    cudf::table_view const& build,
    std::shared_ptr<cudf::detail::row::equality::preprocessed_table> preprocessed_build,
    Comparator const& comparator,
    RowHasher const& row_hasher,
    cudf::null_equality compare_nulls,
    bool compute_metrics,
    rmm::cuda_stream_view stream)
    : _build_has_nested_columns{cudf::has_nested_columns(build)},
      _compare_nulls{compare_nulls},
      _build{build},
      _preprocessed_build{std::move(preprocessed_build)},
      _hash_table{cuco::extent{static_cast<std::size_t>(build.num_rows())},
                  LOAD_FACTOR,
                  cuco::empty_key{cuco::pair{std::numeric_limits<hash_value_type>::max(),
                                             rhs_index_type{cudf::JoinNoMatch}}},
                  comparator,
                  {},
                  cuco::thread_scope_device,
                  cuco_storage_type{},
                  rmm::mr::polymorphic_allocator<char>{},
                  stream.value()},
      _has_metrics{compute_metrics},
      _distinct_count{0},
      _max_duplicate_count{0}
  {
    CUDF_FUNC_RANGE();
    CUDF_EXPECTS(0 != this->_build.num_columns(), "Key remap build table is empty");

    cudf::size_type const build_num_rows{_build.num_rows()};
    if (build_num_rows == 0) { return; }

    auto const key_iter = cudf::detail::make_counting_transform_iterator(
      0, make_key_pair<rhs_index_type, RowHasher>{row_hasher});

    bool const skip_nulls =
      (_compare_nulls == cudf::null_equality::UNEQUAL) && cudf::nullable(build);

    auto const row_bitmask =
      skip_nulls
        ? cudf::detail::bitmask_and(_build, stream, cudf::get_current_device_resource_ref()).first
        : rmm::device_buffer{};
    auto const bitmask_ptr =
      skip_nulls ? reinterpret_cast<cudf::bitmask_type const*>(row_bitmask.data()) : nullptr;

    if (compute_metrics) {
      // Use hash-based atomic counting for metrics computation
      compute_metrics_atomic(build_num_rows, key_iter, bitmask_ptr, stream);
    } else {
      // No metrics - simple insert
      auto set_ref = _hash_table.ref(cuco::op::insert);
      thrust::for_each_n(
        rmm::exec_policy_nosync(stream),
        thrust::make_counting_iterator<cudf::size_type>(0),
        build_num_rows,
        insert_only_fn<decltype(set_ref), decltype(key_iter)>{set_ref, key_iter, bitmask_ptr});
    }
  }

 private:
  template <typename KeyIter>
  void compute_metrics_atomic(cudf::size_type build_num_rows,
                              KeyIter key_iter,
                              cudf::bitmask_type const* bitmask_ptr,
                              rmm::cuda_stream_view stream)
  {
    rmm::device_uvector<cudf::size_type> counts(build_num_rows, stream);
    thrust::fill(rmm::exec_policy_nosync(stream), counts.begin(), counts.end(), 0);

    rmm::device_scalar<cudf::size_type> d_distinct_count(0, stream);

    auto set_ref = _hash_table.ref(cuco::op::insert_and_find);

    auto const grid_size = cudf::util::div_rounding_up_safe(
      build_num_rows, static_cast<cudf::size_type>(KEY_REMAP_BLOCK_SIZE));

    insert_and_count_kernel<<<grid_size, KEY_REMAP_BLOCK_SIZE, 0, stream.value()>>>(
      build_num_rows, set_ref, key_iter, counts.data(), d_distinct_count.data(), bitmask_ptr);

    _distinct_count = d_distinct_count.value(stream);

    _max_duplicate_count = thrust::reduce(rmm::exec_policy_nosync(stream),
                                          counts.begin(),
                                          counts.end(),
                                          cudf::size_type{0},
                                          thrust::maximum<cudf::size_type>{});
  }

 public:
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> probe(
    cudf::table_view const& probe_keys,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const override
  {
    CUDF_FUNC_RANGE();

    cudf::size_type const probe_num_rows{probe_keys.num_rows()};

    if (probe_num_rows == 0) {
      return std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
    }

    if (this->_build.num_rows() == 0) {
      auto result =
        std::make_unique<rmm::device_uvector<cudf::size_type>>(probe_num_rows, stream, mr);
      thrust::fill(
        rmm::exec_policy_nosync(stream), result->begin(), result->end(), cudf::JoinNoMatch);
      return result;
    }

    auto result =
      std::make_unique<rmm::device_uvector<cudf::size_type>>(probe_num_rows, stream, mr);
    auto const output_begin =
      thrust::make_transform_output_iterator(result->begin(), extract_index{});

    auto preprocessed_probe =
      cudf::detail::row::equality::preprocessed_table::create(probe_keys, stream);

    if (cudf::detail::is_primitive_row_op_compatible(_build)) {
      auto const d_hasher = cudf::detail::row::primitive::row_hasher{
        cudf::nullate::DYNAMIC{has_nulls}, preprocessed_probe};
      auto const d_equal = cudf::detail::row::primitive::row_equality_comparator{
        cudf::nullate::DYNAMIC{has_nulls}, preprocessed_probe, _preprocessed_build, _compare_nulls};

      auto const iter = cudf::detail::make_counting_transform_iterator(
        0, make_key_pair<lhs_index_type, decltype(d_hasher)>{d_hasher});

      find_matches(
        iter, probe_comparator<decltype(d_equal), true>{d_equal}, probe_keys, output_begin, stream);
    } else {
      auto const two_table_equal =
        cudf::detail::row::equality::two_table_comparator(preprocessed_probe, _preprocessed_build);

      auto const probe_row_hasher = cudf::detail::row::hash::row_hasher{preprocessed_probe};
      auto const d_probe_hasher = probe_row_hasher.device_hasher(cudf::nullate::DYNAMIC{has_nulls});
      auto const iter           = cudf::detail::make_counting_transform_iterator(
        0, make_key_pair<lhs_index_type, decltype(d_probe_hasher)>{d_probe_hasher});

      if (_build_has_nested_columns) {
        auto const device_comparator =
          two_table_equal.equal_to<true>(cudf::nullate::DYNAMIC{has_nulls}, _compare_nulls);
        find_matches(iter, probe_comparator{device_comparator}, probe_keys, output_begin, stream);
      } else {
        auto const device_comparator =
          two_table_equal.equal_to<false>(cudf::nullate::DYNAMIC{has_nulls}, _compare_nulls);
        find_matches(iter, probe_comparator{device_comparator}, probe_keys, output_begin, stream);
      }
    }
    return result;
  }

  bool has_metrics() const override { return _has_metrics; }

  cudf::size_type get_distinct_count() const override
  {
    CUDF_EXPECTS(_has_metrics, "Metrics were not computed during construction");
    return _distinct_count;
  }

  cudf::size_type get_max_duplicate_count() const override
  {
    CUDF_EXPECTS(_has_metrics, "Metrics were not computed during construction");
    return _max_duplicate_count;
  }

 private:
  template <typename IterType, typename EqualType, typename FoundIterator>
  void find_matches(IterType iter,
                    EqualType const& d_equal,
                    cudf::table_view const& probe_keys,
                    FoundIterator found_begin,
                    rmm::cuda_stream_view stream) const
  {
    auto const probe_num_rows = probe_keys.num_rows();

    if (_compare_nulls == cudf::null_equality::EQUAL or (not cudf::nullable(probe_keys))) {
      _hash_table.find_async(
        iter, iter + probe_num_rows, d_equal, key_hasher{}, found_begin, stream.value());
    } else {
      auto stencil = thrust::counting_iterator<cudf::size_type>{0};
      auto const row_bitmask =
        cudf::detail::bitmask_and(probe_keys, stream, cudf::get_current_device_resource_ref())
          .first;
      auto const pred =
        row_is_valid{reinterpret_cast<cudf::bitmask_type const*>(row_bitmask.data())};

      _hash_table.find_if_async(iter,
                                iter + probe_num_rows,
                                stencil,
                                pred,
                                d_equal,
                                key_hasher{},
                                found_begin,
                                stream.value());
    }
  }

  bool _build_has_nested_columns;
  cudf::null_equality _compare_nulls;
  cudf::table_view _build;
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> _preprocessed_build;
  hash_table_type _hash_table;
  bool _has_metrics;
  cudf::size_type _distinct_count;
  cudf::size_type _max_duplicate_count;
};

/**
 * @brief Factory function to create a key remap hash table.
 */
std::unique_ptr<key_remap_table_interface> create_key_remap_table(cudf::table_view const& build,
                                                                  cudf::null_equality compare_nulls,
                                                                  bool compute_metrics,
                                                                  rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  if (build.num_rows() == 0 || build.num_columns() == 0) { return nullptr; }

  auto preprocessed_build = cudf::detail::row::equality::preprocessed_table::create(build, stream);

  if (cudf::detail::is_primitive_row_op_compatible(build)) {
    auto const d_hasher = cudf::detail::row::primitive::row_hasher{
      cudf::nullate::DYNAMIC{has_nulls}, preprocessed_build};
    auto const d_equal = cudf::detail::row::primitive::row_equality_comparator{
      cudf::nullate::DYNAMIC{has_nulls}, preprocessed_build, preprocessed_build, compare_nulls};

    using comparator_type = build_comparator<decltype(d_equal)>;
    return std::make_unique<key_remap_table<comparator_type>>(build,
                                                              preprocessed_build,
                                                              comparator_type{d_equal},
                                                              d_hasher,
                                                              compare_nulls,
                                                              compute_metrics,
                                                              stream);
  }

  auto const has_nested = cudf::has_nested_columns(build);
  auto const self_equal = cudf::detail::row::equality::self_comparator(preprocessed_build);
  auto const row_hasher = cudf::detail::row::hash::row_hasher{preprocessed_build};
  auto const d_hasher   = row_hasher.device_hasher(cudf::nullate::DYNAMIC{has_nulls});

  if (has_nested) {
    auto const d_equal =
      self_equal.equal_to<true>(cudf::nullate::DYNAMIC{has_nulls}, compare_nulls);

    using comparator_type = build_comparator<decltype(d_equal)>;
    return std::make_unique<key_remap_table<comparator_type>>(build,
                                                              preprocessed_build,
                                                              comparator_type{d_equal},
                                                              d_hasher,
                                                              compare_nulls,
                                                              compute_metrics,
                                                              stream);
  } else {
    auto const d_equal =
      self_equal.equal_to<false>(cudf::nullate::DYNAMIC{has_nulls}, compare_nulls);

    using comparator_type = build_comparator<decltype(d_equal)>;
    return std::make_unique<key_remap_table<comparator_type>>(build,
                                                              preprocessed_build,
                                                              comparator_type{d_equal},
                                                              d_hasher,
                                                              compare_nulls,
                                                              compute_metrics,
                                                              stream);
  }
}

}  // namespace

/**
 * @brief Implementation class for key_remapping
 */
class key_remapping_impl {
 public:
  key_remapping_impl(cudf::table_view const& build,
                     cudf::null_equality compare_nulls,
                     bool compute_metrics,
                     rmm::cuda_stream_view stream)
    : _build{build},
      _compare_nulls{compare_nulls},
      _compute_metrics{compute_metrics},
      _table{create_key_remap_table(build, compare_nulls, compute_metrics, stream)}
  {
  }

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> probe(
    cudf::table_view const& keys,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    CUDF_EXPECTS(keys.num_columns() == _build.num_columns(),
                 "Mismatch in number of columns to be joined on",
                 std::invalid_argument);

    if (keys.num_rows() == 0) {
      return std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
    }

    CUDF_EXPECTS(cudf::have_same_types(_build, keys),
                 "Mismatch in joining column data types",
                 cudf::data_type_error);

    if (_table == nullptr) {
      auto result =
        std::make_unique<rmm::device_uvector<cudf::size_type>>(keys.num_rows(), stream, mr);
      thrust::fill(
        rmm::exec_policy_nosync(stream), result->begin(), result->end(), cudf::JoinNoMatch);
      return result;
    }
    return _table->probe(keys, stream, mr);
  }

  bool has_metrics() const { return _table ? _table->has_metrics() : _compute_metrics; }

  cudf::size_type get_distinct_count() const
  {
    if (_table) { return _table->get_distinct_count(); }
    CUDF_EXPECTS(_compute_metrics, "Metrics were not computed during construction");
    return 0;
  }

  cudf::size_type get_max_duplicate_count() const
  {
    if (_table) { return _table->get_max_duplicate_count(); }
    CUDF_EXPECTS(_compute_metrics, "Metrics were not computed during construction");
    return 0;
  }

  cudf::null_equality get_compare_nulls() const { return _compare_nulls; }

 private:
  cudf::table_view _build;
  cudf::null_equality _compare_nulls;
  bool _compute_metrics;
  std::unique_ptr<key_remap_table_interface> _table;
};

}  // namespace detail

// Public API implementation

key_remapping::key_remapping(cudf::table_view const& build,
                             null_equality compare_nulls,
                             bool compute_metrics,
                             rmm::cuda_stream_view stream)
  : _impl{
      std::make_unique<detail::key_remapping_impl>(build, compare_nulls, compute_metrics, stream)}
{
  CUDF_EXPECTS(build.num_columns() > 0, "Build table must have at least one column");
}

key_remapping::~key_remapping() = default;

namespace {
std::unique_ptr<cudf::column> remap_keys_internal(detail::key_remapping_impl const& impl,
                                                  cudf::table_view const& keys,
                                                  cudf::size_type not_found_sentinel,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  auto indices = impl.probe(keys, stream, mr);

  if (indices->size() == 0) { return cudf::make_empty_column(cudf::type_id::INT32); }

  thrust::replace(rmm::exec_policy_nosync(stream),
                  indices->begin(),
                  indices->end(),
                  cudf::JoinNoMatch,
                  not_found_sentinel);

  auto const row_count = static_cast<cudf::size_type>(indices->size());
  return std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::INT32}, row_count, indices->release(), rmm::device_buffer{}, 0);
}
}  // namespace

std::unique_ptr<cudf::column> key_remapping::remap_build_keys(
  cudf::table_view const& keys,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return remap_keys_internal(*_impl, keys, KEY_REMAP_BUILD_NULL, stream, mr);
}

std::unique_ptr<cudf::column> key_remapping::remap_probe_keys(
  cudf::table_view const& keys,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return remap_keys_internal(*_impl, keys, KEY_REMAP_NOT_FOUND, stream, mr);
}

bool key_remapping::has_metrics() const { return _impl->has_metrics(); }

size_type key_remapping::get_distinct_count() const { return _impl->get_distinct_count(); }

size_type key_remapping::get_max_duplicate_count() const
{
  return _impl->get_max_duplicate_count();
}

}  // namespace cudf
