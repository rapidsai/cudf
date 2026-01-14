/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common_utils.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/hashing.hpp>
#include <cudf/join/join_factorizer.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cooperative_groups.h>
#include <cuco/static_set.cuh>
#include <cuda/functional>
#include <cuda/std/atomic>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/reduce.h>
#include <thrust/replace.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

namespace cudf {
namespace detail {
namespace {

using cudf::hash_value_type;
using cudf::detail::row::lhs_index_type;
using cudf::detail::row::rhs_index_type;

bool constexpr ASSUME_NULLS_PRESENT = true;

double constexpr HASH_TABLE_LOAD_FACTOR = 0.5;

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
class row_validity_checker {
 public:
  row_validity_checker(cudf::bitmask_type const* row_validity_mask)
    : _row_validity_mask{row_validity_mask}
  {
  }

  __device__ bool operator()(cudf::size_type const& i) const noexcept
  {
    return cudf::bit_is_set(_row_validity_mask, i);
  }

 private:
  cudf::bitmask_type const* _row_validity_mask;
};

/**
 * @brief Device functor to create a pair of {hash_value, row_index}.
 */
template <typename T, typename Hasher>
class hash_index_pair_functor {
 public:
  CUDF_HOST_DEVICE constexpr hash_index_pair_functor(Hasher const& hash) : _hash{hash} {}

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
struct extract_row_index {
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
struct two_table_equality_comparator {
  two_table_equality_comparator(Equal const& d_equal) : _d_equal{d_equal} {}

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
struct self_table_equality_comparator {
  self_table_equality_comparator(RowEqual const& d_equal) : _d_equal{d_equal} {}

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

CUDF_HOST_DEVICE auto constexpr FACTORIZE_BLOCK_SIZE = 128;

/**
 * @brief Kernel for inserting keys with counting using block-scoped atomics.
 */
template <typename SetRef, typename KeyIter>
CUDF_KERNEL void insert_and_count_kernel(cudf::size_type num_rows,
                                         SetRef hash_table_ref,
                                         KeyIter key_iterator,
                                         cudf::size_type* counts_ptr,
                                         cudf::size_type* global_distinct_count,
                                         cudf::bitmask_type const* validity_mask_ptr)
{
  auto const block = cooperative_groups::this_thread_block();

  __shared__ cudf::size_type block_insert_count;
  if (block.thread_rank() == 0) { block_insert_count = 0; }
  block.sync();

  auto const stride = cudf::detail::grid_1d::grid_stride();
  for (auto idx = cudf::detail::grid_1d::global_thread_id(); idx < num_rows; idx += stride) {
    bool const is_valid =
      (validity_mask_ptr == nullptr) || cudf::bit_is_set(validity_mask_ptr, idx);

    if (is_valid) {
      auto const key              = key_iterator[idx];
      auto const [iter, inserted] = hash_table_ref.insert_and_find(key);
      auto const stored_idx       = static_cast<cudf::size_type>(iter->second);

      cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> count_ref{
        counts_ptr[stored_idx]};
      count_ref.fetch_add(1, cuda::std::memory_order_relaxed);

      if (inserted) {
        cuda::atomic_ref<cudf::size_type, cuda::thread_scope_block> ref{block_insert_count};
        ref.fetch_add(1, cuda::std::memory_order_relaxed);
      }
    }
  }

  block.sync();

  if (block.thread_rank() == 0 && block_insert_count > 0) {
    cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> global_count_ref{
      *global_distinct_count};
    global_count_ref.fetch_add(block_insert_count, cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief Functor for inserting keys without tracking metrics.
 */
template <typename SetRef, typename KeyIter>
struct key_inserter {
  mutable SetRef hash_table_ref;
  KeyIter key_iterator;
  cudf::bitmask_type const* validity_mask_ptr;

  __device__ void operator()(cudf::size_type idx) const
  {
    bool const is_valid =
      (validity_mask_ptr == nullptr) || cudf::bit_is_set(validity_mask_ptr, idx);
    if (is_valid) { hash_table_ref.insert(key_iterator[idx]); }
  }
};

/**
 * @brief Abstract interface for deduplicating hash table implementations.
 */
class hash_table_base {
 public:
  virtual ~hash_table_base() = default;

  virtual std::unique_ptr<rmm::device_uvector<cudf::size_type>> lookup_keys(
    cudf::table_view const& probe_keys,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const = 0;

  virtual bool has_metrics() const                        = 0;
  virtual cudf::size_type get_distinct_count() const      = 0;
  virtual cudf::size_type get_max_duplicate_count() const = 0;
};

/**
 * @brief Hash table implementation that deduplicates keys and assigns unique IDs.
 */
template <typename Comparator>
class deduplicating_hash_table : public hash_table_base {
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
  deduplicating_hash_table()                                           = delete;
  ~deduplicating_hash_table() override                                 = default;
  deduplicating_hash_table(deduplicating_hash_table const&)            = delete;
  deduplicating_hash_table(deduplicating_hash_table&&)                 = default;
  deduplicating_hash_table& operator=(deduplicating_hash_table const&) = delete;
  deduplicating_hash_table& operator=(deduplicating_hash_table&&)      = default;

  template <typename RowHasher>
  deduplicating_hash_table(
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
                  HASH_TABLE_LOAD_FACTOR,
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
    CUDF_EXPECTS(0 != this->_build.num_columns(), "Factorizer build table is empty");

    cudf::size_type const build_num_rows{_build.num_rows()};
    if (build_num_rows == 0) { return; }

    auto const key_pair_iterator = cudf::detail::make_counting_transform_iterator(
      0, hash_index_pair_functor<rhs_index_type, RowHasher>{row_hasher});

    bool const skip_nulls =
      (_compare_nulls == cudf::null_equality::UNEQUAL) && cudf::nullable(build);

    auto const row_validity_bitmask =
      skip_nulls
        ? cudf::detail::bitmask_and(_build, stream, cudf::get_current_device_resource_ref()).first
        : rmm::device_buffer{};
    auto const validity_mask_ptr =
      skip_nulls ? reinterpret_cast<cudf::bitmask_type const*>(row_validity_bitmask.data())
                 : nullptr;

    if (compute_metrics) {
      build_with_metrics(build_num_rows, key_pair_iterator, validity_mask_ptr, stream);
    } else {
      auto hash_table_ref = _hash_table.ref(cuco::op::insert);
      thrust::for_each_n(rmm::exec_policy_nosync(stream),
                         thrust::make_counting_iterator<cudf::size_type>(0),
                         build_num_rows,
                         key_inserter<decltype(hash_table_ref), decltype(key_pair_iterator)>{
                           hash_table_ref, key_pair_iterator, validity_mask_ptr});
    }
  }

 private:
  template <typename KeyIter>
  void build_with_metrics(cudf::size_type build_num_rows,
                          KeyIter key_pair_iterator,
                          cudf::bitmask_type const* validity_mask_ptr,
                          rmm::cuda_stream_view stream)
  {
    rmm::device_uvector<cudf::size_type> counts(build_num_rows, stream);
    thrust::fill(rmm::exec_policy_nosync(stream), counts.begin(), counts.end(), 0);

    cudf::detail::device_scalar<cudf::size_type> d_distinct_count{0, stream};

    auto hash_table_ref = _hash_table.ref(cuco::op::insert_and_find);

    cudf::detail::grid_1d grid{build_num_rows, FACTORIZE_BLOCK_SIZE};

    insert_and_count_kernel<<<grid.num_blocks, FACTORIZE_BLOCK_SIZE, 0, stream.value()>>>(
      build_num_rows,
      hash_table_ref,
      key_pair_iterator,
      counts.data(),
      d_distinct_count.data(),
      validity_mask_ptr);

    _distinct_count = d_distinct_count.value(stream);

    _max_duplicate_count = thrust::reduce(rmm::exec_policy_nosync(stream),
                                          counts.begin(),
                                          counts.end(),
                                          cudf::size_type{0},
                                          cuda::maximum<cudf::size_type>{});
  }

 public:
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> lookup_keys(
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
    auto const output_iterator =
      thrust::make_transform_output_iterator(result->begin(), extract_row_index{});

    auto preprocessed_probe =
      cudf::detail::row::equality::preprocessed_table::create(probe_keys, stream);

    if (cudf::detail::is_primitive_row_op_compatible(_build)) {
      auto const d_hasher = cudf::detail::row::primitive::row_hasher{
        cudf::nullate::DYNAMIC{ASSUME_NULLS_PRESENT}, preprocessed_probe};
      auto const d_equal = cudf::detail::row::primitive::row_equality_comparator{
        cudf::nullate::DYNAMIC{ASSUME_NULLS_PRESENT},
        preprocessed_probe,
        _preprocessed_build,
        _compare_nulls};

      auto const key_pair_iterator = cudf::detail::make_counting_transform_iterator(
        0, hash_index_pair_functor<lhs_index_type, decltype(d_hasher)>{d_hasher});

      find_matching_keys(key_pair_iterator,
                         two_table_equality_comparator<decltype(d_equal), true>{d_equal},
                         probe_keys,
                         output_iterator,
                         stream);
    } else {
      auto const two_table_equal =
        cudf::detail::row::equality::two_table_comparator(preprocessed_probe, _preprocessed_build);

      auto const probe_row_hasher = cudf::detail::row::hash::row_hasher{preprocessed_probe};
      auto const d_probe_hasher =
        probe_row_hasher.device_hasher(cudf::nullate::DYNAMIC{ASSUME_NULLS_PRESENT});
      auto const key_pair_iterator = cudf::detail::make_counting_transform_iterator(
        0, hash_index_pair_functor<lhs_index_type, decltype(d_probe_hasher)>{d_probe_hasher});

      if (_build_has_nested_columns) {
        auto const device_comparator = two_table_equal.equal_to<true>(
          cudf::nullate::DYNAMIC{ASSUME_NULLS_PRESENT}, _compare_nulls);
        find_matching_keys(key_pair_iterator,
                           two_table_equality_comparator{device_comparator},
                           probe_keys,
                           output_iterator,
                           stream);
      } else {
        auto const device_comparator = two_table_equal.equal_to<false>(
          cudf::nullate::DYNAMIC{ASSUME_NULLS_PRESENT}, _compare_nulls);
        find_matching_keys(key_pair_iterator,
                           two_table_equality_comparator{device_comparator},
                           probe_keys,
                           output_iterator,
                           stream);
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
  void find_matching_keys(IterType key_pair_iterator,
                          EqualType const& d_equal,
                          cudf::table_view const& probe_keys,
                          FoundIterator output_iterator,
                          rmm::cuda_stream_view stream) const
  {
    CUDF_FUNC_RANGE();
    auto const probe_num_rows = probe_keys.num_rows();

    if (_compare_nulls == cudf::null_equality::EQUAL or (not cudf::nullable(probe_keys))) {
      _hash_table.find_async(key_pair_iterator,
                             key_pair_iterator + probe_num_rows,
                             d_equal,
                             key_hasher{},
                             output_iterator,
                             stream.value());
    } else {
      auto stencil = thrust::counting_iterator<cudf::size_type>{0};
      auto const row_validity_bitmask =
        cudf::detail::bitmask_and(probe_keys, stream, cudf::get_current_device_resource_ref())
          .first;
      auto const validity_checker = row_validity_checker{
        reinterpret_cast<cudf::bitmask_type const*>(row_validity_bitmask.data())};

      _hash_table.find_if_async(key_pair_iterator,
                                key_pair_iterator + probe_num_rows,
                                stencil,
                                validity_checker,
                                d_equal,
                                key_hasher{},
                                output_iterator,
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
 * @brief Factory function to create a deduplicating hash table.
 */
std::unique_ptr<hash_table_base> make_deduplicating_hash_table(cudf::table_view const& build,
                                                               cudf::null_equality compare_nulls,
                                                               bool compute_metrics,
                                                               rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  if (build.num_rows() == 0 || build.num_columns() == 0) { return nullptr; }

  auto preprocessed_build = cudf::detail::row::equality::preprocessed_table::create(build, stream);

  if (cudf::detail::is_primitive_row_op_compatible(build)) {
    auto const d_hasher = cudf::detail::row::primitive::row_hasher{
      cudf::nullate::DYNAMIC{ASSUME_NULLS_PRESENT}, preprocessed_build};
    auto const d_equal = cudf::detail::row::primitive::row_equality_comparator{
      cudf::nullate::DYNAMIC{ASSUME_NULLS_PRESENT},
      preprocessed_build,
      preprocessed_build,
      compare_nulls};

    using comparator_type = self_table_equality_comparator<decltype(d_equal)>;
    return std::make_unique<deduplicating_hash_table<comparator_type>>(build,
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
  auto const d_hasher   = row_hasher.device_hasher(cudf::nullate::DYNAMIC{ASSUME_NULLS_PRESENT});

  if (has_nested) {
    auto const d_equal =
      self_equal.equal_to<true>(cudf::nullate::DYNAMIC{ASSUME_NULLS_PRESENT}, compare_nulls);

    using comparator_type = self_table_equality_comparator<decltype(d_equal)>;
    return std::make_unique<deduplicating_hash_table<comparator_type>>(build,
                                                                       preprocessed_build,
                                                                       comparator_type{d_equal},
                                                                       d_hasher,
                                                                       compare_nulls,
                                                                       compute_metrics,
                                                                       stream);
  } else {
    auto const d_equal =
      self_equal.equal_to<false>(cudf::nullate::DYNAMIC{ASSUME_NULLS_PRESENT}, compare_nulls);

    using comparator_type = self_table_equality_comparator<decltype(d_equal)>;
    return std::make_unique<deduplicating_hash_table<comparator_type>>(build,
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
 * @brief Implementation class for join_factorizer
 */
class join_factorizer_impl {
  friend class cudf::join_factorizer;

 public:
  join_factorizer_impl(cudf::table_view const& build,
                       cudf::null_equality compare_nulls,
                       bool compute_metrics,
                       rmm::cuda_stream_view stream)
    : _build{build},
      _compare_nulls{compare_nulls},
      _compute_metrics{compute_metrics},
      _hash_table{make_deduplicating_hash_table(build, compare_nulls, compute_metrics, stream)}
  {
  }

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> lookup_keys(
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

    if (_hash_table == nullptr) {
      auto result =
        std::make_unique<rmm::device_uvector<cudf::size_type>>(keys.num_rows(), stream, mr);
      thrust::fill(
        rmm::exec_policy_nosync(stream), result->begin(), result->end(), cudf::JoinNoMatch);
      return result;
    }
    return _hash_table->lookup_keys(keys, stream, mr);
  }

  bool has_metrics() const { return _hash_table ? _hash_table->has_metrics() : _compute_metrics; }

  cudf::size_type get_distinct_count() const
  {
    if (_hash_table) { return _hash_table->get_distinct_count(); }
    CUDF_EXPECTS(_compute_metrics, "Metrics were not computed during construction");
    return 0;
  }

  cudf::size_type get_max_duplicate_count() const
  {
    if (_hash_table) { return _hash_table->get_max_duplicate_count(); }
    CUDF_EXPECTS(_compute_metrics, "Metrics were not computed during construction");
    return 0;
  }

  cudf::null_equality get_compare_nulls() const { return _compare_nulls; }

 private:
  cudf::table_view const& get_build() const { return _build; }

  cudf::table_view _build;
  cudf::null_equality _compare_nulls;
  bool _compute_metrics;
  std::unique_ptr<hash_table_base> _hash_table;
};

}  // namespace detail

// Public API implementation

join_factorizer::join_factorizer(cudf::table_view const& build,
                                 null_equality compare_nulls,
                                 cudf::factorizer_metrics metrics,
                                 rmm::cuda_stream_view stream)
  : _impl{std::make_unique<detail::join_factorizer_impl>(
      build, compare_nulls, static_cast<bool>(metrics), stream)}
{
  CUDF_EXPECTS(build.num_columns() > 0, "Build table must have at least one column");
}

join_factorizer::~join_factorizer() = default;

namespace {
std::unique_ptr<cudf::column> factorize_keys_impl(detail::join_factorizer_impl const& impl,
                                                  cudf::table_view const& keys,
                                                  cudf::size_type not_found_sentinel,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  auto indices = impl.lookup_keys(keys, stream, mr);

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

std::unique_ptr<cudf::column> join_factorizer::factorize_build_keys(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return factorize_keys_impl(*_impl, _impl->get_build(), FACTORIZE_BUILD_NULL, stream, mr);
}

std::unique_ptr<cudf::column> join_factorizer::factorize_probe_keys(
  cudf::table_view const& keys,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return factorize_keys_impl(*_impl, keys, FACTORIZE_NOT_FOUND, stream, mr);
}

bool join_factorizer::has_metrics() const { return _impl->has_metrics(); }

size_type join_factorizer::get_distinct_count() const { return _impl->get_distinct_count(); }

size_type join_factorizer::get_max_duplicate_count() const
{
  return _impl->get_max_duplicate_count();
}

}  // namespace cudf
