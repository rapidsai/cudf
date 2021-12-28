/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <hash/helper_functions.cuh>

#include <cudf/detail/gather.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace detail {
/*
 *  Device view of the unordered multiset
 */
template <typename Element,
          typename Hasher   = default_hash<Element>,
          typename Equality = equal_to<Element>>
class unordered_multiset_device_view {
 public:
  unordered_multiset_device_view(size_type hash_size,
                                 const size_type* hash_begin,
                                 const Element* hash_data)
    : hash_size{hash_size}, hash_begin{hash_begin}, hash_data{hash_data}, hasher(), equals()
  {
  }

  bool __device__ contains(Element e) const
  {
    size_type loc = hasher(e) % (2 * hash_size);

    for (size_type i = hash_begin[loc]; i < hash_begin[loc + 1]; ++i) {
      if (equals(hash_data[i], e)) return true;
    }

    return false;
  }

 private:
  Hasher hasher;
  Equality equals;
  size_type hash_size;
  const size_type* hash_begin;
  const Element* hash_data;
};

/*
 * Fixed size set on a device.
 */
template <typename Element,
          typename Hasher   = default_hash<Element>,
          typename Equality = equal_to<Element>>
class unordered_multiset {
 public:
  /**
   * @brief Factory to construct a new unordered_multiset
   */
  static unordered_multiset<Element> create(column_view const& col, rmm::cuda_stream_view stream)
  {
    auto d_column = column_device_view::create(col, stream);
    auto d_col    = *d_column;

    auto hash_bins_start =
      cudf::detail::make_zeroed_device_uvector_async<size_type>(2 * d_col.size() + 1, stream);
    auto hash_bins_end =
      cudf::detail::make_zeroed_device_uvector_async<size_type>(2 * d_col.size() + 1, stream);
    auto hash_data = rmm::device_uvector<Element>(d_col.size(), stream);

    Hasher hasher;
    size_type* d_hash_bins_start = hash_bins_start.data();
    size_type* d_hash_bins_end   = hash_bins_end.data();
    Element* d_hash_data         = hash_data.data();

    thrust::for_each(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     thrust::make_counting_iterator<size_type>(col.size()),
                     [d_hash_bins_start, d_col, hasher] __device__(size_t idx) {
                       if (!d_col.is_null(idx)) {
                         Element e     = d_col.element<Element>(idx);
                         size_type tmp = hasher(e) % (2 * d_col.size());
                         atomicAdd(d_hash_bins_start + tmp, size_type{1});
                       }
                     });

    thrust::exclusive_scan(rmm::exec_policy(stream),
                           hash_bins_start.begin(),
                           hash_bins_start.end(),
                           hash_bins_end.begin());

    thrust::copy(rmm::exec_policy(stream),
                 hash_bins_end.begin(),
                 hash_bins_end.end(),
                 hash_bins_start.begin());

    thrust::for_each(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     thrust::make_counting_iterator<size_type>(col.size()),
                     [d_hash_bins_end, d_hash_data, d_col, hasher] __device__(size_t idx) {
                       if (!d_col.is_null(idx)) {
                         Element e           = d_col.element<Element>(idx);
                         size_type tmp       = hasher(e) % (2 * d_col.size());
                         size_type offset    = atomicAdd(d_hash_bins_end + tmp, size_type{1});
                         d_hash_data[offset] = e;
                       }
                     });

    return unordered_multiset(d_col.size(), std::move(hash_bins_start), std::move(hash_data));
  }

  unordered_multiset_device_view<Element, Hasher, Equality> to_device()
  {
    return unordered_multiset_device_view<Element, Hasher, Equality>(
      size, hash_bins.data(), hash_data.data());
  }

 private:
  unordered_multiset(size_type size,
                     rmm::device_uvector<size_type>&& hash_bins,
                     rmm::device_uvector<Element>&& hash_data)
    : size{size}, hash_bins{std::move(hash_bins)}, hash_data{std::move(hash_data)}
  {
  }

  size_type size;
  rmm::device_uvector<size_type> hash_bins;
  rmm::device_uvector<Element> hash_data;
};

//-------------------------------------------------------------------------------------------------
/*
 * Device view of the unordered multiset, specialized for structs.
 */
using table_row_hash     = cudf::row_hasher<default_hash, cudf::nullate::DYNAMIC>;
using table_row_equality = cudf::row_equality_comparator<cudf::nullate::DYNAMIC>;
using d_table_ptr = std::unique_ptr<table_device_view, std::function<void(table_device_view*)>>;

template <>
class unordered_multiset_device_view<cudf::struct_view, table_row_hash, table_row_equality> {
 public:
  unordered_multiset_device_view(table_row_hash hasher,
                                 table_row_equality eq_comp,
                                 size_type hash_size,
                                 size_type const* const hash_begin,
                                 table_device_view const& hash_data)
    : hasher{hasher},
      eq_comp{eq_comp},
      hash_size{hash_size},
      hash_begin{hash_begin},
      hash_data{hash_data}
  {
  }

  bool __device__ contains_rhs_at(size_type idx) const
  {
    auto const loc = hasher(idx) % (2 * hash_size);
    for (size_type i = hash_begin[loc]; i < hash_begin[loc + 1]; ++i) {
      if (eq_comp(i, idx)) { return true; }
    }
    return false;
  }

 private:
  table_row_hash const hasher;
  table_row_equality eq_comp;
  size_type const hash_size;
  size_type const* const hash_begin;
  table_device_view const hash_data;
};

/*
 * Fixed size set on a device, specialized for structs.
 *
 * It input is two structs columns: a `lhs` column for hashing and a `rhs` column for checking of
 * existence (i.e., checking if the elements of the `rhs` column exist in `lhs`).
 */
template <>
class unordered_multiset<cudf::struct_view, table_row_hash, table_row_equality> {
 public:
  static auto create(column_view const& lhs, column_view const& rhs, rmm::cuda_stream_view stream)
  {
    auto hash_bins_start =
      cudf::detail::make_zeroed_device_uvector_async<size_type>(2 * lhs.size() + 1, stream);
    auto hash_bins_end =
      cudf::detail::make_zeroed_device_uvector_async<size_type>(2 * lhs.size() + 1, stream);
    auto const d_hash_bins_start = hash_bins_start.data();
    auto const d_hash_bins_end   = hash_bins_end.data();

    auto const lhs_table           = table_view{{lhs}};
    auto const rhs_table           = table_view{{rhs}};
    auto const has_null_elements   = has_nested_nulls(lhs_table) || has_nested_nulls(rhs_table);
    auto const flatten_nullability = has_null_elements
                                       ? structs::detail::column_nullability::FORCE
                                       : structs::detail::column_nullability::MATCH_INCOMING;

    auto const flattened_lhs =
      cudf::structs::detail::flatten_nested_columns(lhs_table, {}, {}, flatten_nullability);
    auto const d_flattened_lhs_ptr = table_device_view::create(flattened_lhs, stream);
    auto hasher = table_row_hash{cudf::nullate::DYNAMIC{lhs.has_nulls()}, *d_flattened_lhs_ptr};
    thrust::for_each(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     thrust::make_counting_iterator<size_type>(lhs.size()),
                     [d_hash_bins_start, hasher, d_flattened_lhs = *d_flattened_lhs_ptr] __device__(
                       size_t const idx) {
                       // TODO
                       //                       if (!d_col.is_null(idx))
                       {
                         auto const tmp = hasher(idx) % (2 * d_flattened_lhs.num_rows());
                         atomicAdd(d_hash_bins_start + tmp, size_type{1});
                       }
                     });

    thrust::exclusive_scan(rmm::exec_policy(stream),
                           hash_bins_start.begin(),
                           hash_bins_start.end(),
                           hash_bins_end.begin());

    thrust::copy(rmm::exec_policy(stream),
                 hash_bins_end.begin(),
                 hash_bins_end.end(),
                 hash_bins_start.begin());

    auto hash_gather_map = rmm::device_uvector<size_type>(lhs.size(), stream);
    thrust::for_each(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     thrust::make_counting_iterator<size_type>(lhs.size()),
                     [d_hash_bins_end,
                      hasher,
                      d_flattened_lhs = *d_flattened_lhs_ptr,
                      d_gather_map    = hash_gather_map.begin()] __device__(size_t const idx) {
                       // TODO
                       //                       if (!d_col.is_null(idx))
                       {
                         auto const tmp       = hasher(idx) % (2 * d_flattened_lhs.num_rows());
                         auto const offset    = atomicAdd(d_hash_bins_end + tmp, size_type{1});
                         d_gather_map[offset] = idx;
                       }
                     });

    auto reordered_lhs = cudf::detail::gather(lhs_table,
                                              hash_gather_map,
                                              cudf::out_of_bounds_policy::DONT_CHECK,
                                              cudf::detail::negative_index_policy::NOT_ALLOWED,
                                              stream)
                           ->release();

    auto flattened_reordered_lhs = cudf::structs::detail::flatten_nested_columns(
      table_view{{reordered_lhs.front()->view()}}, {}, {}, flatten_nullability);
    auto d_flattened_reordered_lhs_ptr = table_device_view::create(flattened_reordered_lhs, stream);
    auto flattened_rhs =
      cudf::structs::detail::flatten_nested_columns(rhs_table, {}, {}, flatten_nullability);
    auto d_flattened_rhs_ptr = table_device_view::create(flattened_rhs, stream);
    auto eq_comp             = table_row_equality{cudf::nullate::DYNAMIC{lhs.has_nulls()},
                                      *d_flattened_reordered_lhs_ptr,
                                      *d_flattened_rhs_ptr};

    return unordered_multiset(std::move(hash_bins_start),
                              std::move(reordered_lhs.front()),
                              std::move(hasher),
                              std::move(eq_comp),
                              std::move(flattened_reordered_lhs),
                              std::move(flattened_rhs),
                              std::move(d_flattened_reordered_lhs_ptr),
                              std::move(d_flattened_rhs_ptr));
  }

  unordered_multiset_device_view<cudf::struct_view, table_row_hash, table_row_equality> to_device()
  {
    return unordered_multiset_device_view<cudf::struct_view, table_row_hash, table_row_equality>(
      hasher,
      eq_comp,
      static_cast<size_type>(hash_bins.size()),
      hash_bins.data(),
      *d_flattened_reordered_lhs_ptr);
  }

 private:
  unordered_multiset(rmm::device_uvector<size_type>&& hash_bins,
                     std::unique_ptr<column>&& reordered_lhs,
                     table_row_hash&& hasher,
                     table_row_equality&& eq_comp,
                     cudf::structs::detail::flattened_table&& flattened_reordered_lhs,
                     cudf::structs::detail::flattened_table&& flattened_rhs,
                     d_table_ptr&& d_flattened_reordered_lhs_ptr,
                     d_table_ptr&& d_flattened_rhs_ptr)
    : hash_bins{std::move(hash_bins)},
      reordered_lhs{std::move(reordered_lhs)},
      hasher{std::move(hasher)},
      eq_comp{std::move(eq_comp)},
      flattened_reordered_lhs{std::move(flattened_reordered_lhs)},
      flattened_rhs{std::move(flattened_rhs)},
      d_flattened_reordered_lhs_ptr{std::move(d_flattened_reordered_lhs_ptr)},
      d_flattened_rhs_ptr{std::move(d_flattened_rhs_ptr)}

  {
  }
  rmm::device_uvector<size_type> const hash_bins;
  std::unique_ptr<column> const reordered_lhs;
  table_row_hash const hasher;
  table_row_equality const eq_comp;

  cudf::structs::detail::flattened_table const flattened_reordered_lhs;
  cudf::structs::detail::flattened_table const flattened_rhs;
  d_table_ptr const d_flattened_reordered_lhs_ptr;
  d_table_ptr const d_flattened_rhs_ptr;
};

}  // namespace detail
}  // namespace cudf
