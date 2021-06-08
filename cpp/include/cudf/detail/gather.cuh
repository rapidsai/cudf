/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf/detail/copy.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/lists/detail/gather.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/detail/gather.cuh>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <algorithm>

#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>

namespace cudf {
namespace detail {

/**
 * @brief The operation to perform when a gather map index is out of bounds
 */
enum class gather_bitmask_op {
  DONT_CHECK,   // Don't check for out of bounds indices
  PASSTHROUGH,  // Preserve mask at rows with out of bounds indices
  NULLIFY,      // Nullify rows with out of bounds indices
};

// template <gather_bitmask_op Op, typename GatherMap>
// void gather_bitmask(table_device_view input,
//                    GatherMap gather_map_begin,
//                    bitmask_type** masks,
//                    size_type mask_count,
//                    size_type mask_size,
//                    size_type* valid_counts,
//                    rmm::cuda_stream_view stream);

template <typename MapIterator>
void gather_bitmask(table_view const& source,
                    MapIterator& gather_map,
                    std::vector<std::unique_ptr<column>>& target,
                    gather_bitmask_op op,
                    rmm::cuda_stream_view stream,
                    rmm::mr::device_memory_resource* mr);

/**
 * @brief Gathers the specified rows of a set of columns according to a gather map.
 *
 * Gathers the rows of the source columns according to `gather_map` such that row "i"
 * in the resulting table's columns will contain row "gather_map[i]" from the source columns.
 * The number of rows in the result table will be equal to the number of elements in
 * `gather_map`.
 *
 * A negative value `i` in the `gather_map` is interpreted as `i+n`, where
 * `n` is the number of rows in the `source_table`.
 *
 * tparam MapIterator Iterator type for the gather map
 * @param[in] source_table View into the table containing the input columns whose rows will be
 * gathered
 * @param[in] gather_map_begin Beginning of iterator range of integer indices that map the rows in
 * the source columns to rows in the destination columns
 * @param[in] gather_map_end End of iterator range of integer indices that map the rows in the
 * source columns to rows in the destination columns
 * @param[in] bounds_policy Policy to apply to account for possible out-of-bound indices
 * `DONT_CHECK` skips all bound checking for gather map values. `NULLIFY` coerces rows that
 * corresponds to out-of-bound indices in the gather map to be null elements. Callers should
 * use `DONT_CHECK` when they are certain that the gather_map contains only valid indices for
 * better performance. In case there are out-of-bound indices in the gather map, the behavior
 * is undefined. Defaults to `DONT_CHECK`.
 * @param[in] mr Device memory resource used to allocate the returned table's device memory
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @return cudf::table Result of the gather
 */

class iterator_adaper_base {
 public:
  iterator_adaper_base()                                     = default;
  virtual ~iterator_adaper_base()                            = default;
  virtual bool operator!=(iterator_adaper_base const&) const = 0;
  virtual bool operator==(iterator_adaper_base const&) const = 0;
  virtual int32_t operator*()                                = 0;
  virtual iterator_adaper_base& operator++()                 = 0;
};

template <class T>
class iterator_adaper : public iterator_adaper_base {
 public:
  iterator_adaper(T const& iter) : _iter(iter) {}
  iterator_adaper(iterator_adaper const& other) : _iter(other._iter) {}
  iterator_adaper& operator=(iterator_adaper const& other)
  {
    _iter = other._iter;
    return *this;
  }

  bool operator!=(iterator_adaper_base const& other) const override
  {
    auto const& other_t = static_cast<iterator_adaper const&>(other);
    return _iter != other_t._iter;
  }
  bool operator==(iterator_adaper_base const& other) const override
  {
    auto const& other_t = static_cast<iterator_adaper const&>(other);
    return _iter == other_t._iter;
  }
  int32_t operator*() override { return static_cast<int>(*_iter); }
  iterator_adaper_base& operator++() override
  {
    ++_iter;
    return static_cast<iterator_adaper_base&>(*this);
  }

 private:
  T _iter;
};

std::unique_ptr<table> gather_iterator(
  table_view const& source_table,
  iterator_adaper_base& gather_map_begin,
  iterator_adaper_base& gather_map_end,
  out_of_bounds_policy bounds_policy  = out_of_bounds_policy::DONT_CHECK,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

template <typename MapIterator>
std::unique_ptr<table> gather(
  table_view const& source_table,
  MapIterator gather_map_begin,
  MapIterator gather_map_end,
  out_of_bounds_policy bounds_policy  = out_of_bounds_policy::DONT_CHECK,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  iterator_adaper<MapIterator> iter_begin(gather_map_begin);
  iterator_adaper<MapIterator> iter_end(gather_map_begin);
  gather_iterator(source_table,
                  static_cast<iterator_adaper_base&>(iter_begin),
                  static_cast<iterator_adaper_base&>(iter_end),
                  bounds_policy,
                  stream,
                  mr);
}

template <gather_bitmask_op Op, typename MapIterator>
struct gather_bitmask_functor {
  table_device_view input;
  bitmask_type** masks;
  MapIterator gather_map;

  __device__ bool operator()(size_type mask_idx, size_type bit_idx)
  {
    auto row_idx = gather_map[bit_idx];
    auto col     = input.column(mask_idx);

    if (Op != gather_bitmask_op::DONT_CHECK) {
      bool out_of_range = is_signed_iterator<MapIterator>() ? (row_idx < 0 || row_idx >= col.size())
                                                            : row_idx >= col.size();
      if (out_of_range) {
        if (Op == gather_bitmask_op::PASSTHROUGH) {
          return bit_is_set(masks[mask_idx], bit_idx);
        } else if (Op == gather_bitmask_op::NULLIFY) {
          return false;
        }
      }
    }

    return col.is_valid(row_idx);
  }
};

template <gather_bitmask_op Op>
void gather_bitmask(table_device_view input,
                    iterator_adaper_base& gather_map_begin,
                    bitmask_type** masks,
                    size_type mask_count,
                    size_type mask_size,
                    size_type* valid_counts,
                    rmm::cuda_stream_view stream)
{
  if (mask_size == 0) { return; }

  constexpr size_type block_size = 256;
  using Selector                 = gather_bitmask_functor<Op, decltype(gather_map_begin)>;
  auto selector                  = Selector{input, masks, gather_map_begin};
  auto counting_it               = thrust::make_counting_iterator(0);
  auto kernel =
    valid_if_n_kernel<decltype(counting_it), decltype(counting_it), Selector, block_size>;

  cudf::detail::grid_1d grid{mask_size, block_size, 1};
  kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
    counting_it, counting_it, selector, masks, mask_count, mask_size, valid_counts);
}

}  // namespace detail
}  // namespace cudf
