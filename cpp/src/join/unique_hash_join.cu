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

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/unique_hash_join.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/hashing/detail/helper_functions.cuh>
#include <cudf/join.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cuco/static_set.cuh>

#include <cub/block/block_scan.cuh>

#include <cooperative_groups.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace cudf {
namespace detail {
namespace {

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

template <typename CG>
__device__ void flush_buffer(CG const& group,
                             cudf::size_type* block_counter,
                             cuco::pair<cudf::size_type, cudf::size_type>* buffer,
                             cudf::size_type* counter,
                             cudf::size_type* build_indices,
                             cudf::size_type* probe_indices)
{
  auto i       = group.thread_rank();
  auto const n = *block_counter;

  size_type offset;
  if (i == 0) { offset = atomicAdd(counter, n); }
  offset = group.shfl(offset, 0);

  while (i < n) {
    *(build_indices + offset + i) = buffer[i].first;
    *(probe_indices + offset + i) = buffer[i].second;
    i += group.size();
  }
  if (group.thread_rank() == 0) { *block_counter = 0; }
}

template <typename Iter, typename HashTable>
CUDF_KERNEL void unique_join_probe_kernel(Iter iter,
                                          cudf::size_type size,
                                          HashTable hash_table,
                                          cudf::size_type* build_indices,
                                          cudf::size_type* probe_indices,
                                          cudf::size_type* counter)
{
  namespace cg = cooperative_groups;

  auto constexpr cg_size = HashTable::cg_size;

  __shared__ cuco::pair<size_type, size_type> block_buffer[DEFAULT_JOIN_BLOCK_SIZE];
  __shared__ size_type block_counter;

  auto idx               = cudf::detail::grid_1d::global_thread_id() / cg_size;
  auto const stride      = cudf::detail::grid_1d::grid_stride() / cg_size;
  auto const block       = cg::this_thread_block();
  auto const tile        = cg::tiled_partition<cg_size>(block);
  auto const thread_rank = block.thread_rank();

  if (thread_rank == 0) { block_counter = 0; }
  block.sync();

  while ((idx - stride) < size) {
    if (idx < size) {
      auto const found = hash_table.find(tile, *(iter + idx));
      if (thread_rank == 0 and found != hash_table.end()) {
        auto const offset    = atomicAdd(&block_counter, 1);
        block_buffer[offset] = cuco::pair{found->second, idx};
      }
    }
    block.sync();
    if (block_counter > (DEFAULT_JOIN_BLOCK_SIZE / 2)) {
      flush_buffer(block, &block_counter, block_buffer, counter, build_indices, probe_indices);
    }
    idx += stride;
  }
  block.sync();
  if (block_counter > 0) {
    flush_buffer(block, &block_counter, block_buffer, counter, build_indices, probe_indices);
  }
}
}  // namespace

template <typename Hasher, cudf::has_nested HasNested>
unique_hash_join<Hasher, HasNested>::unique_hash_join(cudf::table_view const& build,
                                                      cudf::table_view const& probe,
                                                      bool has_nulls,
                                                      cudf::null_equality compare_nulls,
                                                      rmm::cuda_stream_view stream)
  : _has_nulls{has_nulls},
    _is_empty{build.num_rows() == 0},
    _nulls_equal{compare_nulls},
    _build{build},
    _probe{probe},
    _preprocessed_build{
      cudf::experimental::row::equality::preprocessed_table::create(_build, stream)},
    _preprocessed_probe{
      cudf::experimental::row::equality::preprocessed_table::create(_probe, stream)},
    _hash_table{::compute_hash_table_size(build.num_rows()),
                cuco::empty_key{cuco::pair{std::numeric_limits<hash_value_type>::max(),
                                           lhs_index_type{JoinNoneValue}}},
                prepare_device_equal<HasNested>(
                  _preprocessed_build, _preprocessed_probe, has_nulls, compare_nulls),
                {},
                cudf::detail::cuco_allocator{stream},
                stream.value()}
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(0 != this->_build.num_columns(), "Hash join build table is empty");

  if (this->_is_empty) { return; }

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
    auto const pred = cudf::detail::row_is_valid{row_bitmask};

    // insert valid rows
    this->_hash_table.insert_if_async(
      iter, iter + build_table_num_rows, stencil, pred, stream.value());
  }
}

template <typename Hasher, cudf::has_nested HasNested>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
unique_hash_join<Hasher, HasNested>::inner_join(std::optional<std::size_t> output_size,
                                                rmm::cuda_stream_view stream,
                                                rmm::mr::device_memory_resource* mr) const
{
  CUDF_FUNC_RANGE();

  size_type const probe_table_num_rows{this->_probe.num_rows()};

  std::size_t const join_size = output_size ? *output_size : probe_table_num_rows;

  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);

  auto const probe_row_hasher =
    cudf::experimental::row::hash::row_hasher{this->_preprocessed_probe};
  auto const d_probe_hasher = probe_row_hasher.device_hasher(nullate::DYNAMIC{this->_has_nulls});
  auto const iter           = cudf::detail::make_counting_transform_iterator(
    0, build_keys_fn<decltype(d_probe_hasher), rhs_index_type>{d_probe_hasher});
  auto counter = rmm::device_scalar<cudf::size_type>{stream};
  counter.set_value_to_zero_async(stream);

  cudf::detail::grid_1d grid{probe_table_num_rows, DEFAULT_JOIN_BLOCK_SIZE};
  unique_join_probe_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
    iter,
    probe_table_num_rows,
    this->_hash_table.ref(cuco::experimental::op::find),
    left_indices->data(),
    right_indices->data(),
    counter.data());

  auto const actual_size = counter.value(stream);
  left_indices->resize(actual_size, stream);
  right_indices->resize(actual_size, stream);

  return {std::move(left_indices), std::move(right_indices)};
}

/*
template <typename Equal, typename Hasher>
std::size_t unique_hash_join<Equal, Hasher>::inner_join_size(cudf::table_view const& probe,
                                                             rmm::cuda_stream_view stream) const
{
  CUDF_FUNC_RANGE();

  // Return directly if build table is empty
  if (_is_empty) { return 0; }

  CUDF_EXPECTS(_has_nulls || !cudf::has_nested_nulls(probe),
               "Probe table has nulls while build table was not hashed with null check.");

  auto const preprocessed_probe =
    cudf::experimental::row::equality::preprocessed_table::create(probe, stream);

  return 10;
  cudf::detail::compute_join_output_size(_build,
                                         probe,
                                         _preprocessed_build,
                                         preprocessed_probe,
                                         _hash_table,
                                         cudf::detail::join_kind::INNER_JOIN,
                                         _has_nulls,
                                         _nulls_equal,
                                         stream);
}
*/
}  // namespace detail

template <cudf::has_nested HasNested>
unique_hash_join<HasNested>::~unique_hash_join() = default;

template <cudf::has_nested HasNested>
unique_hash_join<HasNested>::unique_hash_join(cudf::table_view const& build,
                                              cudf::table_view const& probe,
                                              null_equality compare_nulls,
                                              rmm::cuda_stream_view stream)
  : unique_hash_join(build, probe, nullable_join::YES, compare_nulls, stream)
{
}

template <cudf::has_nested HasNested>
unique_hash_join<HasNested>::unique_hash_join(cudf::table_view const& build,
                                              cudf::table_view const& probe,
                                              nullable_join has_nulls,
                                              null_equality compare_nulls,
                                              rmm::cuda_stream_view stream)
  : _impl{std::make_unique<impl_type>(
      build, probe, has_nulls == nullable_join::YES, compare_nulls, stream)}
{
}

template <cudf::has_nested HasNested>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
unique_hash_join<HasNested>::inner_join(std::optional<std::size_t> output_size,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr) const
{
  return _impl->inner_join(output_size, stream, mr);
}

template <cudf::has_nested HasNested>
std::size_t unique_hash_join<HasNested>::inner_join_size(rmm::cuda_stream_view stream) const
{
  return _impl->inner_join_size(stream);
}

}  // namespace cudf
