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
#include "join_common_utils.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/unique_hash_join.cuh>
#include <cudf/hashing/detail/helper_functions.cuh>
#include <cudf/join.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

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
                                           lhs_index_type{cudf::detail::JoinNoneValue}}},
                prepare_device_equal<HasNested>(
                  _preprocessed_build, _preprocessed_probe, has_nulls, compare_nulls),
                {},
                cudf::detail::cuco_allocator{stream},
                stream.value()}
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(not this->_is_empty, "Hash join build table is empty");
}

/*
template <typename Equal, typename Hasher>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
unique_hash_join<Equal, Hasher>::inner_join(cudf::table_view const& probe,
                              std::optional<std::size_t> output_size,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr) const
{
  CUDF_FUNC_RANGE();
  return ;
}

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
