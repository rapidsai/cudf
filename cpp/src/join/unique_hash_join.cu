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

#include <cudf/detail/join.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/hashing/detail/helper_functions.cuh>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/types.hpp>

namespace cudf {
namespace detail {
namespace {
}  // namespace

/*
template <typename Equal, typename Hasher>
unique_hash_join<Equal, Hasher>::unique_hash_join(cudf::table_view const& build,
                             bool has_nulls,
                             cudf::null_equality compare_nulls,
                             rmm::cuda_stream_view stream)
  : _has_nulls(has_nulls),
    _is_empty{build.num_rows() == 0},
    _nulls_equal{compare_nulls},
    _hash_table{::compute_hash_table_size(build.num_rows()),
                cuco::empty_key{cuco::pair{std::numeric_limits<hash_value_type>::max(),
              cudf::detail::JoinNoneValue}}
            },
    _build{build},
    _preprocessed_build{
      cudf::experimental::row::equality::preprocessed_table::create(_build, stream)}
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(0 != build.num_columns(), "Hash join build table is empty");

  if (_is_empty) { return; }

  auto const row_bitmask =
    cudf::detail::bitmask_and(build, stream, rmm::mr::get_current_device_resource()).first;
}

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
                                              null_equality compare_nulls,
                                              rmm::cuda_stream_view stream)
  // If we cannot know beforehand about null existence then let's assume that there are nulls.
  : unique_hash_join(build, nullable_join::YES, compare_nulls, stream)
{
}

template <cudf::has_nested HasNested>
unique_hash_join<HasNested>::unique_hash_join(cudf::table_view const& build,
                                              nullable_join has_nulls,
                                              null_equality compare_nulls,
                                              rmm::cuda_stream_view stream)
  : _impl{std::make_unique<impl_type const>(
      build, has_nulls == nullable_join::YES, compare_nulls, stream)}
{
}

template <cudf::has_nested HasNested>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
unique_hash_join<HasNested>::inner_join(cudf::table_view const& probe,
                                        std::optional<std::size_t> output_size,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr) const
{
  return _impl->inner_join(probe, output_size, stream, mr);
}

template <cudf::has_nested HasNested>
std::size_t unique_hash_join<HasNested>::inner_join_size(cudf::table_view const& probe,
                                                         rmm::cuda_stream_view stream) const
{
  return _impl->inner_join_size(probe, stream);
}

}  // namespace cudf
