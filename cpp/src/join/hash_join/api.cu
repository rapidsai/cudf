/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/join/hash_join.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/join/hash_join.hpp>

#include <memory>

namespace cudf {

hash_join::~hash_join() = default;

hash_join::hash_join(cudf::table_view const& build,
                     null_equality compare_nulls,
                     rmm::cuda_stream_view stream)
  : hash_join(
      build, nullable_join::YES, compare_nulls, cudf::detail::CUCO_DESIRED_LOAD_FACTOR, stream)
{
}

hash_join::hash_join(cudf::table_view const& build,
                     nullable_join has_nulls,
                     null_equality compare_nulls,
                     double load_factor,
                     rmm::cuda_stream_view stream)
  : _impl{std::make_unique<impl_type const>(
      build, has_nulls == nullable_join::YES, compare_nulls, load_factor, stream)}
{
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::inner_join(cudf::table_view const& probe,
                      std::optional<std::size_t> output_size,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join(probe, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::left_join(cudf::table_view const& probe,
                     std::optional<std::size_t> output_size,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr) const
{
  return _impl->left_join(probe, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::full_join(cudf::table_view const& probe,
                     std::optional<std::size_t> output_size,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr) const
{
  return _impl->full_join(probe, output_size, stream, mr);
}

std::size_t hash_join::inner_join_size(cudf::table_view const& probe,
                                       rmm::cuda_stream_view stream) const
{
  return _impl->inner_join_size(probe, stream);
}

std::size_t hash_join::left_join_size(cudf::table_view const& probe,
                                      rmm::cuda_stream_view stream) const
{
  return _impl->left_join_size(probe, stream);
}

std::size_t hash_join::full_join_size(cudf::table_view const& probe,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr) const
{
  return _impl->full_join_size(probe, stream, mr);
}

cudf::join_match_context hash_join::inner_join_match_context(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join_match_context(probe, stream, mr);
}

cudf::join_match_context hash_join::left_join_match_context(cudf::table_view const& probe,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr) const
{
  return _impl->left_join_match_context(probe, stream, mr);
}

cudf::join_match_context hash_join::full_join_match_context(cudf::table_view const& probe,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr) const
{
  return _impl->full_join_match_context(probe, stream, mr);
}

}  // namespace cudf
