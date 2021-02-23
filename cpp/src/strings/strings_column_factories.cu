/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <strings/utilities.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>

namespace cudf {

namespace {
struct string_view_to_pair {
  string_view null_placeholder;
  string_view_to_pair(string_view n) : null_placeholder(n) {}
  __device__ thrust::pair<const char*, size_type> operator()(const string_view& i)
  {
    return (i.data() == null_placeholder.data())
             ? thrust::pair<const char*, size_type>{nullptr, 0}
             : thrust::pair<const char*, size_type>{i.data(), i.size_bytes()};
  }
};

}  // namespace

namespace detail {

std::unique_ptr<column> make_strings_column(
  const device_span<thrust::pair<const char*, size_type>>& strings,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  return cudf::strings::detail::make_strings_column(strings.begin(), strings.end(), stream, mr);
}

std::unique_ptr<column> make_strings_column(const device_span<string_view>& string_views,
                                            const string_view null_placeholder,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  auto it_pair =
    thrust::make_transform_iterator(string_views.begin(), string_view_to_pair{null_placeholder});
  return cudf::strings::detail::make_strings_column(
    it_pair, it_pair + string_views.size(), stream, mr);
}

std::unique_ptr<column> make_strings_column(
  device_span<char> const& chars,
  device_span<size_type> const& offsets,
  size_type null_count,
  rmm::device_buffer&& null_mask,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  return cudf::strings::detail::make_strings_column(chars.begin(),
                                                    chars.end(),
                                                    offsets.begin(),
                                                    offsets.end(),
                                                    null_count,
                                                    std::move(null_mask),
                                                    stream,
                                                    mr);
}

}  // namespace detail

// Create a strings-type column from vector of pointer/size pairs
std::unique_ptr<column> make_strings_column(
  const rmm::device_vector<thrust::pair<const char*, size_type>>& strings,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  return strings::detail::make_strings_column(strings.begin(), strings.end(), stream, mr);
}

// Create a strings-type column from vector of string_view
std::unique_ptr<column> make_strings_column(const rmm::device_vector<string_view>& string_views,
                                            const string_view null_placeholder,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  return detail::make_strings_column(
    detail::device_span<string_view>{const_cast<string_view*>(string_views.data().get()),
                                     string_views.size()},
    null_placeholder,
    stream,
    mr);
}

// Create a strings-type column from device vector of chars and vector of offsets.
std::unique_ptr<column> make_strings_column(rmm::device_vector<char> const& strings,
                                            rmm::device_vector<size_type> const& offsets,
                                            rmm::device_vector<bitmask_type> const& valid_mask,
                                            size_type null_count,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  // build null bitmask
  rmm::device_buffer null_mask{valid_mask.data().get(), valid_mask.size() * sizeof(bitmask_type)};

  return cudf::strings::detail::make_strings_column(strings.begin(),
                                                    strings.end(),
                                                    offsets.begin(),
                                                    offsets.end(),
                                                    null_count,
                                                    std::move(null_mask),
                                                    stream,
                                                    mr);
}

// Create strings column from host vectors
std::unique_ptr<column> make_strings_column(std::vector<char> const& strings,
                                            std::vector<size_type> const& offsets,
                                            std::vector<bitmask_type> const& null_mask,
                                            size_type null_count,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  rmm::device_uvector<char> d_strings{strings.size(), stream};
  rmm::device_uvector<size_type> d_offsets{offsets.size(), stream};
  rmm::device_uvector<bitmask_type> d_null_mask{null_mask.size(), stream};

  CUDA_TRY(cudaMemcpyAsync(
    d_strings.data(), strings.data(), strings.size(), cudaMemcpyDefault, stream.value()));
  CUDA_TRY(cudaMemcpyAsync(d_offsets.data(),
                           offsets.data(),
                           offsets.size() * sizeof(size_type),
                           cudaMemcpyDefault,
                           stream.value()));
  CUDA_TRY(cudaMemcpyAsync(d_null_mask.data(),
                           null_mask.data(),
                           null_mask.size() * sizeof(bitmask_type),
                           cudaMemcpyDefault,
                           stream.value()));

  return make_strings_column(detail::device_span<char>{d_strings},
                             detail::device_span<size_type>{d_offsets},
                             null_count,
                             d_null_mask.release(),
                             stream,
                             mr);
}

//
std::unique_ptr<column> make_strings_column(size_type num_strings,
                                            std::unique_ptr<column> offsets_column,
                                            std::unique_ptr<column> chars_column,
                                            size_type null_count,
                                            rmm::device_buffer&& null_mask,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  if (null_count > 0) CUDF_EXPECTS(null_mask.size() > 0, "Column with nulls must be nullable.");
  CUDF_EXPECTS(num_strings == offsets_column->size() - 1,
               "Invalid offsets column size for strings column.");
  CUDF_EXPECTS(offsets_column->null_count() == 0, "Offsets column should not contain nulls");
  CUDF_EXPECTS(chars_column->null_count() == 0, "Chars column should not contain nulls");

  std::vector<std::unique_ptr<column>> children;
  children.emplace_back(std::move(offsets_column));
  children.emplace_back(std::move(chars_column));
  return std::make_unique<column>(data_type{type_id::STRING},
                                  num_strings,
                                  rmm::device_buffer{0, stream, mr},
                                  null_mask,
                                  null_count,
                                  std::move(children));
}

}  // namespace cudf
