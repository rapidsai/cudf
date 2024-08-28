/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/dictionary/detail/search.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/search.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

namespace cudf {
namespace detail {

namespace {

struct contains_column_dispatch {
  template <typename Element>
  std::unique_ptr<column> operator()(column_view const& haystack,
                                     column_view const& needles,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
  {
    auto result_v = detail::contains(table_view{{haystack}},
                                     table_view{{needles}},
                                     null_equality::EQUAL,
                                     nan_equality::ALL_EQUAL,
                                     stream,
                                     mr);
    return std::make_unique<column>(
      std::move(result_v), detail::copy_bitmask(needles, stream, mr), needles.null_count());
  }
};

template <>
std::unique_ptr<column> contains_column_dispatch::operator()<dictionary32>(
  column_view const& haystack_in,
  column_view const& needles_in,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  dictionary_column_view const haystack(haystack_in);
  dictionary_column_view const needles(needles_in);
  // first combine keys so both dictionaries have the same set
  auto needles_matched = dictionary::detail::add_keys(
    needles, haystack.keys(), stream, rmm::mr::get_current_device_resource());
  auto const needles_view = dictionary_column_view(needles_matched->view());
  auto haystack_matched   = dictionary::detail::set_keys(
    haystack, needles_view.keys(), stream, rmm::mr::get_current_device_resource());
  auto const haystack_view = dictionary_column_view(haystack_matched->view());

  // now just use the indices for the contains
  column_view const haystack_indices = haystack_view.get_indices_annotated();
  column_view const needles_indices  = needles_view.get_indices_annotated();
  return cudf::type_dispatcher(haystack_indices.type(),
                               contains_column_dispatch{},
                               haystack_indices,
                               needles_indices,
                               stream,
                               mr);
}

}  // namespace

std::unique_ptr<column> contains(column_view const& haystack,
                                 column_view const& needles,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  return cudf::type_dispatcher(
    haystack.type(), contains_column_dispatch{}, haystack, needles, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> contains(column_view const& haystack,
                                 column_view const& needles,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(haystack, needles, stream, mr);
}

}  // namespace cudf
