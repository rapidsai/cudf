/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <hash/unordered_multiset.cuh>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/dictionary/detail/search.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/uninitialized_fill.h>

namespace cudf {
namespace detail {

namespace {

struct contains_column_dispatch {
  template <typename Element, typename Haystack>
  struct contains_fn {
    bool __device__ operator()(size_type const idx) const
    {
      if (needles_have_nulls && needles.is_null_nocheck(idx)) {
        // Exit early. The value doesn't matter, and will be masked as a null element.
        return true;
      }

      return haystack.contains(needles.template element<Element>(idx));
    }

    Haystack const haystack;
    column_device_view const needles;
    bool const needles_have_nulls;
  };

  template <typename Element, CUDF_ENABLE_IF(!is_nested<Element>())>
  std::unique_ptr<column> operator()(column_view const& haystack,
                                     column_view const& needles,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    auto result = make_numeric_column(data_type{type_to_id<bool>()},
                                      needles.size(),
                                      copy_bitmask(needles, stream, mr),
                                      needles.null_count(),
                                      stream,
                                      mr);
    if (needles.is_empty()) { return result; }

    auto const out_begin = result->mutable_view().template begin<bool>();
    if (haystack.is_empty()) {
      thrust::uninitialized_fill(
        rmm::exec_policy(stream), out_begin, out_begin + needles.size(), false);
      return result;
    }

    auto const haystack_set = cudf::detail::unordered_multiset<Element>::create(haystack, stream);
    auto const haystack_set_dv = haystack_set.to_device();
    auto const needles_cdv_ptr = column_device_view::create(needles, stream);

    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(needles.size()),
                      out_begin,
                      contains_fn<Element, decltype(haystack_set_dv)>{
                        haystack_set_dv, *needles_cdv_ptr, needles.has_nulls()});

    result->set_null_count(needles.null_count());

    return result;
  }

  template <typename Element, CUDF_ENABLE_IF(is_nested<Element>())>
  std::unique_ptr<column> operator()(column_view const& haystack,
                                     column_view const& needles,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    auto result_v = detail::contains(table_view{{haystack}},
                                     table_view{{needles}},
                                     null_equality::EQUAL,
                                     nan_equality::ALL_EQUAL,
                                     stream,
                                     mr);
    return std::make_unique<column>(
      std::move(result_v), copy_bitmask(needles, stream, mr), needles.null_count());
  }
};

template <>
std::unique_ptr<column> contains_column_dispatch::operator()<dictionary32>(
  column_view const& haystack_in,
  column_view const& needles_in,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr) const
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
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(haystack.type() == needles.type(), "DTYPE mismatch");

  return cudf::type_dispatcher(
    haystack.type(), contains_column_dispatch{}, haystack, needles, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> contains(column_view const& haystack,
                                 column_view const& needles,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(haystack, needles, cudf::get_default_stream(), mr);
}

}  // namespace cudf
