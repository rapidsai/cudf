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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/dictionary/detail/search.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/search.hpp>
#include <cudf/structs/detail/contains.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>

#include <hash/unordered_multiset.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform.h>

namespace cudf {
namespace detail {
namespace {

struct contains_scalar_dispatch {
  template <typename Element>
  bool operator()(column_view const& haystack, scalar const& needle, rmm::cuda_stream_view stream)
  {
    CUDF_EXPECTS(haystack.type() == needle.type(), "scalar and column types must match");

    using Type       = device_storage_type_t<Element>;
    using ScalarType = cudf::scalar_type_t<Element>;
    auto d_haystack  = column_device_view::create(haystack, stream);
    auto s           = static_cast<const ScalarType*>(&needle);

    if (haystack.has_nulls()) {
      auto found_iter = thrust::find(rmm::exec_policy(stream),
                                     d_haystack->pair_begin<Type, true>(),
                                     d_haystack->pair_end<Type, true>(),
                                     thrust::make_pair(s->value(stream), true));

      return found_iter != d_haystack->pair_end<Type, true>();
    } else {
      auto found_iter = thrust::find(rmm::exec_policy(stream),  //
                                     d_haystack->begin<Type>(),
                                     d_haystack->end<Type>(),
                                     s->value(stream));

      return found_iter != d_haystack->end<Type>();
    }
  }
};

template <>
bool contains_scalar_dispatch::operator()<cudf::list_view>(column_view const&,
                                                           scalar const&,
                                                           rmm::cuda_stream_view)
{
  CUDF_FAIL("list_view type not supported yet");
}

template <>
bool contains_scalar_dispatch::operator()<cudf::struct_view>(column_view const& haystack,
                                                             scalar const& needle,
                                                             rmm::cuda_stream_view stream)
{
  return cudf::structs::detail::contains(structs_column_view{haystack}, needle, stream);
}

template <>
bool contains_scalar_dispatch::operator()<cudf::dictionary32>(column_view const& haystack,
                                                              scalar const& needle,
                                                              rmm::cuda_stream_view stream)
{
  auto dict_col = cudf::dictionary_column_view(haystack);
  // first, find the needle in the dictionary's key set
  auto index = cudf::dictionary::detail::get_index(dict_col, needle, stream);
  // if found, check the index is actually in the indices column
  return index->is_valid(stream) ? cudf::type_dispatcher(dict_col.indices().type(),
                                                         contains_scalar_dispatch{},
                                                         dict_col.indices(),
                                                         *index,
                                                         stream)
                                 : false;
}

}  // namespace

bool contains(column_view const& haystack, scalar const& needle, rmm::cuda_stream_view stream)
{
  if (haystack.is_empty()) { return false; }
  if (not needle.is_valid(stream)) { return haystack.has_nulls(); }

  return cudf::type_dispatcher(
    haystack.type(), contains_scalar_dispatch{}, haystack, needle, stream);
}

struct multi_contains_dispatch {
  template <typename Element>
  std::unique_ptr<column> operator()(column_view const& haystack,
                                     column_view const& needles,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    std::unique_ptr<column> result = make_numeric_column(data_type{type_to_id<bool>()},
                                                         needles.size(),
                                                         copy_bitmask(needles),
                                                         needles.null_count(),
                                                         stream,
                                                         mr);

    if (needles.is_empty()) { return result; }

    mutable_column_view result_view = result.get()->mutable_view();

    if (haystack.is_empty()) {
      thrust::fill(
        rmm::exec_policy(stream), result_view.begin<bool>(), result_view.end<bool>(), false);
      return result;
    }

    auto hash_set        = cudf::detail::unordered_multiset<Element>::create(haystack, stream);
    auto device_hash_set = hash_set.to_device();

    auto d_needles_ptr = column_device_view::create(needles, stream);
    auto d_needles     = *d_needles_ptr;

    if (needles.has_nulls()) {
      thrust::transform(rmm::exec_policy(stream),
                        thrust::make_counting_iterator<size_type>(0),
                        thrust::make_counting_iterator<size_type>(needles.size()),
                        result_view.begin<bool>(),
                        [device_hash_set, d_needles] __device__(size_t index) {
                          return d_needles.is_null_nocheck(index) ||
                                 device_hash_set.contains(d_needles.element<Element>(index));
                        });
    } else {
      thrust::transform(rmm::exec_policy(stream),
                        thrust::make_counting_iterator<size_type>(0),
                        thrust::make_counting_iterator<size_type>(needles.size()),
                        result_view.begin<bool>(),
                        [device_hash_set, d_needles] __device__(size_t index) {
                          return device_hash_set.contains(d_needles.element<Element>(index));
                        });
    }

    return result;
  }
};

template <>
std::unique_ptr<column> multi_contains_dispatch::operator()<list_view>(
  column_view const&, column_view const&, rmm::cuda_stream_view, rmm::mr::device_memory_resource*)
{
  CUDF_FAIL("list_view type not supported");
}

template <>
std::unique_ptr<column> multi_contains_dispatch::operator()<struct_view>(
  column_view const&, column_view const&, rmm::cuda_stream_view, rmm::mr::device_memory_resource*)
{
  CUDF_FAIL("struct_view type not supported");
}

template <>
std::unique_ptr<column> multi_contains_dispatch::operator()<dictionary32>(
  column_view const& haystack_in,
  column_view const& needles_in,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  dictionary_column_view const haystack(haystack_in);
  dictionary_column_view const needles(needles_in);
  // first combine keys so both dictionaries have the same set
  auto needles_matched     = dictionary::detail::add_keys(needles, haystack.keys(), stream);
  auto const needles_view  = dictionary_column_view(needles_matched->view());
  auto haystack_matched    = dictionary::detail::set_keys(haystack, needles_view.keys(), stream);
  auto const haystack_view = dictionary_column_view(haystack_matched->view());

  // now just use the indices for the contains
  column_view const haystack_indices = haystack_view.get_indices_annotated();
  column_view const needles_indices  = needles_view.get_indices_annotated();
  return cudf::type_dispatcher(haystack_indices.type(),
                               multi_contains_dispatch{},
                               haystack_indices,
                               needles_indices,
                               stream,
                               mr);
}

std::unique_ptr<column> contains(column_view const& haystack,
                                 column_view const& needles,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(haystack.type() == needles.type(), "DTYPE mismatch");

  return cudf::type_dispatcher(
    haystack.type(), multi_contains_dispatch{}, haystack, needles, stream, mr);
}

}  // namespace detail

bool contains(column_view const& haystack, scalar const& needle)
{
  CUDF_FUNC_RANGE();
  return detail::contains(haystack, needle, rmm::cuda_stream_default);
}

std::unique_ptr<column> contains(column_view const& haystack,
                                 column_view const& needles,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(haystack, needles, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
