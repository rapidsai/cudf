/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/stream_compaction.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/binary_search.h>

namespace cudf {
namespace dictionary {
namespace detail {
namespace {
/**
 * @brief Type-dispatch functor for remapping the old indices to new values based on the new
 * key-set.
 *
 * The dispatch is based on the key type.
 * The output column is the new indices column for the new dictionary column.
 */
struct dispatch_compute_indices {
  template <typename Element>
  typename std::enable_if_t<cudf::is_relationally_comparable<Element, Element>(),
                            std::unique_ptr<column>>
  operator()(dictionary_column_view const& input,
             column_view const& new_keys,
             rmm::mr::device_memory_resource* mr,
             cudaStream_t stream)
  {
    auto dictionary_view = column_device_view::create(input.parent(), stream);
    auto d_dictionary    = *dictionary_view;
    auto dictionary_itr  = thrust::make_transform_iterator(
      thrust::make_counting_iterator<size_type>(0), [d_dictionary] __device__(size_type idx) {
        if (d_dictionary.is_null(idx)) return Element{};
        column_device_view d_keys = d_dictionary.child(1);
        size_type index           = static_cast<size_type>(d_dictionary.element<dictionary32>(idx));
        return d_keys.template element<Element>(index);
      });
    auto new_keys_view = column_device_view::create(new_keys, stream);
    auto d_new_keys    = *new_keys_view;
    auto keys_itr      = thrust::make_transform_iterator(
      thrust::make_counting_iterator<size_type>(0),
      [d_new_keys] __device__(size_type idx) { return d_new_keys.template element<Element>(idx); });

    auto result =
      make_numeric_column(data_type{INT32}, input.size(), mask_state::UNALLOCATED, stream, mr);
    auto d_result = result->mutable_view().data<int32_t>();
    auto execpol  = rmm::exec_policy(stream);
    thrust::lower_bound(execpol->on(stream),
                        keys_itr,
                        keys_itr + new_keys.size(),
                        dictionary_itr,
                        dictionary_itr + input.size(),
                        d_result,
                        thrust::less<Element>());
    result->set_null_count(0);
    return result;
  }

  template <typename Element>
  typename std::enable_if_t<!cudf::is_relationally_comparable<Element, Element>(),
                            std::unique_ptr<column>>
  operator()(dictionary_column_view const& input,
             column_view const& new_keys,
             rmm::mr::device_memory_resource* mr,
             cudaStream_t stream)
  {
    CUDF_FAIL("list_view dictionary set_keys not supported yet");
  }
};

}  // namespace

//
std::unique_ptr<column> set_keys(
  dictionary_column_view const& dictionary_column,
  column_view const& new_keys,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  CUDF_EXPECTS(!new_keys.has_nulls(), "keys parameter must not have nulls");
  auto keys = dictionary_column.keys();
  CUDF_EXPECTS(keys.type() == new_keys.type(), "keys types must match");

  // copy the keys -- use drop_duplicates to make sure they are sorted and unique
  auto table_keys = cudf::detail::drop_duplicates(table_view{{new_keys}},
                                                  std::vector<size_type>{0},
                                                  duplicate_keep_option::KEEP_FIRST,
                                                  null_equality::EQUAL,
                                                  mr,
                                                  stream)
                      ->release();
  std::unique_ptr<column> keys_column(std::move(table_keys.front()));

  // compute the new nulls
  auto matches     = cudf::detail::contains(keys, keys_column->view(), mr, stream);
  auto d_matches   = matches->view().data<bool>();
  auto d_indices   = dictionary_column.indices().data<int32_t>();
  auto d_null_mask = dictionary_column.null_mask();
  auto new_nulls   = cudf::detail::valid_if(
    thrust::make_counting_iterator<size_type>(dictionary_column.offset()),
    thrust::make_counting_iterator<size_type>(dictionary_column.offset() +
                                              dictionary_column.size()),
    [d_null_mask, d_indices, d_matches] __device__(size_type idx) {
      if (d_null_mask && !bit_is_set(d_null_mask, idx)) return false;
      return d_matches[d_indices[idx]];
    },
    stream,
    mr);

  // compute the new indices
  auto indices_column = type_dispatcher(keys_column->type(),
                                        dispatch_compute_indices{},
                                        dictionary_column,
                                        keys_column->view(),
                                        mr,
                                        stream);

  // create column with keys_column and indices_column
  return make_dictionary_column(std::move(keys_column),
                                std::move(indices_column),
                                std::move(new_nulls.first),
                                new_nulls.second);
}
}  // namespace detail

// external API

std::unique_ptr<column> set_keys(dictionary_column_view const& dictionary_column,
                                 column_view const& keys,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::set_keys(dictionary_column, keys, mr);
}

}  // namespace dictionary
}  // namespace cudf
