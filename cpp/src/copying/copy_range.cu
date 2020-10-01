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
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy_range.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/strings/detail/copy_range.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <thrust/iterator/constant_iterator.h>

#include <cuda_runtime.h>

#include <memory>

namespace {
template <typename T>
void in_place_copy_range(cudf::column_view const& source,
                         cudf::mutable_column_view& target,
                         cudf::size_type source_begin,
                         cudf::size_type source_end,
                         cudf::size_type target_begin,
                         cudaStream_t stream = 0)
{
  auto p_source_device_view = cudf::column_device_view::create(source, stream);
  if (source.has_nulls()) {
    cudf::detail::copy_range(
      cudf::detail::make_null_replacement_iterator<T>(*p_source_device_view, T()) + source_begin,
      cudf::detail::make_validity_iterator(*p_source_device_view) + source_begin,
      target,
      target_begin,
      target_begin + (source_end - source_begin),
      stream);
  } else {
    cudf::detail::copy_range(p_source_device_view->begin<T>() + source_begin,
                             thrust::make_constant_iterator(true),  // dummy
                             target,
                             target_begin,
                             target_begin + (source_end - source_begin),
                             stream);
  }
}

struct in_place_copy_range_dispatch {
  cudf::column_view const& source;
  cudf::mutable_column_view& target;

  template <typename T>
  std::enable_if_t<cudf::is_fixed_width<T>(), void> operator()(cudf::size_type source_begin,
                                                               cudf::size_type source_end,
                                                               cudf::size_type target_begin,
                                                               cudaStream_t stream = 0)
  {
    in_place_copy_range<T>(source, target, source_begin, source_end, target_begin, stream);
  }

  template <typename T>
  std::enable_if_t<not cudf::is_fixed_width<T>(), void> operator()(cudf::size_type source_begin,
                                                                   cudf::size_type source_end,
                                                                   cudf::size_type target_begin,
                                                                   cudaStream_t stream = 0)
  {
    CUDF_FAIL("in-place copy does not work for variable width types.");
  }
};

struct out_of_place_copy_range_dispatch {
  cudf::column_view const& source;
  cudf::column_view const& target;

  template <typename T>
  std::unique_ptr<cudf::column> operator()(
    cudf::size_type source_begin,
    cudf::size_type source_end,
    cudf::size_type target_begin,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
    cudaStream_t stream                 = 0)
  {
    auto p_ret = std::make_unique<cudf::column>(target, stream, mr);
    if ((!p_ret->nullable()) && source.has_nulls(source_begin, source_end)) {
      p_ret->set_null_mask(
        cudf::create_null_mask(p_ret->size(), cudf::mask_state::ALL_VALID, stream, mr), 0);
    }

    if (source_end != source_begin) {  // otherwise no-op
      auto ret_view = p_ret->mutable_view();
      in_place_copy_range<T>(source, ret_view, source_begin, source_end, target_begin, stream);
    }

    return p_ret;
  }
};

template <>
std::unique_ptr<cudf::column> out_of_place_copy_range_dispatch::operator()<cudf::string_view>(
  cudf::size_type source_begin,
  cudf::size_type source_end,
  cudf::size_type target_begin,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  auto target_end           = target_begin + (source_end - source_begin);
  auto p_source_device_view = cudf::column_device_view::create(source, stream);
  if (source.has_nulls()) {
    return cudf::strings::detail::copy_range(
      cudf::detail::make_null_replacement_iterator<cudf::string_view>(*p_source_device_view,
                                                                      cudf::string_view()) +
        source_begin,
      cudf::detail::make_validity_iterator(*p_source_device_view) + source_begin,
      cudf::strings_column_view(target),
      target_begin,
      target_end,
      mr,
      stream);
  } else {
    return cudf::strings::detail::copy_range(
      p_source_device_view->begin<cudf::string_view>() + source_begin,
      thrust::make_constant_iterator(true),
      cudf::strings_column_view(target),
      target_begin,
      target_end,
      mr,
      stream);
  }
}

template <>
std::unique_ptr<cudf::column> out_of_place_copy_range_dispatch::operator()<numeric::decimal64>(
  cudf::size_type source_begin,
  cudf::size_type source_end,
  cudf::size_type target_begin,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  CUDF_FAIL("decimal64 type not supported");
}

template <>
std::unique_ptr<cudf::column> out_of_place_copy_range_dispatch::operator()<numeric::decimal32>(
  cudf::size_type source_begin,
  cudf::size_type source_end,
  cudf::size_type target_begin,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  CUDF_FAIL("decimal32 type not supported");
}

template <>
std::unique_ptr<cudf::column> out_of_place_copy_range_dispatch::operator()<cudf::dictionary32>(
  cudf::size_type source_begin,
  cudf::size_type source_end,
  cudf::size_type target_begin,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  // check the keys in the source and target
  cudf::dictionary_column_view const dict_source(source);
  cudf::dictionary_column_view const dict_target(target);
  CUDF_EXPECTS(dict_source.keys().type() == dict_target.keys().type(),
               "dictionary keys must be the same type");

  // combine keys so both dictionaries have the same set
  auto target_matched =
    cudf::dictionary::detail::add_keys(dict_target, dict_source.keys(), mr, stream);
  auto const target_view = cudf::dictionary_column_view(target_matched->view());
  auto source_matched    = cudf::dictionary::detail::set_keys(
    dict_source, target_view.keys(), rmm::mr::get_current_device_resource(), stream);
  auto const source_view = cudf::dictionary_column_view(source_matched->view());

  // build the new indices by calling in_place_copy_range on just the indices
  auto const source_indices = source_view.get_indices_annotated();
  auto target_contents      = target_matched->release();
  auto target_indices(std::move(target_contents.children.front()));
  cudf::mutable_column_view new_indices(
    target_indices->type(),
    dict_target.size(),
    target_indices->mutable_view().head(),
    static_cast<cudf::bitmask_type*>(target_contents.null_mask->data()),
    dict_target.null_count());
  cudf::type_dispatcher(new_indices.type(),
                        in_place_copy_range_dispatch{source_indices, new_indices},
                        source_begin,
                        source_end,
                        target_begin,
                        stream);
  auto null_count = new_indices.null_count();
  auto indices_column =
    std::make_unique<cudf::column>(new_indices.type(),
                                   new_indices.size(),
                                   std::move(*(target_indices->release().data.release())),
                                   rmm::device_buffer{0, stream, mr},
                                   0);

  // take the keys from the matched column allocated using mr
  auto keys_column(std::move(target_contents.children.back()));

  // create column with keys_column and indices_column
  return make_dictionary_column(std::move(keys_column),
                                std::move(indices_column),
                                std::move(*(target_contents.null_mask.release())),
                                null_count);
}

template <>
std::unique_ptr<cudf::column> out_of_place_copy_range_dispatch::operator()<cudf::list_view>(
  cudf::size_type source_begin,
  cudf::size_type source_end,
  cudf::size_type target_begin,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  CUDF_FAIL("list_view type not supported");
}

}  // namespace

namespace cudf {
namespace detail {
void copy_range_in_place(column_view const& source,
                         mutable_column_view& target,
                         size_type source_begin,
                         size_type source_end,
                         size_type target_begin,
                         cudaStream_t stream)
{
  CUDF_EXPECTS(cudf::is_fixed_width(target.type()) == true,
               "In-place copy_range does not support variable-sized types.");
  CUDF_EXPECTS((source_begin >= 0) && (source_end <= source.size()) &&
                 (source_begin <= source_end) && (target_begin >= 0) &&
                 (target_begin <= target.size() - (source_end - source_begin)),
               "Range is out of bounds.");
  CUDF_EXPECTS(target.type() == source.type(), "Data type mismatch.");
  CUDF_EXPECTS((target.nullable() == true) || (source.has_nulls() == false),
               "target should be nullable if source has null values.");

  if (source_end != source_begin) {  // otherwise no-op
    cudf::type_dispatcher(target.type(),
                          in_place_copy_range_dispatch{source, target},
                          source_begin,
                          source_end,
                          target_begin,
                          stream);
  }
}

std::unique_ptr<column> copy_range(column_view const& source,
                                   column_view const& target,
                                   size_type source_begin,
                                   size_type source_end,
                                   size_type target_begin,
                                   rmm::mr::device_memory_resource* mr,
                                   cudaStream_t stream)
{
  CUDF_EXPECTS((source_begin >= 0) && (source_end <= source.size()) &&
                 (source_begin <= source_end) && (target_begin >= 0) &&
                 (target_begin <= target.size() - (source_end - source_begin)),
               "Range is out of bounds.");
  CUDF_EXPECTS(target.type() == source.type(), "Data type mismatch.");

  return cudf::type_dispatcher(target.type(),
                               out_of_place_copy_range_dispatch{source, target},
                               source_begin,
                               source_end,
                               target_begin,
                               mr,
                               stream);
}

}  // namespace detail

void copy_range_in_place(column_view const& source,
                         mutable_column_view& target,
                         size_type source_begin,
                         size_type source_end,
                         size_type target_begin)
{
  CUDF_FUNC_RANGE();
  return detail::copy_range_in_place(source, target, source_begin, source_end, target_begin, 0);
}

std::unique_ptr<column> copy_range(column_view const& source,
                                   column_view const& target,
                                   size_type source_begin,
                                   size_type source_end,
                                   size_type target_begin,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_range(source, target, source_begin, source_end, target_begin, mr, 0);
}

}  // namespace cudf
