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

#include <cudf/detail/copy.hpp>
#include <cudf/detail/copy_if_else.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/unary.hpp>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/detail/replace.hpp>
#include <cudf/dictionary/detail/search.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace cudf {
namespace dictionary {
namespace detail {
namespace {

/**
 * @brief An index accessor that returns a validity flag along with the index value.
 *
 * This is used to make a `pair_iterator` for calling `copy_if_else`.
 */
template <bool has_nulls = false>
struct nullable_index_accessor {
  cudf::detail::input_indexalator iter;
  bitmask_type const* null_mask{};
  size_type const offset{};

  /**
   * @brief Create an accessor from a column_view.
   */
  nullable_index_accessor(column_view const& col) : null_mask{col.null_mask()}, offset{col.offset()}
  {
    if (has_nulls) { CUDF_EXPECTS(col.nullable(), "Unexpected non-nullable column."); }
    iter = cudf::detail::indexalator_factory::make_input_iterator(col);
  }

  /**
   * @brief Create an accessor from a scalar.
   */
  nullable_index_accessor(scalar const& input)
  {
    iter = cudf::detail::indexalator_factory::make_input_iterator(input);
  }

  __device__ thrust::pair<size_type, bool> operator()(size_type i) const
  {
    return {iter[i], (has_nulls ? bit_is_set(null_mask, i + offset) : true)};
  }
};

/**
 * @brief Create an index iterator with a nullable index accessor.
 */
template <bool has_nulls>
auto make_nullable_index_iterator(column_view const& col)
{
  return thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                         nullable_index_accessor<has_nulls>{col});
}

/**
 * @brief Create an index iterator with a nullable index accessor for a scalar.
 */
auto make_scalar_iterator(scalar const& input)
{
  return thrust::make_transform_iterator(thrust::make_constant_iterator<size_type>(0),
                                         nullable_index_accessor<false>{input});
}

/**
 * @brief This utility uses `copy_if_else` to replace null entries using the input bitmask as a
 * predicate.
 *
 * The predicate identifies which column row to copy from and the bitmask specifies which rows
 * are null. Since the `copy_if_else` accepts iterators, we also supply it with pair-iterators
 * created from indexalators and the validity masks.
 *
 * @tparam ReplacementItr must be a pair iterator of (index,valid).
 *
 * @param input lhs for `copy_if_else`
 * @param replacement_iter rhs for `copy_if_else`
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Always returns column of type INT32 (size_type)
 */
template <typename ReplacementIter>
std::unique_ptr<column> replace_indices(column_view const& input,
                                        ReplacementIter replacement_iter,
                                        rmm::mr::device_memory_resource* mr,
                                        cudaStream_t stream)
{
  auto const input_view = column_device_view::create(input, stream);
  auto const d_input    = *input_view;
  auto predicate        = [d_input] __device__(auto i) { return d_input.is_valid(i); };

  auto input_pair_iterator = make_nullable_index_iterator<true>(input);
  return cudf::detail::copy_if_else(true,
                                    input_pair_iterator,
                                    input_pair_iterator + input.size(),
                                    replacement_iter,
                                    predicate,
                                    mr,
                                    stream);
}

}  // namespace

/**
 * @copydoc cudf::dictionary::detail::replace_nulls(cudf::column_view const&,cudf::column_view
 * const&,rmm::mr::device_memory_resource*,cudaStream_t)
 */
std::unique_ptr<column> replace_nulls(dictionary_column_view const& input,
                                      dictionary_column_view const& replacement,
                                      rmm::mr::device_memory_resource* mr,
                                      cudaStream_t stream)
{
  if (input.size() == 0) { return cudf::empty_like(input.parent()); }
  if (!input.has_nulls()) { return std::make_unique<cudf::column>(input.parent()); }
  CUDF_EXPECTS(input.keys().type() == replacement.keys().type(), "keys must match");
  CUDF_EXPECTS(replacement.size() == input.size(), "column sizes must match");

  // first combine the keys so both input dictionaries have the same set
  auto matched = match_dictionaries({input, replacement}, mr, stream);

  // now build the new indices by doing replace-null using the updated input indices
  auto const input_indices =
    dictionary_column_view(matched.front()->view()).get_indices_annotated();
  auto const repl_indices = dictionary_column_view(matched.back()->view()).get_indices_annotated();
  auto new_indices =
    repl_indices.has_nulls()
      ? replace_indices(input_indices, make_nullable_index_iterator<true>(repl_indices), mr, stream)
      : replace_indices(
          input_indices, make_nullable_index_iterator<false>(repl_indices), mr, stream);

  // auto keys_column = ;
  return make_dictionary_column(
    std::move(matched.front()->release().children.back()), std::move(new_indices), mr, stream);
}

/**
 * @copydoc cudf::dictionary::detail::replace_nulls(cudf::column_view const&,cudf::scalar
 * const&,rmm::mr::device_memory_resource*,cudaStream_t)
 */
std::unique_ptr<column> replace_nulls(dictionary_column_view const& input,
                                      scalar const& replacement,
                                      rmm::mr::device_memory_resource* mr,
                                      cudaStream_t stream)
{
  if (input.size() == 0) { return cudf::empty_like(input.parent()); }
  if (!input.has_nulls() || !replacement.is_valid()) {
    return std::make_unique<cudf::column>(input.parent());
  }
  CUDF_EXPECTS(input.keys().type() == replacement.type(), "keys must match scalar type");

  // first add the replacment to the keys so only the indices need to be processed
  auto const default_mr = rmm::mr::get_current_device_resource();
  auto input_matched    = dictionary::detail::add_keys(
    input, make_column_from_scalar(replacement, 1, default_mr, stream)->view(), mr, stream);
  auto const input_view   = dictionary_column_view(input_matched->view());
  auto const scalar_index = get_index(input_view, replacement, default_mr, stream);

  // now build the new indices by doing replace-null on the updated indices
  auto const input_indices = input_view.get_indices_annotated();
  auto new_indices =
    replace_indices(input_indices, make_scalar_iterator(*scalar_index), mr, stream);
  new_indices->set_null_mask(rmm::device_buffer{0, stream, mr}, 0);

  return make_dictionary_column(
    std::move(input_matched->release().children.back()), std::move(new_indices), mr, stream);
}

}  // namespace detail
}  // namespace dictionary
}  // namespace cudf
