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

#include <cudf/detail/copy_if_else.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/unary.hpp>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/detail/replace.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>

namespace cudf {
namespace dictionary {
namespace detail {
namespace {

template <bool has_nulls = false>
struct nullable_index_accessor {
  cudf::detail::input_indexalator itr;
  bitmask_type const* null_mask;
  size_type const offset;
  nullable_index_accessor(column_view const& col) : null_mask{col.null_mask()}, offset{col.offset()}
  {
    if (has_nulls) { CUDF_EXPECTS(col.nullable(), "Unexpected non-nullable column."); }
    itr = cudf::detail::indexalator_factory::make_input_iterator(col);
  }

  __device__ thrust::pair<size_type, bool> operator()(size_type i) const
  {
    return {itr[i], (has_nulls ? bit_is_set(null_mask, i + offset) : true)};
  }
};

template <bool has_nulls>
static auto make_pair_iterator(column_view const& col)
{
  return thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                         nullable_index_accessor<has_nulls>{col});
}

template <bool has_nulls>
std::unique_ptr<column> replace_indices(column_view const& input,
                                        column_view const& replacement,
                                        rmm::mr::device_memory_resource* mr,
                                        cudaStream_t stream)
{
  size_type const size  = input.size();
  auto const input_view = column_device_view::create(input, stream);
  auto const d_input    = *input_view;
  auto predicate        = [d_input] __device__(auto i) { return d_input.is_valid(i); };

  auto input_pair_iterator = make_pair_iterator<has_nulls>(input);
  if (replacement.nullable()) {
    auto replacement_pair_iterator = make_pair_iterator<true>(replacement);
    return cudf::detail::copy_if_else(true,
                                      input_pair_iterator,
                                      input_pair_iterator + size,
                                      replacement_pair_iterator,
                                      predicate,
                                      mr,
                                      stream);
  } else {
    auto replacement_pair_iterator = make_pair_iterator<false>(replacement);
    return cudf::detail::copy_if_else(has_nulls,
                                      input_pair_iterator,
                                      input_pair_iterator + size,
                                      replacement_pair_iterator,
                                      predicate,
                                      mr,
                                      stream);
  }
}

}  // namespace

/**
 * @brief Create a new dictionary column by replace nulls with values
 * from second dictionary.
 */
std::unique_ptr<column> replace_nulls(dictionary_column_view const& input,
                                      dictionary_column_view const& replacement,
                                      rmm::mr::device_memory_resource* mr,
                                      cudaStream_t stream)
{
  CUDF_EXPECTS(input.keys().type() == replacement.keys().type(), "keys must match");

  // first combine keys so both dictionaries have the same set
  auto input_matched    = dictionary::detail::add_keys(input, replacement.keys(), mr, stream);
  auto const input_view = dictionary_column_view(input_matched->view());
  auto repl_matched     = dictionary::detail::set_keys(
    replacement, input_view.keys(), rmm::mr::get_current_device_resource(), stream);
  auto const repl_view = dictionary_column_view(repl_matched->view());

  // now build the new indices by doing replace-null on the indices
  auto const input_indices = input_view.get_indices_annotated();
  auto const repl_indices  = repl_view.get_indices_annotated();
  //
  auto new_indices = input_indices.has_nulls()
                       ? replace_indices<true>(input_indices, repl_indices, mr, stream)
                       : replace_indices<false>(input_indices, repl_indices, mr, stream);

  auto const indices_type = new_indices->type();
  auto const indices_size = new_indices->size();
  auto const null_count   = new_indices->null_count();
  auto contents           = new_indices->release();
  auto const new_type     = get_indices_type_for_size(input_view.keys().size());
  // build the indices for the output column
  auto indices_column = [&] {
    if (new_type.id() == cudf::type_id::UINT32) {
      return std::make_unique<column>(cudf::data_type{cudf::type_id::UINT32},
                                      indices_size,
                                      std::move(*(contents.data.release())),
                                      rmm::device_buffer{0, stream, mr},
                                      0);
    }
    cudf::column_view cast_view{indices_type, indices_size, contents.data->data()};
    return cudf::detail::cast(cast_view, new_type, mr, stream);
  }();

  // take the keys from the matched column allocated using mr
  std::unique_ptr<column> keys_column(std::move(input_matched->release().children.back()));

  // create column with keys_column and indices_column
  return make_dictionary_column(std::move(keys_column),
                                std::move(indices_column),
                                std::move(*(contents.null_mask.release())),
                                null_count);
}

}  // namespace detail
}  // namespace dictionary
}  // namespace cudf
