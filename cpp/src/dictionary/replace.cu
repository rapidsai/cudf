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
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/detail/replace.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>

namespace cudf {
namespace dictionary {
namespace detail {
namespace {

template <typename T, bool has_nulls>
std::unique_ptr<column> replace_indices(column_device_view const& input,
                                        column_device_view const& replacement,
                                        rmm::mr::device_memory_resource* mr,
                                        cudaStream_t stream)
{
  size_type const size = input.size();
  auto predicate       = [input] __device__(auto i) { return input.is_valid(i); };

  auto input_pair_iterator = cudf::detail::make_pair_iterator<T, has_nulls>(input);
  if (replacement.nullable()) {
    auto replacement_pair_iterator = cudf::detail::make_pair_iterator<T, true>(replacement);
    return cudf::detail::copy_if_else(true,
                                      input_pair_iterator,
                                      input_pair_iterator + size,
                                      replacement_pair_iterator,
                                      predicate,
                                      mr,
                                      stream);
  } else {
    auto replacement_pair_iterator = cudf::detail::make_pair_iterator<T, false>(replacement);
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

  // now build the new indices by doing a scatter on just the matched indices
  auto const input_indices = column_device_view::create(input_view.get_indices_annotated(), stream);
  auto const repl_indices  = column_device_view::create(repl_view.get_indices_annotated(), stream);

  // call copy_if_else using bitmask of input as predicate
  auto new_indices =
    input_view.has_nulls()
      ? replace_indices<uint32_t, true>(*input_indices, *repl_indices, mr, stream)
      : replace_indices<uint32_t, false>(*input_indices, *repl_indices, mr, stream);

  // record some data before calling release()
  auto const indices_type = new_indices->type();
  auto const output_size  = new_indices->size();
  auto const null_count   = new_indices->null_count();
  auto contents           = new_indices->release();
  auto indices_column     = std::make_unique<column>(indices_type,
                                                 static_cast<size_type>(output_size),
                                                 std::move(*(contents.data.release())),
                                                 rmm::device_buffer{0, stream, mr},
                                                 0);

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
