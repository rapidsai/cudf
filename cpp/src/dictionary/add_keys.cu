/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

namespace cudf {
namespace dictionary {
namespace detail {
namespace {
struct negate_fn {
  __device__ bool operator()(size_type idx) const noexcept { return !b[idx]; }
  bool const* b;
};
}  // namespace

/**
 * @brief Create a new dictionary column by adding the new keys elements
 * to the existing dictionary_column.
 *
 * ```
 * Example:
 * d1 = {[a, b, c, d, f], {4, 0, 3, 1, 2, 2, 2, 4, 0}}
 * d2 = add_keys( d1, [d, b, e] )
 * d2 is now {[a, b, c, d, e, f], [5, 0, 3, 1, 2, 2, 2, 5, 0]}
 * ```
 */
std::unique_ptr<column> add_keys(dictionary_column_view const& input,
                                 column_view const& new_keys,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!new_keys.has_nulls(), "Keys must not have nulls", std::invalid_argument);
  auto old_keys = input.keys();
  CUDF_EXPECTS(
    cudf::have_same_types(new_keys, old_keys), "Keys must be the same type", cudf::data_type_error);

  // first, find duplicate keys
  auto c = cudf::detail::contains(old_keys, new_keys, stream, mr);
  auto b = column_device_view::create(c->view(), stream);
  // filter out duplicates from the input
  auto nks = std::move(
    cudf::detail::copy_if(cudf::table_view({new_keys}), negate_fn{b->data<bool>()}, stream, mr)
      ->release()
      .front());
  // build new keys by concatenating new ones to the end
  auto keys_column =
    (nks->size() > 0)
      ? cudf::detail::concatenate(std::vector<column_view>{old_keys, nks->view()}, stream, mr)
      : std::make_unique<column>(old_keys, stream, mr);
  // this leaves the indices untouched so just copy them
  auto indices_column = std::make_unique<column>(input.get_indices_annotated(), stream, mr);
  indices_column->set_null_mask(rmm::device_buffer{}, 0);
  return make_dictionary_column(std::move(keys_column),
                                std::move(indices_column),
                                cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                input.null_count());
}

}  // namespace detail

std::unique_ptr<column> add_keys(dictionary_column_view const& dictionary_column,
                                 column_view const& keys,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::add_keys(dictionary_column, keys, stream, mr);
}

}  // namespace dictionary
}  // namespace cudf
