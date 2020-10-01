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

#include <cudf/detail/concatenate.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/thrust_rmm_allocator.h>

namespace cudf {
namespace dictionary {
namespace detail {
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
 *
 */
std::unique_ptr<column> add_keys(
  dictionary_column_view const& dictionary_column,
  column_view const& new_keys,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0)
{
  CUDF_EXPECTS(!new_keys.has_nulls(), "Keys must not have nulls");
  auto old_keys = dictionary_column.keys();  // [a,b,c,d,f]
  CUDF_EXPECTS(new_keys.type() == old_keys.type(), "Keys must be the same type");
  // first, concatenate the keys together
  // [a,b,c,d,f] + [d,b,e] = [a,b,c,d,f,d,b,e]
  auto combined_keys = cudf::detail::concatenate(
    std::vector<column_view>{old_keys, new_keys}, rmm::mr::get_current_device_resource(), stream);
  // sort and remove any duplicates from the combined keys
  // drop_duplicates([a,b,c,d,f,d,b,e]) = [a,b,c,d,e,f]
  auto table_keys = cudf::detail::drop_duplicates(table_view{{*combined_keys}},
                                                  std::vector<size_type>{0},  // only one key column
                                                  duplicate_keep_option::KEEP_FIRST,
                                                  null_equality::EQUAL,
                                                  mr,
                                                  stream)
                      ->release();
  std::unique_ptr<column> keys_column(std::move(table_keys.front()));
  // create a map for the indices
  // lower_bound([a,b,c,d,e,f],[a,b,c,d,f]) = [0,1,2,3,5]
  auto map_indices = cudf::detail::lower_bound(
    table_view{{keys_column->view()}},
    table_view{{old_keys}},
    std::vector<order>{order::ASCENDING},
    std::vector<null_order>{null_order::AFTER},  // should be no nulls here
    mr,
    stream);
  // now create the indices column -- map old values to the new ones
  // gather([4,0,3,1,2,2,2,4,0],[0,1,2,3,5]) = [5,0,3,1,2,2,2,5,0]
  column_view indices_view(dictionary_column.indices().type(),
                           dictionary_column.size(),
                           dictionary_column.indices().head(),
                           nullptr,
                           0,
                           dictionary_column.offset());
  // the result may contain nulls if the input contains nulls
  // and the corresponding index is therefore invalid/undefined
  auto table_indices = cudf::detail::gather(table_view{{map_indices->view()}},
                                            indices_view,
                                            cudf::detail::out_of_bounds_policy::IGNORE,
                                            cudf::detail::negative_index_policy::NOT_ALLOWED,
                                            mr,
                                            stream)
                         ->release();
  // The output of lower_bound is INT32 but we need to convert to unsigned indices.
  auto const indices_type = get_indices_type_for_size(keys_column->size());
  auto indices_column     = [&] {
    column_view gather_result = table_indices.front()->view();
    auto const indices_size   = gather_result.size();
    // we can just use the lower-bound/gather data directly for UINT32 case
    if (indices_type.id() == type_id::UINT32) {
      auto contents = table_indices.front()->release();
      return std::make_unique<column>(data_type{type_id::UINT32},
                                      indices_size,
                                      std::move(*(contents.data.release())),
                                      rmm::device_buffer{0, stream, mr},
                                      0);
    }
    // otherwise we need to convert the gather result
    column_view cast_view(gather_result.type(), indices_size, gather_result.head(), nullptr, 0);
    return cudf::detail::cast(cast_view, indices_type, mr, stream);
  }();

  // create new dictionary column with keys_column and indices_column
  // null mask has not changed
  return make_dictionary_column(std::move(keys_column),
                                std::move(indices_column),
                                copy_bitmask(dictionary_column.parent(), stream, mr),
                                dictionary_column.null_count());
}

}  // namespace detail

std::unique_ptr<column> add_keys(dictionary_column_view const& dictionary_column,
                                 column_view const& keys,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::add_keys(dictionary_column, keys, mr);
}

}  // namespace dictionary
}  // namespace cudf
