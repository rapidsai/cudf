/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <io/utilities/parsing_utils.cuh>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <memory>

namespace cudf::io::json::experimental {

template <typename str_tuple_it, typename B>
std::unique_ptr<column> parse_data(str_tuple_it str_tuples,
                                   size_type col_size,
                                   data_type col_type,
                                   B&& null_mask,
                                   cudf::io::parse_options_view const& options,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  if (col_type == cudf::data_type{cudf::type_id::STRING}) {
    rmm::device_uvector<size_type> offsets(col_size + 1, stream);
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       col_size,
                       [str_tuples,
                        sizes     = device_span<size_type>{offsets},
                        null_mask = static_cast<bitmask_type*>(null_mask.data()),
                        options] __device__(size_type row) {
                         if (not bit_is_set(null_mask, row)) {
                           sizes[row] = 0;
                           return;
                         }
                         auto const in = str_tuples[row];

                         auto const is_null_literal = serialized_trie_contains(
                           options.trie_na, {in.first, static_cast<size_t>(in.second)});
                         if (is_null_literal) {
                           sizes[row] = 0;
                           clear_bit(null_mask, row);
                           return;
                         }

                         sizes[row] = in.second;
                       });

    thrust::exclusive_scan(
      rmm::exec_policy(stream), offsets.begin(), offsets.end(), offsets.begin());

    rmm::device_uvector<char> chars(offsets.back_element(stream), stream);
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       col_size,
                       [str_tuples,
                        chars     = device_span<char>{chars},
                        offsets   = device_span<size_type>{offsets},
                        null_mask = static_cast<bitmask_type*>(null_mask.data()),
                        options] __device__(size_type row) {
                         if (not bit_is_set(null_mask, row)) { return; }
                         auto const in = str_tuples[row];
                         for (int i = 0; i < in.second; ++i) {
                           chars[offsets[row] + i] = *(in.first + i);
                         }
                       });

    return make_strings_column(
      col_size, std::move(offsets), std::move(chars), std::move(null_mask));
  }

  auto out_col = make_fixed_width_column(
    col_type, col_size, std::move(null_mask), cudf::UNKNOWN_NULL_COUNT, stream, mr);
  auto output_dv_ptr = mutable_column_device_view::create(*out_col, stream);

  // use existing code (`ConvertFunctor`) to convert values
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    col_size,
    [str_tuples, col = *output_dv_ptr, options, col_type] __device__(size_type row) {
      if (col.is_null(row)) { return; }
      auto const in = str_tuples[row];

      auto const is_null_literal =
        serialized_trie_contains(options.trie_na, {in.first, static_cast<size_t>(in.second)});

      if (is_null_literal) {
        col.set_null(row);
        return;
      }

      auto const is_parsed = cudf::type_dispatcher(col_type,
                                                   ConvertFunctor{},
                                                   in.first,
                                                   in.first + in.second,
                                                   col.data<char>(),
                                                   row,
                                                   col_type,
                                                   options,
                                                   false);
      if (not is_parsed) { col.set_null(row); }
    });

  return out_col;
}

}  // namespace cudf::io::json::experimental
