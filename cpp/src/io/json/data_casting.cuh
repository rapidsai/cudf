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

template <typename str_tuple_it>
rmm::device_uvector<thrust::pair<const char*, size_type>> coalesce_input(
  str_tuple_it str_tuples, size_type col_size, rmm::cuda_stream_view stream)
{
  auto result = rmm::device_uvector<thrust::pair<const char*, size_type>>(col_size, stream);
  thrust::copy_n(rmm::exec_policy(stream), str_tuples, col_size, result.begin());
  return result;
}

template <typename str_tuple_it, typename B>
std::unique_ptr<column> parse_data(str_tuple_it str_tuples,
                                   size_type col_size,
                                   data_type col_type,
                                   B&& null_mask,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  auto parse_opts = parse_options{',', '\n', '\"', '.'};

  parse_opts.trie_true  = cudf::detail::create_serialized_trie({"true"}, stream);
  parse_opts.trie_false = cudf::detail::create_serialized_trie({"false"}, stream);
  parse_opts.trie_na    = cudf::detail::create_serialized_trie({"", "null"}, stream);

  if (col_type == cudf::data_type{cudf::type_id::STRING}) {
    auto const strings_span = coalesce_input(str_tuples, col_size, stream);
    return make_strings_column(strings_span, stream);
  }

  auto out_col = make_fixed_width_column(
    col_type, col_size, std::move(null_mask), cudf::UNKNOWN_NULL_COUNT, stream, mr);
  auto output_dv_ptr = mutable_column_device_view::create(*out_col, stream);

  // use existing code (`ConvertFunctor`) to convert values
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    col_size,
    [str_tuples, col = *output_dv_ptr, opts = parse_opts.view(), col_type] __device__(
      size_type row_idx) {
      auto const in = str_tuples[row_idx];

      auto const is_null_literal =
        serialized_trie_contains(opts.trie_na, {in.first, static_cast<size_t>(in.second)});

      if (is_null_literal) {
        col.set_null(row_idx);
        return;
      }

      auto const is_parsed = cudf::type_dispatcher(col_type,
                                                   ConvertFunctor{},
                                                   in.first,
                                                   in.first + in.second,
                                                   col.data<char>(),
                                                   row_idx,
                                                   col_type,
                                                   opts,
                                                   false);
      if (not is_parsed) { col.set_null(row_idx); }
    });

  return out_col;
}

}  // namespace cudf::io::json::experimental
