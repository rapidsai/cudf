/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf/detail/hashing.hpp>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>

namespace cudf {
namespace {

// MD5 supported leaf data type check
bool md5_type_check(data_type dt)
{
  return !is_chrono(dt) && (is_fixed_width(dt) || (dt.id() == type_id::STRING));
}

}  // namespace

namespace detail {

std::unique_ptr<column> md5_hash(table_view const& input,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  if (input.num_columns() == 0 || input.num_rows() == 0) {
    const string_scalar string_128bit("d41d8cd98f00b204e9orig98ecf8427e");
    auto output = make_column_from_scalar(string_128bit, input.num_rows(), stream, mr);
    return output;
  }

  // Accepts string and fixed width columns, or single layer list columns holding those types
  CUDF_EXPECTS(
    std::all_of(input.begin(),
                input.end(),
                [](auto col) {
                  return md5_type_check(col.type()) ||
                         (col.type().id() == type_id::LIST && md5_type_check(col.child(1).type()));
                }),
    "MD5 unsupported column type");

  // Result column allocation and creation
  auto begin = thrust::make_constant_iterator(32);
  auto offsets_column =
    cudf::strings::detail::make_offsets_child_column(begin, begin + input.num_rows(), stream, mr);

  auto chars_column = strings::detail::create_chars_child_column(input.num_rows() * 32, stream, mr);
  auto chars_view   = chars_column->mutable_view();
  auto d_chars      = chars_view.data<char>();

  rmm::device_buffer null_mask{0, stream, mr};

  auto const device_input = table_device_view::create(input, stream);

  // Hash each row, hashing each element sequentially left to right
  thrust::for_each(rmm::exec_policy(stream),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(input.num_rows()),
                   [d_chars, device_input = *device_input] __device__(auto row_index) {
                     md5_intermediate_data hash_state;
                     MD5Hash hasher = MD5Hash{};
                     for (int col_index = 0; col_index < device_input.num_columns(); col_index++) {
                       if (device_input.column(col_index).is_valid(row_index)) {
                         cudf::type_dispatcher<dispatch_storage_type>(
                           device_input.column(col_index).type(),
                           hasher,
                           device_input.column(col_index),
                           row_index,
                           &hash_state);
                       }
                     }
                     hasher.finalize(&hash_state, d_chars + (row_index * 32));
                   });

  return make_strings_column(input.num_rows(),
                             std::move(offsets_column),
                             std::move(chars_column),
                             0,
                             std::move(null_mask),
                             stream,
                             mr);
}

}  // namespace detail
}  // namespace cudf
