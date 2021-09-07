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

// SHA supported leaf data type check
bool sha_type_check(data_type dt)
{
  return !is_chrono(dt) && (is_fixed_width(dt) || (dt.id() == type_id::STRING));
}

}  // namespace

namespace detail {

std::unique_ptr<column> sha1_hash(table_view const& input,
                                  cudaStream_t stream,
                                  rmm::mr::device_memory_resource* mr)
{
  if (input.num_columns() == 0 || input.num_rows() == 0) {
    // Return the SHA-1 hash of a zero-length input.
    const string_scalar string_160bit("da39a3ee5e6b4b0d3255bfef95601890afd80709");
    auto output = make_column_from_scalar(string_160bit, input.num_rows(), stream, mr);
    return output;
  }

  CUDF_EXPECTS(
    std::all_of(input.begin(), input.end(), [](auto col) { return sha_type_check(col.type()); }),
    "SHA-1 unsupported column type");

  // Result column allocation and creation
  auto begin = thrust::make_constant_iterator(40);
  auto offsets_column =
    cudf::strings::detail::make_offsets_child_column(begin, begin + input.num_rows(), stream, mr);

  auto chars_column = strings::detail::create_chars_child_column(input.num_rows() * 40, stream, mr);
  auto chars_view   = chars_column->mutable_view();
  auto d_chars      = chars_view.data<char>();

  rmm::device_buffer null_mask{0, stream, mr};

  auto const device_input = table_device_view::create(input, stream);

  // Hash each row, hashing each element sequentially left to right
  thrust::for_each(rmm::exec_policy(stream),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(input.num_rows()),
                   [d_chars, device_input = *device_input] __device__(auto row_index) {
                     sha1_intermediate_data hash_state;
                     SHA1Hash hasher = SHA1Hash{};
                     for (int col_index = 0; col_index < device_input.num_columns(); col_index++) {
                       if (device_input.column(col_index).is_valid(row_index)) {
                         cudf::type_dispatcher(device_input.column(col_index).type(),
                                               hasher,
                                               device_input.column(col_index),
                                               row_index,
                                               &hash_state);
                       }
                     }
                     hasher.finalize(&hash_state, d_chars + (row_index * 40));
                   });

  return make_strings_column(input.num_rows(),
                             std::move(offsets_column),
                             std::move(chars_column),
                             0,
                             std::move(null_mask),
                             stream,
                             mr);
}

std::unique_ptr<column> sha256_hash(table_view const& input,
                                    bool truncate_output,
                                    cudaStream_t stream,
                                    rmm::mr::device_memory_resource* mr)
{
  return nullptr;
}

std::unique_ptr<column> sha512_hash(table_view const& input,
                                    bool truncate_output,
                                    cudaStream_t stream,
                                    rmm::mr::device_memory_resource* mr)
{
  return nullptr;
}

}  // namespace detail
}  // namespace cudf
