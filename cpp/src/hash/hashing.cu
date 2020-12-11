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
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/hashing.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/scatter.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/partitioning.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace {

// MD5 supported leaf data type check
bool md5_type_check(data_type dt)
{
  return !is_chrono(dt) && (is_fixed_width(dt) || (dt.id() == type_id::STRING));
}

}  // namespace

namespace detail {

std::unique_ptr<column> hash(table_view const& input,
                             hash_id hash_function,
                             std::vector<uint32_t> const& initial_hash,
                             uint32_t seed,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  switch (hash_function) {
    case (hash_id::HASH_MURMUR3): return murmur_hash3_32(input, initial_hash, stream, mr);
    case (hash_id::HASH_MD5): return md5_hash(input, stream, mr);
    case (hash_id::HASH_SERIAL_MURMUR3):
      return serial_murmur_hash3_32<MurmurHash3_32>(input, seed, stream, mr);
    case (hash_id::HASH_SPARK_MURMUR3):
      return serial_murmur_hash3_32<SparkMurmurHash3_32>(input, seed, stream, mr);
    default: return nullptr;
  }
}

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

  auto chars_column = strings::detail::create_chars_child_column(
    input.num_rows(), 0, input.num_rows() * 32, stream, mr);
  auto chars_view = chars_column->mutable_view();
  auto d_chars    = chars_view.data<char>();

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
                         cudf::type_dispatcher(device_input.column(col_index).type(),
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

template <template <typename> class hash_function>
std::unique_ptr<column> serial_murmur_hash3_32(table_view const& input,
                                               uint32_t seed,
                                               rmm::cuda_stream_view stream,
                                               rmm::mr::device_memory_resource* mr)
{
  auto output = make_numeric_column(
    data_type(type_id::INT32), input.num_rows(), mask_state::UNALLOCATED, stream, mr);

  if (input.num_columns() == 0 || input.num_rows() == 0) { return output; }

  auto const device_input = table_device_view::create(input, stream);
  auto output_view        = output->mutable_view();

  if (has_nulls(input)) {
    thrust::tabulate(rmm::exec_policy(stream),
                     output_view.begin<int32_t>(),
                     output_view.end<int32_t>(),
                     [device_input = *device_input, seed] __device__(auto row_index) {
                       return thrust::reduce(
                         thrust::seq,
                         device_input.begin(),
                         device_input.end(),
                         seed,
                         [rindex = row_index] __device__(auto hash, auto column) {
                           return cudf::type_dispatcher(
                             column.type(),
                             element_hasher_with_seed<hash_function, true>{hash, hash},
                             column,
                             rindex);
                         });
                     });
  } else {
    thrust::tabulate(rmm::exec_policy(stream),
                     output_view.begin<int32_t>(),
                     output_view.end<int32_t>(),
                     [device_input = *device_input, seed] __device__(auto row_index) {
                       return thrust::reduce(
                         thrust::seq,
                         device_input.begin(),
                         device_input.end(),
                         seed,
                         [rindex = row_index] __device__(auto hash, auto column) {
                           return cudf::type_dispatcher(
                             column.type(),
                             element_hasher_with_seed<hash_function, false>{hash, hash},
                             column,
                             rindex);
                         });
                     });
  }

  return output;
}

std::unique_ptr<column> murmur_hash3_32(table_view const& input,
                                        std::vector<uint32_t> const& initial_hash,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  // TODO this should be UINT32
  auto output = make_numeric_column(
    data_type(type_id::INT32), input.num_rows(), mask_state::UNALLOCATED, stream, mr);

  // Return early if there's nothing to hash
  if (input.num_columns() == 0 || input.num_rows() == 0) { return output; }

  bool const nullable     = has_nulls(input);
  auto const device_input = table_device_view::create(input, stream);
  auto output_view        = output->mutable_view();

  // Compute the hash value for each row depending on the specified hash function
  if (!initial_hash.empty()) {
    CUDF_EXPECTS(initial_hash.size() == size_t(input.num_columns()),
                 "Expected same size of initial hash values as number of columns");
    auto device_initial_hash = rmm::device_vector<uint32_t>(initial_hash);

    if (nullable) {
      thrust::tabulate(rmm::exec_policy(stream),
                       output_view.begin<int32_t>(),
                       output_view.end<int32_t>(),
                       row_hasher_initial_values<MurmurHash3_32, true>(
                         *device_input, device_initial_hash.data().get()));
    } else {
      thrust::tabulate(rmm::exec_policy(stream),
                       output_view.begin<int32_t>(),
                       output_view.end<int32_t>(),
                       row_hasher_initial_values<MurmurHash3_32, false>(
                         *device_input, device_initial_hash.data().get()));
    }
  } else {
    if (nullable) {
      thrust::tabulate(rmm::exec_policy(stream),
                       output_view.begin<int32_t>(),
                       output_view.end<int32_t>(),
                       row_hasher<MurmurHash3_32, true>(*device_input));
    } else {
      thrust::tabulate(rmm::exec_policy(stream),
                       output_view.begin<int32_t>(),
                       output_view.end<int32_t>(),
                       row_hasher<MurmurHash3_32, false>(*device_input));
    }
  }

  return output;
}

}  // namespace detail

std::unique_ptr<column> hash(table_view const& input,
                             hash_id hash_function,
                             std::vector<uint32_t> const& initial_hash,
                             uint32_t seed,
                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::hash(input, hash_function, initial_hash, seed, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> murmur_hash3_32(table_view const& input,
                                        std::vector<uint32_t> const& initial_hash,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::murmur_hash3_32(input, initial_hash, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
