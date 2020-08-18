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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/tabulate.h>

#include <algorithm>

namespace cudf {
namespace detail {
namespace {

template <typename IterType>
std::vector<column_view> to_leaf_columns(IterType iter_begin, IterType iter_end)
{
  std::vector<column_view> leaf_columns;
  std::for_each(iter_begin, iter_end, [&leaf_columns](column_view const& col) {
    if (is_nested(col.type())) {
      CUDF_EXPECTS(col.type().id() == type_id::STRUCT, "unsupported nested type");
      auto child_columns = to_leaf_columns(col.child_begin(), col.child_end());
      leaf_columns.insert(leaf_columns.end(), child_columns.begin(), child_columns.end());
    } else {
      leaf_columns.emplace_back(col);
    }
  });
  return leaf_columns;
}

}  // namespace

template <template <typename> class hash_function>
std::unique_ptr<column> serial_murmur_hash3_32(table_view const& input,
                                               uint32_t seed,
                                               rmm::cuda_stream_view stream,
                                               rmm::mr::device_memory_resource* mr)
{
  auto output = make_numeric_column(
    data_type(type_id::INT32), input.num_rows(), mask_state::UNALLOCATED, stream, mr);

  if (input.num_columns() == 0 || input.num_rows() == 0) { return output; }

  table_view const leaf_table(to_leaf_columns(input.begin(), input.end()));
  auto const device_input = table_device_view::create(leaf_table, stream);
  auto output_view        = output->mutable_view();

  if (has_nulls(leaf_table)) {
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

std::unique_ptr<column> hash(table_view const& input,
                             hash_id hash_function,
                             cudf::host_span<uint32_t const> initial_hash,
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
    case (hash_id::HASH_SHA1): return sha1(input, stream, mr);
    case (hash_id::HASH_SHA224): return sha256_base(input, true, stream, mr);
    case (hash_id::HASH_SHA256): return sha256_base(input, false, stream, mr);
    case (hash_id::HASH_SHA384): return sha512_base(input, true, stream, mr);
    case (hash_id::HASH_SHA512): return sha512_base(input, false, stream, mr);

    default: return nullptr;
  }
}

<<<<<<< HEAD
=======
std::unique_ptr<column> md5_hash(table_view const& input,
                                 rmm::mr::device_memory_resource* mr,
                                 cudaStream_t stream)
{
  if (input.num_columns() == 0 || input.num_rows() == 0) {
    const string_scalar string_128bit("d41d8cd98f00b204e9orig98ecf8427e");
    auto output = make_column_from_scalar(string_128bit, input.num_rows(), mr, stream);
    return output;
  }

  CUDF_EXPECTS(
    std::all_of(input.begin(),
                input.end(),
                [](auto col) {
                  return !is_chrono(col.type()) &&
                         (is_fixed_width(col.type()) || (col.type().id() == type_id::STRING));
                }),
    "MD5 unsupported column type");

  // Result column allocation and creation
  auto begin = thrust::make_constant_iterator(32);
  auto offsets_column =
    cudf::strings::detail::make_offsets_child_column(begin, begin + input.num_rows(), mr, stream);
  auto offsets_view  = offsets_column->view();
  auto d_new_offsets = offsets_view.data<int32_t>();

  auto chars_column = strings::detail::create_chars_child_column(
    input.num_rows(), 0, input.num_rows() * 32, mr, stream);
  auto chars_view = chars_column->mutable_view();
  auto d_chars    = chars_view.data<char>();

  rmm::device_buffer null_mask{0, stream, mr};

  bool const nullable     = has_nulls(input);
  auto const device_input = table_device_view::create(input, stream);

  // Hash each row, hashing each element sequentially left to right
  thrust::for_each(
    rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(input.num_rows()),
    [d_chars, device_input = *device_input, has_nulls = nullable] __device__(auto row_index) {
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

std::unique_ptr<column> murmur_hash3_32(table_view const& input,
                                        std::vector<uint32_t> const& initial_hash,
                                        rmm::mr::device_memory_resource* mr,
                                        cudaStream_t stream)
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
      thrust::tabulate(rmm::exec_policy(stream)->on(stream),
                       output_view.begin<int32_t>(),
                       output_view.end<int32_t>(),
                       row_hasher_initial_values<MurmurHash3_32, true>(
                         *device_input, device_initial_hash.data().get()));
    } else {
      thrust::tabulate(rmm::exec_policy(stream)->on(stream),
                       output_view.begin<int32_t>(),
                       output_view.end<int32_t>(),
                       row_hasher_initial_values<MurmurHash3_32, false>(
                         *device_input, device_initial_hash.data().get()));
    }
  } else {
    if (nullable) {
      thrust::tabulate(rmm::exec_policy(stream)->on(stream),
                       output_view.begin<int32_t>(),
                       output_view.end<int32_t>(),
                       row_hasher<MurmurHash3_32, true>(*device_input));
    } else {
      thrust::tabulate(rmm::exec_policy(stream)->on(stream),
                       output_view.begin<int32_t>(),
                       output_view.end<int32_t>(),
                       row_hasher<MurmurHash3_32, false>(*device_input));
    }
  }

  return output;
}

std::unique_ptr<column> sha1(table_view const& input,
                             rmm::mr::device_memory_resource* mr,
                             cudaStream_t stream) {
}

std::unique_ptr<column> sha256_base(table_view const& input,
                                    bool truncate_output,
                                    rmm::mr::device_memory_resource* mr,
                                    cudaStream_t stream) {
}

std::unique_ptr<column> sha512_base(table_view const& input,
                                    bool truncate_output,
                                    rmm::mr::device_memory_resource* mr,
                                    cudaStream_t stream) {
}

>>>>>>> 7e5dec8a8d... initial sha structure
}  // namespace detail

std::unique_ptr<column> hash(table_view const& input,
                             hash_id hash_function,
                             cudf::host_span<uint32_t const> initial_hash,
                             uint32_t seed,
                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::hash(input, hash_function, initial_hash, seed, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
