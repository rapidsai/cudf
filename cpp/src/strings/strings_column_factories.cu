/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>

namespace cudf {

namespace strings::detail {
std::vector<std::unique_ptr<column>> make_strings_column_batch(
  std::vector<cudf::device_span<thrust::pair<char const*, size_type> const>> strings_batch,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<column>> output;
  std::vector<std::unique_ptr<column>>> offset_columns;
  std::vector<size_type> total_bytes;
  std::vector<size_type> strings_sizes;
  std::vector<thrust::transform_iterator<size_type> offsets_transformer_itr;
  std::vector<int64_t> chars_sizes;
  std::vector<rmm::device_buffer> null_masks;
  std::vector<size_type> null_counts;

  std::transform(
    strings_batch.begin(),
    strings_batch.end(),
    std::back_inserter(strings_sizes),
    [] (auto &strings) {
      return thrust::distance(strings.begin(), strings.end());
    }
  );

  std::transform(
    strings_batch.begin(),
    strings_batch.end(),
    std::back_inserter(offsets_transformer_itr),
    [stream, mr] (auto &strings) {
      size_type strings_count = thrust::distance(strings.begin(), strings.end());
      auto offsets_transformer =
        cuda::proclaim_return_type<size_type>([] __device__(string_index_pair item) -> size_type {
          return (item.first != nullptr ? static_cast<size_type>(item.second) : size_type{0});
        });
      return thrust::make_transform_iterator(strings.begin(), offsets_transformer);
    }
  );

  [offset_columns, total_bytes] = cudf::strings::detail::make_offsets_child_column_batch(
    offsets_transformer_itr, strings_sizes, stream, mr);

  

  // create null mask
  auto validator = [] __device__(string_index_pair const item) { return item.first != nullptr; };
  [] = cudf::detail::valid_if_n_kernel(strings_batch, sizes, validator, stream, mr);
  auto const null_count = new_nulls.second;
  auto null_mask =
    (null_count > 0) ? std::move(new_nulls.first) : rmm::device_buffer{0, stream, mr};


  // build chars column
  std::transform(
    thrust::make_zip_iterator(thrust::make_tuple(offset_columns.begin(), total_bytes.begin(), strings_sizes.begin(), strings_batch.begin(), nu))

    std::back_inserter(output),
    [] (auto &elem) {
      auto strings_count = thrust::get<2>(elem)
      auto bytes = thrust::get<1>(elem)
      auto null_count = thrust::get<4>(elem)
      auto begin = thrust::get<3>(elem)
      auto d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(thrust::get<0>(elem)->view());
      auto chars_data = [d_offsets, bytes = bytes, begin, strings_count, null_count, stream, mr] {
      auto const avg_bytes_per_row = bytes / std::max(strings_count - null_count, 1);
      // use a character-parallel kernel for long string lengths
      if (avg_bytes_per_row > FACTORY_BYTES_PER_ROW_THRESHOLD) {
        auto const str_begin = thrust::make_transform_iterator(
          begin, cuda::proclaim_return_type<string_view>([] __device__(auto ip) {
            return string_view{ip.first, ip.second};
          }));

        return gather_chars(str_begin,
                            thrust::make_counting_iterator<size_type>(0),
                            thrust::make_counting_iterator<size_type>(strings_count),
                            d_offsets,
                            bytes,
                            stream,
                            mr);
      } else {
        // this approach is 2-3x faster for a large number of smaller string lengths
        auto chars_data = rmm::device_uvector<char>(bytes, stream, mr);
        auto d_chars    = chars_data.data();
        auto copy_chars = [d_chars] __device__(auto item) {
          string_index_pair const str = thrust::get<0>(item);
          int64_t const offset        = thrust::get<1>(item);
          if (str.first != nullptr) memcpy(d_chars + offset, str.first, str.second);
        };
        thrust::for_each_n(rmm::exec_policy(stream),
                          thrust::make_zip_iterator(thrust::make_tuple(begin, d_offsets)),
                          strings_count,
                          copy_chars);
        return chars_data;
      }
    }();

    return make_strings_column(strings_count,
                              std::move(offsets_column),
                              chars_data.release(),
                              null_count,
                              std::move(null_mask));
    }
  )

  return output;
}

}  // namespace strings::detail

std::vector<std::unique_ptr<column>> make_strings_column_batch(
  std::vector<cudf::device_span<thrust::pair<char const*, size_type> const>> strings_batch,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return cudf::strings::detail::make_strings_column_batch(strings_batch, stream, mr);
}

// Create a strings-type column from vector of pointer/size pairs
std::unique_ptr<column> make_strings_column(
  device_span<thrust::pair<char const*, size_type> const> strings,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  return cudf::strings::detail::make_strings_column(strings.begin(), strings.end(), stream, mr);
}

namespace {
struct string_view_to_pair {
  string_view null_placeholder;
  string_view_to_pair(string_view n) : null_placeholder(n) {}
  __device__ thrust::pair<char const*, size_type> operator()(string_view const& i)
  {
    return (i.data() == null_placeholder.data())
             ? thrust::pair<char const*, size_type>{nullptr, 0}
             : thrust::pair<char const*, size_type>{i.data(), i.size_bytes()};
  }
};

}  // namespace

std::unique_ptr<column> make_strings_column(device_span<string_view const> string_views,
                                            string_view null_placeholder,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto it_pair =
    thrust::make_transform_iterator(string_views.begin(), string_view_to_pair{null_placeholder});
  return cudf::strings::detail::make_strings_column(
    it_pair, it_pair + string_views.size(), stream, mr);
}

std::unique_ptr<column> make_strings_column(size_type num_strings,
                                            std::unique_ptr<column> offsets_column,
                                            rmm::device_buffer&& chars_buffer,
                                            size_type null_count,
                                            rmm::device_buffer&& null_mask)
{
  CUDF_FUNC_RANGE();

  if (null_count > 0) { CUDF_EXPECTS(null_mask.size() > 0, "Column with nulls must be nullable."); }
  CUDF_EXPECTS(num_strings == offsets_column->size() - 1,
               "Invalid offsets column size for strings column.");
  CUDF_EXPECTS(offsets_column->null_count() == 0, "Offsets column should not contain nulls");

  std::vector<std::unique_ptr<column>> children;
  children.emplace_back(std::move(offsets_column));

  return std::make_unique<column>(data_type{type_id::STRING},
                                  num_strings,
                                  std::move(chars_buffer),
                                  std::move(null_mask),
                                  null_count,
                                  std::move(children));
}

std::unique_ptr<column> make_strings_column(size_type num_strings,
                                            rmm::device_uvector<size_type>&& offsets,
                                            rmm::device_uvector<char>&& chars,
                                            rmm::device_buffer&& null_mask,
                                            size_type null_count)
{
  CUDF_FUNC_RANGE();

  if (num_strings == 0) { return make_empty_column(type_id::STRING); }

  auto const offsets_size = static_cast<size_type>(offsets.size());

  if (null_count > 0) CUDF_EXPECTS(null_mask.size() > 0, "Column with nulls must be nullable.");

  CUDF_EXPECTS(num_strings == offsets_size - 1, "Invalid offsets column size for strings column.");

  auto offsets_column = std::make_unique<column>(  //
    data_type{type_id::INT32},
    offsets_size,
    offsets.release(),
    rmm::device_buffer(),
    0);

  auto children = std::vector<std::unique_ptr<column>>();

  children.emplace_back(std::move(offsets_column));

  return std::make_unique<column>(data_type{type_id::STRING},
                                  num_strings,
                                  chars.release(),
                                  std::move(null_mask),
                                  null_count,
                                  std::move(children));
}

}  // namespace cudf
