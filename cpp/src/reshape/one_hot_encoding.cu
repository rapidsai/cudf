/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/cmath>

namespace cudf {
namespace detail {

namespace {

template <typename InputType>
struct one_hot_encoding_functor {
  one_hot_encoding_functor(column_device_view input, column_device_view category)
    : d_input(input), d_category(category)
  {
  }

  bool __device__ operator()(size_type i)
  {
    auto element_index  = i % d_input.size();
    auto category_index = i / d_input.size();

    if (d_category.is_valid(category_index) and d_input.is_valid(element_index)) {
      if constexpr (is_floating_point<InputType>()) {
        if (cuda::std::isnan(d_category.element<InputType>(category_index)) and
            cuda::std::isnan(d_input.element<InputType>(element_index)))
          return true;
      }
      return d_category.element<InputType>(category_index) ==
             d_input.element<InputType>(element_index);
    }
    return !d_category.is_valid(category_index) and !d_input.is_valid(element_index);
  }

 private:
  column_device_view d_input, d_category;
};

}  // anonymous namespace

struct one_hot_encoding_launcher {
  template <typename InputType, CUDF_ENABLE_IF(is_numeric<InputType>())>
  std::pair<std::unique_ptr<column>, table_view> operator()(column_view const& input_column,
                                                            column_view const& categories,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::mr::device_memory_resource* mr)
  {
    auto total_size    = input_column.size() * categories.size();
    auto all_encodings = make_numeric_column(
      data_type{type_id::BOOL8}, total_size, mask_state::UNALLOCATED, stream, mr);

    auto d_input_column    = column_device_view::create(input_column, stream);
    auto d_category_column = column_device_view::create(categories, stream);

    one_hot_encoding_functor<InputType> one_hot_encoding_compute_f(*d_input_column,
                                                                   *d_category_column);

    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(total_size),
                      all_encodings->mutable_view().begin<bool>(),
                      one_hot_encoding_compute_f);

    auto split_iter = make_counting_transform_iterator(
      1, [&input_column](auto i) { return i * input_column.size(); });
    std::vector<size_type> split_indices(split_iter, split_iter + categories.size() - 1);

    // TODO: use detail interface, gh9226
    auto views = cudf::split(all_encodings->view(), split_indices);
    table_view encodings_view{views};

    return std::make_pair(std::move(all_encodings), encodings_view);
  }

  template <typename InputType, typename... Arg, CUDF_ENABLE_IF(not is_numeric<InputType>())>
  std::pair<std::unique_ptr<column>, table_view> operator()(Arg&&...)
  {
    CUDF_FAIL(
      "One-hot encoding for variable length column is not supported. Consider encoding the column "
      "with dictionary::encode.");
  }
};

std::pair<std::unique_ptr<column>, table_view> one_hot_encoding(column_view const& input_column,
                                                                column_view const& categories,
                                                                rmm::cuda_stream_view stream,
                                                                rmm::mr::device_memory_resource* mr)
{
  return type_dispatcher(
    input_column.type(), one_hot_encoding_launcher{}, input_column, categories, stream, mr);
}

}  // namespace detail

std::pair<std::unique_ptr<column>, table_view> one_hot_encoding(
  column_view const& input_column,
  column_view const& categories,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  CUDF_EXPECTS(input_column.type() == categories.type(),
               "Mismatch type between input and categories.");
  return detail::one_hot_encoding(input_column, categories, rmm::cuda_stream_default, mr);
}
}  // namespace cudf
