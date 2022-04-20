/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <algorithm>

namespace cudf {
namespace detail {

namespace {

template <typename InputType>
struct one_hot_encode_functor {
  one_hot_encode_functor(column_device_view input, column_device_view category, bool nulls)
    : _equality_comparator{nullate::DYNAMIC{nulls}, input, category, null_equality::EQUAL},
      _input_size{input.size()}
  {
  }

  bool __device__ operator()(size_type i)
  {
    size_type const element_index  = i % _input_size;
    size_type const category_index = i / _input_size;
    return _equality_comparator.template operator()<InputType>(element_index, category_index);
  }

 private:
  element_equality_comparator<nullate::DYNAMIC> const _equality_comparator;
  size_type const _input_size;
};

}  // anonymous namespace

struct one_hot_encode_launcher {
  template <typename InputType, CUDF_ENABLE_IF(is_equality_comparable<InputType, InputType>())>
  std::pair<std::unique_ptr<column>, table_view> operator()(column_view const& input_column,
                                                            column_view const& categories,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::mr::device_memory_resource* mr)
  {
    auto const total_size = input_column.size() * categories.size();
    auto all_encodings    = make_numeric_column(
      data_type{type_id::BOOL8}, total_size, mask_state::UNALLOCATED, stream, mr);

    auto d_input_column    = column_device_view::create(input_column, stream);
    auto d_category_column = column_device_view::create(categories, stream);
    one_hot_encode_functor<InputType> one_hot_encoding_compute_f(
      *d_input_column, *d_category_column, input_column.nullable() || categories.nullable());

    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(total_size),
                      all_encodings->mutable_view().begin<bool>(),
                      one_hot_encoding_compute_f);

    auto split_iter = make_counting_transform_iterator(
      1, [width = input_column.size()](auto i) { return i * width; });
    std::vector<size_type> split_indices(split_iter, split_iter + categories.size() - 1);

    // TODO: use detail interface, gh9226
    auto views = cudf::split(all_encodings->view(), split_indices);
    table_view encodings_view{views};

    return std::pair(std::move(all_encodings), encodings_view);
  }

  template <typename InputType,
            typename... Args,
            CUDF_ENABLE_IF(not is_equality_comparable<InputType, InputType>())>
  std::pair<std::unique_ptr<column>, table_view> operator()(Args&&...)
  {
    CUDF_FAIL("Cannot encode column type without well-defined equality operator.");
  }
};

std::pair<std::unique_ptr<column>, table_view> one_hot_encode(column_view const& input,
                                                              column_view const& categories,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(input.type() == categories.type(), "Mismatch type between input and categories.");

  if (categories.is_empty()) { return std::pair(make_empty_column(type_id::BOOL8), table_view{}); }

  if (input.is_empty()) {
    auto empty_data = make_empty_column(type_id::BOOL8);
    std::vector<column_view> views(categories.size(), empty_data->view());
    return std::pair(std::move(empty_data), table_view{views});
  }

  return type_dispatcher(input.type(), one_hot_encode_launcher{}, input, categories, stream, mr);
}

}  // namespace detail

std::pair<std::unique_ptr<column>, table_view> one_hot_encode(column_view const& input,
                                                              column_view const& categories,
                                                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::one_hot_encode(input, categories, rmm::cuda_stream_default, mr);
}
}  // namespace cudf
