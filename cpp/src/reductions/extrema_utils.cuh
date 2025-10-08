/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#pragma once

#include "nested_type_minmax_util.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cub/device/device_reduce.cuh>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>

namespace cudf::reduction::simple::detail {

/**
 * @brief A type-dispatcher functor to create a scalar with a given value.
 */
template <typename InputType>
class make_scalar_fn {
  static_assert(cudf::is_numeric<InputType>(), "InputType must be numeric");

  template <typename OutputType>
  static constexpr bool is_supported()
  {
    return cudf::is_numeric<OutputType>();
  }

 public:
  template <typename OutputType>
  [[nodiscard]] std::unique_ptr<scalar> operator()(InputType input,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr) const
    requires(is_supported<OutputType>())
  {
    using ScalarType = scalar_type_t<OutputType>;
    return std::make_unique<ScalarType>(static_cast<OutputType>(input), true, stream, mr);
  }

  template <typename OutputType>
  std::unique_ptr<scalar> operator()(InputType,
                                     rmm::cuda_stream_view,
                                     rmm::device_async_resource_ref) const
    requires(not is_supported<OutputType>())
  {
    CUDF_FAIL("make_scalar_fn is not supported for this type");
  }
};

/**
 * @brief An adapter to make a functor's operator() non-inlinable.
 *
 * This is to reduce compile time when compiling heavy binary comparators with thrust algorithms,
 * especially `thrust::reduce` and `thrust::min/max_element`.
 * @tparam Functor The functor to prevent inlining.
 */
template <typename Functor>
struct non_inline_adapter_fn {
  Functor f;
  non_inline_adapter_fn(Functor&& f_) : f{std::forward<Functor>(f_)} {}
  template <typename... Args>
  [[nodiscard]] __attribute__((noinline)) __device__ auto operator()(Args&&... args) const
  {
    return f(std::forward<Args>(args)...);
  }
};

/**
 * @brief Call reduce and return a scalar of the specified type, specialized for ARGMIN and
 * ARGMAX aggregations.
 *
 * @tparam K The aggregation to execute on the column, must be either ARGMIN or ARGMAX.
 */
template <aggregation::Kind K>
class arg_minmax_dispatcher {
  static_assert(K == aggregation::ARGMIN or K == aggregation::ARGMAX,
                "Aggregation kind must be either ARGMIN or ARGMAX");

  template <typename ElementType>
  static constexpr bool is_supported()
  {
    return !cudf::is_dictionary<ElementType>() && !std::is_same_v<ElementType, void>;
  }

  template <typename Comparator>
  [[nodiscard]] size_type find_minmax_idx(size_type size,
                                          Comparator comp,
                                          rmm::cuda_stream_view stream) const
  {
    auto const input_it     = thrust::make_counting_iterator<size_type>(0);
    auto const extremum_pos = [&] {
      if constexpr (K == aggregation::ARGMIN) {
        return thrust::min_element(rmm::exec_policy_nosync(stream),
                                   input_it,
                                   input_it + size,
                                   non_inline_adapter_fn<Comparator>{std::move(comp)});
      } else {
        return thrust::max_element(rmm::exec_policy_nosync(stream),
                                   input_it,
                                   input_it + size,
                                   non_inline_adapter_fn<Comparator>{std::move(comp)});
      }
    }();
    return static_cast<size_type>(cuda::std::distance(input_it, extremum_pos));
  }

  template <typename ElementType>
  [[nodiscard]] size_type find_minmax_idx_numeric(column_view const& input,
                                                  rmm::cuda_stream_view stream) const
  {
    auto const find_extremum = [&](auto const& it) {
      if constexpr (K == aggregation::ARGMIN) {
        return thrust::min_element(rmm::exec_policy_nosync(stream), it, it + input.size());
      } else {
        return thrust::max_element(rmm::exec_policy_nosync(stream), it, it + input.size());
      }
    };

    using Op = std::conditional_t<K == aggregation::ARGMIN,
                                  reduction::detail::op::min,
                                  reduction::detail::op::max>;
    if (input.has_nulls()) {
      auto const d_input     = column_device_view::create(input, stream);
      auto const transformer = Op{}.template get_null_replacing_element_transformer<ElementType>();
      auto const it =
        thrust::make_transform_iterator(d_input->pair_begin<ElementType, true>(), transformer);
      return static_cast<size_type>(cuda::std::distance(it, find_extremum(it)));
    } else {
      auto const it = input.template begin<ElementType>();
      return static_cast<size_type>(cuda::std::distance(it, find_extremum(it)));
    }
  }

 public:
  /**
   * @brief Called by the type-dispatcher to reduce the input column.
   *
   * @tparam ElementType The input column type
   * @param input Input column (must be numeric)
   * @param output_type Requested type of the scalar result
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned scalar's device memory
   */
  template <typename ElementType>
  [[nodiscard]] std::unique_ptr<scalar> operator()(column_view const& input,
                                                   data_type const output_type,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr) const
    requires(is_supported<ElementType>())
  {
    CUDF_EXPECTS(cudf::is_index_type(output_type), "Output type must be an index type.");
    auto const& values =
      is_dictionary(input.type()) ? dictionary_column_view(input).get_indices_annotated() : input;

    auto const idx = [&] {
      if constexpr (not cudf::is_nested<ElementType>()) {
        if constexpr (cudf::is_numeric<ElementType>() and not cudf::is_fixed_point<ElementType>()) {
          return find_minmax_idx_numeric<ElementType>(values, stream);
        } else {  // fixed-point or strings, which are not supported by null replacement transformer
          // Nulls are considered "greater" (ARGMIN), or "less" (ARGMAX) than non-null values.
          auto const null_orders = std::vector<null_order>{
            K == aggregation::ARGMIN ? null_order::AFTER : null_order::BEFORE};
          auto const comparator = cudf::detail::row::lexicographic::self_comparator{
            table_view{{values}}, {}, null_orders, stream};
          return find_minmax_idx(
            input.size(),
            comparator.less<false /*has_nested_columns*/>(nullate::DYNAMIC{input.has_nulls()}),
            stream);
        }
      } else {  // nested types
        using Op = std::conditional_t<K == aggregation::ARGMIN,
                                      reduction::detail::op::min,
                                      reduction::detail::op::max>;
        auto const binop_generator =
          reduction::detail::comparison_binop_generator::create<Op>(values, stream);
        return find_minmax_idx(input.size(), binop_generator.less(), stream);
      }
    }();

    return type_dispatcher(output_type, make_scalar_fn<size_type>{}, idx, stream, mr);
  }

  template <typename ElementType>
  std::unique_ptr<scalar> operator()(column_view const&,
                                     data_type const,
                                     rmm::cuda_stream_view,
                                     rmm::device_async_resource_ref) const
    requires(not is_supported<ElementType>())
  {
    CUDF_FAIL("ARGMIN/ARGMAX is not supported for this type");
  }
};

}  // namespace cudf::reduction::simple::detail
