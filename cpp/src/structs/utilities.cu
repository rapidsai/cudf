/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <cudf/table/row_operators.cuh>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <rmm/exec_policy.hpp>
#include <structs/utilities.hpp>
#include "cudf/binaryop.hpp"
#include "cudf/column/column_device_view.cuh"
#include "cudf/table/table_device_view.cuh"
#include "cudf/table/table_view.hpp"
#include "cudf/types.hpp"
#include "cudf/utilities/traits.hpp"
#include "cudf/utilities/type_dispatcher.hpp"
#include "thrust/logical.h"

namespace cudf {
namespace structs {
namespace detail {

struct StructCompFunctor {
  template <typename output_type, bool const has_nulls>
  void equality_row(table_device_view lhs,
                    table_device_view rhs,
                    mutable_column_view out,
                    rmm::cuda_stream_view stream)
  {
    row_equality_comparator<has_nulls> comparator(lhs, rhs, true);
    auto d_out = mutable_column_device_view::create(out);

    thrust::tabulate(rmm::exec_policy(stream),
                     out.begin<output_type>(),
                     out.end<output_type>(),
                     [comparator, d_out = *d_out] __device__(size_type row_index) {
                       return d_out.is_valid(row_index) && comparator(row_index, row_index);
                     });
  }

  template <typename output_type, bool const has_nulls>
  void nequal_row(table_device_view lhs,
                  table_device_view rhs,
                  mutable_column_view out,
                  rmm::cuda_stream_view stream)
  {
    row_equality_comparator<has_nulls> comparator(lhs, rhs, true);
    auto d_out = mutable_column_device_view::create(out);

    thrust::tabulate(rmm::exec_policy(stream),
                     out.begin<output_type>(),
                     out.end<output_type>(),
                     [comparator, d_out = *d_out] __device__(size_type row_index) {
                       return d_out.is_valid(row_index) && !comparator(row_index, row_index);
                     });
  }

  template <typename output_type, bool const has_nulls>
  void lt_row(table_device_view lhs,
              table_device_view rhs,
              mutable_column_view out,
              rmm::cuda_stream_view stream)
  {
    row_lexicographic_comparator<has_nulls> comparator(lhs, rhs, nullptr, nullptr);
    auto d_out = mutable_column_device_view::create(out);

    thrust::tabulate(rmm::exec_policy(stream),
                     out.begin<output_type>(),
                     out.end<output_type>(),
                     [comparator, d_out = *d_out] __device__(size_type row_index) {
                       return d_out.is_valid(row_index) && comparator(row_index, row_index);
                     });
  }

  template <typename output_type, bool const has_nulls>
  void gte_row(table_device_view lhs,
               table_device_view rhs,
               mutable_column_view out,
               rmm::cuda_stream_view stream)
  {
    row_lexicographic_comparator<has_nulls> comparator(lhs, rhs, nullptr, nullptr);
    auto d_out = mutable_column_device_view::create(out);

    thrust::tabulate(rmm::exec_policy(stream),
                     out.begin<output_type>(),
                     out.end<output_type>(),
                     [comparator, d_out = *d_out] __device__(size_type row_index) {
                       return d_out.is_valid(row_index) && !comparator(row_index, row_index);
                     });
  }

  template <typename output_type, bool const has_nulls>
  void gt_row(table_device_view lhs,
              table_device_view rhs,
              mutable_column_view out,
              rmm::cuda_stream_view stream)
  {
    std::vector<order> op_modifier{};
    std::vector<null_order> norder{};
    std::for_each(thrust::counting_iterator<size_type>(0),
                  thrust::counting_iterator<size_type>(lhs.num_columns()),
                  [&](auto child_ind) {
                    op_modifier.push_back(order::DESCENDING);
                    norder.push_back(null_order::BEFORE);
                  });

    auto const op_modifier_dv = cudf::detail::make_device_uvector_async(op_modifier, stream);
    auto comparator = row_lexicographic_comparator<has_nulls>(lhs, rhs, op_modifier_dv.data());
    auto d_out      = mutable_column_device_view::create(out);

    thrust::tabulate(rmm::exec_policy(stream),
                     out.begin<output_type>(),
                     out.end<output_type>(),
                     [comparator, d_out = *d_out] __device__(size_type row_index) {
                       return d_out.is_valid(row_index) && comparator(row_index, row_index);
                     });
  }

  template <typename output_type, bool const has_nulls>
  void lte_row(table_device_view lhs,
               table_device_view rhs,
               mutable_column_view out,
               rmm::cuda_stream_view stream)
  {
    std::vector<order> op_modifier{};
    std::vector<null_order> norder{};
    std::for_each(thrust::counting_iterator<size_type>(0),
                  thrust::counting_iterator<size_type>(lhs.num_columns()),
                  [&](auto child_ind) {
                    op_modifier.push_back(order::DESCENDING);
                    norder.push_back(null_order::BEFORE);
                  });

    auto const op_modifier_dv = cudf::detail::make_device_uvector_async(op_modifier, stream);
    auto comparator = row_lexicographic_comparator<has_nulls>(lhs, rhs, op_modifier_dv.data());
    auto d_out      = mutable_column_device_view::create(out);

    thrust::tabulate(rmm::exec_policy(stream),
                     out.begin<output_type>(),
                     out.end<output_type>(),
                     [comparator, d_out = *d_out] __device__(size_type row_index) {
                       return d_out.is_valid(row_index) && !comparator(row_index, row_index);
                     });
  }

  template <typename T, std::enable_if_t<is_numeric<T>()>* = nullptr>
  void __host__ operator()(table_view const& lhs,
                           table_view const& rhs,
                           mutable_column_view& out,
                           binary_operator op,
                           rmm::cuda_stream_view stream)
  {
    bool const has_nulls = has_nested_nulls(lhs) || has_nested_nulls(rhs);

    auto d_lhs = table_device_view::create(lhs);
    auto d_rhs = table_device_view::create(rhs);

    if (has_nulls) {
      switch (op) {
        case binary_operator::EQUAL: equality_row<T, true>(*d_lhs, *d_rhs, out, stream); break;
        case binary_operator::NOT_EQUAL: nequal_row<T, true>(*d_lhs, *d_rhs, out, stream); break;
        case binary_operator::LESS: lt_row<T, true>(*d_lhs, *d_rhs, out, stream); break;
        case binary_operator::GREATER: gt_row<T, true>(*d_lhs, *d_rhs, out, stream); break;
        case binary_operator::LESS_EQUAL: lte_row<T, true>(*d_lhs, *d_rhs, out, stream); break;
        case binary_operator::GREATER_EQUAL: gte_row<T, true>(*d_lhs, *d_rhs, out, stream); break;
        // case binary_operator::NULL_EQUALS: break;
        default: CUDF_FAIL("Unsupported operator for these types");
      }
    } else {
      switch (op) {
        case binary_operator::EQUAL: equality_row<T, false>(*d_lhs, *d_rhs, out, stream); break;
        case binary_operator::NOT_EQUAL: nequal_row<T, false>(*d_lhs, *d_rhs, out, stream); break;
        case binary_operator::LESS: lt_row<T, false>(*d_lhs, *d_rhs, out, stream); break;
        case binary_operator::GREATER: gt_row<T, false>(*d_lhs, *d_rhs, out, stream); break;
        case binary_operator::LESS_EQUAL: lte_row<T, false>(*d_lhs, *d_rhs, out, stream); break;
        case binary_operator::GREATER_EQUAL: gte_row<T, false>(*d_lhs, *d_rhs, out, stream); break;
        // case binary_operator::NULL_EQUALS: break;
        default: CUDF_FAIL("Unsupported operator for these types");
      }
    }
  }

  template <typename T, std::enable_if_t<!is_numeric<T>()>* = nullptr>
  void __host__ operator()(table_view const& lhs,
                           table_view const& rhs,
                           mutable_column_view& out,
                           binary_operator op,
                           rmm::cuda_stream_view stream)
  {
    CUDF_FAIL("unsupported output type");
  }
};

std::unique_ptr<column> struct_binary_operation(column_view const& lhs,
                                                column_view const& rhs,
                                                binary_operator op,
                                                data_type output_type,
                                                rmm::cuda_stream_view stream,
                                                rmm::mr::device_memory_resource* mr)
{
  auto const lhs_superimposed = superimpose_parent_nulls(lhs);
  auto const lhs_flattener    = flatten_nested_columns(
    table_view{{std::get<0>(lhs_superimposed)}}, {}, {}, column_nullability::MATCH_INCOMING);
  table_view lhs_flat = std::get<0>(lhs_flattener);

  auto const rhs_superimposed = superimpose_parent_nulls(rhs);
  auto const rhs_flattener    = flatten_nested_columns(
    table_view{{std::get<0>(rhs_superimposed)}}, {}, {}, column_nullability::MATCH_INCOMING);
  table_view rhs_flat = std::get<0>(rhs_flattener);

  auto out =
    cudf::detail::make_fixed_width_column_for_output(lhs, rhs, op, output_type, stream, mr);
  auto out_view = out->mutable_view();

  type_dispatcher(out_view.type(), StructCompFunctor{}, lhs_flat, rhs_flat, out_view, op, stream);

  return out;
}

}  // namespace detail
}  // namespace structs
}  // namespace cudf
