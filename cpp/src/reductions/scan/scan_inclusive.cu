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

#include "scan.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/table/row_operators.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/logical.h>
#include <thrust/scan.h>

namespace cudf {
namespace detail {

// logical-and scan of the null mask of the input view
rmm::device_buffer mask_scan(column_view const& input_view,
                             scan_type inclusive,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  rmm::device_buffer mask =
    detail::create_null_mask(input_view.size(), mask_state::UNINITIALIZED, stream, mr);
  auto d_input   = column_device_view::create(input_view, stream);
  auto valid_itr = detail::make_validity_iterator(*d_input);

  auto first_null_position = [&] {
    size_type const first_null = thrust::find_if_not(rmm::exec_policy(stream),
                                                     valid_itr,
                                                     valid_itr + input_view.size(),
                                                     thrust::identity<bool>{}) -
                                 valid_itr;
    size_type const exclusive_offset = (inclusive == scan_type::EXCLUSIVE) ? 1 : 0;
    return std::min(input_view.size(), first_null + exclusive_offset);
  }();

  set_null_mask(static_cast<bitmask_type*>(mask.data()), 0, first_null_position, true, stream);
  set_null_mask(
    static_cast<bitmask_type*>(mask.data()), first_null_position, input_view.size(), false, stream);
  return mask;
}

namespace {

/**
 * @brief Dispatcher for running Scan operation on input column
 *
 * @tparam Op device binary operator
 */
template <typename Op>
struct scan_dispatcher {
 private:
  template <typename T>
  static constexpr bool is_string_supported()
  {
    return std::is_same<T, string_view>::value &&
           (std::is_same<Op, DeviceMin>::value || std::is_same<Op, DeviceMax>::value);
  }

  template <typename T>
  static constexpr bool is_supported()
  {
    return std::is_arithmetic<T>::value || is_string_supported<T>();
  }

  // for arithmetic types
  template <typename T, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
  auto inclusive_scan(column_view const& input_view,
                      null_policy,
                      rmm::cuda_stream_view stream,
                      rmm::mr::device_memory_resource* mr)
  {
    auto output_column = detail::allocate_like(
      input_view, input_view.size(), mask_allocation_policy::NEVER, stream, mr);
    mutable_column_view result = output_column->mutable_view();

    auto d_input = column_device_view::create(input_view, stream);
    auto const begin =
      make_null_replacement_iterator(*d_input, Op::template identity<T>(), input_view.has_nulls());
    thrust::inclusive_scan(
      rmm::exec_policy(stream), begin, begin + input_view.size(), result.data<T>(), Op{});

    CHECK_CUDA(stream.value());
    return output_column;
  }

  // for string type: only MIN and MAX are supported
  template <typename T, std::enable_if_t<is_string_supported<T>()>* = nullptr>
  std::unique_ptr<column> inclusive_scan(column_view const& input_view,
                                         null_policy,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
  {
    auto d_input = column_device_view::create(input_view, stream);

    rmm::device_uvector<T> result(input_view.size(), stream);
    auto begin =
      make_null_replacement_iterator(*d_input, Op::template identity<T>(), input_view.has_nulls());
    thrust::inclusive_scan(
      rmm::exec_policy(stream), begin, begin + input_view.size(), result.data(), Op{});

    CHECK_CUDA(stream.value());
    return make_strings_column(result, Op::template identity<string_view>(), stream, mr);
  }

 public:
  /**
   * @brief creates new column from input column by applying scan operation
   *
   * @param input     input column view
   * @param inclusive inclusive or exclusive scan
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned column's device memory
   * @return
   *
   * @tparam T type of input column
   */
  template <typename T, typename std::enable_if_t<is_supported<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& input,
                                     null_policy null_handling,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return inclusive_scan<T>(input, null_handling, stream, mr);
  }

  template <typename T, typename... Args>
  std::enable_if_t<!is_supported<T>(), std::unique_ptr<column>> operator()(Args&&...)
  {
    CUDF_FAIL("Non-arithmetic types not supported for inclusive scan");
  }
};

bool struct_has_nested_nulls(column_view const& struct_col)
{
  return std::any_of(struct_col.child_begin(), struct_col.child_end(), [](auto const& col) {
    return has_nested_nulls(table_view{{col}});
  });
}

bool has_nested_children(column_view const& struct_col)
{
  return std::any_of(struct_col.child_begin(), struct_col.child_end(), [](auto const& col) {
    return is_nested(col.type());
  });
}

template <bool nested_nulls>
void generate_rank_struct_comparisons(column_view const& order_by,
                                      mutable_column_view out,
                                      rmm::cuda_stream_view stream)
{
  auto const d_order_by = column_device_view::create(order_by, stream);
  thrust::tabulate(
    rmm::exec_policy(stream),
    out.begin<size_type>(),
    out.end<size_type>(),
    [d_order_by = *d_order_by, has_nulls = order_by.has_nulls()] __device__(size_type row_index) {
      if (row_index == 0) {
        return row_index + 1;
      } else if (has_nulls) {
        bool const lhs_is_null{d_order_by.is_null_nocheck(row_index)};
        bool const rhs_is_null{d_order_by.is_null_nocheck(row_index - 1)};
        if (lhs_is_null and rhs_is_null) {
          return 0;
        } else if (lhs_is_null != rhs_is_null) {
          return row_index + 1;
        }
      }

      return thrust::all_of(
               thrust::seq,
               thrust::make_counting_iterator<size_type>(0),
               thrust::make_counting_iterator<size_type>(0) + d_order_by.num_child_columns(),
               [row_index, d_order_by] __device__(size_type child_index) {
                 column_device_view const& col = d_order_by.child(child_index);
                 element_equality_comparator<nested_nulls> element_comparator{col, col, true};
                 return type_dispatcher(col.type(), element_comparator, row_index, row_index - 1);
               })
               ? 0
               : row_index + 1;
    });
}

template <bool has_nulls>
void generate_rank_comparisons(column_view const& order_by,
                               mutable_column_view out,
                               rmm::cuda_stream_view stream)
{
  auto const d_order_by = column_device_view::create(order_by, stream);
  element_equality_comparator<has_nulls> element_comparator(*d_order_by, *d_order_by, true);
  thrust::tabulate(rmm::exec_policy(stream),
                   out.begin<size_type>(),
                   out.end<size_type>(),
                   [type = d_order_by->type(), element_comparator] __device__(size_type row_index) {
                     return (row_index == 0 ||
                             !type_dispatcher(type, element_comparator, row_index, row_index - 1))
                              ? row_index + 1
                              : 0;
                   });
}

template <bool nested_nulls>
void generate_dense_rank_struct_comparisons(column_view const& order_by,
                                            mutable_column_view out,
                                            rmm::cuda_stream_view stream)
{
  auto const d_order_by = column_device_view::create(order_by, stream);
  thrust::tabulate(
    rmm::exec_policy(stream),
    out.begin<size_type>(),
    out.end<size_type>(),
    [d_order_by = *d_order_by, has_nulls = order_by.has_nulls()] __device__(size_type row_index) {
      if (row_index == 0) {
        return true;
      } else if (has_nulls) {
        bool const lhs_is_null{d_order_by.is_null_nocheck(row_index)};
        bool const rhs_is_null{d_order_by.is_null_nocheck(row_index - 1)};
        if (lhs_is_null and rhs_is_null) {
          return false;
        } else if (lhs_is_null != rhs_is_null) {
          return true;
        }
      }

      return !thrust::all_of(
        thrust::seq,
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(0) + d_order_by.num_child_columns(),
        [row_index, d_order_by] __device__(size_type child_index) {
          column_device_view const& col = d_order_by.child(child_index);
          element_equality_comparator<nested_nulls> element_comparator{col, col, true};
          return type_dispatcher(col.type(), element_comparator, row_index, row_index - 1);
        });
    });
}

template <bool has_nulls>
void generate_dense_rank_comparisons(column_view const& order_by,
                                     mutable_column_view out,
                                     rmm::cuda_stream_view stream)
{
  auto const d_order_by = column_device_view::create(order_by, stream);
  element_equality_comparator<has_nulls> element_comparator(*d_order_by, *d_order_by, true);
  thrust::tabulate(rmm::exec_policy(stream),
                   out.begin<size_type>(),
                   out.end<size_type>(),
                   [type = d_order_by->type(), element_comparator] __device__(size_type row_index) {
                     return (row_index == 0 ||
                             !type_dispatcher(type, element_comparator, row_index, row_index - 1));
                   });
}

}  // namespace

std::unique_ptr<column> inclusive_rank_scan(column_view const& order_by,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  auto ranks = make_fixed_width_column(
    data_type{type_to_id<size_type>()}, order_by.size(), mask_state::ALL_VALID, stream, mr);
  auto mutable_ranks = ranks->mutable_view();
  if (order_by.type().id() == type_id::LIST) {
    CUDF_FAIL("List types not supported");
  } else if (order_by.type().id() == type_id::STRUCT) {
    bool nested_nulls = struct_has_nested_nulls(order_by);
    bool is_nested    = has_nested_children(order_by);

    if (is_nested) {
      CUDF_FAIL("Nested struct and list types not supported");
    } else if (nested_nulls) {
      generate_rank_struct_comparisons<true>(order_by, mutable_ranks, stream);
    } else {
      generate_rank_struct_comparisons<false>(order_by, mutable_ranks, stream);
    }
  } else if (order_by.has_nulls()) {
    generate_rank_comparisons<true>(order_by, mutable_ranks, stream);
  } else {
    generate_rank_comparisons<false>(order_by, mutable_ranks, stream);
  }

  thrust::inclusive_scan(rmm::exec_policy(stream),
                         mutable_ranks.begin<size_type>(),
                         mutable_ranks.end<size_type>(),
                         mutable_ranks.begin<size_type>(),
                         DeviceMax{});
  return ranks;
}

std::unique_ptr<column> inclusive_dense_rank_scan(column_view const& order_by,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::mr::device_memory_resource* mr)
{
  auto ranks = make_fixed_width_column(
    data_type{type_to_id<size_type>()}, order_by.size(), mask_state::ALL_VALID, stream, mr);
  auto mutable_ranks = ranks->mutable_view();
  if (order_by.type().id() == type_id::LIST) {
    CUDF_FAIL("List types not supported");
  } else if (order_by.type().id() == type_id::STRUCT) {
    bool nested_nulls = struct_has_nested_nulls(order_by);
    bool is_nested    = has_nested_children(order_by);

    if (is_nested) {
      CUDF_FAIL("Nested struct and list types not supported");
    } else if (nested_nulls) {
      generate_dense_rank_struct_comparisons<true>(order_by, mutable_ranks, stream);
    } else {
      generate_dense_rank_struct_comparisons<false>(order_by, mutable_ranks, stream);
    }
  } else if (order_by.has_nulls()) {
    generate_dense_rank_comparisons<true>(order_by, mutable_ranks, stream);
  } else {
    generate_dense_rank_comparisons<false>(order_by, mutable_ranks, stream);
  }

  thrust::inclusive_scan(rmm::exec_policy(stream),
                         mutable_ranks.begin<size_type>(),
                         mutable_ranks.end<size_type>(),
                         mutable_ranks.begin<size_type>());
  return ranks;
}

std::unique_ptr<column> scan_inclusive(
  column_view const& input,
  std::unique_ptr<aggregation> const& agg,
  null_policy null_handling,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto output = scan_agg_dispatch<scan_dispatcher>(input, agg, null_handling, stream, mr);

  if (agg->kind == aggregation::RANK || agg->kind == aggregation::DENSE_RANK) {
    return output;
  } else if (null_handling == null_policy::EXCLUDE) {
    output->set_null_mask(detail::copy_bitmask(input, stream, mr), input.null_count());
  } else if (input.nullable()) {
    output->set_null_mask(mask_scan(input, scan_type::INCLUSIVE, stream, mr), UNKNOWN_NULL_COUNT);
  }

  return output;
}
}  // namespace detail
}  // namespace cudf
