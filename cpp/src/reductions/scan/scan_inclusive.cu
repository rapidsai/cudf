/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <reductions/scan/scan.cuh>
#include <reductions/struct_minmax_util.cuh>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/reduction.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

#include <type_traits>

namespace cudf {
namespace detail {

// logical-and scan of the null mask of the input view
std::pair<rmm::device_buffer, size_type> mask_scan(column_view const& input_view,
                                                   scan_type inclusive,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::mr::device_memory_resource* mr)
{
  rmm::device_buffer mask =
    detail::create_null_mask(input_view.size(), mask_state::UNINITIALIZED, stream, mr);
  auto d_input   = column_device_view::create(input_view, stream);
  auto valid_itr = detail::make_validity_iterator(*d_input);

  auto first_null_position = [&] {
    size_type const first_null =
      thrust::find_if_not(
        rmm::exec_policy(stream), valid_itr, valid_itr + input_view.size(), thrust::identity{}) -
      valid_itr;
    size_type const exclusive_offset = (inclusive == scan_type::EXCLUSIVE) ? 1 : 0;
    return std::min(input_view.size(), first_null + exclusive_offset);
  }();

  set_null_mask(static_cast<bitmask_type*>(mask.data()), 0, first_null_position, true, stream);
  set_null_mask(
    static_cast<bitmask_type*>(mask.data()), first_null_position, input_view.size(), false, stream);
  return {std::move(mask), input_view.size() - first_null_position};
}

namespace {

/**
 * @brief Min/Max inclusive scan operator
 *
 * This operator will accept index values, check them and then
 * run the `Op` operation on the individual element objects.
 * The returned result is the appropriate index value.
 *
 * This was specifically created to workaround a thrust issue
 * https://github.com/NVIDIA/thrust/issues/1479
 * where invalid values are passed to the operator.
 */
template <typename Element, typename Op>
struct min_max_scan_operator {
  column_device_view const col;      ///< strings column device view
  Element const null_replacement{};  ///< value used when element is null
  bool const has_nulls;              ///< true if col has null elements

  min_max_scan_operator(column_device_view const& col, bool has_nulls = true)
    : col{col}, null_replacement{Op::template identity<Element>()}, has_nulls{has_nulls}
  {
    // verify validity bitmask is non-null, otherwise, is_null_nocheck() will crash
    if (has_nulls) CUDF_EXPECTS(col.nullable(), "column with nulls must have a validity bitmask");
  }

  __device__ inline size_type operator()(size_type lhs, size_type rhs) const
  {
    // thrust::inclusive_scan may pass us garbage values so we need to protect ourselves;
    // in these cases the return value does not matter since the result is not used
    if (lhs < 0 || rhs < 0 || lhs >= col.size() || rhs >= col.size()) return 0;
    Element d_lhs =
      has_nulls && col.is_null_nocheck(lhs) ? null_replacement : col.element<Element>(lhs);
    Element d_rhs =
      has_nulls && col.is_null_nocheck(rhs) ? null_replacement : col.element<Element>(rhs);
    return Op{}(d_lhs, d_rhs) == d_lhs ? lhs : rhs;
  }
};

template <typename Op, typename T>
struct scan_functor {
  static std::unique_ptr<column> invoke(column_view const& input_view,
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

    CUDF_CHECK_CUDA(stream.value());
    return output_column;
  }
};

template <typename Op>
struct scan_functor<Op, cudf::string_view> {
  static std::unique_ptr<column> invoke(column_view const& input_view,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
  {
    auto d_input = column_device_view::create(input_view, stream);

    // build indices of the scan operation results
    rmm::device_uvector<size_type> result(input_view.size(), stream);
    thrust::inclusive_scan(
      rmm::exec_policy(stream),
      thrust::counting_iterator<size_type>(0),
      thrust::counting_iterator<size_type>(input_view.size()),
      result.begin(),
      min_max_scan_operator<cudf::string_view, Op>{*d_input, input_view.has_nulls()});

    // call gather using the indices to build the output column
    auto result_table = cudf::detail::gather(cudf::table_view({input_view}),
                                             result,
                                             out_of_bounds_policy::DONT_CHECK,
                                             negative_index_policy::NOT_ALLOWED,
                                             stream,
                                             mr);
    return std::move(result_table->release().front());
  }
};

template <typename Op>
struct scan_functor<Op, cudf::struct_view> {
  static std::unique_ptr<column> invoke(column_view const& input,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
  {
    // Create a gather map containing indices of the prefix min/max elements.
    auto gather_map = rmm::device_uvector<size_type>(input.size(), stream);
    auto const binop_generator =
      cudf::reduction::detail::comparison_binop_generator::create<Op>(input, stream);
    thrust::inclusive_scan(rmm::exec_policy(stream),
                           thrust::counting_iterator<size_type>(0),
                           thrust::counting_iterator<size_type>(input.size()),
                           gather_map.begin(),
                           binop_generator.binop());

    // Gather the children columns of the input column. Must use `get_sliced_child` to properly
    // handle input in case it is a sliced view.
    auto const input_children = [&] {
      auto const it = cudf::detail::make_counting_transform_iterator(
        0, [structs_view = structs_column_view{input}, &stream](auto const child_idx) {
          return structs_view.get_sliced_child(child_idx, stream);
        });
      return std::vector<column_view>(it, it + input.num_children());
    }();

    // Gather the children elements of the prefix min/max struct elements for the output.
    auto scanned_children = cudf::detail::gather(table_view{input_children},
                                                 gather_map,
                                                 out_of_bounds_policy::DONT_CHECK,
                                                 negative_index_policy::NOT_ALLOWED,
                                                 stream,
                                                 mr)
                              ->release();

    // Don't need to set a null mask because that will be handled at the caller.
    return make_structs_column(
      input.size(), std::move(scanned_children), 0, rmm::device_buffer{0, stream, mr}, stream, mr);
  }
};

/**
 * @brief Dispatcher for running a Scan operation on an input column
 *
 * @tparam Op device binary operator
 */
template <typename Op>
struct scan_dispatcher {
 private:
  template <typename T>
  static constexpr bool is_supported()
  {
    if constexpr (std::is_same_v<T, cudf::struct_view>) {
      return std::is_same_v<Op, DeviceMin> || std::is_same_v<Op, DeviceMax>;
    } else {
      return std::is_invocable_v<Op, T, T> && !cudf::is_dictionary<T>();
    }
  }

 public:
  /**
   * @brief Creates a new column from the input column by applying the scan operation
   *
   * @param input Input column view
   * @param null_handling How null row entries are to be processed
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned column's device memory
   * @return
   *
   * @tparam T type of input column
   */
  template <typename T, std::enable_if_t<is_supported<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& input,
                                     null_policy,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return scan_functor<Op, T>::invoke(input, stream, mr);
  }

  template <typename T, typename... Args>
  std::enable_if_t<!is_supported<T>(), std::unique_ptr<column>> operator()(Args&&...)
  {
    CUDF_FAIL("Unsupported type for inclusive scan operation");
  }
};

}  // namespace

std::unique_ptr<column> scan_inclusive(column_view const& input,
                                       scan_aggregation const& agg,
                                       null_policy null_handling,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  auto output = scan_agg_dispatch<scan_dispatcher>(input, agg, null_handling, stream, mr);

  if (null_handling == null_policy::EXCLUDE) {
    output->set_null_mask(detail::copy_bitmask(input, stream, mr), input.null_count());
  } else if (input.nullable()) {
    auto [mask, null_count] = mask_scan(input, scan_type::INCLUSIVE, stream, mr);
    output->set_null_mask(mask, null_count);
  }

  // If the input is a structs column, we also need to push down nulls from the parent output column
  // into the children columns.
  if (input.type().id() == type_id::STRUCT && output->has_nulls()) {
    auto const num_rows   = output->size();
    auto const null_count = output->null_count();
    auto content          = output->release();

    // Build new children columns.
    const auto null_mask = reinterpret_cast<bitmask_type const*>(content.null_mask->data());
    std::for_each(content.children.begin(),
                  content.children.end(),
                  [null_mask, null_count, stream, mr](auto& child) {
                    child = structs::detail::superimpose_nulls(
                      null_mask, null_count, std::move(child), stream, mr);
                  });

    // Replace the children columns.
    output = cudf::make_structs_column(
      num_rows, std::move(content.children), null_count, std::move(*content.null_mask), stream, mr);
  }

  return output;
}
}  // namespace detail
}  // namespace cudf
