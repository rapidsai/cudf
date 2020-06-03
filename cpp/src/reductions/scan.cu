#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>

#include <rmm/rmm.h>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>

namespace cudf {
namespace detail {
/**
 * @brief Dispatcher for running Scan operation on input column
 * Dispatches scan operation on `Op` and creates output column
 *
 * @tparam Op device binary operator
 */
template <typename Op>
struct ScanDispatcher {
 private:
  template <typename T>
  static constexpr bool is_string_supported()
  {
    return std::is_same<T, string_view>::value &&
           (std::is_same<Op, cudf::DeviceMin>::value || std::is_same<Op, cudf::DeviceMax>::value);
  }
  // return true if T is arithmetic type (including bool)
  template <typename T>
  static constexpr bool is_supported()
  {
    return std::is_arithmetic<T>::value || is_string_supported<T>();
  }

  // for arithmetic types
  template <typename T, std::enable_if_t<std::is_arithmetic<T>::value, T>* = nullptr>
  auto exclusive_scan(const column_view& input_view,
                      null_policy null_handling,
                      rmm::mr::device_memory_resource* mr,
                      cudaStream_t stream)
  {
    const size_type size = input_view.size();
    auto output_column =
      detail::allocate_like(input_view, size, mask_allocation_policy::NEVER, mr, stream);
    if (null_handling == null_policy::EXCLUDE) {
      output_column->set_null_mask(copy_bitmask(input_view, stream, mr), input_view.null_count());
    }
    mutable_column_view output = output_column->mutable_view();
    auto d_input               = column_device_view::create(input_view, stream);

    if (input_view.has_nulls()) {
      auto input = make_null_replacement_iterator(*d_input, Op::template identity<T>());
      thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream),
                             input,
                             input + size,
                             output.data<T>(),
                             Op::template identity<T>(),
                             Op{});
    } else {
      auto input = d_input->begin<T>();
      thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream),
                             input,
                             input + size,
                             output.data<T>(),
                             Op::template identity<T>(),
                             Op{});
    }

    CHECK_CUDA(stream);
    return output_column;
  }

  // for string type
  template <typename T, std::enable_if_t<is_string_supported<T>(), T>* = nullptr>
  std::unique_ptr<column> exclusive_scan(const column_view& input_view,
                                         null_policy null_handling,
                                         rmm::mr::device_memory_resource* mr,
                                         cudaStream_t stream)
  {
    CUDF_FAIL("String types supports only inclusive min/max for `cudf::scan`");
  }

  rmm::device_buffer mask_inclusive_scan(const column_view& input_view,
                                         rmm::mr::device_memory_resource* mr,
                                         cudaStream_t stream)
  {
    rmm::device_buffer mask =
      create_null_mask(input_view.size(), mask_state::UNINITIALIZED, stream, mr);
    auto d_input = column_device_view::create(input_view, stream);
    auto v       = detail::make_validity_iterator(*d_input);
    auto first_null_position =
      thrust::find_if_not(
        rmm::exec_policy(stream)->on(stream), v, v + input_view.size(), thrust::identity<bool>{}) -
      v;
    cudf::set_null_mask(
      static_cast<cudf::bitmask_type*>(mask.data()), 0, first_null_position, true);
    cudf::set_null_mask(
      static_cast<cudf::bitmask_type*>(mask.data()), first_null_position, input_view.size(), false);
    return mask;
  }

  // for arithmetic types
  template <typename T, std::enable_if_t<std::is_arithmetic<T>::value, T>* = nullptr>
  auto inclusive_scan(const column_view& input_view,
                      null_policy null_handling,
                      rmm::mr::device_memory_resource* mr,
                      cudaStream_t stream)
  {
    const size_type size = input_view.size();
    auto output_column =
      detail::allocate_like(input_view, size, mask_allocation_policy::NEVER, mr, stream);
    if (null_handling == null_policy::EXCLUDE) {
      output_column->set_null_mask(copy_bitmask(input_view, stream, mr), input_view.null_count());
    } else {
      if (input_view.nullable()) {
        output_column->set_null_mask(mask_inclusive_scan(input_view, mr, stream),
                                     cudf::UNKNOWN_NULL_COUNT);
      }
    }

    auto d_input               = column_device_view::create(input_view, stream);
    mutable_column_view output = output_column->mutable_view();

    if (input_view.has_nulls()) {
      auto input = make_null_replacement_iterator(*d_input, Op::template identity<T>());
      thrust::inclusive_scan(
        rmm::exec_policy(stream)->on(stream), input, input + size, output.data<T>(), Op{});
    } else {
      auto input = d_input->begin<T>();
      thrust::inclusive_scan(
        rmm::exec_policy(stream)->on(stream), input, input + size, output.data<T>(), Op{});
    }

    CHECK_CUDA(stream);
    return output_column;
  }

  // for string type
  template <typename T, std::enable_if_t<is_string_supported<T>(), T>* = nullptr>
  std::unique_ptr<column> inclusive_scan(const column_view& input_view,
                                         null_policy null_handling,
                                         rmm::mr::device_memory_resource* mr,
                                         cudaStream_t stream)
  {
    const size_type size = input_view.size();
    rmm::device_vector<T> result(size);

    auto d_input = column_device_view::create(input_view, stream);

    if (input_view.has_nulls()) {
      auto input = make_null_replacement_iterator(*d_input, Op::template identity<T>());
      thrust::inclusive_scan(
        rmm::exec_policy(stream)->on(stream), input, input + size, result.data().get(), Op{});
    } else {
      auto input = d_input->begin<T>();
      thrust::inclusive_scan(
        rmm::exec_policy(stream)->on(stream), input, input + size, result.data().get(), Op{});
    }
    CHECK_CUDA(stream);

    auto output_column = make_strings_column(result, Op::template identity<T>(), stream, mr);
    if (null_handling == null_policy::EXCLUDE) {
      output_column->set_null_mask(copy_bitmask(input_view, stream, mr), input_view.null_count());
    } else {
      if (input_view.nullable()) {
        output_column->set_null_mask(mask_inclusive_scan(input_view, mr, stream),
                                     cudf::UNKNOWN_NULL_COUNT);
      }
    }
    return output_column;
  }

 public:
  /**
   * @brief creates new column from input column by applying scan operation
   *
   * @param input     input column view
   * @param inclusive inclusive or exclusive scan
   * @param mr Device memory resource used to allocate the returned column's device memory
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @return
   *
   * @tparam T type of input column
   */
  template <typename T, typename std::enable_if_t<is_supported<T>(), T>* = nullptr>
  std::unique_ptr<column> operator()(const column_view& input,
                                     scan_type inclusive,
                                     null_policy null_handling,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    std::unique_ptr<column> output;
    if (inclusive == scan_type::INCLUSIVE)
      output = inclusive_scan<T>(input, null_handling, mr, stream);
    else
      output = exclusive_scan<T>(input, null_handling, mr, stream);
    if (null_handling == null_policy::EXCLUDE) {
      CUDF_EXPECTS(input.null_count() == output->null_count(),
                   "Input / output column null count mismatch");
    }
    return output;
  }

  template <typename T, typename std::enable_if_t<!is_supported<T>(), T>* = nullptr>
  std::unique_ptr<column> operator()(const column_view& input,
                                     scan_type inclusive,
                                     null_policy null_handling,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    CUDF_FAIL("Non-arithmetic types not supported for `cudf::scan`");
  }
};

std::unique_ptr<column> scan(const column_view& input,
                             std::unique_ptr<aggregation> const& agg,
                             scan_type inclusive,
                             null_policy null_handling,
                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                             cudaStream_t stream                 = 0)
{
  CUDF_EXPECTS(is_numeric(input.type()) || is_compound(input.type()),
               "Unexpected non-numeric or non-string type.");

  switch (agg->kind) {
    case aggregation::SUM:
      return cudf::type_dispatcher(input.type(),
                                   ScanDispatcher<cudf::DeviceSum>(),
                                   input,
                                   inclusive,
                                   null_handling,
                                   mr,
                                   stream);
    case aggregation::MIN:
      return cudf::type_dispatcher(input.type(),
                                   ScanDispatcher<cudf::DeviceMin>(),
                                   input,
                                   inclusive,
                                   null_handling,
                                   mr,
                                   stream);
    case aggregation::MAX:
      return cudf::type_dispatcher(input.type(),
                                   ScanDispatcher<cudf::DeviceMax>(),
                                   input,
                                   inclusive,
                                   null_handling,
                                   mr,
                                   stream);
    case aggregation::PRODUCT:
      return cudf::type_dispatcher(input.type(),
                                   ScanDispatcher<cudf::DeviceProduct>(),
                                   input,
                                   inclusive,
                                   null_handling,
                                   mr,
                                   stream);
    default: CUDF_FAIL("Unsupported aggregation operator for scan");
  }
}
}  // namespace detail

std::unique_ptr<column> scan(const column_view& input,
                             std::unique_ptr<aggregation> const& agg,
                             scan_type inclusive,
                             null_policy null_handling,
                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::scan(input, agg, inclusive, null_handling, mr);
}

}  // namespace cudf
