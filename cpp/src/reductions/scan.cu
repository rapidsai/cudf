#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/iterator.cuh>

#include <rmm/rmm.h>
#include <cudf/utilities/error.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/reduction.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/detail/null_mask.hpp>


namespace cudf {
namespace experimental {

namespace detail {

  /**
   * @brief Dispatcher for running Scan operation on input column
   * Dispatches scan operartion on `Op` and creates output column
   *
   * @tparam Op device binary operator
   */
template <typename Op>
struct ScanDispatcher {
  private:
  template <typename T>
  static constexpr bool is_string_supported() {
    return std::is_same<T, string_view>::value &&
     (std::is_same<Op, cudf::DeviceMin>::value ||
      std::is_same<Op, cudf::DeviceMax>::value);
  }
  // return true if T is arithmetic type (including cudf::experimental::bool8)
  template <typename T>
  static constexpr bool is_supported() {
    return std::is_arithmetic<T>::value || is_string_supported<T>();
  }

  //for arithmetic types
  template <typename T,
    std::enable_if_t<std::is_arithmetic<T>::value, T>* = nullptr>
  auto exclusive_scan(const column_view& input_view, bool skipna,
                      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    const size_type size = input_view.size();
    auto output_column = experimental::detail::allocate_like(
        input_view, size, experimental::mask_allocation_policy::NEVER, mr,
        stream);
    if (skipna) {
      output_column->set_null_mask(copy_bitmask(input_view, stream, mr),
                                   input_view.null_count());
    }
    mutable_column_view output = output_column->mutable_view();
    auto d_input = column_device_view::create(input_view, stream);

    if (input_view.has_nulls()) {
      auto input = make_null_replacement_iterator(*d_input, Op::template identity<T>());
      thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream), 
                            input, input + size, output.data<T>(),
                            Op::template identity<T>(),
                            Op{});
    } else {
      auto input = d_input->begin<T>();
      thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream), 
                            input, input + size, output.data<T>(),
                            Op::template identity<T>(),
                            Op{});
    }

    CHECK_CUDA(stream);
    return output_column;
  }

  //for string type
  template <typename T,
    std::enable_if_t<is_string_supported<T>(), T>* = nullptr>
  std::unique_ptr<column> exclusive_scan(const column_view& input_view, bool skipna,
                      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    CUDF_FAIL("String types supports only inclusive min/max for `cudf::scan`");
  }

  rmm::device_buffer mask_inclusive_scan(const column_view &input_view,
                                         rmm::mr::device_memory_resource *mr,
                                         cudaStream_t stream)
  {
    rmm::device_buffer mask =
        create_null_mask(input_view.size(), UNINITIALIZED, stream, mr);
    auto d_input = column_device_view::create(input_view, stream);
    auto v = experimental::detail::make_validity_iterator(*d_input);
    auto first_null_position = thrust::find_if_not(
      rmm::exec_policy(stream)->on(stream),
      v, v + input_view.size(), 
      thrust::identity<bool>{}) -  v;
    cudf::detail::clear_bits_from(static_cast<bitmask_type *>(mask.data()),
                                  input_view.size(), first_null_position, 
                                  stream);
    return mask;
  }

  //for arithmetic types
  template <typename T,
    std::enable_if_t<std::is_arithmetic<T>::value, T>* = nullptr>
  auto inclusive_scan(const column_view& input_view, bool skipna,
                      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    const size_type size = input_view.size();
    auto output_column = experimental::detail::allocate_like(
        input_view, size, experimental::mask_allocation_policy::NEVER, mr,
        stream);
    if(skipna) {
    output_column->set_null_mask(copy_bitmask(input_view, stream, mr),
                                 input_view.null_count());
    } else {
      if (input_view.nullable()) {
        output_column->set_null_mask(mask_inclusive_scan(input_view, mr, stream), cudf::UNKNOWN_NULL_COUNT);
      }
    }
    
    auto d_input = column_device_view::create(input_view, stream);
    mutable_column_view output = output_column->mutable_view();

    if (input_view.has_nulls()) {
      auto input = make_null_replacement_iterator(*d_input, Op::template identity<T>());
      thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream), 
                            input, input + size, output.data<T>(),
                            Op{});
    } else {
      auto input = d_input->begin<T>();
      thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream), 
                            input, input + size, output.data<T>(),
                            Op{});
    }

    CHECK_CUDA(stream);
    return output_column;
  }

//for string type
template <typename T,
    std::enable_if_t<is_string_supported<T>(), T>* = nullptr>
  std::unique_ptr<column> inclusive_scan(const column_view& input_view, bool skipna,
                      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    const size_type size = input_view.size();
    rmm::device_vector<T> result(size);

    auto d_input = column_device_view::create(input_view, stream);

    if (input_view.has_nulls()) {
      auto input = make_null_replacement_iterator(*d_input, Op::template identity<T>());
      thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream), 
                            input, input + size, result.data().get(),
                            Op{});
    } else {
      auto input = d_input->begin<T>();
      thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream), 
                            input, input + size, result.data().get(),
                            Op{});
    }
    CHECK_CUDA(stream);

    auto output_column = make_strings_column(result, Op::template identity<T>(), stream, mr);
    if(skipna) {
      output_column->set_null_mask(copy_bitmask(input_view, stream, mr),
                                   input_view.null_count());
    } else {
      if (input_view.nullable()) {
        output_column->set_null_mask(
            mask_inclusive_scan(input_view, mr, stream),
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
   * @param mr The resource to use for all allocations
   * @param stream The stream on which to execute all allocations and copies
   * @return 
   *
   * @tparam T type of input column
   */
  template <typename T,
            typename std::enable_if_t<is_supported<T>(), T>* = nullptr>
  std::unique_ptr<column> operator()(const column_view& input, 
                  bool inclusive, bool skipna,
                  rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    std::unique_ptr<column> output;
      if (inclusive)
        output = inclusive_scan<T>(input, skipna, mr, stream);
      else
        output = exclusive_scan<T>(input, skipna, mr, stream);
    if (skipna) {
      CUDF_EXPECTS(input.null_count() == output->null_count(),
                   "Input / output column null count mismatch");
    }
    return output;
  }

  template <typename T,
            typename std::enable_if_t<!is_supported<T>(), T>* = nullptr>
  std::unique_ptr<column> operator()(const column_view& input,
                  bool inclusive, bool skipna,
                  rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    CUDF_FAIL("Non-arithmetic types not supported for `cudf::scan`");
  }
};

std::unique_ptr<column> scan(const column_view& input,
                             scan_op op, bool inclusive, bool skipna,
                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                             cudaStream_t stream=0)
{
  CUDF_EXPECTS(is_numeric(input.type()) || is_compound(input.type()), "Unexpected non-numeric or non-string type.");

  switch (op) {
    case scan_op::SUM:
        return cudf::experimental::type_dispatcher(input.type(),
            ScanDispatcher<cudf::DeviceSum>(), input, inclusive, skipna, mr, stream);
    case scan_op::MIN:
        return cudf::experimental::type_dispatcher(input.type(),
            ScanDispatcher<cudf::DeviceMin>(), input, inclusive, skipna, mr, stream);
    case scan_op::MAX:
        return cudf::experimental::type_dispatcher(input.type(),
            ScanDispatcher<cudf::DeviceMax>(), input, inclusive, skipna, mr, stream);
    case scan_op::PRODUCT:
        return cudf::experimental::type_dispatcher(input.type(),
            ScanDispatcher<cudf::DeviceProduct>(), input, inclusive, skipna, mr, stream);
    default:
        CUDF_FAIL("The input enum `scan::operators` is out of the range");
    }
}
} // namespace detail

std::unique_ptr<column> scan(const column_view& input,
                             scan_op op, bool inclusive, bool skipna,
                             rmm::mr::device_memory_resource* mr)
{
  return detail::scan(input, op, inclusive, skipna, mr);
}

}  // namespace experimental
}  // namespace cudf
