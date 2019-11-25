#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/iterator.cuh>

#include <rmm/rmm.h>
#include <cudf/utilities/error.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <utilities/device_atomics.cuh>
#include <cub/device/device_scan.cuh>
#include <cudf/reduction.hpp>


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
  // return true if T is arithmetic type (including cudf::experimental::bool8)
  template <typename T>
  static constexpr bool is_supported() {
    return std::is_arithmetic<T>::value;
  }

  template <typename T, typename InputIterator>
  void exclusive_scan(const InputIterator input, T* output, size_t size,
                      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    rmm::device_buffer temp_storage;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveScan(temp_storage.data(), temp_storage_bytes,
                                   input, output, Op{},
                                   Op::template identity<T>(), size, stream);
    // Prepare temp storage
    temp_storage = rmm::device_buffer{temp_storage_bytes, stream, mr};

    cub::DeviceScan::ExclusiveScan(temp_storage.data(), temp_storage_bytes,
                                   input, output, Op{},
                                   Op::template identity<T>(), size, stream);
    CUDA_CHECK_LAST();
  }

  template <typename T, typename InputIterator>
  void inclusive_scan(const InputIterator input, T* output, size_t size,
                      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    rmm::device_buffer temp_storage;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveScan(temp_storage.data(), temp_storage_bytes,
                                   input, output, Op{}, size, stream);
    // Prepare temp storage
    temp_storage = rmm::device_buffer{temp_storage_bytes, stream, mr};
    cub::DeviceScan::InclusiveScan(temp_storage.data(), temp_storage_bytes,
                                   input, output, Op{}, size, stream);
    CUDA_CHECK_LAST();
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
                  bool inclusive, 
                  rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    std::unique_ptr<column> output_column = make_numeric_column(
        input.type(), input.size(), 
        copy_bitmask(input, stream, mr), //copy bit mask
        input.null_count(), stream, mr);

    mutable_column_view output = output_column->mutable_view();
    const size_t size = input.size();

    if (input.has_nulls()) {
      auto d_input = column_device_view::create(input, stream);
      auto it = make_null_replacement_iterator(*d_input, Op::template identity<T>());
      if (inclusive)
        inclusive_scan(it, output.data<T>(), size, mr, stream);
      else
        exclusive_scan(it, output.data<T>(), size, mr, stream);
    } else {
      auto it = input.data<T>();  // since scan is for arithmetic types only
      if (inclusive)
        inclusive_scan(it, output.data<T>(), size, mr, stream);
      else
        exclusive_scan(it, output.data<T>(), size, mr, stream);
    }
    CUDF_EXPECTS(input.null_count() == output.null_count(),
        "Input / output column null count mismatch");
    return output_column;
  }

  template <typename T,
            typename std::enable_if_t<!is_supported<T>(), T>* = nullptr>
  std::unique_ptr<column> operator()(const column_view& input,
                  bool inclusive, 
                  rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    CUDF_FAIL("Non-arithmetic types not supported for `cudf::scan`");
  }
};

std::unique_ptr<column> scan(const column_view& input,
                             scan_op op, bool inclusive,
                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                             cudaStream_t stream=0)
{
  CUDF_EXPECTS(is_numeric(input.type()), "Unexpected non-numeric type.");

  switch (op) {
    case scan_op::SUM:
        return cudf::experimental::type_dispatcher(input.type(),
            ScanDispatcher<cudf::DeviceSum>(), input, inclusive, mr, stream);
    case scan_op::MIN:
        return cudf::experimental::type_dispatcher(input.type(),
            ScanDispatcher<cudf::DeviceMin>(), input, inclusive, mr, stream);
    case scan_op::MAX:
        return cudf::experimental::type_dispatcher(input.type(),
            ScanDispatcher<cudf::DeviceMax>(), input, inclusive, mr, stream);
    case scan_op::PRODUCT:
        return cudf::experimental::type_dispatcher(input.type(),
            ScanDispatcher<cudf::DeviceProduct>(), input, inclusive, mr, stream);
    default:
        CUDF_FAIL("The input enum `scan::operators` is out of the range");
    }
}
} // namespace detail

std::unique_ptr<column> scan(const column_view& input,
                             scan_op op, bool inclusive,
                             rmm::mr::device_memory_resource* mr)
{
  return detail::scan(input, op, inclusive, mr);
}

}  // namespace experimental
}  // namespace cudf
