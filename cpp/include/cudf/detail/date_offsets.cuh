#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/datetime.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace datetime {
namespace detail {

template <typename Timestamp>
CUDA_DEVICE_CALLABLE Timestamp
add_business_days(cudf::timestamp_scalar_device_view<Timestamp> const initial, std::size_t n)
{
  // just add `n` days:
  return initial.value() + cuda::std::chrono::days(n);
}

struct date_range_functor {
  template <typename T>
  typename std::enable_if_t<cudf::is_timestamp_t<T>::value, std::unique_ptr<cudf::column>>
  operator()(cudf::scalar const& input,
             std::size_t n,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
  {
    auto const device_input =
      get_scalar_device_view(static_cast<cudf::scalar_type_t<T>&>(const_cast<scalar&>(input)));
    auto output_column_type = cudf::data_type{cudf::type_to_id<T>()};
    auto output             = cudf::make_fixed_width_column(
      output_column_type, n, cudf::mask_state::UNALLOCATED, stream, mr);
    auto output_view = static_cast<cudf::mutable_column_view>(*output);

    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<std::size_t>(0),
                      thrust::make_counting_iterator<std::size_t>(n),
                      output_view.begin<T>(),
                      [device_input, n] __device__(std::size_t i) {
                        return add_business_days<T>(device_input, i);
                      });

    return output;
  }

  template <typename T>
  typename std::enable_if_t<!cudf::is_timestamp_t<T>::value, std::unique_ptr<cudf::column>>
  operator()(cudf::scalar const& input,
             std::size_t n,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
  {
    CUDF_FAIL("Cannot make a date_range of a non-datetime type");
  }
};

}  // namespace detail
}  // namespace datetime
}  // namespace cudf
