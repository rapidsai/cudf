#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/datetime.hpp>
#include <cudf/detail/datetime_ops.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace detail {
struct calendrical_month_sequence_functor {
  template <typename T>
  typename std::enable_if_t<cudf::is_timestamp_t<T>::value, std::unique_ptr<cudf::column>>
  operator()(size_type n,
             scalar const& input,
             size_type months,
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
                      [initial = device_input, months] __device__(std::size_t i) {
                        return datetime::detail::add_calendrical_months_with_scale_back(
                          initial.value(), i * months);
                      });

    return output;
  }

  template <typename T>
  typename std::enable_if_t<!cudf::is_timestamp_t<T>::value, std::unique_ptr<cudf::column>>
  operator()(size_type n,
             scalar const& input,
             size_type months,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
  {
    CUDF_FAIL("Cannot make a date_range of a non-datetime type");
  }
};

}  // namespace detail
}  // namespace cudf
