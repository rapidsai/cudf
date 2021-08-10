#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/datetime.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cudf/detail/date_offsets.cuh>

namespace cudf {
namespace datetime {
namespace detail {
std::unique_ptr<cudf::column> date_range(cudf::scalar const& initial,
                                         std::size_t n,
                                         DateOffset offset,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(cudf::is_timestamp(initial.type()), "Column type should be timestamp");
  auto output_col_type = initial.type();

  // Return empty column if n = 0
  if (n == 0) return cudf::make_empty_column(output_col_type);

  auto launch = date_range_functor{};

  return type_dispatcher(initial.type(), launch, initial, n, offset, stream, mr);
}
}  // namespace detail

std::unique_ptr<cudf::column> date_range_month(cudf::scalar const& initial,
                                               size_t n,
                                               size_t months,
                                               rmm::mr::device_memory_resource* mr)
{
  return detail::date_range(
    initial, n, detail::DateOffset{months, 0}, rmm::cuda_stream_default, mr);
}

std::unique_ptr<cudf::column> date_range_nanosecond(cudf::scalar const& initial,
                                                    size_t n,
                                                    size_t nanoseconds,
                                                    rmm::mr::device_memory_resource* mr)
{
  return detail::date_range(
    initial, n, detail::DateOffset{0, nanoseconds}, rmm::cuda_stream_default, mr);
}

}  // namespace datetime
}  // namespace cudf
