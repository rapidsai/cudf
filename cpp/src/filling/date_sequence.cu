#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/datetime.hpp>
#include <cudf/detail/date_sequence.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace detail {
std::unique_ptr<cudf::column> date_sequence(size_type size,
                                            scalar const& init,
                                            size_type months,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(cudf::is_timestamp(init.type()), "Column type should be timestamp");
  auto output_col_type = init.type();

  // Return empty column if n = 0
  if (size == 0) return cudf::make_empty_column(output_col_type);

  auto launch = date_sequence_functor{};

  return type_dispatcher(init.type(), launch, size, init, months, stream, mr);
}
}  // namespace detail

std::unique_ptr<cudf::column> date_sequence(size_type size,
                                            scalar const& init,
                                            size_type months,
                                            rmm::mr::device_memory_resource* mr)
{
  return detail::date_sequence(size, init, months, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
