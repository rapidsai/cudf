#include <memory>
#include <vector>

#include <cudf/types.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <quantiles/quantiles_util.hpp>

namespace cudf {
namespace experimental {
namespace detail {

std::unique_ptr<column>
quantiles_indices(table_view const& input,
                  std::vector<double> q,
                  interpolation interp,
                  std::vector<order> const& column_order,
                  std::vector<null_order> const& null_precedence,
                  rmm::mr::device_memory_resource* mr =
                    rmm::mr::get_default_resource(),
                  cudaStream_t stream = 0)
{
    auto sorted_idx = sorted_order(input, column_order, null_precedence);

    auto q_device = static_cast<rmm::device_vector<double>>(q);
    auto q_data_type = data_type{ type_to_id<size_type>() };
    auto q_indices = make_numeric_column(q_data_type, q.size());

    auto select_quantile_idx = [interp, size=input.num_rows()]
        __device__ (double q) {
            auto idop = thrust::identity<size_type>();
            return select_quantile<size_type>(idop, size, q, interp);
        };

    thrust::transform(q_device.begin(),
                      q_device.end(),
                      q_indices->mutable_view().data<size_type>(),
                      select_quantile_idx);

    return q_indices;
}

}

std::unique_ptr<table>
quantiles(table_view const& input,
          std::vector<double> q,
          interpolation interp,
          std::vector<order> const& column_order,
          std::vector<null_order> const& null_precedence,
          rmm::mr::device_memory_resource* mr)
{
    auto quantiles_map = detail::quantiles_indices(input,
                                                   q,
                                                   interp,
                                                   column_order,
                                                   null_precedence);

    return detail::gather(input, quantiles_map->view());
}

}
}
