#include <memory>
#include <vector>

#include <cudf/types.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/table/table_view.hpp>
#include "cudf/detail/gather.hpp"

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

    return sorted_idx;
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
