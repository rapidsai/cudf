#include <memory>
#include <vector>

#include <cudf/types.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/detail/gather.cuh>
#include <quantiles/quantiles_util.hpp>

namespace cudf {
namespace experimental {

std::unique_ptr<table>
quantiles(table_view const& input,
          std::vector<double> const& q,
          interpolation interp,
          std::vector<order> const& column_order,
          std::vector<null_order> const& null_precedence,
          rmm::mr::device_memory_resource* mr)
{
    CUDF_EXPECTS(interp == interpolation::HIGHER ||
                 interp == interpolation::LOWER ||
                 interp == interpolation::NEAREST,
                 "multi-column quantiles require a non-arithmetic interpolation strategy.");

    CUDF_EXPECTS(input.num_rows() > 0,
                 "multi-column quantiles require at least one input row.");

    auto sorted_idx = detail::sorted_order(input, column_order, null_precedence);

    auto sortmap_lookup = [sortmap=sorted_idx->view().data<size_type>()]
        __device__(size_type idx) {
            return sortmap[idx];
        };

    auto quantile_idx_lookup = [sortmap_lookup, interp, size=input.num_rows()]
        __device__ (double q) {
            return detail::select_quantile<size_type>(sortmap_lookup,
                                                      size,
                                                      q,
                                                      interp);
        };

    auto q_device = static_cast<rmm::device_vector<double>>(q);
    auto quantile_idx_iter = thrust::make_transform_iterator(q_device.begin(),
                                                             quantile_idx_lookup);

    return detail::gather(input,
                          quantile_idx_iter,
                          quantile_idx_iter + q.size(),
                          false,
                          false,
                          false,
                          mr);
}

}
}
