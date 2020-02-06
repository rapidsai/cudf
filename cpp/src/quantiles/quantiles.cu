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

namespace detail {

template<typename SortMapIterator>
std::unique_ptr<table>
quantiles(table_view const& input,
          SortMapIterator sortmap,
          std::vector<double> const& q,
          interpolation interp,
          rmm::mr::device_memory_resource* mr)
{
    auto quantile_idx_lookup = [sortmap, interp, size=input.num_rows()]
        __device__ (double q) {
            auto selector = [sortmap] __device__ (auto idx) {
                return sortmap[idx];
            };
            return detail::select_quantile<size_type>(selector, size, q, interp);
        };

    rmm::device_vector<double> q_device{q};

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

std::unique_ptr<table>
quantiles(table_view const& input,
          std::vector<double> const& q,
          interpolation interp,
          cudf::sorted is_input_sorted,
          std::vector<order> const& column_order,
          std::vector<null_order> const& null_precedence,
          rmm::mr::device_memory_resource* mr)
{
    CUDF_EXPECTS(q.size() > 0,
                 "multi-column quantiles require at least one requested quantile.");

    CUDF_EXPECTS(interp == interpolation::HIGHER ||
                 interp == interpolation::LOWER ||
                 interp == interpolation::NEAREST,
                 "multi-column quantiles require a non-arithmetic interpolation strategy.");

    CUDF_EXPECTS(input.num_rows() > 0,
                 "multi-column quantiles require at least one input row.");

    if (is_input_sorted == sorted::YES)
    {
        return detail::quantiles(input,
                                 thrust::make_counting_iterator<size_type>(0),
                                 q,
                                 interp,
                                 mr);
    }
    else
    {
        auto sorted_idx = detail::sorted_order(input, column_order, null_precedence);
        return detail::quantiles(input,
                                 sorted_idx->view().data<size_type>(),
                                 q,
                                 interp,
                                 mr);
    }
}

}
}
