
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/copying.hpp>
#include <cudf/utilities/error.hpp>
#include <memory>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cudf/detail/gather.cuh>

namespace cudf {

namespace experimental {

namespace {

struct tile_functor {
    size_type count;
    size_type __device__ operator()(size_type i)
    {
        return i % count;
    }
};

} // anonymous namespace

std::unique_ptr<table>
tile(const table_view &in,
    size_type count,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0)
{
    CUDF_EXPECTS(count >= 0, "Count cannot be negative");

    auto in_num_rows = in.num_rows();

    if (count == 0 or in_num_rows == 0)
    {
        return empty_like(in);
    }

    auto out_num_rows = in_num_rows * count;
    auto counting_it = thrust::make_counting_iterator<size_type>(0);
    auto tiled_it = thrust::make_transform_iterator(counting_it,
                                                    tile_functor{in_num_rows});

    return detail::gather(in, tiled_it, tiled_it + out_num_rows, mr, stream);
}

} // namespace experimental

} // namespace cudf