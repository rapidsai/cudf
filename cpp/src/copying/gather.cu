#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/count.h>

#include <memory>

namespace cudf {
namespace detail {

struct dispatch_map_type {
  template <typename map_type, std::enable_if_t<is_index_type<map_type>()>* = nullptr>
  std::unique_ptr<table> operator()(
    table_view const& source_table,
    column_view const& gather_map,
    size_type num_destination_rows,
    out_of_bounds_policy bounds,
    negative_index_policy neg_indices,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream                 = 0)
  {
    if (bounds == out_of_bounds_policy::FAIL) {
      cudf::size_type begin =
        neg_indices == negative_index_policy::ALLOWED ? -source_table.num_rows() : 0;
      CUDF_EXPECTS(num_destination_rows ==
                     thrust::count_if(rmm::exec_policy()->on(0),
                                      gather_map.begin<map_type>(),
                                      gather_map.end<map_type>(),
                                      bounds_checker<map_type>{begin, source_table.num_rows()}),
                   "Index out of bounds.");
    }

    if (neg_indices == negative_index_policy::ALLOWED) {
      auto idx_converter = index_converter<map_type>{source_table.num_rows()};
      return gather(source_table,
                    thrust::make_transform_iterator(gather_map.begin<map_type>(), idx_converter),
                    thrust::make_transform_iterator(gather_map.end<map_type>(), idx_converter),
                    bounds == out_of_bounds_policy::IGNORE,
                    mr,
                    stream);
    } else {
      return gather(source_table,
                    gather_map.begin<map_type>(),
                    gather_map.end<map_type>(),
                    bounds == out_of_bounds_policy::IGNORE,
                    mr,
                    stream);
    }
  }

  template <typename map_type,
            typename... Args,
            std::enable_if_t<not is_index_type<map_type>()>* = nullptr>
  std::unique_ptr<table> operator()(Args&&... args)
  {
    CUDF_FAIL("Gather map must be an integral type.");
  }
};  // namespace detail

std::unique_ptr<table> gather(table_view const& source_table,
                              column_view const& gather_map,
                              out_of_bounds_policy bounds,
                              negative_index_policy neg_indices,
                              rmm::mr::device_memory_resource* mr,
                              cudaStream_t stream)
{
  CUDF_EXPECTS(gather_map.has_nulls() == false, "gather_map contains nulls");

  std::unique_ptr<table> destination_table = cudf::type_dispatcher(gather_map.type(),
                                                                   dispatch_map_type{},
                                                                   source_table,
                                                                   gather_map,
                                                                   gather_map.size(),
                                                                   bounds,
                                                                   neg_indices,
                                                                   mr,
                                                                   stream);

  return destination_table;
}

}  // namespace detail

std::unique_ptr<table> gather(table_view const& source_table,
                              column_view const& gather_map,
                              bool check_bounds,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  auto index_policy = is_unsigned(gather_map.type()) ? detail::negative_index_policy::NOT_ALLOWED
                                                     : detail::negative_index_policy::ALLOWED;

  return detail::gather(
    source_table,
    gather_map,
    check_bounds ? detail::out_of_bounds_policy::FAIL : detail::out_of_bounds_policy::NULLIFY,
    index_policy,
    mr);
}

}  // namespace cudf
