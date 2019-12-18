#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/aggregation.hpp>
#include <cudf/types.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/iterator/discard_iterator.h>

#include <memory>

namespace cudf {
namespace experimental {
namespace groupby {
namespace detail {


template <aggregation::Kind k>
struct reduce_functor {
  template <typename T>
  std::enable_if_t<std::is_arithmetic<T>::value, std::unique_ptr<column> >
  operator()(column_view const& values,
             rmm::device_vector<cudf::size_type> const& group_labels,
             size_type num_groups,
             cudaStream_t stream)
  {
    using OpType = cudf::experimental::detail::corresponding_operator_t<k>;
    using ResultType = cudf::experimental::detail::target_type_t<T, k>;

    auto result = make_numeric_column(data_type(type_to_id<ResultType>()), 
                                      num_groups);
    auto op = OpType{};

    if (values.nullable()) {
      T default_value = OpType::template identity<T>();
      auto device_values = column_device_view::create(values);
      auto val_it = cudf::experimental::detail::make_null_replacement_iterator(
                        *device_values, default_value);
      
      thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                            group_labels.begin(), group_labels.end(),
                            val_it, thrust::make_discard_iterator(),
                            result->mutable_view().begin<ResultType>(),
                            thrust::equal_to<size_type>(), op);
    } else {
      auto it = thrust::make_transform_iterator(values.data<T>(),
                            [] __device__ (auto i) { return i; });
      thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                            group_labels.begin(), group_labels.end(),
                            it, thrust::make_discard_iterator(),
                            result->mutable_view().begin<ResultType>(),
                            thrust::equal_to<size_type>(), op);
    }
    
    return result;
  }

  template <typename T, typename... Args>
  std::enable_if_t<!std::is_arithmetic<T>::value, std::unique_ptr<column> >
  operator()(Args&&... args) {
    CUDF_FAIL("Only numeric types are supported in variance");
  }
};

// TODO (dm): take memory resource
std::unique_ptr<column> group_sum(
    column_view const& values,
    rmm::device_vector<size_type> const& group_labels,
    size_type num_groups,
    cudaStream_t stream = 0);

std::unique_ptr<column> group_var(
    column_view const& values,
    column_view const& group_means,
    rmm::device_vector<size_type> const& group_labels,
    rmm::device_vector<size_type> const& group_sizes,
    size_type ddof,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream = 0);

/**
 * @brief Internal API to calculate groupwise quantiles
 * 
 * @param values Grouped and sorted (within group) values to get quantiles from
 * @param group_offsets Offsets of groups' starting points within @p values
 * @param group_sizes Number of valid elements per group
 * @param quantiles List of quantiles q where q lies in [0,1]
 * @param interp Method to use when desired value lies between data points
 * @param stream Stream to perform computation in
 */
std::unique_ptr<column> group_quantiles(
    column_view const& values,
    rmm::device_vector<size_type> const& group_offsets,
    rmm::device_vector<size_type> const& group_sizes,
    std::vector<double> const& quantiles,
    interpolation interp,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream = 0);

}  // namespace detail
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
