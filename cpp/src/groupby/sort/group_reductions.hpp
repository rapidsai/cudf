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
  static constexpr bool is_supported(){
    if (cudf::is_numeric<T>())
      return true;
    else if (cudf::is_timestamp<T>() and k == aggregation::MIN)
      return true;
    else
      return false;
  }

  template <typename T>
  std::enable_if_t<is_supported<T>(), std::unique_ptr<column> >
  operator()(column_view const& values,
             column_view const& group_sizes,
             rmm::device_vector<cudf::size_type> const& group_labels,
             cudaStream_t stream)
  {
    using OpType = cudf::experimental::detail::corresponding_operator_t<k>;
    using ResultType = cudf::experimental::detail::target_type_t<T, k>;
    size_type num_groups = group_sizes.size();

    auto result = make_fixed_width_column(data_type(type_to_id<ResultType>()), 
                                          num_groups,
                                          values.nullable() 
                                            ? mask_state::UNINITIALIZED
                                            : mask_state::UNALLOCATED);
    auto op = OpType{};

    if (values.size() == 0) {
      return result;
    }

    if (values.nullable()) {
      T default_value = OpType::template identity<T>();
      auto device_values = column_device_view::create(values);
      auto val_it = cudf::experimental::detail::make_null_replacement_iterator(
                        *device_values, default_value);

      // Without this transform, thrust throws a runtime error
      auto it = thrust::make_transform_iterator(val_it,
                            [] __device__ (auto i) { return i; });
      
      thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
        // Input keys
          group_labels.begin(), group_labels.end(),
        // Input values
          it,
        // Output keys
          thrust::make_discard_iterator(),
        // Output values
          result->mutable_view().begin<ResultType>(),
        // comparator and operation
          thrust::equal_to<size_type>(), op);

      auto result_view = mutable_column_device_view::create(*result);
      auto group_size_view = column_device_view::create(group_sizes);

      thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator(0), group_sizes.size(),
        [d_result=*result_view, d_group_sizes=*group_size_view]
        __device__ (size_type i){
          size_type group_size = d_group_sizes.element<size_type>(i);
          if (group_size == 0)
            d_result.set_null(i);
          else
            d_result.set_valid(i);
        });
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
  std::enable_if_t<not is_supported<T>(), std::unique_ptr<column> >
  operator()(Args&&... args) {
    CUDF_FAIL("Only numeric types are supported in variance");
  }
};

// TODO (dm): take memory resource
std::unique_ptr<column> group_sum(
    column_view const& values,
    column_view const& group_sizes,
    rmm::device_vector<size_type> const& group_labels,
    cudaStream_t stream = 0);

std::unique_ptr<column> group_min(
    column_view const& values,
    column_view const& group_sizes,
    rmm::device_vector<size_type> const& group_labels,
    cudaStream_t stream = 0);

std::unique_ptr<column> group_count(
    column_view const& values,
    rmm::device_vector<size_type> const& group_labels,
    size_type num_groups,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream = 0);

std::unique_ptr<column> group_var(
    column_view const& values,
    column_view const& group_means,
    column_view const& group_sizes,
    rmm::device_vector<size_type> const& group_labels,
    size_type ddof,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream = 0);

/**
 * @brief Internal API to calculate groupwise quantiles
 * 
 * @param values Grouped and sorted (within group) values to get quantiles from
 * @param group_sizes Number of valid elements per group
 * @param group_offsets Offsets of groups' starting points within @p values
 * @param quantiles List of quantiles q where q lies in [0,1]
 * @param interp Method to use when desired value lies between data points
 * @param stream Stream to perform computation in
 */
std::unique_ptr<column> group_quantiles(
    column_view const& values,
    column_view const& group_sizes,
    rmm::device_vector<size_type> const& group_offsets,
    std::vector<double> const& quantiles,
    interpolation interp,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream = 0);

}  // namespace detail
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
