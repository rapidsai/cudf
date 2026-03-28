/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tdigest_aggregation.cuh"

namespace cudf {
namespace tdigest {
namespace detail {

bool is_cpu_cluster_computation_disabled = false;

namespace {

// make a centroid from a scalar with a weight of 1.
template <typename T>
struct make_centroid {
  column_device_view const col;

  centroid operator() __device__(size_type index) const
  {
    auto const is_valid = col.is_valid(index);
    auto const mean     = is_valid ? convert_to_floating<double>(col.element<T>(index)) : 0.0;
    auto const weight   = is_valid ? 1.0 : 0.0;
    return {mean, weight, is_valid};
  }
};

// make a centroid from a scalar with a weight of 1. this functor
// assumes any value index it is passed is not null
template <typename T>
struct make_centroid_no_nulls {
  column_device_view const col;

  centroid operator() __device__(size_type index) const
  {
    return {convert_to_floating<double>(col.element<T>(index)), 1.0, true};
  }
};

// return the min/max value of scalar inputs by group index
template <typename T>
struct get_scalar_minmax_grouped {
  column_device_view const col;
  device_span<size_type const> group_offsets;
  size_type const* group_valid_counts;

  __device__ cuda::std::tuple<double, double> operator()(size_type group_index)
  {
    auto const valid_count = group_valid_counts[group_index];
    return valid_count > 0
             ? cuda::std::make_tuple(
                 convert_to_floating<double>(col.element<T>(group_offsets[group_index])),
                 convert_to_floating<double>(
                   col.element<T>(group_offsets[group_index] + valid_count - 1)))
             : cuda::std::make_tuple(0.0, 0.0);
  }
};

// return the min/max value of scalar inputs
template <typename T>
struct get_scalar_minmax {
  column_device_view const col;
  size_type const valid_count;

  __device__ cuda::std::tuple<double, double> operator()(size_type)
  {
    return valid_count > 0
             ? cuda::std::make_tuple(convert_to_floating<double>(col.element<T>(0)),
                                     convert_to_floating<double>(col.element<T>(valid_count - 1)))
             : cuda::std::make_tuple(0.0, 0.0);
  }
};

struct typed_group_tdigest {
  template <typename T>
  std::unique_ptr<column> operator()(column_view const& col,
                                     cudf::device_span<size_type const> group_offsets,
                                     cudf::device_span<size_type const> group_labels,
                                     cudf::device_span<size_type const> group_valid_counts,
                                     size_type num_groups,
                                     int delta,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
    requires(cudf::is_numeric<T>() || cudf::is_fixed_point<T>())
  {
    // first, generate cluster weight information for each input group
    auto cinfo = [&]() {
      // if we will be at least partially using the CPU here, move the important values into pinned
      // and reference those instead
      if (use_cpu_for_cluster_computation(num_groups)) {
        auto temp_mr = cudf::get_pinned_memory_resource();
        auto p_group_offsets =
          cudf::detail::make_device_uvector_async(group_offsets, stream, temp_mr);
        auto p_group_valid_counts =
          cudf::detail::make_device_uvector_async(group_valid_counts, stream, temp_mr);
        auto ret = generate_group_cluster_info(
          delta,
          num_groups,
          nearest_value_scalar_weights_grouped{p_group_offsets.begin()},
          scalar_group_info_grouped{p_group_valid_counts.begin(), p_group_offsets.begin()},
          cumulative_scalar_weight_grouped{
            cuda::std::span<size_type const>{p_group_offsets.begin(), p_group_offsets.size()}},
          col.null_count() > 0,
          stream,
          mr);
        stream.synchronize();
        return ret;
      }
      return generate_group_cluster_info(
        delta,
        num_groups,
        nearest_value_scalar_weights_grouped{group_offsets.data()},
        scalar_group_info_grouped{group_valid_counts.data(), group_offsets.data()},
        cumulative_scalar_weight_grouped{
          cuda::std::span<size_type const>{group_offsets.data(), group_offsets.size()}},
        col.null_count() > 0,
        stream,
        mr);
    }();

    // device column view. handy because the .element() function
    // automatically handles fixed-point conversions for us
    auto d_col = cudf::column_device_view::create(col, stream);

    // compute min and max columns
    auto min_col = cudf::make_numeric_column(
      data_type{type_id::FLOAT64}, num_groups, mask_state::UNALLOCATED, stream, mr);
    auto max_col = cudf::make_numeric_column(
      data_type{type_id::FLOAT64}, num_groups, mask_state::UNALLOCATED, stream, mr);
    thrust::transform(
      rmm::exec_policy_nosync(stream),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(0) + num_groups,
      thrust::make_zip_iterator(cuda::std::make_tuple(min_col->mutable_view().begin<double>(),
                                                      max_col->mutable_view().begin<double>())),
      get_scalar_minmax_grouped<T>{*d_col, group_offsets, group_valid_counts.data()});

    // for simple input values, the "centroids" all have a weight of 1.
    auto scalar_to_centroid =
      cudf::detail::make_counting_transform_iterator(0, make_centroid<T>{*d_col});

    // generate the final tdigest
    return compute_tdigests(delta,
                            scalar_to_centroid,
                            scalar_to_centroid + col.size(),
                            cumulative_scalar_weight_grouped{cuda::std::span<size_type const>{
                              group_offsets.begin(), group_offsets.size()}},
                            std::move(min_col),
                            std::move(max_col),
                            cinfo,
                            col.null_count() > 0,
                            stream,
                            mr);
  }

  template <typename T, typename... Args>
  std::unique_ptr<column> operator()(Args&&...)
    requires(!cudf::is_numeric<T>() && !cudf::is_fixed_point<T>())
  {
    CUDF_FAIL("Non-numeric type in group_tdigest");
  }
};

struct typed_reduce_tdigest {
  // this function assumes col is sorted in ascending order with nulls at the end
  template <typename T>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     int delta,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
    requires(cudf::is_numeric<T>() || cudf::is_fixed_point<T>())
  {
    CUDF_FUNC_RANGE();

    // treat this the same as the groupby path with a single group.  Note:  even though
    // there is only 1 group there are still multiple keys within the group that represent
    // the clustering of (N input values) -> (1 output centroid), so the final computation
    // remains a reduce_by_key() and not a reduce().
    //
    // additionally we get a few optimizations.
    // - since we only ever have 1 "group" that is sorted with nulls at the end,
    //   we can simply process just the non-null values and act as if the column
    //   is non-nullable, allowing us to process fewer values than if we were doing a groupby.
    //
    // - several of the functors used during the reduction are cheaper than during a groupby.

    auto const valid_count = col.size() - col.null_count();

    // first, generate cluster weight information for each input group
    auto cinfo =
      generate_group_cluster_info(delta,
                                  1,
                                  nearest_value_scalar_weights{valid_count},
                                  scalar_group_info{static_cast<double>(valid_count), valid_count},
                                  cumulative_scalar_weight{},
                                  false,
                                  stream,
                                  mr);

    // device column view. handy because the .element() function
    // automatically handles fixed-point conversions for us
    auto d_col = cudf::column_device_view::create(col, stream);

    // compute min and max columns
    auto min_col = cudf::make_numeric_column(
      data_type{type_id::FLOAT64}, 1, mask_state::UNALLOCATED, stream, mr);
    auto max_col = cudf::make_numeric_column(
      data_type{type_id::FLOAT64}, 1, mask_state::UNALLOCATED, stream, mr);
    thrust::transform(
      rmm::exec_policy_nosync(stream),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(0) + 1,
      thrust::make_zip_iterator(cuda::std::make_tuple(min_col->mutable_view().begin<double>(),
                                                      max_col->mutable_view().begin<double>())),
      get_scalar_minmax<T>{*d_col, valid_count});

    // for simple input values, the "centroids" all have a weight of 1.
    auto scalar_to_centroid =
      cudf::detail::make_counting_transform_iterator(0, make_centroid_no_nulls<T>{*d_col});

    // generate the final tdigest and wrap it in a struct_scalar
    return to_tdigest_scalar(compute_tdigests(delta,
                                              scalar_to_centroid,
                                              scalar_to_centroid + valid_count,
                                              cumulative_scalar_weight{},
                                              std::move(min_col),
                                              std::move(max_col),
                                              cinfo,
                                              false,
                                              stream,
                                              mr),
                             stream,
                             mr);
  }

  template <typename T, typename... Args>
  std::unique_ptr<scalar> operator()(Args&&...)
    requires(!cudf::is_numeric<T>() && !cudf::is_fixed_point<T>())
  {
    CUDF_FAIL("Non-numeric type in group_tdigest");
  }
};

}  // anonymous namespace

std::unique_ptr<scalar> reduce_tdigest(column_view const& col,
                                       int max_centroids,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  if (col.size() == 0) { return cudf::tdigest::detail::make_empty_tdigest_scalar(stream, mr); }

  // since this isn't coming out of a groupby, we need to sort the inputs in ascending
  // order with nulls at the end.
  table_view t({col});
  auto sorted = cudf::detail::sort(
    t, {order::ASCENDING}, {null_order::AFTER}, stream, cudf::get_current_device_resource_ref());

  auto const delta = max_centroids;
  return cudf::type_dispatcher(
    col.type(), typed_reduce_tdigest{}, sorted->get_column(0), delta, stream, mr);
}

std::unique_ptr<column> group_tdigest(column_view const& col,
                                      cudf::device_span<size_type const> group_offsets,
                                      cudf::device_span<size_type const> group_labels,
                                      cudf::device_span<size_type const> group_valid_counts,
                                      size_type num_groups,
                                      int max_centroids,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  if (col.size() == 0) { return cudf::tdigest::detail::make_empty_tdigests_column(1, stream, mr); }

  auto const delta = max_centroids;
  return cudf::type_dispatcher(col.type(),
                               typed_group_tdigest{},
                               col,
                               group_offsets,
                               group_labels,
                               group_valid_counts,
                               num_groups,
                               delta,
                               stream,
                               mr);
}

}  // namespace detail
}  // namespace tdigest
}  // namespace cudf
