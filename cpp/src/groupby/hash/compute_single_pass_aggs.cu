/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_single_pass_aggs.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/utilities/element_argminmax.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/reduction/detail/sum_overflow.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/memory_resource>
#include <cuda/std/algorithm>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/stream>
#include <thrust/adjacent_difference.h>
#include <thrust/scan.h>
#include <thrust/tabulate.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

namespace cudf::groupby::detail::hash {
namespace {

/// Reads a fixed-width element, going through the keys when the column is a dictionary.
template <typename T>
struct value_accessor {
  column_device_view col;
  bool is_dictionary;

  __device__ T operator()(size_type row) const
  {
    if (is_dictionary) {
      auto const keys = col.child(dictionary_column_view::keys_column_index);
      return keys.element<T>(static_cast<size_type>(col.element<dictionary32>(row)));
    }
    return col.element<T>(row);
  }
};

/// Maps a grouped position to the value of the input row at that position, substituting
/// `null_value` for null rows and optionally squaring the value.
template <typename Source, typename Target, bool Square>
struct grouped_value_fn {
  size_type const* grouped_rows;
  value_accessor<Source> value;
  Target null_value;
  bool has_nulls;

  __device__ Target operator()(size_type position) const
  {
    auto const row = grouped_rows[position];
    if (has_nulls && value.col.is_null_nocheck(row)) { return null_value; }
    auto const result = static_cast<Target>(value(row));
    if constexpr (Square) { return result * result; }
    return result;
  }
};

/// Maps a grouped position to the validity of the input row at that position.
struct grouped_validity_fn {
  size_type const* grouped_rows;
  column_device_view col;

  __device__ bool operator()(size_type position) const
  {
    return col.is_valid_nocheck(grouped_rows[position]);
  }
};

/// Maps a grouped position to a SUM_OVERFLOW accumulator, treating nulls as a zero contribution.
template <typename DeviceType>
struct grouped_sum_overflow_fn {
  size_type const* grouped_rows;
  value_accessor<DeviceType> value;
  bool has_nulls;

  __device__ cudf::reduction::detail::sum_overflow_result<DeviceType> operator()(
    size_type position) const
  {
    auto const row = grouped_rows[position];
    if (has_nulls && value.col.is_null_nocheck(row)) { return {DeviceType{0}, 0}; }
    return {value(row), 0};
  }
};

/// Splits a reduced accumulator into the sum and overflow-flag children of the output struct.
template <typename DeviceType>
struct split_sum_overflow_fn {
  __device__ cuda::std::tuple<DeviceType, bool> operator()(
    cudf::reduction::detail::sum_overflow_result<DeviceType> const& accumulator) const
  {
    return {accumulator.sum, accumulator.wraps != 0};
  }
};

/// Sums accumulated together when several additive aggregations are requested on one column.
template <typename Result>
struct fused_sums {
  Result sum;
  Result sum_of_squares;
  size_type count;
};

template <typename Result>
struct fused_sums_plus {
  __device__ fused_sums<Result> operator()(fused_sums<Result> const& lhs,
                                           fused_sums<Result> const& rhs) const
  {
    return {lhs.sum + rhs.sum, lhs.sum_of_squares + rhs.sum_of_squares, lhs.count + rhs.count};
  }
};

/// Maps a grouped position to the sums contributed by the input row at that position.
template <typename Source, typename Result>
struct grouped_fused_sums_fn {
  size_type const* grouped_rows;
  value_accessor<Source> value;
  bool has_nulls;

  __device__ fused_sums<Result> operator()(size_type position) const
  {
    auto const row = grouped_rows[position];
    if (has_nulls && value.col.is_null_nocheck(row)) { return {Result{0}, Result{0}, 0}; }
    auto const result = static_cast<Result>(value(row));
    return {result, result * result, 1};
  }
};

/// Splits the reduced sums into the SUM, SUM_OF_SQUARES and COUNT_VALID outputs.
template <typename Result>
struct split_fused_sums_fn {
  __device__ cuda::std::tuple<Result, Result, size_type> operator()(
    fused_sums<Result> const& sums) const
  {
    return {sums.sum, sums.sum_of_squares, sums.count};
  }
};

/// The by-key scan slows down with wide accumulators, so on that path only the two sums are fused
/// and the valid count is reduced separately.
template <typename Result>
using fused_sum_pair = cuda::std::pair<Result, Result>;

template <typename Result>
struct fused_sum_pair_plus {
  __device__ fused_sum_pair<Result> operator()(fused_sum_pair<Result> const& lhs,
                                               fused_sum_pair<Result> const& rhs) const
  {
    return {lhs.first + rhs.first, lhs.second + rhs.second};
  }
};

template <typename Source, typename Result>
struct grouped_fused_sum_pair_fn {
  size_type const* grouped_rows;
  value_accessor<Source> value;
  bool has_nulls;

  __device__ fused_sum_pair<Result> operator()(size_type position) const
  {
    auto const row = grouped_rows[position];
    if (has_nulls && value.col.is_null_nocheck(row)) { return {Result{0}, Result{0}}; }
    auto const result = static_cast<Result>(value(row));
    return {result, result * result};
  }
};

template <typename Result>
struct split_fused_sum_pair_fn {
  __device__ cuda::std::tuple<Result, Result> operator()(fused_sum_pair<Result> const& sums) const
  {
    return {sums.first, sums.second};
  }
};

constexpr bool is_fusable_sum(aggregation::Kind kind)
{
  return kind == aggregation::SUM || kind == aggregation::SUM_OF_SQUARES ||
         kind == aggregation::COUNT_VALID;
}

/// Groups this small on average are reduced by key, since one block per segment would leave most
/// of the device idle.
constexpr size_type min_avg_rows_per_segment = 96;

/// Groups longer than this are reduced chunk by chunk so that every block has a bounded range.
constexpr size_type rows_per_chunk = 1 << 14;

template <typename ValueIterator, typename OutputIterator, typename Op, typename T>
void reduce_segments(device_span<size_type const> offsets,
                     ValueIterator values,
                     OutputIterator output,
                     Op op,
                     T init,
                     cuda::stream_ref stream)
{
  auto const env =
    cuda::std::execution::env{cuda::std::execution::prop{cuda::get_stream_t{}, stream},
                              cuda::std::execution::prop{cuda::mr::get_memory_resource_t{},
                                                         cudf::get_current_device_resource_ref()}};
  CUDF_CUDA_TRY(
    cub::DeviceSegmentedReduce::Reduce(values,
                                       output,
                                       static_cast<cuda::std::int64_t>(offsets.size() - 1),
                                       offsets.begin(),
                                       offsets.begin() + 1,
                                       op,
                                       init,
                                       env));
}

/// Reduces the grouped values of every group into one output element per group.
template <typename ValueIterator, typename OutputIterator, typename Op, typename T>
void reduce_groups(grouped_rows const& grouped,
                   ValueIterator values,
                   OutputIterator output,
                   Op op,
                   T init,
                   cuda::stream_ref stream)
{
  if (!grouped.labels.is_empty()) {
    auto const env = cuda::std::execution::env{
      cuda::std::execution::prop{cuda::get_stream_t{}, stream},
      cuda::std::execution::prop{cuda::mr::get_memory_resource_t{},
                                 cudf::get_current_device_resource_ref()}};
    CUDF_CUDA_TRY(cub::DeviceReduce::ReduceByKey(grouped.labels.begin(),
                                                 cuda::make_discard_iterator(),
                                                 values,
                                                 output,
                                                 cuda::make_discard_iterator(),
                                                 op,
                                                 grouped.labels.size(),
                                                 env));
    return;
  }
  if (grouped.group_chunks.is_empty()) {
    reduce_segments(grouped.offsets, values, output, op, init, stream);
    return;
  }
  rmm::device_uvector<T> partials(
    grouped.chunk_offsets.size() - 1, stream, cudf::get_current_device_resource_ref());
  reduce_segments(grouped.chunk_offsets, values, partials.begin(), op, init, stream);
  reduce_segments(grouped.group_chunks, partials.begin(), output, op, init, stream);
}

struct reduction_context {
  column_view const& values;
  column_device_view const& d_values;
  data_type values_type;  ///< Type of the values, or of the keys for dictionary values
  grouped_rows const& grouped;
  size_type num_groups;
  bool nullable;  ///< Whether the result carries a null mask
  cuda::stream_ref stream;
  rmm::device_async_resource_ref mr;

  template <typename T>
  value_accessor<T> accessor() const
  {
    return {d_values, is_dictionary(values.type())};
  }
};

/// A group is valid when any of its rows is valid.
std::pair<rmm::device_buffer, size_type> reduce_group_validity(reduction_context const& ctx)
{
  rmm::device_uvector<bool> group_valid(
    ctx.num_groups, ctx.stream, cudf::get_current_device_resource_ref());
  reduce_groups(ctx.grouped,
                cudf::detail::make_counting_transform_iterator(
                  0, grouped_validity_fn{ctx.grouped.rows.data(), ctx.d_values}),
                group_valid.begin(),
                cuda::std::logical_or<bool>{},
                false,
                ctx.stream);
  return cudf::detail::valid_if(
    group_valid.begin(), group_valid.end(), cuda::std::identity{}, ctx.stream, ctx.mr);
}

void set_group_null_mask(column& result, reduction_context const& ctx)
{
  if (!ctx.nullable || ctx.num_groups == 0) { return; }
  auto [null_mask, null_count] = reduce_group_validity(ctx);
  result.set_null_mask(std::move(null_mask), null_count);
}

std::unique_ptr<column> make_size_type_column(reduction_context const& ctx)
{
  return make_fixed_width_column(data_type{type_to_id<size_type>()},
                                 ctx.num_groups,
                                 mask_state::UNALLOCATED,
                                 ctx.stream,
                                 ctx.mr);
}

std::unique_ptr<column> count_groups(reduction_context const& ctx, bool valid_only)
{
  auto result = make_size_type_column(ctx);
  if (ctx.num_groups == 0) { return result; }

  if (valid_only && ctx.values.has_nulls()) {
    auto const valid_counts = cudf::detail::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<size_type>(
        [is_valid = grouped_validity_fn{ctx.grouped.rows.data(), ctx.d_values}] __device__(
          size_type position) { return static_cast<size_type>(is_valid(position)); }));
    reduce_groups(ctx.grouped,
                  valid_counts,
                  result->mutable_view().begin<size_type>(),
                  cuda::std::plus<size_type>{},
                  size_type{0},
                  ctx.stream);
  } else {
    thrust::adjacent_difference(
      rmm::exec_policy_nosync(ctx.stream, cudf::get_current_device_resource_ref()),
      ctx.grouped.offsets.begin() + 1,
      ctx.grouped.offsets.end(),
      result->mutable_view().begin<size_type>());
  }
  return result;
}

/// The device representation of a column element. Chrono and fixed-point columns reduce as their
/// integer reps, so the reduction kernels are only instantiated once per representation.
template <typename T>
struct rep_type {
  using type = device_storage_type_t<T>;
};

template <typename T>
  requires(cudf::is_chrono<T>())
struct rep_type<T> {
  using type = typename T::rep;
};

template <typename T>
using rep_type_t = typename rep_type<T>::type;

template <aggregation::Kind K, typename T>
constexpr bool is_reduction_supported()
{
  switch (K) {
    case aggregation::SUM:
      return cudf::is_numeric<T>() || cudf::is_duration<T>() || cudf::is_fixed_point<T>();
    case aggregation::PRODUCT:
    case aggregation::SUM_OF_SQUARES: return cudf::detail::is_product_supported<T>();
    case aggregation::MIN:
    case aggregation::MAX: return cudf::is_fixed_width<T>() && is_relationally_comparable<T, T>();
    case aggregation::ARGMIN:
    case aggregation::ARGMAX: return is_relationally_comparable<T, T>();
    case aggregation::SUM_OVERFLOW: return cudf::detail::sum_overflow_supported<T>;
    default: return false;
  }
}

template <aggregation::Kind K>
struct reduce_fn {
  template <typename T>
    requires(is_reduction_supported<K, T>() &&
             (K == aggregation::SUM || K == aggregation::PRODUCT ||
              K == aggregation::SUM_OF_SQUARES || K == aggregation::MIN || K == aggregation::MAX))
  std::unique_ptr<column> operator()(reduction_context const& ctx) const
  {
    using Source = rep_type_t<T>;
    using Result = rep_type_t<cudf::detail::target_type_t<T, K>>;
    using Op     = cudf::detail::corresponding_operator_t<K>;

    auto result = make_fixed_width_column(cudf::detail::target_type(ctx.values_type, K),
                                          ctx.num_groups,
                                          mask_state::UNALLOCATED,
                                          ctx.stream,
                                          ctx.mr);
    if (ctx.num_groups == 0) { return result; }

    using value_fn      = grouped_value_fn<Source, Result, K == aggregation::SUM_OF_SQUARES>;
    auto const identity = Op::template identity<Result>();
    auto const values   = cudf::detail::make_counting_transform_iterator(
      0,
      value_fn{ctx.grouped.rows.data(), ctx.accessor<Source>(), identity, ctx.values.has_nulls()});
    reduce_groups(
      ctx.grouped, values, result->mutable_view().begin<Result>(), Op{}, identity, ctx.stream);
    set_group_null_mask(*result, ctx);
    return result;
  }

  template <typename T>
    requires(is_reduction_supported<K, T>() &&
             (K == aggregation::ARGMIN || K == aggregation::ARGMAX))
  std::unique_ptr<column> operator()(reduction_context const& ctx) const
  {
    auto result = make_size_type_column(ctx);
    if (ctx.num_groups == 0) { return result; }

    // The grouped rows are the input row indices themselves, so reducing them with the
    // element comparator yields the input index of each group's extremum. The sentinel identity
    // loses against every valid row and is left in place for all-null groups.
    constexpr auto is_argmin = K == aggregation::ARGMIN;
    reduce_groups(ctx.grouped,
                  ctx.grouped.rows.begin(),
                  result->mutable_view().begin<size_type>(),
                  cudf::detail::element_argminmax_fn<rep_type_t<T>>{
                    ctx.d_values, ctx.values.has_nulls(), is_argmin},
                  is_argmin ? cudf::detail::ARGMIN_SENTINEL : cudf::detail::ARGMAX_SENTINEL,
                  ctx.stream);
    set_group_null_mask(*result, ctx);
    return result;
  }

  template <typename T>
    requires(is_reduction_supported<K, T>() && K == aggregation::SUM_OVERFLOW)
  std::unique_ptr<column> operator()(reduction_context const& ctx) const
  {
    using Source      = rep_type_t<T>;
    using accumulator = cudf::reduction::detail::sum_overflow_result<Source>;

    auto sum_child = make_fixed_width_column(
      ctx.values_type, ctx.num_groups, mask_state::UNALLOCATED, ctx.stream, ctx.mr);
    auto overflow_child = make_fixed_width_column(
      data_type{type_id::BOOL8}, ctx.num_groups, mask_state::UNALLOCATED, ctx.stream, ctx.mr);
    if (ctx.num_groups > 0) {
      auto const values = cudf::detail::make_counting_transform_iterator(
        0,
        grouped_sum_overflow_fn<Source>{
          ctx.grouped.rows.data(), ctx.accessor<Source>(), ctx.values.has_nulls()});
      auto const children = cuda::transform_output_iterator{
        cuda::make_zip_iterator(sum_child->mutable_view().begin<Source>(),
                                overflow_child->mutable_view().begin<bool>()),
        split_sum_overflow_fn<Source>{}};
      reduce_groups(ctx.grouped,
                    values,
                    children,
                    cudf::reduction::detail::overflow_sum_op<Source>{},
                    accumulator{},
                    ctx.stream);
    }

    auto [null_mask, null_count] = ctx.nullable && ctx.num_groups > 0
                                     ? reduce_group_validity(ctx)
                                     : std::pair{rmm::device_buffer{}, size_type{0}};
    std::vector<std::unique_ptr<column>> children;
    children.push_back(std::move(sum_child));
    children.push_back(std::move(overflow_child));
    return create_structs_hierarchy(
      ctx.num_groups, std::move(children), null_count, std::move(null_mask), ctx.stream, ctx.mr);
  }

  template <typename T>
    requires(!is_reduction_supported<K, T>())
  std::unique_ptr<column> operator()(reduction_context const&) const
  {
    CUDF_FAIL("Unsupported type for hash groupby aggregation");
  }
};

/// Computes the SUM, SUM_OF_SQUARES and COUNT_VALID aggregations requested on one column, as
/// extracted for MEAN, M2, VARIANCE and STD, with a single segmented reduction.
struct fused_sums_fn {
  template <typename T>
    requires(cudf::detail::is_product_supported<T>())
  std::vector<std::unique_ptr<column>> operator()(reduction_context const& ctx,
                                                  host_span<aggregation::Kind const> kinds,
                                                  std::span<int8_t const> is_intermediate) const
  {
    using Source = rep_type_t<T>;
    using Result = rep_type_t<cudf::detail::target_type_t<T, aggregation::SUM>>;
    static_assert(
      cuda::std::
        is_same_v<Result, rep_type_t<cudf::detail::target_type_t<T, aggregation::SUM_OF_SQUARES>>>);

    // Every sum is reduced even when it is not requested; those land in temporary columns.
    auto const make_output = [&](aggregation::Kind kind) {
      auto const requested = std::find(kinds.begin(), kinds.end(), kind) != kinds.end();
      return make_fixed_width_column(cudf::detail::target_type(ctx.values_type, kind),
                                     ctx.num_groups,
                                     mask_state::UNALLOCATED,
                                     ctx.stream,
                                     requested ? ctx.mr : cudf::get_current_device_resource_ref());
    };
    auto sum            = make_output(aggregation::SUM);
    auto sum_of_squares = make_output(aggregation::SUM_OF_SQUARES);
    auto count          = !ctx.grouped.labels.is_empty() ? count_groups(ctx, true)
                                                         : make_output(aggregation::COUNT_VALID);
    auto const counts   = count->view().template begin<size_type>();
    if (ctx.num_groups > 0 && !ctx.grouped.labels.is_empty()) {
      auto const values = cudf::detail::make_counting_transform_iterator(
        0,
        grouped_fused_sum_pair_fn<Source, Result>{
          ctx.grouped.rows.data(), ctx.accessor<Source>(), ctx.values.has_nulls()});
      auto const outputs = cuda::transform_output_iterator{
        cuda::make_zip_iterator(sum->mutable_view().template begin<Result>(),
                                sum_of_squares->mutable_view().template begin<Result>()),
        split_fused_sum_pair_fn<Result>{}};
      reduce_groups(ctx.grouped,
                    values,
                    outputs,
                    fused_sum_pair_plus<Result>{},
                    fused_sum_pair<Result>{Result{0}, Result{0}},
                    ctx.stream);
    } else if (ctx.num_groups > 0) {
      auto const values = cudf::detail::make_counting_transform_iterator(
        0,
        grouped_fused_sums_fn<Source, Result>{
          ctx.grouped.rows.data(), ctx.accessor<Source>(), ctx.values.has_nulls()});
      auto const outputs = cuda::transform_output_iterator{
        cuda::make_zip_iterator(sum->mutable_view().template begin<Result>(),
                                sum_of_squares->mutable_view().template begin<Result>(),
                                count->mutable_view().template begin<size_type>()),
        split_fused_sums_fn<Result>{}};
      reduce_groups(ctx.grouped,
                    values,
                    outputs,
                    fused_sums_plus<Result>{},
                    fused_sums<Result>{Result{0}, Result{0}, 0},
                    ctx.stream);
    }

    std::vector<std::unique_ptr<column>> results;
    for (std::size_t i = 0; i < kinds.size(); ++i) {
      auto result = kinds[i] == aggregation::SUM              ? std::move(sum)
                    : kinds[i] == aggregation::SUM_OF_SQUARES ? std::move(sum_of_squares)
                                                              : std::move(count);
      // A sum is null when its group has no valid row, which the valid count already tells.
      auto const nullable =
        !is_intermediate[i] && kinds[i] != aggregation::COUNT_VALID && ctx.values.has_nulls();
      if (nullable && ctx.num_groups > 0) {
        auto [null_mask, null_count] = cudf::detail::valid_if(
          counts,
          counts + ctx.num_groups,
          [] __device__(size_type count) { return count > 0; },
          ctx.stream,
          ctx.mr);
        result->set_null_mask(std::move(null_mask), null_count);
      }
      results.push_back(std::move(result));
    }
    return results;
  }

  template <typename T>
    requires(!cudf::detail::is_product_supported<T>())
  std::vector<std::unique_ptr<column>> operator()(reduction_context const&,
                                                  host_span<aggregation::Kind const>,
                                                  std::span<int8_t const>) const
  {
    CUDF_FAIL("Unsupported type for fused hash groupby sums");
  }
};

/// Calls `f.template operator()<K>()` for the reduction kind `kind`.
template <typename F>
auto dispatch_reduction_kind(aggregation::Kind kind, F&& f)
{
  switch (kind) {
    case aggregation::SUM: return f.template operator()<aggregation::SUM>();
    case aggregation::PRODUCT: return f.template operator()<aggregation::PRODUCT>();
    case aggregation::SUM_OF_SQUARES: return f.template operator()<aggregation::SUM_OF_SQUARES>();
    case aggregation::MIN: return f.template operator()<aggregation::MIN>();
    case aggregation::MAX: return f.template operator()<aggregation::MAX>();
    case aggregation::ARGMIN: return f.template operator()<aggregation::ARGMIN>();
    case aggregation::ARGMAX: return f.template operator()<aggregation::ARGMAX>();
    case aggregation::SUM_OVERFLOW: return f.template operator()<aggregation::SUM_OVERFLOW>();
    default: CUDF_FAIL("Unsupported hash groupby aggregation");
  }
}

struct compute_reduction_fn {
  reduction_context const& ctx;

  template <aggregation::Kind K>
  std::unique_ptr<column> operator()() const
  {
    return type_dispatcher(ctx.values_type, reduce_fn<K>{}, ctx);
  }
};

template <aggregation::Kind K>
struct is_reduction_supported_fn {
  template <typename T>
  bool operator()() const
  {
    return is_reduction_supported<K, T>();
  }
};

struct is_reduction_kind_supported_fn {
  data_type values_type;

  template <aggregation::Kind K>
  bool operator()() const
  {
    return type_dispatcher(values_type, is_reduction_supported_fn<K>{});
  }
};

std::unique_ptr<column> compute_aggregation(aggregation::Kind kind, reduction_context const& ctx)
{
  switch (kind) {
    case aggregation::COUNT_VALID: return count_groups(ctx, true);
    case aggregation::COUNT_ALL: return count_groups(ctx, false);
    default: return dispatch_reduction_kind(kind, compute_reduction_fn{ctx});
  }
}

}  // namespace

bool is_single_pass_agg_supported(data_type values_type, aggregation::Kind kind)
{
  // Values of STRUCT and LIST types are not aggregated by the hash groupby.
  if (cudf::is_nested(values_type)) { return false; }
  switch (kind) {
    case aggregation::COUNT_VALID:
    case aggregation::COUNT_ALL: return true;
    case aggregation::SUM:
    case aggregation::PRODUCT:
    case aggregation::SUM_OF_SQUARES:
    case aggregation::MIN:
    case aggregation::MAX:
    case aggregation::ARGMIN:
    case aggregation::ARGMAX:
    case aggregation::SUM_OVERFLOW:
      return dispatch_reduction_kind(kind, is_reduction_kind_supported_fn{values_type});
    default: return false;
  }
}

grouped_rows make_grouped_rows(device_span<size_type const> rows,
                               device_span<size_type const> offsets,
                               cuda::stream_ref stream)
{
  auto const temp_mr    = cudf::get_current_device_resource_ref();
  auto const num_rows   = static_cast<size_type>(rows.size());
  auto const num_groups = static_cast<size_type>(offsets.size()) - 1;
  grouped_rows grouped{rows,
                       offsets,
                       rmm::device_uvector<size_type>{0, stream, temp_mr},
                       rmm::device_uvector<size_type>{0, stream, temp_mr},
                       rmm::device_uvector<size_type>{0, stream, temp_mr}};
  if (num_groups == 0) { return grouped; }

  if (num_rows / num_groups < min_avg_rows_per_segment) {
    grouped.labels.resize(num_rows, stream);
    cudf::detail::label_segments(
      offsets.begin(), offsets.end(), grouped.labels.begin(), grouped.labels.end(), stream);
    return grouped;
  }

  auto const policy       = rmm::exec_policy_nosync(stream, temp_mr);
  auto const chunk_counts = cudf::detail::make_counting_transform_iterator(
    0, [offsets = offsets.begin()] __device__(size_type group) -> size_type {
      return cudf::util::div_rounding_up_safe(offsets[group + 1] - offsets[group], rows_per_chunk);
    });
  grouped.group_chunks.resize(num_groups + 1, stream);
  grouped.group_chunks.set_element_to_zero_async(0, stream);
  thrust::inclusive_scan(
    policy, chunk_counts, chunk_counts + num_groups, grouped.group_chunks.begin() + 1);
  auto const num_chunks = grouped.group_chunks.back_element(stream);
  if (num_chunks == num_groups) {
    // No group spans several chunks, so the groups are the segments.
    grouped.group_chunks.resize(0, stream);
    return grouped;
  }

  grouped.chunk_offsets.resize(num_chunks + 1, stream);
  thrust::tabulate(policy,
                   grouped.chunk_offsets.begin(),
                   grouped.chunk_offsets.end(),
                   [offsets      = offsets.begin(),
                    group_chunks = grouped.group_chunks.begin(),
                    num_groups,
                    num_chunks,
                    num_rows] __device__(size_type chunk) -> size_type {
                     if (chunk == num_chunks) { return num_rows; }
                     auto const group = static_cast<size_type>(
                       cuda::std::upper_bound(group_chunks, group_chunks + num_groups + 1, chunk) -
                       group_chunks - 1);
                     return offsets[group] + (chunk - group_chunks[group]) * rows_per_chunk;
                   });
  return grouped;
}

std::vector<std::unique_ptr<column>> compute_single_pass_aggs(
  table_view const& values,
  host_span<aggregation::Kind const> agg_kinds,
  std::span<int8_t const> is_agg_intermediate,
  grouped_rows const& grouped,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(values.num_columns() == static_cast<size_type>(agg_kinds.size()),
               "The number of values columns and aggregation kinds must be the same.");
  CUDF_EXPECTS(values.num_columns() == static_cast<size_type>(is_agg_intermediate.size()),
               "The number of values columns and intermediate flags must be the same.");

  auto const num_groups = static_cast<size_type>(grouped.offsets.size()) - 1;
  auto const num_aggs   = agg_kinds.size();

  // Returns one past the last of the consecutive additive aggregations on the column of `begin`
  // that can be computed together with the aggregation at `begin`.
  auto const fused_end = [&](std::size_t begin, data_type values_type) {
    auto const& col = values.column(begin);
    if (!is_fusable_sum(agg_kinds[begin]) ||
        !is_single_pass_agg_supported(values_type, aggregation::SUM_OF_SQUARES)) {
      return begin + 1;
    }
    auto end = begin + 1;
    while (end < num_aggs && is_fusable_sum(agg_kinds[end]) &&
           cudf::detail::is_shallow_equivalent(col, values.column(end)) &&
           std::find(agg_kinds.begin() + begin, agg_kinds.begin() + end, agg_kinds[end]) ==
             agg_kinds.begin() + end) {
      ++end;
    }
    // By key, only the two sums are fused, so fusing pays off only when both are requested.
    auto const has_kind = [&](aggregation::Kind kind) {
      return std::find(agg_kinds.begin() + begin, agg_kinds.begin() + end, kind) !=
             agg_kinds.begin() + end;
    };
    if (!grouped.labels.is_empty() &&
        !(has_kind(aggregation::SUM) && has_kind(aggregation::SUM_OF_SQUARES))) {
      return begin + 1;
    }
    return end;
  };

  std::vector<std::unique_ptr<column>> results;
  results.reserve(num_aggs);
  for (std::size_t i = 0; i < num_aggs;) {
    auto const& col  = values.column(i);
    auto const d_col = column_device_view::create(col, stream);
    auto const values_type =
      is_dictionary(col.type()) ? dictionary_column_view(col).keys().type() : col.type();
    auto const kind = agg_kinds[i];
    // Counts are never null, and intermediate results skip the null mask to avoid the extra work.
    auto const nullable = !is_agg_intermediate[i] && kind != aggregation::COUNT_VALID &&
                          kind != aggregation::COUNT_ALL && col.has_nulls();
    auto const ctx =
      reduction_context{col, *d_col, values_type, grouped, num_groups, nullable, stream, mr};

    auto const end = fused_end(i, values_type);
    if (end > i + 1) {
      auto fused =
        type_dispatcher(values_type,
                        fused_sums_fn{},
                        ctx,
                        host_span<aggregation::Kind const>{agg_kinds}.subspan(i, end - i),
                        is_agg_intermediate.subspan(i, end - i));
      std::move(fused.begin(), fused.end(), std::back_inserter(results));
    } else {
      results.push_back(compute_aggregation(kind, ctx));
    }
    i = end;
  }
  return results;
}

}  // namespace cudf::groupby::detail::hash
