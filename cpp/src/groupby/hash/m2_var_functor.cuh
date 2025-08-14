/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/tabulate.h>

namespace cudf::groupby::detail::hash {

namespace {

template <typename Source>
__device__ constexpr bool is_m2_var_supported()
{
  return is_numeric<Source>() && !is_fixed_point<Source>();
}

struct m2_functor {
  template <typename Source, typename... Args>
  void operator()(Args...)  //
    requires(!is_m2_var_supported<Source>())
  {
    CUDF_FAIL("Invalid source type for M2 aggregation.");
  }

  template <typename Target, typename SumSqrType, typename SumType, typename CountType>
  void evaluate(Target* target,
                SumSqrType const* sum_sqr,
                SumType const* sum,
                CountType const* count,
                size_type size,
                rmm::cuda_stream_view stream) const noexcept
  {
    thrust::tabulate(rmm::exec_policy_nosync(stream),
                     target,
                     target + size,
                     [sum_sqr, sum, count] __device__(size_type const idx) {
                       auto const group_count = count[idx];
                       if (group_count == 0) { return Target{}; }
                       auto const group_sum_sqr = static_cast<Target>(sum_sqr[idx]);
                       auto const group_sum     = static_cast<Target>(sum[idx]);
                       auto const result = group_sum_sqr - group_sum * group_sum / group_count;
                       return result;
                     });
  }

  template <typename Source>
  void operator()(mutable_column_view const& target,
                  column_view const& sum_sqr,
                  column_view const& sum,
                  column_view const& count,
                  rmm::cuda_stream_view stream) const noexcept
    requires(is_m2_var_supported<Source>())
  {
    using Target     = cudf::detail::target_type_t<Source, aggregation::M2>;
    using SumSqrType = cudf::detail::target_type_t<Source, aggregation::SUM_OF_SQUARES>;
    using SumType    = cudf::detail::target_type_t<Source, aggregation::SUM>;
    using CountType  = cudf::detail::target_type_t<Source, aggregation::COUNT_VALID>;

    // Separate the implementation into another function, which has fewer instantiations since
    // the data types (target/sum/count etc) are mostly the same.
    evaluate(target.begin<Target>(),
             sum_sqr.begin<SumSqrType>(),
             sum.begin<SumType>(),
             count.begin<CountType>(),
             target.size(),
             stream);
  }
};

}  // namespace

inline void compute_m2(data_type source_type,
                       mutable_column_view const& target,
                       column_view const& sum_sqr,
                       column_view const& sum,
                       column_view const& count,
                       rmm::cuda_stream_view stream)
{
  type_dispatcher(source_type, m2_functor{}, target, sum_sqr, sum, count, stream);
}

inline void compute_variance(mutable_column_view const& target,
                             column_view const& m2,
                             column_view const& count,
                             bool* validity,
                             size_type ddof,
                             rmm::cuda_stream_view stream)
{
  using Target    = double;
  using M2Type    = double;
  using CountType = int32_t;

  CUDF_EXPECTS(target.type().id() == cudf::type_to_id<Target>(),
               "Data type of VARIANCE aggregation must be FLOAT64.");
  CUDF_EXPECTS(m2.type().id() == cudf::type_to_id<M2Type>(),
               "Data type of M2 aggregation must be FLOAT64.");
  CUDF_EXPECTS(count.type().id() == cudf::type_to_id<CountType>(),
               "Data type of COUNT_VALID aggregation must be INT32.");

  auto const out_it = thrust::make_zip_iterator(target.begin<Target>(), validity);
  thrust::tabulate(rmm::exec_policy_nosync(stream),
                   out_it,
                   out_it + target.size(),
                   [m2 = m2.begin<M2Type>(), count = count.begin<CountType>(), ddof] __device__(
                     size_type const idx) -> cuda::std::pair<Target, bool> {
                     auto const group_count = count[idx];
                     auto const df          = group_count - ddof;
                     if (group_count == 0 || df <= 0) { return {Target{}, false}; }
                     return {m2[idx] / df, true};
                   });
}

}  // namespace cudf::groupby::detail::hash
