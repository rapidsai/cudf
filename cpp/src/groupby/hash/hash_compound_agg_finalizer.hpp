/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/types.hpp>

namespace cudf::groupby::detail::hash {

class hash_compound_agg_finalizer final : public cudf::detail::aggregation_finalizer {
  column_view col;
  data_type input_type;
  cudf::detail::result_cache* cache;
  bitmask_type const* d_row_bitmask;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

 public:
  using cudf::detail::aggregation_finalizer::visit;

  hash_compound_agg_finalizer(column_view col,
                              cudf::detail::result_cache* cache,
                              bitmask_type const* d_row_bitmask,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);
  virtual ~hash_compound_agg_finalizer() = default;

  // Enables conversion of ARGMIN/ARGMAX into MIN/MAX
  auto gather_argminmax(cudf::aggregation const& agg);

  void visit(cudf::detail::min_aggregation const& agg) override;

  void visit(cudf::detail::max_aggregation const& agg) override;

  void visit(cudf::detail::mean_aggregation const& agg) override;

  void visit(cudf::detail::m2_aggregation const& agg) override;

  void visit(cudf::detail::var_aggregation const& agg) override;

  void visit(cudf::detail::std_aggregation const& agg) override;
};

}  // namespace cudf::groupby::detail::hash
