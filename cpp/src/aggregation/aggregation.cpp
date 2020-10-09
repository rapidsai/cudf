/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <memory>

namespace cudf {
/// Factory to create a SUM aggregation
std::unique_ptr<aggregation> make_sum_aggregation()
{
  return std::make_unique<aggregation>(aggregation::SUM);
}
/// Factory to create a PRODUCT aggregation
std::unique_ptr<aggregation> make_product_aggregation()
{
  return std::make_unique<aggregation>(aggregation::PRODUCT);
}
/// Factory to create a MIN aggregation
std::unique_ptr<aggregation> make_min_aggregation()
{
  return std::make_unique<aggregation>(aggregation::MIN);
}
/// Factory to create a MAX aggregation
std::unique_ptr<aggregation> make_max_aggregation()
{
  return std::make_unique<aggregation>(aggregation::MAX);
}
/// Factory to create a COUNT aggregation
std::unique_ptr<aggregation> make_count_aggregation(null_policy null_handling)
{
  auto kind =
    (null_handling == null_policy::INCLUDE) ? aggregation::COUNT_ALL : aggregation::COUNT_VALID;
  return std::make_unique<aggregation>(kind);
}
/// Factory to create a ANY aggregation
std::unique_ptr<aggregation> make_any_aggregation()
{
  return std::make_unique<aggregation>(aggregation::ANY);
}
/// Factory to create a ALL aggregation
std::unique_ptr<aggregation> make_all_aggregation()
{
  return std::make_unique<aggregation>(aggregation::ALL);
}
/// Factory to create a SUM_OF_SQUARES aggregation
std::unique_ptr<aggregation> make_sum_of_squares_aggregation()
{
  return std::make_unique<aggregation>(aggregation::SUM_OF_SQUARES);
}
/// Factory to create a MEAN aggregation
std::unique_ptr<aggregation> make_mean_aggregation()
{
  return std::make_unique<aggregation>(aggregation::MEAN);
}
/// Factory to create a VARIANCE aggregation
std::unique_ptr<aggregation> make_variance_aggregation(size_type ddof)
{
  return std::make_unique<detail::std_var_aggregation>(aggregation::VARIANCE, ddof);
};
/// Factory to create a STD aggregation
std::unique_ptr<aggregation> make_std_aggregation(size_type ddof)
{
  return std::make_unique<detail::std_var_aggregation>(aggregation::STD, ddof);
};
/// Factory to create a MEDIAN aggregation
std::unique_ptr<aggregation> make_median_aggregation()
{
  // TODO I think this should just return a quantile_aggregation?
  return std::make_unique<aggregation>(aggregation::MEDIAN);
}
/// Factory to create a QUANTILE aggregation
std::unique_ptr<aggregation> make_quantile_aggregation(std::vector<double> const& q,
                                                       interpolation i)
{
  return std::make_unique<detail::quantile_aggregation>(q, i);
}
/// Factory to create a ARGMAX aggregation
std::unique_ptr<aggregation> make_argmax_aggregation()
{
  return std::make_unique<aggregation>(aggregation::ARGMAX);
}
/// Factory to create a ARGMIN aggregation
std::unique_ptr<aggregation> make_argmin_aggregation()
{
  return std::make_unique<aggregation>(aggregation::ARGMIN);
}
/// Factory to create a NUNIQUE aggregation
std::unique_ptr<aggregation> make_nunique_aggregation(null_policy null_handling)
{
  return std::make_unique<detail::nunique_aggregation>(aggregation::NUNIQUE, null_handling);
}
/// Factory to create a NTH_ELEMENT aggregation
std::unique_ptr<aggregation> make_nth_element_aggregation(size_type n, null_policy null_handling)
{
  return std::make_unique<detail::nth_element_aggregation>(
    aggregation::NTH_ELEMENT, n, null_handling);
}
/// Factory to create a ROW_NUMBER aggregation
std::unique_ptr<aggregation> make_row_number_aggregation()
{
  return std::make_unique<aggregation>(aggregation::ROW_NUMBER);
}
/// Factory to create a COLLECT aggregation
std::unique_ptr<aggregation> make_collect_aggregation()
{
  return std::make_unique<aggregation>(aggregation::COLLECT);
}
/// Factory to create a LAG aggregation
std::unique_ptr<aggregation> make_lag_aggregation(size_type offset)
{
  return std::make_unique<cudf::detail::lead_lag_aggregation>(aggregation::LAG, offset);
}
/// Factory to create a LEAD aggregation
std::unique_ptr<aggregation> make_lead_aggregation(size_type offset)
{
  return std::make_unique<cudf::detail::lead_lag_aggregation>(aggregation::LEAD, offset);
}
/// Factory to create a UDF aggregation
std::unique_ptr<aggregation> make_udf_aggregation(udf_type type,
                                                  std::string const& user_defined_aggregator,
                                                  data_type output_type)
{
  aggregation* a =
    new detail::udf_aggregation{type == udf_type::PTX ? aggregation::PTX : aggregation::CUDA,
                                user_defined_aggregator,
                                output_type};
  return std::unique_ptr<aggregation>(a);
}

namespace detail {
namespace {
struct target_type_functor {
  template <typename Source, aggregation::Kind k>
  constexpr data_type operator()() const noexcept
  {
    return data_type{type_to_id<target_type_t<Source, k>>()};
  }
};

struct is_valid_aggregation_impl {
  template <typename Source, aggregation::Kind k>
  constexpr bool operator()() const noexcept
  {
    return is_valid_aggregation<Source, k>();
  }
};
}  // namespace

// Return target data_type for the given source_type and aggregation
data_type target_type(data_type source, aggregation::Kind k)
{
  return dispatch_type_and_aggregation(source, k, target_type_functor{});
}

// Verifies the aggregation `k` is valid on the type `source`
bool is_valid_aggregation(data_type source, aggregation::Kind k)
{
  return dispatch_type_and_aggregation(source, k, is_valid_aggregation_impl{});
}
}  // namespace detail
}  // namespace cudf
