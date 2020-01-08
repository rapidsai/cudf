/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
namespace experimental {

/// Factory to create a SUM aggregation
std::unique_ptr<aggregation> make_sum_aggregation() {
  return std::make_unique<aggregation>(aggregation::SUM);
}
/// Factory to create a MIN aggregation
std::unique_ptr<aggregation> make_min_aggregation() {
  return std::make_unique<aggregation>(aggregation::MIN);
}
/// Factory to create a MAX aggregation
std::unique_ptr<aggregation> make_max_aggregation() {
  return std::make_unique<aggregation>(aggregation::MAX);
}
/// Factory to create a COUNT aggregation
std::unique_ptr<aggregation> make_count_aggregation() {
  return std::make_unique<aggregation>(aggregation::COUNT);
}
/// Factory to create a MEAN aggregation
std::unique_ptr<aggregation> make_mean_aggregation() {
  return std::make_unique<aggregation>(aggregation::MEAN);
}
/// Factory to create a VARIANCE aggregation
std::unique_ptr<aggregation> make_variance_aggregation(size_type ddof) {
  return std::make_unique<detail::std_var_aggregation>(aggregation::VARIANCE, ddof);
};
/// Factory to create a STD aggregation
std::unique_ptr<aggregation> make_std_aggregation(size_type ddof) {
  return std::make_unique<detail::std_var_aggregation>(aggregation::STD, ddof);
};
/// Factory to create a MEDIAN aggregation
std::unique_ptr<aggregation> make_median_aggregation() {
  // TODO I think this should just return a quantile_aggregation?
  return std::make_unique<aggregation>(aggregation::MEDIAN);
}
/// Factory to create a QUANTILE aggregation
std::unique_ptr<aggregation> make_quantile_aggregation(
    std::vector<double> const& q, interpolation i) {
  return std::make_unique<detail::quantile_aggregation>(q, i);
}
/// Factory to create a ARGMAX aggregation
std::unique_ptr<aggregation> make_argmax_aggregation() {
  return std::make_unique<aggregation>(aggregation::ARGMAX);
}
/// Factory to create a ARGMIN aggregation
std::unique_ptr<aggregation> make_argmin_aggregation() {
  return std::make_unique<aggregation>(aggregation::ARGMIN);
}

namespace detail {
namespace {
struct target_type_functor {
  template <typename Source, aggregation::Kind k>
  constexpr data_type operator()() const noexcept {
    return data_type{type_to_id<target_type_t<Source, k>>()};
  }
};

struct is_valid_aggregation_impl {
  template <typename Source, aggregation::Kind k>
  constexpr bool operator()() const noexcept {
    return is_valid_aggregation<Source, k>();
  }
};
}  // namespace

// Return target data_type for the given source_type and aggregation
data_type target_type(data_type source, aggregation::Kind k) {
  return dispatch_type_and_aggregation(source, k, target_type_functor{});
}

// Verifies the aggregation `k` is valid on the type `source`
bool is_valid_aggregation(data_type source, aggregation::Kind k) {
  return dispatch_type_and_aggregation(source, k, is_valid_aggregation_impl{});
}
}  // namespace detail
}  // namespace experimental
}  // namespace cudf
