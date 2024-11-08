/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cudf/groupby.hpp>
#include <cudf/utilities/span.hpp>

#include <memory>
#include <tuple>
#include <vector>

namespace cudf::groupby::detail::hash {

// flatten aggs to filter in single pass aggs
std::tuple<table_view, std::vector<aggregation::Kind>, std::vector<std::unique_ptr<aggregation>>>
flatten_single_pass_aggs(host_span<aggregation_request const> requests);

}  // namespace cudf::groupby::detail::hash
