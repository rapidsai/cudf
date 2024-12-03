/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/tdigest/tdigest_column_view.hpp>

namespace cudf {
namespace tdigest {

tdigest_column_view::tdigest_column_view(column_view const& col) : column_view(col)
{
  // sanity check that this is actually tdigest data
  CUDF_EXPECTS(col.type().id() == type_id::STRUCT, "Encountered invalid tdigest column");
  CUDF_EXPECTS(col.size() > 0, "tdigest columns must have > 0 rows");
  CUDF_EXPECTS(col.offset() == 0, "Encountered a sliced tdigest column");
  CUDF_EXPECTS(not col.nullable(), "Encountered nullable tdigest column");

  structs_column_view const scv(col);
  CUDF_EXPECTS(scv.num_children() == 3, "Encountered invalid tdigest column");
  CUDF_EXPECTS(scv.child(min_column_index).type().id() == type_id::FLOAT64,
               "Encountered invalid tdigest column");
  CUDF_EXPECTS(scv.child(max_column_index).type().id() == type_id::FLOAT64,
               "Encountered invalid tdigest column");

  lists_column_view const lcv(scv.child(centroid_column_index));
  auto data = lcv.child();
  CUDF_EXPECTS(data.type().id() == type_id::STRUCT, "Encountered invalid tdigest column");
  CUDF_EXPECTS(data.num_children() == 2,
               "Encountered tdigest column with an invalid number of children");
  auto mean = data.child(mean_column_index);
  CUDF_EXPECTS(mean.type().id() == type_id::FLOAT64, "Encountered invalid tdigest mean column");
  auto weight = data.child(weight_column_index);
  CUDF_EXPECTS(weight.type().id() == type_id::FLOAT64, "Encountered invalid tdigest weight column");
}

lists_column_view tdigest_column_view::centroids() const { return child(centroid_column_index); }

column_view tdigest_column_view::means() const
{
  auto c = centroids();
  structs_column_view const inner(c.parent().child(lists_column_view::child_column_index));
  return inner.child(mean_column_index);
}

column_view tdigest_column_view::weights() const
{
  auto c = centroids();
  structs_column_view const inner(c.parent().child(lists_column_view::child_column_index));
  return inner.child(weight_column_index);
}

double const* tdigest_column_view::min_begin() const
{
  return child(min_column_index).begin<double>();
}

double const* tdigest_column_view::max_begin() const
{
  return child(max_column_index).begin<double>();
}

}  // namespace tdigest
}  // namespace cudf
