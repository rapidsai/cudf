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

#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/detail/reduction_functions.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>

namespace cudf {
namespace experimental {
namespace detail {

struct reduce_dispatch_functor {
  column_view const col;
  data_type output_dtype;
  rmm::mr::device_memory_resource *mr;
  cudaStream_t stream;

  reduce_dispatch_functor(column_view const &col, data_type output_dtype,
                          rmm::mr::device_memory_resource *mr,
                          cudaStream_t stream)
      : col(col), output_dtype(output_dtype), mr(mr), stream(stream) {}

  template <aggregation::Kind k>
  std::unique_ptr<scalar> operator()(std::unique_ptr<aggregation> const &agg) {
    switch (k) {
    case aggregation::SUM:
      return reduction::sum(col, output_dtype, mr, stream);
      break;
    case aggregation::MIN:
      return reduction::min(col, output_dtype, mr, stream);
      break;
    case aggregation::MAX:
      return reduction::max(col, output_dtype, mr, stream);
      break;
    case aggregation::ANY:
      return reduction::any(col, output_dtype, mr, stream);
      break;
    case aggregation::ALL:
      return reduction::all(col, output_dtype, mr, stream);
      break;
    case aggregation::PRODUCT:
      return reduction::product(col, output_dtype, mr, stream);
      break;
    case aggregation::SUM_OF_SQUARES:
      return reduction::sum_of_squares(col, output_dtype, mr, stream);
      break;
    case aggregation::MEAN:
      return reduction::mean(col, output_dtype, mr, stream);
      break;
    case aggregation::VARIANCE: {
      auto var_agg = static_cast<std_var_aggregation const*>(agg.get());
      return reduction::variance(col, output_dtype, var_agg->_ddof, mr, stream);
      }
      break;
    case aggregation::STD: {
      auto var_agg = static_cast<std_var_aggregation const*>(agg.get());
      return reduction::standard_deviation(col, output_dtype, var_agg->_ddof, mr, stream);
      }
      break;
    default:
      CUDF_FAIL("Unsupported reduction operator");
    }
  }
};

std::unique_ptr<scalar> reduce(
    column_view const& col, 
    std::unique_ptr<aggregation> const &agg,
    data_type output_dtype,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0)
{
  std::unique_ptr<scalar> result = make_default_constructed_scalar(output_dtype);
  result->set_valid(false, stream);

  // check if input column is empty
  if (col.size() <= col.null_count()) return result;

  result = aggregation_dispatcher(
      agg->kind, reduce_dispatch_functor{col, output_dtype, mr, stream}, agg);
  return result;
}
}  // namespace detail

 std::unique_ptr<scalar> reduce(
    column_view const& col, std::unique_ptr<aggregation> const &agg,
    data_type output_dtype,
    rmm::mr::device_memory_resource* mr)
{
  return detail::reduce(col, agg, output_dtype, mr);
}

}  // namespace experimental
}  // namespace cudf

