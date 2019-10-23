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
#include "digitize.hpp"
#include <cudf/digitize.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <utilities/error_utils.hpp>
#include <thrust/binary_search.h>

namespace cudf {

namespace {

struct binary_search_bound {
  template<typename T>
  auto operator()(column_view const& col, column_view const& bins, bool upper_bound,
                  rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    auto output = cudf::make_numeric_column(data_type{INT32}, col.size(), cudf::UNALLOCATED, stream, mr);

    if (upper_bound) {
      thrust::upper_bound(rmm::exec_policy()->on(stream), bins.begin<T>(), bins.end<T>(),
        col.begin<T>(), col.end<T>(), output.begin<int32_t>(), thrust::less_equal<T>());
    } else {
      thrust::lower_bound(rmm::exec_policy()->on(stream), bins.begin<T>(), bins.end<T>(),
        col.begin<T>(), col.end<T>(), output.begin<int32_t>(), thrust::less_equal<T>());
    }

    return output;
  }
};

}  // namespace

namespace detail {

std::unique_ptr<column>
digitize(column_view const& col, column_view const& bins, bool right,
         rmm::mr::device_memory_resource* mr, cudaStream_t stream)
{
  auto const dtype = col.type();
  CUDF_EXPECTS(dtype == bins.type(), "Column type mismatch");

  // TODO should this make use of the comparable type traits?
  CUDF_EXPECTS(is_numeric(dtype) || is_timestamp(dtype), "Type must be numeric or timestamp");

  // TODO: Handle when col or bins have null values
  CUDF_EXPECTS(0 == col.null_count(), "Null values unsupported");
  CUDF_EXPECTS(0 == bins.null_count(), "Null values unsupported");

  return experimental::type_dispatcher(dtype, binary_search_bound{},
    col, bins, right, stream);
}

}  // namespace detail

std::unique_ptr<column>
digitize(column_view const& col, column_view const& bins, bool right,
         rmm::mr::device_memory_resource* mr)
{
  return detail::digitize(col, bins, right, mr);
}

}  // namespace cudf
