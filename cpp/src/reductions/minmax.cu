/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
// The translation unit for reduction `minmax`

#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/reduction.hpp>
#include <cudf/column/column_device_view.cuh>

#include <thrust/transform_reduce.h>
namespace cudf {
namespace detail {

namespace {
    /**
    * @brief functor that takes in a single value, x,
    * and returns a minmax_pair whose minimum and 
    * maximum values are initialized to x.
    * 
    */
    struct minmax_unary_op
    : public thrust::unary_function< scalar, minmax_pair >
    {
      minmax_pair operator()(const scalar& x) const
      {
        minmax_pair result(x, x);
        return result;
      }
    };

    /**
    * @brief functor that accepts two minmax_pair structs and returns a new
    * minmax_pair whose minimum and maximum values are the min() and max()
    * respectively of the minimums and maximums of the input pairs
    * 
    */
    template <typename T>
    struct minmax_binary_op
    : public thrust::binary_function< std::pair<T, bool>, std::pair<T, bool>, minmax_pair >
    {
      __host__ __device__
      minmax_pair operator()(const std::pair<T, bool>& x, const std::pair<T, bool>& y) const
      {
        if (x.second && y.second) {
          return minmax_pair{thrust::min(x.min_val, y.min_val),
                             thrust::max(x.max_val, y.max_val)};
        } else if (x.second && !y.second) {
          return minmax_pair{x.first, x.first};
        } else if (!x.second && y.second) {
          return minmax_pair{y.first, y.first};
        } else if (!x.second && !y.second) {
          return minmax_pair{cudf::DeviceMax<T>.identity(), cudf::DeviceMin<T>.identity()};
        }
      }
    };

    struct minmax_impl {
      template <typename T>
      std::unique_ptr<minmax_pair> operator()(const cudf::column_view & col, rmm::mr::device_memory_resource* mr,
        cudaStream_t stream)
      {
        // setup arguments
        minmax_binary_op<T> binary_op;
      
        // initialize reduction with the first value
        minmax_pair init{cudf::DeviceMax.identity(), cudf::DeviceMin.identity()};
      
     
        auto device_col = column_device_view::create(col, stream);

        // compute minimum and maximum values
        auto result = thrust::transform_reduce(
          thrust::make_counting_iterator<size_type>(0),
          thrust::make_counting_iterator<size_type>(col.size()),
          [d_col = *device_col] __device__ (size_type index) {
            return std::pair<T, bool>(d_col.element<T>(index), d_col.is_valid(index));
          }, init, binary_op);

        auto pair        = new minmax_pair(std::move(result));
        return std::unique_ptr<minmax_pair>(pair);
      }
    };
    
}  // namespace

    std::unique_ptr<cudf::minmax_pair> minmax(const cudf::column_view &col,
      rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource(),
      cudaStream_t stream = 0)
    {
      return type_dispatcher(col.type(), minmax_impl{}, col, mr, stream);
    }    
}  // namespace detail

std::unique_ptr<cudf::minmax_pair> minmax(const cudf::column_view &col,
                                          rmm::mr::device_memory_resource *mr)
{
  return cudf::detail::minmax(col, mr);
}

} // namespace cudf
