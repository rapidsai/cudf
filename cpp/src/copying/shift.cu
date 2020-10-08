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

#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <algorithm>
#include <iterator>
#include <memory>

namespace cudf {
namespace {
inline bool __device__ out_of_bounds(size_type size, size_type idx)
{
  return idx < 0 || idx >= size;
}

struct shift_functor {
  template <typename T, typename... Args>
  std::enable_if_t<not cudf::is_fixed_width<T>(), std::unique_ptr<column>> operator()(
    Args&&... args)
  {
    CUDF_FAIL("shift does not support non-fixed-width types.");
  }

  template <typename T>
  std::enable_if_t<cudf::is_fixed_width<T>(), std::unique_ptr<column>> operator()(
    column_view const& input,
    size_type offset,
    scalar const& fill_value,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
  {
    using Type       = device_storage_type_t<T>;
    using ScalarType = cudf::scalar_type_t<Type>;
    auto& scalar     = static_cast<ScalarType const&>(fill_value);

    auto device_input = column_device_view::create(input);
    auto output =
      detail::allocate_like(input, input.size(), mask_allocation_policy::NEVER, mr, stream);
    auto device_output = mutable_column_device_view::create(*output);

    auto size        = input.size();
    auto index_begin = thrust::make_counting_iterator<size_type>(0);
    auto index_end   = thrust::make_counting_iterator<size_type>(size);

    if (input.nullable() || not scalar.is_valid()) {
      auto func_validity = [size,
                            offset,
                            fill  = scalar.validity_data(),
                            input = *device_input] __device__(size_type idx) {
        auto src_idx = idx - offset;
        return out_of_bounds(size, src_idx) ? *fill : input.is_valid(src_idx);
      };

      auto mask_pair = detail::valid_if(index_begin, index_end, func_validity, stream, mr);

      output->set_null_mask(std::move(std::get<0>(mask_pair)));
      output->set_null_count(std::get<1>(mask_pair));
    }

    auto data = device_output->data<Type>();

    // avoid assigning elements we know to be invalid.
    if (not scalar.is_valid()) {
      if (offset > 0) {
        index_begin = thrust::make_counting_iterator<size_type>(offset);
        data        = data + offset;
      } else if (offset < 0) {
        index_end = thrust::make_counting_iterator<size_type>(size + offset);
      }
    }

    auto func_value =
      [size, offset, fill = scalar.data(), input = *device_input] __device__(size_type idx) {
        auto src_idx = idx - offset;
        return out_of_bounds(size, src_idx) ? *fill : input.element<Type>(src_idx);
      };

    thrust::transform(
      rmm::exec_policy(stream)->on(stream), index_begin, index_end, data, func_value);

    return output;
  }
};

}  // anonymous namespace

std::unique_ptr<column> shift(column_view const& input,
                              size_type offset,
                              scalar const& fill_value,
                              rmm::mr::device_memory_resource* mr,
                              cudaStream_t stream)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(input.type() == fill_value.type(),
               "shift requires each fill value type to match the corresponding column type.");

  if (input.size() == 0) { return empty_like(input); }

  return type_dispatcher(input.type(), shift_functor{}, input, offset, fill_value, mr, stream);
}

}  // namespace cudf
