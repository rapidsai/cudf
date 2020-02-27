/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
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

#include <cudf/compiled_binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/detail/utilities/device_operators.cuh>

#include <cudf/detail/iterator.cuh>
#include <rmm/thrust_rmm_allocator.h>
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf {
namespace experimental {
namespace detail {

namespace {
  template<typename outT, typename lhsT, typename rhsT, typename Op>
  std::unique_ptr<column> binary_operation(
                      data_type output_type,
                      column_view const& lhs,
                      column_view const& rhs,
                      Op const op,
                      rmm::mr::device_memory_resource* mr,
                      cudaStream_t stream) {
    auto new_mask = bitmask_and(table_view({lhs, rhs}), mr, stream);
    auto out_col = make_fixed_width_column(output_type, lhs.size(), new_mask,
                                       cudf::UNKNOWN_NULL_COUNT, stream, mr);
    mutable_column_view out = out_col->mutable_view();
    auto dout = cudf::mutable_column_device_view::create(out, stream);
    auto dlhs = cudf::column_device_view::create(lhs, stream);
    auto drhs = cudf::column_device_view::create(rhs, stream);
  if (lhs.has_nulls() || rhs.has_nulls()) {
    auto itlhs = thrust::make_transform_iterator(
        dlhs->pair_begin<lhsT, false>(), [] __device__ (auto p) { return static_cast<outT>(p.first); });
    auto itrhs = thrust::make_transform_iterator(
        drhs->pair_begin<rhsT, false>(), [] __device__ (auto p) { return static_cast<outT>(p.first); });
    outT* itout = dout->data<outT>();
    //auto itout = dout->begin<outT>();
    thrust::transform(rmm::exec_policy(stream)->on(stream),
        itlhs, itlhs+lhs.size(), itrhs, itout,  op);
    /*
    thrust::for_each_n(
        thrust::make_counting_iterator<size_type>(0), lhs.size(),
        [itout, itlhs, itrhs, op] __device__ (auto i) {
          itout[i] = 
          //op.template operator()( //<outT>(
          op(
          static_cast<outT>(*(itlhs+i)), 
          static_cast<outT>(*(itrhs+i)));
        });
        */
  } else {
    auto itlhs = thrust::make_transform_iterator(
        dlhs->begin<lhsT>(), [] __device__ (auto p) { return static_cast<outT>(p); });
    auto itrhs = thrust::make_transform_iterator(
        drhs->begin<rhsT>(), [] __device__ (auto p) { return static_cast<outT>(p); });
    //auto itlhs = dlhs->begin<lhsT>();
    //auto itrhs = drhs->begin<rhsT>();
    outT* itout = dout->data<outT>();
    //auto itout = dout->begin<outT>();
    thrust::transform(rmm::exec_policy(stream)->on(stream),
        itlhs, itlhs+lhs.size(), itrhs, itout,  op);
    //thrust::for_each(itlhs, itlhs+lhs.size(), [] __device__ (auto p) { p; } );
    //thrust::for_each(itrhs, itrhs+rhs.size(), [] __device__ (auto p) { p; } );
    /*
    thrust::for_each_n(
        thrust::make_counting_iterator<size_type>(0), lhs.size(),
        [itout, itlhs, itrhs, op] __device__ (auto i) {
          itout[i] = 
          //op.template operator()(//<outT>(
          op(
          static_cast<outT>(*(itlhs+i)), 
          static_cast<outT>(*(itrhs+i)));
        });
        */
  }
  return out_col;
  }

  template<typename lhsT, typename rhsT, typename Op>
  struct out_dispatcher {
    template <typename outT>
    static constexpr bool is_supported() { return
        is_fixed_width<outT>() && //(!std::is_same<outT, cudf::experimental::bool8>::value) && 
        is_fixed_width<lhsT>() && //(!std::is_same<lhsT, cudf::experimental::bool8>::value) && 
        is_fixed_width<rhsT>() && //(!std::is_same<rhsT, cudf::experimental::bool8>::value) && 
        std::is_convertible<lhsT, outT>::value &&
        std::is_convertible<rhsT, outT>::value;
    }
 
    template<typename outT,
      std::enable_if_t<is_supported<outT>()>* = nullptr>
    std::unique_ptr<column> operator()(
                      data_type output_type,
                      column_view const& lhs,
                      column_view const& rhs,
                      Op const& op,
                      rmm::mr::device_memory_resource* mr,
                      cudaStream_t stream)
    {
      return binary_operation<outT, lhsT, rhsT, Op>(output_type, lhs, rhs, op, mr, stream);
    }
    template<typename outT,
      std::enable_if_t<!is_supported<outT>()>* = nullptr>
    std::unique_ptr<column> operator()(
                      data_type output_type,
                      column_view const& lhs,
                      column_view const& rhs,
                      Op const& op,
                      rmm::mr::device_memory_resource* mr,
                      cudaStream_t stream)
    {
      CUDF_FAIL("Unsupported type for binary op");
    }
  };
  template<typename rhsT, typename Op>
  struct lhs_dispatcher {
    template<typename lhsT>
    std::unique_ptr<column> operator()(
                      data_type output_type,
                      column_view const& lhs,
                      column_view const& rhs,
                      Op const& op,
                      rmm::mr::device_memory_resource* mr,
                      cudaStream_t stream)
    {
      return cudf::experimental::type_dispatcher(output_type, out_dispatcher<lhsT, rhsT, Op>{}, 
          output_type, lhs, rhs, op, mr, stream);
    }
  };
  template<typename Op>
  struct rhs_dispatcher {
    template<typename rhsT>
    std::unique_ptr<column> operator()(
                      data_type output_type,
                      column_view const& lhs,
                      column_view const& rhs,
                      Op const& op,
                      rmm::mr::device_memory_resource* mr,
                      cudaStream_t stream)
    {
      return cudf::experimental::type_dispatcher(lhs.type(), lhs_dispatcher<rhsT, Op>{}, 
          output_type, lhs, rhs, op, mr, stream);
    }
  };
} // namespace anonymous
 
std::unique_ptr<column> experimental_binary_operation1(column_view const& lhs,
                                         column_view const& rhs,
                                         //binary_operator op,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr,
                                         cudaStream_t stream=0) {
  CUDF_EXPECTS((lhs.size() == rhs.size()), "Column sizes don't match");

  using Op = cudf::DeviceSum;
  return cudf::experimental::type_dispatcher(rhs.type(), rhs_dispatcher<Op>{}, 
      output_type, lhs, rhs, Op{}, mr, stream);
}
}  // namespace detail

std::unique_ptr<column> experimental_binary_operation1(column_view const& lhs,
                                         column_view const& rhs,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr) {
  return detail::experimental_binary_operation1(lhs, rhs, output_type, mr);
}


}  // namespace experimental
}  // namespace cudf
