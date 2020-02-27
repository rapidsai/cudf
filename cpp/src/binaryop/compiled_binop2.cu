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
#include <cudf/detail/copy.hpp>

#include <cudf/detail/iterator.cuh>
#include <rmm/thrust_rmm_allocator.h>
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf {
namespace experimental {
namespace detail {
  //----
namespace {
  template<typename lhsT, typename rhsT, typename Op>
  struct out_dispatcher {
    mutable_column_device_view out;
    column_device_view lhs;
    column_device_view rhs;
    Op op;
    __host__ __device__ out_dispatcher(
                mutable_column_device_view& out,
                column_device_view const& lhs,
                column_device_view const& rhs,
                Op const op) : out(out), lhs(lhs), rhs(rhs), op(op) { }

    template <typename outT>
    static constexpr bool is_supported() { return
        is_fixed_width<outT>() && (!std::is_same<outT, cudf::experimental::bool8>::value) && 
        is_fixed_width<lhsT>() && (!std::is_same<lhsT, cudf::experimental::bool8>::value) && 
        is_fixed_width<rhsT>() && (!std::is_same<rhsT, cudf::experimental::bool8>::value) && 
        std::is_convertible<lhsT, outT>::value &&
        std::is_convertible<rhsT, outT>::value;
    }

    template<typename outT,
      std::enable_if_t<is_supported<outT>()>* = nullptr>
    __device__ void operator()(size_type i)
    {
      (out.data<outT>())[i] = 
               op(static_cast<outT>(lhs.element<lhsT>(i)),
                  static_cast<outT>(rhs.element<rhsT>(i)));
    }
    template<typename outT,
      std::enable_if_t<!is_supported<outT>()>* = nullptr>
    __device__ void operator()(size_type i)
    { }
  };

  template<typename rhsT, typename Op>
  struct lhs_dispatcher {
    mutable_column_device_view out;
    column_device_view lhs;
    column_device_view rhs;
    Op op;
    __host__ __device__ lhs_dispatcher(
                mutable_column_device_view& out,
                column_device_view const& lhs,
                column_device_view const& rhs,
                Op const op) : out(out), lhs(lhs), rhs(rhs), op(op) { }
    template<typename lhsT>
    __device__ void operator()(size_type i)
    {
      cudf::experimental::type_dispatcher(out.type(), out_dispatcher<lhsT, rhsT, Op>{out, lhs, rhs, op}, i);
    }
  };

  template<typename Op>
  struct rhs_dispatcher {
    mutable_column_device_view out;
    column_device_view lhs;
    column_device_view rhs;
    Op op;
    __host__ __device__ rhs_dispatcher(
                mutable_column_device_view const& out,
                column_device_view const& lhs,
                column_device_view const& rhs,
                Op const op) : out(out), lhs(lhs), rhs(rhs), op(op) { }
    template<typename rhsT>
    __device__ void operator()(size_type i)
    {
      cudf::experimental::type_dispatcher(out.type(), lhs_dispatcher<rhsT, Op>{out, lhs, rhs, op}, i);
    }
  };
} // namespace anonymous

  //----
std::unique_ptr<column> experimental_binary_operation2(column_view const& lhs,
                                         column_view const& rhs,
                                         //binary_operator op,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr,
                                         cudaStream_t stream=0) {
  CUDF_EXPECTS((lhs.size() == rhs.size()), "Column sizes don't match");

  auto new_mask = bitmask_and(table_view({lhs, rhs}), mr, stream);
  auto out_col = make_fixed_width_column(output_type, lhs.size(), new_mask,
                                     cudf::UNKNOWN_NULL_COUNT, stream, mr);
  auto out_view = out_col->mutable_view();
  auto dout = mutable_column_device_view::create(out_view);
  auto dlhs = column_device_view::create(lhs);
  auto drhs = column_device_view::create(rhs);
  using Op = cudf::DeviceSum;
  thrust::for_each_n(
      thrust::make_counting_iterator<size_type>(0), rhs.size(),
      [out = *dout, lhs=*dlhs, rhs=*drhs] __device__ (size_type i) -> void {
      cudf::experimental::type_dispatcher(rhs.type(), rhs_dispatcher<Op>{out, lhs, rhs, Op{}}, i);
      });
  return out_col;
}
}  // namespace detail

std::unique_ptr<column> experimental_binary_operation2(column_view const& lhs,
                                         column_view const& rhs,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr) {
  return detail::experimental_binary_operation2(lhs, rhs, output_type, mr);
}


}  // namespace experimental
}  // namespace cudf
