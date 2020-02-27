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
//---------------------------------------------------------------------------------------
struct sum {
  mutable_column_device_view out;
  column_device_view lhs;
  column_device_view rhs;
  cudf::DeviceSum op{};
  __host__ __device__  sum(
                mutable_column_device_view const& out,
                column_device_view const& lhs,
                column_device_view const& rhs) : out(out), lhs(lhs), rhs(rhs) {}

  template<typename outT, typename lhsT, typename rhsT>
  static constexpr bool is_supported() { return
      is_fixed_width<outT>() && 
      is_fixed_width<lhsT>() &&
      is_fixed_width<rhsT>() &&
      std::is_convertible<lhsT, outT>::value &&
      std::is_convertible<rhsT, outT>::value;
  }
  template<typename outT, typename lhsT, typename rhsT,
    std::enable_if_t<is_supported<outT, lhsT, rhsT>()>* = nullptr>
  __device__ void operator()(size_type i) {
    (out.data<outT>())[i] = 
               op(static_cast<outT>(lhs.element<lhsT>(i)),
                  static_cast<outT>(rhs.element<rhsT>(i)));
  }
  template<typename outT, typename lhsT, typename rhsT,
    std::enable_if_t<!is_supported<outT, lhsT, rhsT>()>* = nullptr>
  __device__ void operator()(size_type i) { }
};
//---------------------------------------------------------------------------------------


//#pragma nv_exec_check_disable
template <typename functor_t, typename... Ts,
         typename... Args>
         __device__ void
         tt_inner_for_each_n(functor_t f, size_type size,
             Args&&... args) {
           int tid = threadIdx.x;
           int blkid = blockIdx.x;
           int blksz = blockDim.x;
           int gridsz = gridDim.x;

           int start = tid + blkid * blksz;
           int step = blksz * gridsz;

           for (cudf::size_type i=start; i<size; i+=step) {
             //return
             f.template operator()<Ts...>(i,
                 std::forward<Args>(args)...);
           }
         }

  //recursive template parameter pack this too.
  template<typename Op, typename... Ts>
  struct level_dispatcher {
    template<typename lhsT>
    __device__ void operator()(
    size_type n,
    Op const op)
    {
      tt_inner_for_each_n<Op, lhsT, Ts...>(op, n);
    }
    template<typename lhsT, typename... Types>
    __device__ void operator()(
    size_type n,
    Op const op,
    data_type type1,
    Types&&... types)
    {
      cudf::experimental::type_dispatcher(type1, level_dispatcher<Op, lhsT, Ts...>{}, n, op,
          std::forward<Types>(types)...);
    }
  };

template<class Op>
__global__
void three_operand(
                   data_type type1, 
                   data_type type2, 
                   data_type type3,
                   size_type n,
                   Op op)
{
  cudf::experimental::type_dispatcher(type3, level_dispatcher<Op>{}, n, op, type1, type2);
}

namespace detail {

std::unique_ptr<column> experimental_binary_operation3(
                                         column_view const& lhs,
                                         column_view const& rhs,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr,
                                         cudaStream_t stream=0) {
  auto new_mask = bitmask_and(table_view({lhs, rhs}), mr, stream);
  auto out = make_fixed_width_column(output_type, lhs.size(), new_mask,
                                     cudf::UNKNOWN_NULL_COUNT, stream, mr);
  auto out_view = out->mutable_view();
  auto dout = mutable_column_device_view::create(out_view);
  auto dlhs = column_device_view::create(lhs);
  auto drhs = column_device_view::create(rhs);
  sum p{*dout, *dlhs, *drhs};

  int block_size;// = 256;
  int min_grid_size;
  CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, three_operand<sum>));
  // Calculate actual block count to use based on records count
  const int grid_size = (out_view.size() + block_size - 1) / block_size;

  three_operand<sum><<< min_grid_size, block_size >>>(out_view.type(), lhs.type(), rhs.type(), out_view.size(), p);
  return out;
}

}  // namespace detail
 
std::unique_ptr<column> experimental_binary_operation3(column_view const& lhs,
                                         column_view const& rhs,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr) {
  return detail::experimental_binary_operation3(lhs, rhs, output_type, mr);
}
}  // namespace experimental
}  // namespace cudf
