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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/table/table_view.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include "binary_ops.hpp"

namespace cudf {
namespace experimental {
namespace binops {
namespace compiled {

namespace {

template <typename Lhs, typename Rhs, typename Out>
struct apply_binop {
  binary_operator op;
  apply_binop(binary_operator op) : op(op) {}
  CUDA_DEVICE_CALLABLE Out operator()(Lhs const& x, Rhs const& y) const {
    switch (op) {
      case binary_operator::EQUAL: return this->equal(x, y);
      case binary_operator::NOT_EQUAL: return this->not_equal(x, y);
      case binary_operator::LESS: return this->less(x, y);
      case binary_operator::GREATER: return this->greater(x, y);
      case binary_operator::LESS_EQUAL: return this->less_equal(x, y);
      case binary_operator::GREATER_EQUAL: return this->greater_equal(x, y);
      default: return Out{};
    }
  }
  CUDA_DEVICE_CALLABLE Out equal(Lhs const& x, Rhs const& y) const { return static_cast<Out>(x == y); }
  CUDA_DEVICE_CALLABLE Out not_equal(Lhs const& x, Rhs const& y) const { return static_cast<Out>(x != y); }
  CUDA_DEVICE_CALLABLE Out less(Lhs const& x, Rhs const& y) const { return static_cast<Out>(x < y); }
  CUDA_DEVICE_CALLABLE Out greater(Lhs const& x, Rhs const& y) const { return static_cast<Out>(x > y); }
  CUDA_DEVICE_CALLABLE Out less_equal(Lhs const& x, Rhs const& y) const { return static_cast<Out>(x <= y); }
  CUDA_DEVICE_CALLABLE Out greater_equal(Lhs const& x, Rhs const& y) const { return static_cast<Out>(x >= y); }
};

template <typename Lhs, typename Rhs, typename Out>
struct apply_binop_scalar_lhs_rhs : apply_binop<Lhs, Rhs, Out> {
  cudf::experimental::scalar_device_type_t<Rhs> scalar;
  apply_binop_scalar_lhs_rhs(binary_operator op, cudf::experimental::scalar_device_type_t<Rhs> scalar)
    : apply_binop<Lhs, Rhs, Out>(op), scalar(scalar) {}
  CUDA_DEVICE_CALLABLE Out operator()(Lhs const& x) const {
    return apply_binop<Lhs, Rhs, Out>::operator()(x, scalar.value());
  }
};

template <typename Lhs, typename Rhs, typename Out>
struct apply_binop_scalar_rhs_lhs : apply_binop<Lhs, Rhs, Out> {
  cudf::experimental::scalar_device_type_t<Rhs> scalar;
  apply_binop_scalar_rhs_lhs(binary_operator op, cudf::experimental::scalar_device_type_t<Rhs> scalar)
    : apply_binop<Lhs, Rhs, Out>(op), scalar(scalar) {}
  CUDA_DEVICE_CALLABLE Out operator()(Lhs const& x) const {
    return apply_binop<Lhs, Rhs, Out>::operator()(scalar.value(), x);
  }
};

template <typename Lhs, typename Rhs, typename Out>
struct binary_op {

  std::unique_ptr<column> operator()(column_view const& lhs, scalar const& rhs, binary_operator op, data_type out_type, bool const reversed, rmm::mr::device_memory_resource* mr, cudaStream_t stream) {
    auto new_mask = binops::detail::scalar_col_valid_mask_and(lhs, rhs, stream, mr);
    auto out = make_fixed_width_column(out_type, lhs.size(), new_mask,
                                       rhs.is_valid(stream) ? cudf::UNKNOWN_NULL_COUNT : lhs.size(), stream, mr);

    if (lhs.size() > 0 && rhs.is_valid(stream)) {
      auto out_view = out->mutable_view();
      auto out_itr = out_view.begin<Out>();
      auto lhs_device_view = column_device_view::create(lhs, stream);
      auto rhs_scalar = static_cast<cudf::experimental::scalar_type_t<Rhs> const&>(rhs);
      auto rhs_scalar_view = get_scalar_device_view(rhs_scalar);
      if (lhs.has_nulls()) {
        auto lhs_itr = cudf::experimental::detail::make_null_replacement_iterator(*lhs_device_view, Lhs{});
        reversed ?
          thrust::transform(rmm::exec_policy(stream)->on(stream), lhs_itr, lhs_itr + lhs.size(), out_itr, apply_binop_scalar_rhs_lhs<Lhs, Rhs, Out>{op, rhs_scalar_view}) :
          thrust::transform(rmm::exec_policy(stream)->on(stream), lhs_itr, lhs_itr + lhs.size(), out_itr, apply_binop_scalar_lhs_rhs<Lhs, Rhs, Out>{op, rhs_scalar_view}) ;
      } else {
        auto lhs_itr = thrust::make_transform_iterator(thrust::make_counting_iterator(size_type{0}),
                                                      [col=*lhs_device_view] __device__ (size_type i) { return col.element<Lhs>(i); });
        reversed ?
          thrust::transform(rmm::exec_policy(stream)->on(stream), lhs_itr, lhs_itr + lhs.size(), out_itr, apply_binop_scalar_rhs_lhs<Lhs, Rhs, Out>{op, rhs_scalar_view}) :
          thrust::transform(rmm::exec_policy(stream)->on(stream), lhs_itr, lhs_itr + lhs.size(), out_itr, apply_binop_scalar_lhs_rhs<Lhs, Rhs, Out>{op, rhs_scalar_view}) ;
      }
    }

    CHECK_CUDA(stream);

    return out;
  }

  std::unique_ptr<column> operator()(column_view const& lhs, column_view const& rhs, binary_operator op, data_type out_type, rmm::mr::device_memory_resource* mr, cudaStream_t stream) {
    auto new_mask = bitmask_and(table_view({lhs, rhs}), mr, stream);
    auto out = make_fixed_width_column(out_type, lhs.size(), new_mask,
                                       cudf::UNKNOWN_NULL_COUNT, stream, mr);

    if (lhs.size() > 0) {
      auto out_view = out->mutable_view();
      auto out_itr = out_view.begin<Out>();
      auto lhs_device_view = column_device_view::create(lhs, stream);
      auto rhs_device_view = column_device_view::create(rhs, stream);
      if (lhs.has_nulls() && rhs.has_nulls()) {
        auto lhs_itr = cudf::experimental::detail::make_null_replacement_iterator(*lhs_device_view, Lhs{});
        auto rhs_itr = cudf::experimental::detail::make_null_replacement_iterator(*rhs_device_view, Rhs{});
        thrust::transform(rmm::exec_policy(stream)->on(stream), lhs_itr, lhs_itr + lhs.size(), rhs_itr, out_itr, apply_binop<Lhs, Rhs, Out>{op});
      } else if (lhs.has_nulls()) {
        auto lhs_itr = cudf::experimental::detail::make_null_replacement_iterator(*lhs_device_view, Lhs{});
        auto rhs_itr = thrust::make_transform_iterator(thrust::make_counting_iterator(size_type{0}),
                                                      [col=*rhs_device_view] __device__ (size_type i) { return col.element<Rhs>(i); });
        thrust::transform(rmm::exec_policy(stream)->on(stream), lhs_itr, lhs_itr + lhs.size(), rhs_itr, out_itr, apply_binop<Lhs, Rhs, Out>{op});
      } else if (rhs.has_nulls()) {
        auto lhs_itr = thrust::make_transform_iterator(thrust::make_counting_iterator(size_type{0}),
                                                      [col=*lhs_device_view] __device__ (size_type i) { return col.element<Lhs>(i); });
        auto rhs_itr = cudf::experimental::detail::make_null_replacement_iterator(*rhs_device_view, Rhs{});
        thrust::transform(rmm::exec_policy(stream)->on(stream), lhs_itr, lhs_itr + lhs.size(), rhs_itr, out_itr, apply_binop<Lhs, Rhs, Out>{op});
      } else {
        auto lhs_itr = thrust::make_transform_iterator(thrust::make_counting_iterator(size_type{0}),
                                                      [col=*lhs_device_view] __device__ (size_type i) { return col.element<Lhs>(i); });
        auto rhs_itr = thrust::make_transform_iterator(thrust::make_counting_iterator(size_type{0}),
                                                      [col=*rhs_device_view] __device__ (size_type i) { return col.element<Rhs>(i); });
        thrust::transform(rmm::exec_policy(stream)->on(stream), lhs_itr, lhs_itr + lhs.size(), rhs_itr, out_itr, apply_binop<Lhs, Rhs, Out>{op});
      }
    }

    CHECK_CUDA(stream);

    return out;
  }
};

}  // namespace

std::unique_ptr<column> binary_operation(scalar const& lhs, column_view const& rhs, binary_operator op, data_type output_type, rmm::mr::device_memory_resource* mr, cudaStream_t stream) {
  // hard-coded to only work with cudf::string_view so we don't explode compile times
  CUDF_EXPECTS(lhs.type().id() == cudf::STRING, "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(rhs.type().id() == cudf::STRING, "Invalid/Unsupported rhs datatype");
  CUDF_EXPECTS(is_boolean(output_type), "Invalid/Unsupported output datatype");
  return binary_op<cudf::string_view, cudf::string_view, cudf::experimental::bool8>{}(rhs, lhs, op, output_type, true, mr, stream);
}

std::unique_ptr<column> binary_operation(column_view const& lhs, scalar const& rhs, binary_operator op, data_type output_type, rmm::mr::device_memory_resource* mr, cudaStream_t stream) {
  // hard-coded to only work with cudf::string_view so we don't explode compile times
  CUDF_EXPECTS(lhs.type().id() == cudf::STRING, "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(rhs.type().id() == cudf::STRING, "Invalid/Unsupported rhs datatype");
  CUDF_EXPECTS(is_boolean(output_type), "Invalid/Unsupported output datatype");
  return binary_op<cudf::string_view, cudf::string_view, cudf::experimental::bool8>{}(lhs, rhs, op, output_type, false, mr, stream);
}

std::unique_ptr<column> binary_operation(column_view const& lhs, column_view const& rhs, binary_operator op, data_type output_type, rmm::mr::device_memory_resource* mr, cudaStream_t stream) {
  // hard-coded to only work with cudf::string_view so we don't explode compile times
  CUDF_EXPECTS(lhs.type().id() == cudf::STRING, "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(rhs.type().id() == cudf::STRING, "Invalid/Unsupported rhs datatype");
  CUDF_EXPECTS(is_boolean(output_type), "Invalid/Unsupported output datatype");
  return binary_op<cudf::string_view, cudf::string_view, cudf::experimental::bool8>{}(lhs, rhs, op, output_type, mr, stream);
}

}  // namespace compiled
}  // namespace binops
}  // namespace experimental
}  // namespace cudf
