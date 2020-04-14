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

// This functor performs null aware comparison between two columns or a column and a scalar by
// iterating over them on the device
struct null_equals_comparator {
    // This functor does the actual comparison between column value and a scalar
    // or two column values
    template <typename ColT, typename RhsDeviceViewT>
    struct compare_functor {
        column_device_view const lhs_dev_view_;  // Column device view - lhs
        RhsDeviceViewT const rhs_dev_view_;  // Scalar or a column device view - rhs

        compare_functor(column_device_view const& lhs_dev_view,
                        RhsDeviceViewT const& rhs_dev_view)
            : lhs_dev_view_(lhs_dev_view),
              rhs_dev_view_(rhs_dev_view) {}

        // This is used to compare a scalar and a column value
        template <typename RhsViewT = RhsDeviceViewT>
        CUDA_DEVICE_CALLABLE
        typename std::enable_if_t<!std::is_same<RhsViewT, column_device_view>::value, bool>
        operator()(cudf::size_type i) const {
            if (!rhs_dev_view_.is_valid() && !lhs_dev_view_.is_valid(i)) return true;

            if (rhs_dev_view_.is_valid() && lhs_dev_view_.is_valid(i)) {
                return (lhs_dev_view_.element<ColT>(i) == rhs_dev_view_.value());
            }

            return false;
        }

        // This is used to compare 2 column values
        template <typename RhsViewT = RhsDeviceViewT>
        CUDA_DEVICE_CALLABLE
        typename std::enable_if_t<std::is_same<RhsViewT, column_device_view>::value, bool>
        operator()(cudf::size_type i) const {
            if (!rhs_dev_view_.is_valid(i) && !lhs_dev_view_.is_valid(i)) return true;

            if (rhs_dev_view_.is_valid(i) && lhs_dev_view_.is_valid(i)) {
                return lhs_dev_view_.element<ColT>(i) == rhs_dev_view_.template element<ColT>(i);
            }

            return false;
        }
    };

    template <typename ColT>
    auto get_device_view(cudf::scalar const& scalar_item) const {
        return get_scalar_device_view(
                static_cast<cudf::experimental::scalar_type_t<ColT>&>(
                    const_cast<scalar&>(scalar_item)));
    }

    template <typename ColT>
    auto get_device_view(column_device_view const& col_item) const {
        return col_item;
    }

    // This is invoked to perform comparison for all cudf types barring dictionary types
    template <typename ColT, typename RhsT>
    typename std::enable_if_t<!std::is_same<ColT, cudf::dictionary32>::value, void>
    operator()(mutable_column_view& out,
               column_device_view const& lhs_dev_view,
               RhsT const& rhs,
               cudaStream_t stream) const {
        // The rhs item can be a scalar or a column. Use the same interface to get its device view
        auto const rhs_dev_view = get_device_view<ColT>(rhs);

        compare_functor<ColT, typename std::remove_const<decltype(rhs_dev_view)>::type>
            cfunc{lhs_dev_view, rhs_dev_view};

        thrust::transform(rmm::exec_policy(stream)->on(stream),
                          thrust::make_counting_iterator(0),
                          thrust::make_counting_iterator(lhs_dev_view.size()),
                          out.begin<bool>(),
                          cfunc);
    }

    template <typename ColT, typename RhsT>
    typename std::enable_if_t<std::is_same<ColT, cudf::dictionary32>::value, void>
    operator()(mutable_column_view& out,
               column_device_view const& lhs,
               RhsT const& rhs,
               cudaStream_t stream) const {
        CUDF_FAIL("NULL aware comparator for column and scalar not supported for dictionary types");
    }
};

}  // namespace

std::unique_ptr<column> binary_operation(scalar const& lhs, column_view const& rhs, binary_operator op, data_type output_type, rmm::mr::device_memory_resource* mr, cudaStream_t stream) {
  // hard-coded to only work with cudf::string_view so we don't explode compile times
  CUDF_EXPECTS(lhs.type().id() == cudf::STRING, "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(rhs.type().id() == cudf::STRING, "Invalid/Unsupported rhs datatype");
  CUDF_EXPECTS(is_boolean(output_type), "Invalid/Unsupported output datatype");
  return binary_op<cudf::string_view, cudf::string_view, bool>{}(rhs, lhs, op, output_type, true, mr, stream);
}

std::unique_ptr<column> binary_operation(column_view const& lhs, scalar const& rhs, binary_operator op, data_type output_type, rmm::mr::device_memory_resource* mr, cudaStream_t stream) {
  // hard-coded to only work with cudf::string_view so we don't explode compile times
  CUDF_EXPECTS(lhs.type().id() == cudf::STRING, "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(rhs.type().id() == cudf::STRING, "Invalid/Unsupported rhs datatype");
  CUDF_EXPECTS(is_boolean(output_type), "Invalid/Unsupported output datatype");
  return binary_op<cudf::string_view, cudf::string_view, bool>{}(lhs, rhs, op, output_type, false, mr, stream);
}

std::unique_ptr<column> binary_operation(column_view const& lhs, column_view const& rhs, binary_operator op, data_type output_type, rmm::mr::device_memory_resource* mr, cudaStream_t stream) {
  // hard-coded to only work with cudf::string_view so we don't explode compile times
  CUDF_EXPECTS(lhs.type().id() == cudf::STRING, "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(rhs.type().id() == cudf::STRING, "Invalid/Unsupported rhs datatype");
  CUDF_EXPECTS(is_boolean(output_type), "Invalid/Unsupported output datatype");
  return binary_op<cudf::string_view, cudf::string_view, bool>{}(lhs, rhs, op, output_type, mr, stream);
}

std::unique_ptr<column> null_equals(
    column_view const& lhs,
    scalar const& rhs,
    data_type output_type,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream) {
    // Check for datatype
    CUDF_EXPECTS(lhs.type() == rhs.type(), "Both inputs must be of the same type");
    CUDF_EXPECTS(lhs.size() > 0, "Column has to be non empty");
    CUDF_EXPECTS(output_type.id() == type_id::BOOL8, "Output column type has to be bool");

    // Make a bool8 numeric column that is non nullable
    auto out = make_numeric_column(data_type{type_id::BOOL8}, lhs.size(), mask_state::UNALLOCATED,
                                   stream, mr);
    auto out_col_view = out->mutable_view();

    // Create device views for column(s)
    auto const lhs_dev_view = cudf::column_device_view::create(lhs, stream);

    cudf::experimental::type_dispatcher(lhs.type(),
                                        null_equals_comparator{},
                                        out_col_view, *lhs_dev_view, rhs,
                                        stream);

    return out;
}

std::unique_ptr<column> null_equals(
    column_view const& lhs,
    column_view const& rhs,
    data_type output_type,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream) {
    // Check for datatype
    CUDF_EXPECTS(lhs.type() == rhs.type(), "Both columns must be of the same type");
    CUDF_EXPECTS(lhs.size() == rhs.size(), "Both columns must be of the same size");
    CUDF_EXPECTS(lhs.size() > 0, "Columns have to be non empty");
    CUDF_EXPECTS(output_type.id() == type_id::BOOL8, "Output column type has to be bool");

    // Make a bool8 numeric column that is non nullable
    auto out = make_numeric_column(data_type{type_id::BOOL8}, lhs.size(), mask_state::UNALLOCATED,
                                   stream, mr);
    auto out_col_view = out->mutable_view();

    // Create device views for column(s)
    auto const lhs_dev_view = cudf::column_device_view::create(lhs, stream);
    auto const rhs_dev_view = cudf::column_device_view::create(rhs, stream);

    cudf::experimental::type_dispatcher(lhs.type(),
                                        null_equals_comparator{},
                                        out_col_view, *lhs_dev_view, *rhs_dev_view,
                                        stream);

    return out;
}

}  // namespace compiled
}  // namespace binops
}  // namespace experimental
}  // namespace cudf
