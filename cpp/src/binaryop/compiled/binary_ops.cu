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
namespace binops {
namespace compiled {

namespace {

template <typename Lhs, typename Rhs, typename Out>
struct apply_binop {
  binary_operator op;
  apply_binop(binary_operator op) : op(op) {}
  CUDA_DEVICE_CALLABLE Out operator()(Lhs const& x, Rhs const& y) const
  {
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
  CUDA_DEVICE_CALLABLE Out equal(Lhs const& x, Rhs const& y) const
  {
    return static_cast<Out>(x == y);
  }
  CUDA_DEVICE_CALLABLE Out not_equal(Lhs const& x, Rhs const& y) const
  {
    return static_cast<Out>(x != y);
  }
  CUDA_DEVICE_CALLABLE Out less(Lhs const& x, Rhs const& y) const
  {
    return static_cast<Out>(x < y);
  }
  CUDA_DEVICE_CALLABLE Out greater(Lhs const& x, Rhs const& y) const
  {
    return static_cast<Out>(x > y);
  }
  CUDA_DEVICE_CALLABLE Out less_equal(Lhs const& x, Rhs const& y) const
  {
    return static_cast<Out>(x <= y);
  }
  CUDA_DEVICE_CALLABLE Out greater_equal(Lhs const& x, Rhs const& y) const
  {
    return static_cast<Out>(x >= y);
  }
};

template <typename Lhs, typename Rhs, typename Out>
struct apply_binop_scalar_lhs_rhs : apply_binop<Lhs, Rhs, Out> {
  cudf::scalar_device_type_t<Rhs> scalar;
  apply_binop_scalar_lhs_rhs(binary_operator op, cudf::scalar_device_type_t<Rhs> scalar)
    : apply_binop<Lhs, Rhs, Out>(op), scalar(scalar)
  {
  }
  CUDA_DEVICE_CALLABLE Out operator()(Lhs const& x) const
  {
    return apply_binop<Lhs, Rhs, Out>::operator()(x, scalar.value());
  }
};

template <typename Lhs, typename Rhs, typename Out>
struct apply_binop_scalar_rhs_lhs : apply_binop<Rhs, Lhs, Out> {
  cudf::scalar_device_type_t<Rhs> scalar;
  apply_binop_scalar_rhs_lhs(binary_operator op, cudf::scalar_device_type_t<Rhs> scalar)
    : apply_binop<Rhs, Lhs, Out>(op), scalar(scalar)
  {
  }
  CUDA_DEVICE_CALLABLE Out operator()(Lhs const& x) const
  {
    return apply_binop<Rhs, Lhs, Out>::operator()(scalar.value(), x);
  }
};

template <typename Lhs, typename Rhs, typename Out>
struct binary_op {
  std::unique_ptr<column> operator()(column_view const& lhs,
                                     scalar const& rhs,
                                     binary_operator op,
                                     data_type out_type,
                                     bool const reversed,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    auto new_mask = binops::detail::scalar_col_valid_mask_and(lhs, rhs, stream, mr);
    auto out      = make_fixed_width_column(out_type,
                                       lhs.size(),
                                       std::move(new_mask),
                                       rhs.is_valid(stream) ? cudf::UNKNOWN_NULL_COUNT : lhs.size(),
                                       stream,
                                       mr);

    if (lhs.size() > 0 && rhs.is_valid(stream)) {
      auto out_view        = out->mutable_view();
      auto out_itr         = out_view.begin<Out>();
      auto lhs_device_view = column_device_view::create(lhs, stream);
      auto rhs_scalar      = static_cast<cudf::scalar_type_t<Rhs> const&>(rhs);
      auto rhs_scalar_view = get_scalar_device_view(rhs_scalar);
      if (lhs.has_nulls()) {
        auto lhs_itr = cudf::detail::make_null_replacement_iterator(*lhs_device_view, Lhs{});
        reversed
          ? thrust::transform(rmm::exec_policy(stream)->on(stream),
                              lhs_itr,
                              lhs_itr + lhs.size(),
                              out_itr,
                              apply_binop_scalar_rhs_lhs<Lhs, Rhs, Out>{op, rhs_scalar_view})
          : thrust::transform(rmm::exec_policy(stream)->on(stream),
                              lhs_itr,
                              lhs_itr + lhs.size(),
                              out_itr,
                              apply_binop_scalar_lhs_rhs<Lhs, Rhs, Out>{op, rhs_scalar_view});
      } else {
        auto lhs_itr = thrust::make_transform_iterator(
          thrust::make_counting_iterator(size_type{0}),
          [col = *lhs_device_view] __device__(size_type i) { return col.element<Lhs>(i); });
        reversed
          ? thrust::transform(rmm::exec_policy(stream)->on(stream),
                              lhs_itr,
                              lhs_itr + lhs.size(),
                              out_itr,
                              apply_binop_scalar_rhs_lhs<Lhs, Rhs, Out>{op, rhs_scalar_view})
          : thrust::transform(rmm::exec_policy(stream)->on(stream),
                              lhs_itr,
                              lhs_itr + lhs.size(),
                              out_itr,
                              apply_binop_scalar_lhs_rhs<Lhs, Rhs, Out>{op, rhs_scalar_view});
      }
    }

    CHECK_CUDA(stream);

    return out;
  }

  std::unique_ptr<column> operator()(column_view const& lhs,
                                     column_view const& rhs,
                                     binary_operator op,
                                     data_type out_type,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    auto new_mask = bitmask_and(table_view({lhs, rhs}), mr, stream);
    auto out      = make_fixed_width_column(
      out_type, lhs.size(), std::move(new_mask), cudf::UNKNOWN_NULL_COUNT, stream, mr);

    if (lhs.size() > 0) {
      auto out_view        = out->mutable_view();
      auto out_itr         = out_view.begin<Out>();
      auto lhs_device_view = column_device_view::create(lhs, stream);
      auto rhs_device_view = column_device_view::create(rhs, stream);
      if (lhs.has_nulls() && rhs.has_nulls()) {
        auto lhs_itr = cudf::detail::make_null_replacement_iterator(*lhs_device_view, Lhs{});
        auto rhs_itr = cudf::detail::make_null_replacement_iterator(*rhs_device_view, Rhs{});
        thrust::transform(rmm::exec_policy(stream)->on(stream),
                          lhs_itr,
                          lhs_itr + lhs.size(),
                          rhs_itr,
                          out_itr,
                          apply_binop<Lhs, Rhs, Out>{op});
      } else if (lhs.has_nulls()) {
        auto lhs_itr = cudf::detail::make_null_replacement_iterator(*lhs_device_view, Lhs{});
        auto rhs_itr = thrust::make_transform_iterator(
          thrust::make_counting_iterator(size_type{0}),
          [col = *rhs_device_view] __device__(size_type i) { return col.element<Rhs>(i); });
        thrust::transform(rmm::exec_policy(stream)->on(stream),
                          lhs_itr,
                          lhs_itr + lhs.size(),
                          rhs_itr,
                          out_itr,
                          apply_binop<Lhs, Rhs, Out>{op});
      } else if (rhs.has_nulls()) {
        auto lhs_itr = thrust::make_transform_iterator(
          thrust::make_counting_iterator(size_type{0}),
          [col = *lhs_device_view] __device__(size_type i) { return col.element<Lhs>(i); });
        auto rhs_itr = cudf::detail::make_null_replacement_iterator(*rhs_device_view, Rhs{});
        thrust::transform(rmm::exec_policy(stream)->on(stream),
                          lhs_itr,
                          lhs_itr + lhs.size(),
                          rhs_itr,
                          out_itr,
                          apply_binop<Lhs, Rhs, Out>{op});
      } else {
        auto lhs_itr = thrust::make_transform_iterator(
          thrust::make_counting_iterator(size_type{0}),
          [col = *lhs_device_view] __device__(size_type i) { return col.element<Lhs>(i); });
        auto rhs_itr = thrust::make_transform_iterator(
          thrust::make_counting_iterator(size_type{0}),
          [col = *rhs_device_view] __device__(size_type i) { return col.element<Rhs>(i); });
        thrust::transform(rmm::exec_policy(stream)->on(stream),
                          lhs_itr,
                          lhs_itr + lhs.size(),
                          rhs_itr,
                          out_itr,
                          apply_binop<Lhs, Rhs, Out>{op});
      }
    }

    CHECK_CUDA(stream);

    return out;
  }
};

// This functor does the actual comparison between string column value and a scalar string
// or between two string column values using a comparator
template <typename LhsDeviceViewT, typename RhsDeviceViewT, typename OutT, typename CompareFunc>
struct compare_functor {
  LhsDeviceViewT const lhs_dev_view_;  // Scalar or a column device view - lhs
  RhsDeviceViewT const rhs_dev_view_;  // Scalar or a column device view - rhs
  CompareFunc const cfunc_;            // Comparison function

  compare_functor(LhsDeviceViewT const& lhs_dev_view,
                  RhsDeviceViewT const& rhs_dev_view,
                  CompareFunc cf)
    : lhs_dev_view_(lhs_dev_view), rhs_dev_view_(rhs_dev_view), cfunc_(cf)
  {
  }

  // This is used to compare a scalar and a column value
  template <typename LhsViewT = LhsDeviceViewT, typename RhsViewT = RhsDeviceViewT>
  CUDA_DEVICE_CALLABLE
    typename std::enable_if_t<std::is_same<LhsViewT, column_device_view>::value &&
                                !std::is_same<RhsViewT, column_device_view>::value,
                              OutT>
    operator()(cudf::size_type i) const
  {
    return cfunc_(lhs_dev_view_.is_valid(i),
                  rhs_dev_view_.is_valid(),
                  lhs_dev_view_.is_valid(i) ? lhs_dev_view_.template element<cudf::string_view>(i)
                                            : cudf::string_view{},
                  rhs_dev_view_.is_valid() ? rhs_dev_view_.value() : cudf::string_view{});
  }

  // This is used to compare a scalar and a column value
  template <typename LhsViewT = LhsDeviceViewT, typename RhsViewT = RhsDeviceViewT>
  CUDA_DEVICE_CALLABLE
    typename std::enable_if_t<!std::is_same<LhsViewT, column_device_view>::value &&
                                std::is_same<RhsViewT, column_device_view>::value,
                              OutT>
    operator()(cudf::size_type i) const
  {
    return cfunc_(lhs_dev_view_.is_valid(),
                  rhs_dev_view_.is_valid(i),
                  lhs_dev_view_.is_valid() ? lhs_dev_view_.value() : cudf::string_view{},
                  rhs_dev_view_.is_valid(i) ? rhs_dev_view_.template element<cudf::string_view>(i)
                                            : cudf::string_view{});
  }

  // This is used to compare 2 column values
  template <typename LhsViewT = LhsDeviceViewT, typename RhsViewT = RhsDeviceViewT>
  CUDA_DEVICE_CALLABLE
    typename std::enable_if_t<std::is_same<LhsViewT, column_device_view>::value &&
                                std::is_same<RhsViewT, column_device_view>::value,
                              OutT>
    operator()(cudf::size_type i) const
  {
    return cfunc_(lhs_dev_view_.is_valid(i),
                  rhs_dev_view_.is_valid(i),
                  lhs_dev_view_.is_valid(i) ? lhs_dev_view_.template element<cudf::string_view>(i)
                                            : cudf::string_view{},
                  rhs_dev_view_.is_valid(i) ? rhs_dev_view_.template element<cudf::string_view>(i)
                                            : cudf::string_view{});
  }
};

// This functor performs null aware binop between two columns or a column and a scalar by
// iterating over them on the device
struct null_considering_binop {
  auto get_device_view(cudf::scalar const& scalar_item) const
  {
    return get_scalar_device_view(
      static_cast<cudf::scalar_type_t<cudf::string_view>&>(const_cast<scalar&>(scalar_item)));
  }

  auto get_device_view(column_device_view const& col_item) const { return col_item; }

  template <typename LhsViewT, typename RhsViewT, typename OutT, typename CompareFunc>
  void populate_out_col(LhsViewT const& lhsv,
                        RhsViewT const& rhsv,
                        cudf::size_type col_size,
                        cudaStream_t stream,
                        CompareFunc cfunc,
                        OutT* out_col) const
  {
    // Create binop functor instance
    compare_functor<LhsViewT, RhsViewT, OutT, CompareFunc> binop_func{lhsv, rhsv, cfunc};

    // Execute it on every element
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(col_size),
                      out_col,
                      binop_func);
  }

  // This is invoked to perform comparison between cudf string types
  template <typename LhsT, typename RhsT>
  std::unique_ptr<column> operator()(LhsT const& lhs,
                                     RhsT const& rhs,
                                     binary_operator op,
                                     data_type output_type,
                                     cudf::size_type col_size,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) const
  {
    std::unique_ptr<column> out;
    // Create device views for inputs
    auto const lhs_dev_view = get_device_view(lhs);
    auto const rhs_dev_view = get_device_view(rhs);

    switch (op) {
      case binary_operator::NULL_EQUALS: {
        // Validate input
        CUDF_EXPECTS(output_type.id() == type_id::BOOL8, "Output column type has to be bool");

        // Make a bool8 numeric output column
        out = make_numeric_column(
          data_type{type_id::BOOL8}, col_size, mask_state::ALL_VALID, stream, mr);

        // Create a compare function lambda
        auto equal_func = [] __device__(bool lhs_valid,
                                        bool rhs_valid,
                                        cudf::string_view lhs_value,
                                        cudf::string_view rhs_value) {
          if (!lhs_valid && !rhs_valid) return true;
          if (lhs_valid && rhs_valid) return (lhs_value == rhs_value);
          return false;
        };

        // Populate output column
        populate_out_col(lhs_dev_view,
                         rhs_dev_view,
                         col_size,
                         stream,
                         equal_func,
                         mutable_column_view{*out}.begin<bool>());

        break;
      }

      case binary_operator::NULL_MAX:
      case binary_operator::NULL_MIN: {
        // Validate input
        CUDF_EXPECTS(output_type.id() == lhs.type().id(),
                     "Output column type should match input column type");

        // Shallow copy of the resultant strings
        rmm::device_vector<cudf::string_view> out_col_strings(col_size);

        // Invalid output column strings - null rows
        cudf::string_view const invalid_str{nullptr, 0};

        // Create a compare function lambda
        auto minmax_func = [op, invalid_str] __device__(bool lhs_valid,
                                                        bool rhs_valid,
                                                        cudf::string_view lhs_value,
                                                        cudf::string_view rhs_value) {
          if (!lhs_valid && !rhs_valid)
            return invalid_str;
          else if (lhs_valid && rhs_valid) {
            return (op == binary_operator::NULL_MAX)
                     ? thrust::maximum<cudf::string_view>()(lhs_value, rhs_value)
                     : thrust::minimum<cudf::string_view>()(lhs_value, rhs_value);
          } else if (lhs_valid)
            return lhs_value;
          else
            return rhs_value;
        };

        // Populate output column
        populate_out_col(
          lhs_dev_view, rhs_dev_view, col_size, stream, minmax_func, out_col_strings.data().get());

        // Create an output column with the resultant strings
        out = make_strings_column(out_col_strings, invalid_str, stream, mr);

        break;
      }

      default: {
        CUDF_FAIL("Null aware binop not supported");
      }
    }

    return out;
  }
};

}  // namespace

std::unique_ptr<column> binary_operation(scalar const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr,
                                         cudaStream_t stream)
{
  // hard-coded to only work with cudf::string_view so we don't explode compile times
  CUDF_EXPECTS(lhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(rhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported rhs datatype");
  if (is_null_dependent(op)) {
    if (rhs.size() == 0) return cudf::make_empty_column(output_type);
    auto rhs_device_view = cudf::column_device_view::create(rhs, stream);
    return null_considering_binop{}(lhs, *rhs_device_view, op, output_type, rhs.size(), mr, stream);
  } else {
    CUDF_EXPECTS(is_boolean(output_type), "Invalid/Unsupported output datatype");
    // Should pass the right type of scalar and column_view when specializing binary_op
    return binary_op<cudf::string_view, cudf::string_view, bool>{}(
      rhs, lhs, op, output_type, true, mr, stream);
  }
}

std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         scalar const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr,
                                         cudaStream_t stream)
{
  // hard-coded to only work with cudf::string_view so we don't explode compile times
  CUDF_EXPECTS(lhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(rhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported rhs datatype");
  if (is_null_dependent(op)) {
    if (lhs.size() == 0) return cudf::make_empty_column(output_type);
    auto lhs_device_view = cudf::column_device_view::create(lhs, stream);
    return null_considering_binop{}(*lhs_device_view, rhs, op, output_type, lhs.size(), mr, stream);
  } else {
    CUDF_EXPECTS(is_boolean(output_type), "Invalid/Unsupported output datatype");
    return binary_op<cudf::string_view, cudf::string_view, bool>{}(
      lhs, rhs, op, output_type, false, mr, stream);
  }
}

std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr,
                                         cudaStream_t stream)
{
  // hard-coded to only work with cudf::string_view so we don't explode compile times
  CUDF_EXPECTS(lhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(rhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported rhs datatype");
  if (is_null_dependent(op)) {
    CUDF_EXPECTS(lhs.size() == rhs.size(), "Column sizes do not match");
    if (lhs.size() == 0) return cudf::make_empty_column(output_type);
    auto lhs_device_view = cudf::column_device_view::create(lhs, stream);
    auto rhs_device_view = cudf::column_device_view::create(rhs, stream);
    return null_considering_binop{}(
      *lhs_device_view, *rhs_device_view, op, output_type, lhs.size(), mr, stream);
  } else {
    CUDF_EXPECTS(is_boolean(output_type), "Invalid/Unsupported output datatype");
    return binary_op<cudf::string_view, cudf::string_view, bool>{}(
      lhs, rhs, op, output_type, mr, stream);
  }
}

}  // namespace compiled
}  // namespace binops
}  // namespace cudf
