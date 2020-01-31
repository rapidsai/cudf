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
#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/detail/utilities/cuda.cuh>

#include <join/join_common_utils.hpp>
#include <join/join_kernels.cuh>

namespace cudf {

namespace join {

namespace detail {

class join_operation_comparator {
public:
  __host__ join_operation_comparator(join_comparison_operator comparison) {
    cmp = true;
    if (comparison == join_comparison_operator::LESS_THAN) {
      value = cudf::experimental::weak_ordering::LESS;
    } else if (comparison == join_comparison_operator::LESS_THAN_OR_EQUAL) {
      cmp = false;
      value = cudf::experimental::weak_ordering::GREATER;
    } else if (comparison == join_comparison_operator::EQUAL) {
      value = cudf::experimental::weak_ordering::EQUIVALENT;
    } else if (comparison == join_comparison_operator::NOT_EQUAL) {
      cmp = false;
      value = cudf::experimental::weak_ordering::EQUIVALENT;
    } else if (comparison == join_comparison_operator::GREATER_THAN) {
      value = cudf::experimental::weak_ordering::GREATER;
    } else if (comparison == join_comparison_operator::GREATER_THAN_OR_EQUAL) {
      cmp = false;
      value = cudf::experimental::weak_ordering::LESS;
    }
  }

  bool __device__ operator()(cudf::experimental::weak_ordering v) {
    return (v == value) == cmp;
  }

private:
  bool                               cmp;
  cudf::experimental::weak_ordering  value;
};
  

class join_operation_dispatch {
public:
  __host__ join_operation_dispatch(table_device_view const &left, table_device_view const &right, join_operation const & operation):
    _left(left), _right(right), _compare(operation.op),
    _left_column_index(operation.left_column_idx),
    _right_column_index(operation.right_column_idx) {}

  data_type __device__ type() const noexcept { return _left.column(_left_column_index).type(); }

  template <typename Element>
  bool __device__ operator()(cudf::size_type left_index, cudf::size_type right_index) {

    auto left = _left.column(_left_column_index).element<Element>(left_index);
    auto right = _right.column(_right_column_index).element<Element>(right_index);

    auto cmp = cudf::experimental::relational_compare<Element>(left, right);

    return _compare(cmp);
  }

private:
  table_device_view           _left;
  table_device_view           _right;
  join_operation_comparator   _compare;
  cudf::size_type             _left_column_index;
  cudf::size_type             _right_column_index;
};

__device__ bool match_operations(size_t num_operations, 
                                 join_operation_dispatch const* operations,
                                 cudf::size_type l,
                                 cudf::size_type r) {
  bool match = true;

  for (size_t k = 0 ; (k < num_operations) && match ; ++k) {
    match = cudf::experimental::type_dispatcher(operations[k].type(),
                                                operations[k],
                                                l, r);
  }
                                        
  return match;
}


rmm::device_vector<int64_t> nested_join_indices(table_view const& left,
                                                table_view const& right,
                                                std::vector<join_operation> const& join_ops,
                                                cudaStream_t stream) {


  // TODO:  This should be done before we come into this
  //         If we do this outside then I'm not sure I need JoinKind
  //         as a template parameter here...
  //
  //Trivial left join case - exit early
  //if ((JoinKind == join_kind::LEFT_JOIN) && (right.num_rows() == 0)) {
  //  return get_trivial_left_join_indices(left, stream);
  //}

  auto d_left = table_device_view::create(left, stream);
  auto d_right = table_device_view::create(right, stream);

  const cudf::size_type left_num_rows{left.num_rows()};
  const cudf::size_type right_num_rows{right.num_rows()};

  rmm::device_vector<join_operation_dispatch> operations;
  for (auto op : join_ops) {
    operations.push_back(join_operation_dispatch{*d_left, *d_right, op});
  }

  auto num_operations = operations.size();
  join_operation_dispatch *d_operations = operations.data().get();

  //
  //   Naive implementation
  //
  auto output_size = thrust::count_if(rmm::exec_policy(stream)->on(stream),
                                      thrust::make_counting_iterator<cudf::size_type>(0),
                                      thrust::make_counting_iterator<cudf::size_type>(left_num_rows * right_num_rows),
                                      [d_operations, num_operations, right_num_rows] __device__ (cudf::size_type index) {
                                        cudf::size_type l = index / right_num_rows;
                                        cudf::size_type r = index % right_num_rows;

                                        return match_operations(num_operations, d_operations, l, r);
                                      });
                                      
  
  // If the output size is zero, return immediately
  if (output_size == 0) {
    return rmm::device_vector<int64_t>{};
  }

  rmm::device_vector<int64_t> output_indices(output_size);

  thrust::copy_if(rmm::exec_policy(stream)->on(stream),
                  thrust::make_counting_iterator<cudf::size_type>(0),
                  thrust::make_counting_iterator<cudf::size_type>(left_num_rows * right_num_rows),
                  output_indices.begin(),
                  [d_operations, num_operations, right_num_rows] __device__ (cudf::size_type index) {
                    cudf::size_type l = index / right_num_rows;
                    cudf::size_type r = index % right_num_rows;

                    return match_operations(num_operations, d_operations, l, r);
                  });

  return output_indices;
}

}//namespace detail

}//namespace join

}//namespace cudf
