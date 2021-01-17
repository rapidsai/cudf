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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/rolling.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <src/rolling/rolling_detail.hpp>

#include <thrust/iterator/constant_iterator.h>

#include <algorithm>
#include <vector>
struct CollectListTest : public cudf::test::BaseFixture {};

#include <cudf/binaryop.hpp>
#include <thrust/execution_policy.h>
#include <cudf/strings/detail/utilities.cuh>
#include <numeric>
#include <iostream>

cudf::size_type get_num_child_rows(cudf::column_view const& list_offsets,
                                   rmm::cuda_stream_view stream = rmm::cuda_stream_default)
{
  // Number of rows in child-column == last offset value.
  cudf::size_type num_child_rows{};
  CUDA_TRY(cudaMemcpyAsync(&num_child_rows,
                           list_offsets.data<cudf::size_type>() + list_offsets.size() - 1,
                           sizeof(cudf::size_type),
                           cudaMemcpyDeviceToHost,
                           stream.value()));
  stream.synchronize();
  return num_child_rows;
}

void print(std::string const& msg, cudf::column_view col)
{
  std::cout << msg << std::endl;
  cudf::test::print(col);
  std::cout << std::endl;
}

void print(std::string const& msg, std::unique_ptr<cudf::column> const& col)
{
    print(msg, col->view());
}

void foo()
{
  using namespace cudf;
  using namespace cudf::test;

  auto size_data_type = data_type{type_to_id<size_type>()};

  auto ints_column = fixed_width_column_wrapper<size_type>{70,71,72,73,74};

  auto prev_column = fixed_width_column_wrapper<size_type>{1,2,2,2,2};
  auto foll_column = fixed_width_column_wrapper<size_type>{1,1,1,1,0};

  CUDF_EXPECTS(static_cast<column_view>(prev_column).size() == static_cast<column_view>(foll_column).size(), "");

  auto sizes = cudf::binary_operation(prev_column, foll_column, binary_operator::ADD, data_type{size_data_type});
  auto offsets = cudf::strings::detail::make_offsets_child_column(sizes->view().begin<size_type>(), sizes->view().end<size_type>());
  print("Offsets:", offsets);

  // Bail if offsets.size() < 2;

  auto scatter_map = make_fixed_width_column(size_data_type, offsets->size()-2);
  thrust::copy(thrust::device, offsets->view().begin<size_type>()+1, offsets->view().end<size_type>()-1, scatter_map->mutable_view().begin<size_type>());
  print("Scatter_map: ", scatter_map);

  auto scatter_input = make_fixed_width_column(size_data_type, offsets->size()-2);
  thrust::fill_n(thrust::device, scatter_input->mutable_view().begin<size_type>(), offsets->size()-2, size_type{1});

  auto num_child_rows = get_num_child_rows(offsets->view());

  auto child_index_column_input = make_fixed_width_column(size_data_type , num_child_rows); 
  thrust::fill_n(thrust::device, child_index_column_input->mutable_view().begin<size_type>(), num_child_rows, 0);

  thrust::scatter(thrust::device, scatter_input->view().begin<size_type>(), scatter_input->view().end<size_type>(), 
    scatter_map->view().begin<size_type>(), child_index_column_input->mutable_view().begin<size_type>());

  auto per_row_group_mapping = make_fixed_width_column(size_data_type, num_child_rows);
  thrust::inclusive_scan(thrust::device, 
                         child_index_column_input->view().begin<size_type>(), 
                         child_index_column_input->view().end<size_type>(), 
                         per_row_group_mapping->mutable_view().begin<size_type>());
  print("Per_row_group_mapping: ", per_row_group_mapping);

  auto gather_map = make_fixed_width_column(size_data_type, num_child_rows);
  thrust::for_each_n(
      thrust::device,
      thrust::make_counting_iterator<size_type>(0),
      num_child_rows,
      [d_offsets = offsets->view().begin<size_type>(),               // [0,   2,     5,     8,     11, 13]
       d_groups  = per_row_group_mapping->view().begin<size_type>(), // [0,0, 1,1,1, 2,2,2, 3,3,3, 4,4]
       d_prev    = static_cast<column_view>(prev_column).data<size_type>(),
       d_output  = gather_map->mutable_view().begin<size_type>()]          
       __device__(auto i) {
           auto group = d_groups[i];
           auto group_start_offset = d_offsets[group];
           auto relative_index = i - group_start_offset;

           d_output[i] = (group - d_prev[group] + 1) + relative_index;
       }
  );
  print("Gather_map: ", gather_map);

  auto input_columns = std::vector<std::unique_ptr<column>>{};
  input_columns.emplace_back(std::make_unique<column>(ints_column));
  auto input_table = cudf::table{std::move(input_columns)};

  auto output_table = cudf::gather(input_table.view(), gather_map->view());

  print("Gathered column: ", output_table->get_column(0).view());

}

TEST_F(CollectListTest, ProofOfConcept)
{
  foo();
}

TEST_F(CollectListTest, Integration)
{
  using namespace cudf;
  using namespace cudf::test;

  auto size_data_type = data_type{type_to_id<size_type>()};

  auto ints_column = fixed_width_column_wrapper<size_type>{70,71,72,73,74};

  auto prev_column = fixed_width_column_wrapper<size_type>{1,2,2,2,2};
  auto foll_column = fixed_width_column_wrapper<size_type>{1,1,1,1,0};

  CUDF_EXPECTS(static_cast<column_view>(prev_column).size() == static_cast<column_view>(foll_column).size(), "");

  auto sizes = cudf::rolling_window(ints_column, prev_column, foll_column, 1, make_collect_aggregation());

  print("Sizes: ", *sizes);
 
}

CUDF_TEST_PROGRAM_MAIN()