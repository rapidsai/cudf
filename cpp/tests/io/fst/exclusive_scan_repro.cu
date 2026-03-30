/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/testing_main.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>

#include <vector>

// The operator from logical_stack.cuh - copied here for standalone reproduction
template <typename StackSymbolT>
struct PropagateLastWrite {
  __host__ __device__ StackSymbolT operator()(StackSymbolT const& lhs,
                                               StackSymbolT const& rhs) const
  {
    bool is_rhs_read  = (rhs == read_symbol);
    bool is_lhs_write = (lhs != read_symbol);
    return (is_rhs_read && is_lhs_write) ? lhs : rhs;
  }
  StackSymbolT read_symbol;
};

// CPU reference implementation of exclusive scan with PropagateLastWrite
std::vector<char> cpu_exclusive_scan(std::vector<char> const& input,
                                      char init_value,
                                      char read_symbol)
{
  std::vector<char> output(input.size());
  char aggregate = init_value;

  for (size_t i = 0; i < input.size(); i++) {
    // Exclusive scan: output[i] gets the aggregate BEFORE processing input[i]
    output[i] = aggregate;

    // Update aggregate using PropagateLastWrite logic
    char const& lhs   = aggregate;
    char const& rhs   = input[i];
    bool is_rhs_read  = (rhs == read_symbol);
    bool is_lhs_write = (lhs != read_symbol);
    aggregate         = (is_rhs_read && is_lhs_write) ? lhs : rhs;
  }

  return output;
}

struct ExclusiveScanReproTest : public cudf::test::BaseFixture {};

TEST_F(ExclusiveScanReproTest, PropagateLastWriteBug)
{
  constexpr size_t num_elements     = 8160;
  constexpr char read_symbol        = 'x';
  constexpr char empty_stack_symbol = '_';

  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

  // Create input with exact pattern from failing logical_stack_test
  std::vector<char> h_input(num_elements, read_symbol);

  // Set the exact scattered values from the failing test
  h_input[8073] = '[';
  h_input[8074] = '{';
  h_input[8075] = '[';
  h_input[8076] = '{';  // Last '{' before the gap
  // Positions 8077-8147 are 'x' (read_symbol)
  h_input[8148] = '_';
  h_input[8151] = '{';
  h_input[8152] = '_';
  h_input[8154] = '[';
  h_input[8155] = '_';
  h_input[8157] = '[';
  h_input[8159] = '_';

  // Earlier scattered values
  h_input[8020] = '_';
  h_input[8023] = '{';
  h_input[8057] = '[';
  h_input[8060] = '{';
  h_input[8061] = '[';
  h_input[8068] = '{';

  // Compute CPU reference result
  std::vector<char> h_expected = cpu_exclusive_scan(h_input, empty_stack_symbol, read_symbol);

  // Allocate device memory
  rmm::device_uvector<char> d_input{num_elements, stream_view};
  rmm::device_uvector<char> d_output{num_elements, stream_view};

  CUDF_CUDA_TRY(cudaMemcpyAsync(d_input.data(),
                                 h_input.data(),
                                 num_elements * sizeof(char),
                                 cudaMemcpyHostToDevice,
                                 stream.value()));

  // Get temp storage size
  size_t temp_storage_bytes = 0;
  PropagateLastWrite<char> op{read_symbol};

  CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveScan(nullptr,
                                                temp_storage_bytes,
                                                d_input.data(),
                                                d_output.data(),
                                                op,
                                                empty_stack_symbol,
                                                num_elements,
                                                stream.value()));

  rmm::device_uvector<char> d_temp{temp_storage_bytes, stream_view};

  // Run the GPU scan
  CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveScan(d_temp.data(),
                                                temp_storage_bytes,
                                                d_input.data(),
                                                d_output.data(),
                                                op,
                                                empty_stack_symbol,
                                                num_elements,
                                                stream.value()));

  // Copy GPU output back
  std::vector<char> h_output(num_elements);
  CUDF_CUDA_TRY(cudaMemcpyAsync(h_output.data(),
                                 d_output.data(),
                                 num_elements * sizeof(char),
                                 cudaMemcpyDeviceToHost,
                                 stream.value()));
  stream.synchronize();

  // Compare GPU output against CPU reference for all positions
  for (size_t i = 0; i < num_elements; i++) {
    EXPECT_EQ(h_output[i], h_expected[i]) << "Mismatch at position " << i << ": GPU='"
                                           << h_output[i] << "', CPU='" << h_expected[i] << "'";
  }
}

CUDF_TEST_PROGRAM_MAIN()
