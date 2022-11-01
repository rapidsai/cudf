/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/rmm_pool_raii.hpp>
#include <nvbench/nvbench.cuh>

#include <io/fst/lookup_tables.cuh>
#include <io/utilities/hostdevice_vector.hpp>  //TODO find better replacement

#include <tests/io/fst/common.hpp>

#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/discard_iterator.h>

#include <cstdlib>

namespace cudf {
namespace {
auto make_test_json_data(nvbench::state& state)
{
  auto const string_size{size_type(state.get_int64("string_size"))};

  // Test input
  std::string input = R"(  {)"
                      R"("category": "reference",)"
                      R"("index:" [4,12,42],)"
                      R"("author": "Nigel Rees",)"
                      R"("title": "Sayings of the Century",)"
                      R"("price": 8.95)"
                      R"(}  )"
                      R"({)"
                      R"("category": "reference",)"
                      R"("index:" [4,{},null,{"a":[]}],)"
                      R"("author": "Nigel Rees",)"
                      R"("title": "Sayings of the Century",)"
                      R"("price": 8.95)"
                      R"(}  {} [] [ ])";

  auto d_input_scalar          = cudf::make_string_scalar(input);
  auto& d_string_scalar        = static_cast<cudf::string_scalar&>(*d_input_scalar);
  const size_type repeat_times = string_size / input.size();
  return cudf::strings::repeat_string(d_string_scalar, repeat_times);
}

using namespace cudf::test::io::json;
// Type used to represent the atomic symbol type used within the finite-state machine
using SymbolT = char;
// Type sufficiently large to index symbols within the input and output (may be unsigned)
using SymbolOffsetT = uint32_t;
// Helper class to set up transition table, symbol group lookup table, and translation table
using DfaFstT = cudf::io::fst::detail::Dfa<char, NUM_SYMBOL_GROUPS, TT_NUM_STATES>;
constexpr std::size_t single_item = 1;

}  // namespace

void BM_FST_JSON(nvbench::state& state)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cudf::rmm_pool_raii rmm_pool;

  CUDF_EXPECTS(state.get_int64("string_size") <= std::numeric_limits<size_type>::max(),
               "Benchmarks only support up to size_type's maximum number of items");
  auto const string_size{size_type(state.get_int64("string_size"))};
  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

  auto input_string = make_test_json_data(state);
  auto& d_input     = static_cast<cudf::scalar_type_t<std::string>&>(*input_string);

  state.add_element_count(d_input.size());

  // Prepare input & output buffers
  hostdevice_vector<SymbolT> output_gpu(d_input.size(), stream_view);
  hostdevice_vector<SymbolOffsetT> output_gpu_size(single_item, stream_view);
  hostdevice_vector<SymbolOffsetT> out_indexes_gpu(d_input.size(), stream_view);

  // Run algorithm
  DfaFstT parser{pda_sgs, pda_state_tt, pda_out_tt, stream.value()};

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // Allocate device-side temporary storage & run algorithm
    parser.Transduce(d_input.data(),
                     static_cast<SymbolOffsetT>(d_input.size()),
                     output_gpu.device_ptr(),
                     out_indexes_gpu.device_ptr(),
                     output_gpu_size.device_ptr(),
                     start_state,
                     stream.value());
  });
}

void BM_FST_JSON_no_outidx(nvbench::state& state)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cudf::rmm_pool_raii rmm_pool;

  CUDF_EXPECTS(state.get_int64("string_size") <= std::numeric_limits<size_type>::max(),
               "Benchmarks only support up to size_type's maximum number of items");
  auto const string_size{size_type(state.get_int64("string_size"))};
  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

  auto input_string = make_test_json_data(state);
  auto& d_input     = static_cast<cudf::scalar_type_t<std::string>&>(*input_string);

  state.add_element_count(d_input.size());

  // Prepare input & output buffers
  hostdevice_vector<SymbolT> output_gpu(d_input.size(), stream_view);
  hostdevice_vector<SymbolOffsetT> output_gpu_size(single_item, stream_view);
  hostdevice_vector<SymbolOffsetT> out_indexes_gpu(d_input.size(), stream_view);

  // Run algorithm
  DfaFstT parser{pda_sgs, pda_state_tt, pda_out_tt, stream.value()};

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // Allocate device-side temporary storage & run algorithm
    parser.Transduce(d_input.data(),
                     static_cast<SymbolOffsetT>(d_input.size()),
                     output_gpu.device_ptr(),
                     thrust::make_discard_iterator(),
                     output_gpu_size.device_ptr(),
                     start_state,
                     stream.value());
  });
}

void BM_FST_JSON_no_out(nvbench::state& state)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cudf::rmm_pool_raii rmm_pool;

  CUDF_EXPECTS(state.get_int64("string_size") <= std::numeric_limits<size_type>::max(),
               "Benchmarks only support up to size_type's maximum number of items");
  auto const string_size{size_type(state.get_int64("string_size"))};
  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

  auto input_string = make_test_json_data(state);
  auto& d_input     = static_cast<cudf::scalar_type_t<std::string>&>(*input_string);

  state.add_element_count(d_input.size());

  // Prepare input & output buffers
  hostdevice_vector<SymbolOffsetT> output_gpu_size(single_item, stream_view);

  // Run algorithm
  DfaFstT parser{pda_sgs, pda_state_tt, pda_out_tt, stream.value()};

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // Allocate device-side temporary storage & run algorithm
    parser.Transduce(d_input.data(),
                     static_cast<SymbolOffsetT>(d_input.size()),
                     thrust::make_discard_iterator(),
                     thrust::make_discard_iterator(),
                     output_gpu_size.device_ptr(),
                     start_state,
                     stream.value());
  });
}

void BM_FST_JSON_no_str(nvbench::state& state)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cudf::rmm_pool_raii rmm_pool;

  CUDF_EXPECTS(state.get_int64("string_size") <= std::numeric_limits<size_type>::max(),
               "Benchmarks only support up to size_type's maximum number of items");
  auto const string_size{size_type(state.get_int64("string_size"))};
  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

  auto input_string = make_test_json_data(state);
  auto& d_input     = static_cast<cudf::scalar_type_t<std::string>&>(*input_string);

  state.add_element_count(d_input.size());

  // Prepare input & output buffers
  hostdevice_vector<SymbolOffsetT> output_gpu_size(single_item, stream_view);
  hostdevice_vector<SymbolOffsetT> out_indexes_gpu(d_input.size(), stream_view);

  // Run algorithm
  DfaFstT parser{pda_sgs, pda_state_tt, pda_out_tt, stream.value()};

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // Allocate device-side temporary storage & run algorithm
    parser.Transduce(d_input.data(),
                     static_cast<SymbolOffsetT>(d_input.size()),
                     thrust::make_discard_iterator(),
                     out_indexes_gpu.device_ptr(),
                     output_gpu_size.device_ptr(),
                     start_state,
                     stream.value());
  });
}

NVBENCH_BENCH(BM_FST_JSON)
  .set_name("FST_JSON")
  .add_int64_power_of_two_axis("string_size", nvbench::range(20, 30, 1));

NVBENCH_BENCH(BM_FST_JSON_no_outidx)
  .set_name("FST_JSON_no_outidx")
  .add_int64_power_of_two_axis("string_size", nvbench::range(20, 30, 1));

NVBENCH_BENCH(BM_FST_JSON_no_out)
  .set_name("FST_JSON_no_out")
  .add_int64_power_of_two_axis("string_size", nvbench::range(20, 30, 1));

NVBENCH_BENCH(BM_FST_JSON_no_str)
  .set_name("FST_JSON_no_str")
  .add_int64_power_of_two_axis("string_size", nvbench::range(20, 30, 1));

}  // namespace cudf
