/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include "jitify.hpp"

#include <cudf/strings/detail/char_tables.hpp>
#include <cudf/strings/udf/dstring.cuh>
#include <cudf/strings/udf/udf_apis.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuda_runtime.h>

namespace {

rmm::device_buffer create_dstring_array(cudf::size_type size, rmm::cuda_stream_view stream)
{
  auto const output_vector_size = size * sizeof(cudf::strings::udf::dstring);
  rmm::device_buffer result(output_vector_size, stream);
  cudaMemset(result.data(), 0, output_vector_size);
  return result;
}

struct free_dstring_fn {
  cudf::strings::udf::dstring *d_strings;
  __device__ void operator()(cudf::size_type idx) { d_strings[idx].clear(); }
};

void free_dstring_array(void *d_buffer, std::size_t buffer_size, rmm::cuda_stream_view stream)
{
  auto const size = static_cast<cudf::size_type>(buffer_size / sizeof(cudf::strings::udf::dstring));
  auto d_strings  = reinterpret_cast<cudf::strings::udf::dstring *>(d_buffer);
  thrust::for_each_n(
    rmm::exec_policy(stream), thrust::make_counting_iterator(0), size, free_dstring_fn{d_strings});
}

void set_malloc_heap_size(size_t heap_size)
{
  size_t max_malloc_heap_size = 0;
  cudaDeviceGetLimit(&max_malloc_heap_size, cudaLimitMallocHeapSize);
  if (max_malloc_heap_size < heap_size) {
    max_malloc_heap_size = heap_size;
    if (cudaDeviceSetLimit(cudaLimitMallocHeapSize, max_malloc_heap_size) != cudaSuccess) {
      fprintf(stderr, "could not set malloc heap size to %ldMB\n", (heap_size / (1024 * 1024)));
      throw std::runtime_error("");
    }
  }
}

struct dstring_to_string_view_transform_fn {
  __device__ cudf::string_view operator()(cudf::strings::udf::dstring const &dstr)
  {
    return cudf::string_view{dstr.data(), dstr.size_bytes()};
  }
};

}  // namespace

static jitify::JitCache kernel_cache;

std::unique_ptr<udf_module> create_udf_module(std::string const &udf_code,
                                              std::vector<std::string> const &options)
{
  std::vector<std::string> jitify_options;
  jitify_options.push_back("-std=c++17");
  jitify_options.push_back("-DCUDF_JIT_UDF");
  jitify_options.push_back("-I../include");
  jitify_options.push_back("-I../cpp/include");
  std::copy(options.begin(), options.end(), std::back_inserter(jitify_options));
  // nvrtc did not recognize --expt-relaxed-constexpr
  // also it has trouble including thrust headers

  try {
    auto program = kernel_cache.program(udf_code.c_str(), 0, jitify_options);
    return std::make_unique<udf_module>(new jitify::Program(std::move(program)));
  } catch (std::runtime_error &exc) {
    return nullptr;
  }
}

udf_module::~udf_module() { delete program; }

namespace {

using column_span = std::pair<void *, cudf::size_type>;

struct udf_data_fn {
  cudf::column_view const input;

  template <typename T, typename std::enable_if_t<cudf::is_fixed_width<T>()> * = nullptr>
  column_span operator()(std::vector<rmm::device_uvector<cudf::string_view>> &)
  {
    return std::make_pair((void *)input.data<T>(), input.size());
  }

  template <typename T, typename std::enable_if_t<std::is_same_v<T, cudf::string_view>> * = nullptr>
  column_span operator()(std::vector<rmm::device_uvector<cudf::string_view>> &strings_vectors)
  {
    auto sv =
      cudf::strings::detail::create_string_vector_from_column(cudf::strings_column_view(input));
    strings_vectors.emplace_back(std::move(sv));
    auto const &sv_ref = strings_vectors.back();
    return std::make_pair((void *)sv_ref.data(), (cudf::size_type)sv_ref.size());
  }

  template <typename T,
            typename std::enable_if_t<!std::is_same<T, cudf::string_view>::value and
                                      !cudf::is_fixed_width<T>()> * = nullptr>
  column_span operator()(std::vector<rmm::device_uvector<cudf::string_view>> &)
  {
    return column_span{nullptr, 0};  // throw error here?
  }
};

std::unique_ptr<rmm::device_buffer> to_string_view_array(cudf::column_view const input,
                                                         rmm::cuda_stream_view stream)
{
  return std::make_unique<rmm::device_buffer>(
    std::move(cudf::strings::detail::create_string_vector_from_column(
                cudf::strings_column_view(input), stream)
                .release()));
}

std::unique_ptr<cudf::column> from_dstring_array(void *d_buffer,
                                                 std::size_t buffer_size,
                                                 rmm::cuda_stream_view stream)
{
  auto const size = static_cast<cudf::size_type>(buffer_size / sizeof(cudf::strings::udf::dstring));
  auto d_input    = reinterpret_cast<cudf::strings::udf::dstring *>(d_buffer);

  // create string_views of the dstrings
  auto indices = rmm::device_uvector<cudf::string_view>(size, stream);
  thrust::transform(rmm::exec_policy(stream),
                    d_input,
                    d_input + size,
                    indices.data(),
                    dstring_to_string_view_transform_fn{});

  auto results = cudf::make_strings_column(indices, cudf::string_view(nullptr, 0), stream);

  // free the individual dstring elements
  free_dstring_array(d_buffer, buffer_size, stream);

  // return new column
  return results;
}

template <typename KernelType>
void set_global_variables(KernelType &kernel)
{
  try {
    // set global variable data needed for the is-char-type functions
    if (kernel.get_global_ptr("cudf::strings::udf::g_character_flags_table")) {
      auto flags_table = cudf::strings::detail::get_character_flags_table();
      kernel.set_global_array("cudf::strings::udf::g_character_flags_table", &flags_table, 1);
    }
  } catch (...) {
    // this global variable is optional
  }
  try {
    // set global variable data needed for upper/lower functions
    if (kernel.get_global_ptr("cudf::strings::udf::g_character_cases_table")) {
      auto cases_table = cudf::strings::detail::get_character_cases_table();
      kernel.set_global_array("cudf::strings::udf::g_character_cases_table", &cases_table, 1);
    }
    if (kernel.get_global_ptr("cudf::strings::udf::g_special_case_mapping_table")) {
      auto special_cases_table = cudf::strings::detail::get_special_case_mapping_table();
      kernel.set_global_array(
        "cudf::strings::udf::g_special_case_mapping_table", &special_cases_table, 1);
    }
  } catch (...) {
    // these global variables are optional
  }
}

}  // namespace

std::unique_ptr<cudf::column> call_udf(udf_module const &udf,
                                       std::string const &udf_name,
                                       cudf::size_type output_size,
                                       std::vector<cudf::column_view> input,
                                       size_t heap_size)
{
  set_malloc_heap_size(heap_size);

  rmm::cuda_stream_view stream = rmm::cuda_stream_default;

  // setup kernel parameters
  std::vector<rmm::device_uvector<cudf::string_view>> strings_vectors;  // for strings columns
  std::vector<void *> args;                       // holds column pointers and sizes;
  jitify::detail::vector<std::string> arg_types;  // this parameter is not strictly required
  // create args for each input column
  for (int col = 0; col < (int)input.size(); ++col) {
    column_span data = cudf::type_dispatcher<cudf::dispatch_storage_type>(
      input[col].type(), udf_data_fn{input[col]}, strings_vectors);
    args.push_back(data.first);
    arg_types.push_back(jitify::reflection::reflect<void *>());
    args.push_back((void *)(long)data.second);
    arg_types.push_back(jitify::reflection::reflect<cudf::size_type>());
  }
  // transform required because jit launch() args are expected to be pointers to pointers
  std::vector<void *> jitargs;
  std::transform(
    args.begin(), args.end(), std::back_inserter(jitargs), [](auto &pv) { return &pv; });

  // allocate an output array
  rmm::device_buffer output = create_dstring_array(output_size, stream);

  // add the output strings column parameter
  void *d_out = output.data();
  jitargs.push_back(&d_out);
  arg_types.push_back(jitify::reflection::reflect<void *>());
  // add the kernel thread count parameter
  jitargs.push_back(reinterpret_cast<void *>(&output_size));
  arg_types.push_back(jitify::reflection::reflect<cudf::size_type>());

  // setup kernel launch parameters
  auto const num_blocks = ((output_size - 1) / 128) + 1;
  dim3 grid(num_blocks);
  dim3 block(128);

  jitify::Program *pp = udf.program;
  auto kernel         = pp->kernel(udf_name.c_str()).instantiate();
  set_global_variables(kernel);
  auto launcher = kernel.configure(grid, block);
  // launch the kernel passing the parameters
  CUresult result = launcher.launch(jitargs, arg_types);
  if (result) {
    const char *result_str = "ok";
    cuGetErrorName(result, &result_str);
    fprintf(stderr, "launch result = %d [%s]\n", (int)result, result_str);
  }
  auto const err = cudaDeviceSynchronize();
  if (err) { fprintf(stderr, "%s=(%d) ", udf_name.c_str(), (int)err); }

  // convert the output array into a strings column
  // this also frees the individual dstring objects
  return from_dstring_array(output.data(), output.size(), stream);
}

std::unique_ptr<rmm::device_buffer> to_string_view_array(cudf::column_view const input)
{
  return to_string_view_array(input, rmm::cuda_stream_default);
}

std::unique_ptr<cudf::column> from_dstring_array(void *d_buffer, std::size_t buffer_size)
{
  return from_dstring_array(d_buffer, buffer_size, rmm::cuda_stream_default);
}
