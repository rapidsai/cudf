/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit_filter_join_indices_kernel.cuh"
#include "jit/filter_join_kernel.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/join/join.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cub/cub.cuh>
#include <cuco/static_set.cuh>
#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>

#include <jit/cache.hpp>
#include <jit/helpers.hpp>
#include <jit/parser.hpp>

#include <memory>
#include <utility>

// Note: No need for JITIFY_HASH_CUSTOM_TYPE for function templates

namespace cudf {
namespace detail {

namespace {

// Get the JIT kernel program for join filtering
jitify2::Kernel get_join_filter_kernel(std::string const& kernel_name,
                                      std::string const& cuda_source,
                                      jitify2::StringVec const& template_args)
{
  CUDF_FUNC_RANGE();

  // Create a program from source files
  static jitify2::Program program = jitify2::Program::create("filter_join_kernel",
                                                             {"join/jit/filter_join_kernel.cu"});

  // Preprocess the program
  auto preprog = program.value().preprocess({"-std=c++20"});

  // Get program cache
  auto& cache = cudf::jit::get_program_cache(preprog.value());

  // Get kernel with custom header sources
  std::map<std::string, std::string> header_sources = {
    {"cudf/detail/operation-udf.hpp", cuda_source}
  };

  return cache.get_kernel(kernel_name, template_args, header_sources, {"-arch=sm_."});
}

// Build template parameters for JIT kernel
jitify2::StringVec build_join_filter_template_params(
  std::vector<column_view> const& left_columns,
  std::vector<column_view> const& right_columns,
  bool has_user_data)
{
  jitify2::StringVec template_params;
  
  // Add has_user_data template parameter
  template_params.emplace_back(jitify2::reflection::reflect(has_user_data));
  
  // Add left column accessors
  for (size_t i = 0; i < left_columns.size(); ++i) {
    auto const& col = left_columns[i];
    std::string type_name = cudf::type_to_name(col.type());
    template_params.emplace_back(
      jitify2::reflection::Template("cudf::jit::join_left_column_accessor")
        .instantiate(type_name, std::to_string(i)));
  }
  
  // Add right column accessors  
  for (size_t i = 0; i < right_columns.size(); ++i) {
    auto const& col = right_columns[i];
    std::string type_name = cudf::type_to_name(col.type());
    template_params.emplace_back(
      jitify2::reflection::Template("cudf::jit::join_right_column_accessor")
        .instantiate(type_name, std::to_string(i)));
  }
  
  return template_params;
}

// Build the JIT kernel for join filtering
jitify2::ConfiguredKernel build_join_filter_kernel(
  std::string const& predicate_code,
  std::vector<column_view> const& left_columns,
  std::vector<column_view> const& right_columns,
  bool is_ptx,
  bool has_user_data,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  
  // Parse predicate code
  auto const cuda_source = is_ptx 
    ? cudf::jit::parse_single_function_ptx(
        predicate_code, 
        "GENERIC_JOIN_FILTER_OP",
        cudf::jit::build_ptx_params(
          cudf::jit::column_type_names(left_columns),
          cudf::jit::column_type_names(right_columns),
          has_user_data))
    : cudf::jit::parse_single_function_cuda(predicate_code, "GENERIC_JOIN_FILTER_OP");
  
  // Build template parameters
  auto template_args = build_join_filter_template_params(left_columns, right_columns, has_user_data);
  
  // Get compiled kernel
  auto kernel = get_join_filter_kernel(
    jitify2::reflection::Template("cudf::join::jit::filter_join_kernel")
      .instantiate(template_args),
    cuda_source,
    template_args);
    
  return kernel->configure_1d_max_occupancy(0, 0, nullptr, stream.value());
}

// Launch the JIT kernel for join filtering
void launch_join_filter_kernel(
  jitify2::ConfiguredKernel& kernel,
  cudf::table_view const& left,
  cudf::table_view const& right,
  cudf::device_span<size_type const> left_indices,
  cudf::device_span<size_type const> right_indices,
  bool* predicate_results,
  std::optional<void*> user_data,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  
  // Create device views of tables
  std::vector<column_view> left_cols(left.begin(), left.end());
  std::vector<column_view> right_cols(right.begin(), right.end());

  auto [left_handles, left_device_views] =
    cudf::jit::column_views_to_device<column_device_view, column_view>(left_cols, stream, mr);
  auto [right_handles, right_device_views] =
    cudf::jit::column_views_to_device<column_device_view, column_view>(right_cols, stream, mr);
  
  // Set up kernel parameters
  cudf::device_span<cudf::size_type const> left_span = left_indices;
  cudf::device_span<cudf::size_type const> right_span = right_indices;
  cudf::column_device_view_core const* left_tables_ptr = left_device_views.data();
  cudf::column_device_view_core const* right_tables_ptr = right_device_views.data();
  void* user_data_ptr = user_data.value_or(nullptr);
  
  std::array<void*, 6> args{
    &left_span,
    &right_span, 
    &left_tables_ptr,
    &right_tables_ptr,
    &predicate_results,
    &user_data_ptr
  };
  
  kernel->launch_raw(args.data());
}

// Same join semantics handling as the AST version
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
apply_join_semantics(cudf::table_view const& left,
                    cudf::table_view const& right,
                    cudf::device_span<size_type const> left_indices,
                    cudf::device_span<size_type const> right_indices,
                    rmm::device_uvector<bool> const& predicate_results,
                    join_kind join_kind,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr)
{
  auto make_empty_result = [&]() {
    return std::pair{std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr)};
  };

  auto make_result_vectors = [&](size_t size) {
    return std::pair{std::make_unique<rmm::device_uvector<size_type>>(size, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(size, stream, mr)};
  };

  auto predicate_results_ptr = predicate_results.data();
  auto left_ptr              = left_indices.data();
  auto right_ptr             = right_indices.data();

  // Handle different join semantics - same logic as AST version
  if (join_kind == join_kind::INNER_JOIN) {
    // INNER_JOIN: only keep pairs that satisfy the predicate
    auto valid_predicate = [=] __device__(size_type i) -> bool { return predicate_results_ptr[i]; };

    auto const num_valid =
      thrust::count_if(rmm::exec_policy_nosync(stream),
                       thrust::counting_iterator{0},
                       thrust::counting_iterator{static_cast<size_type>(left_indices.size())},
                       valid_predicate);

    if (num_valid == 0) { return make_empty_result(); }

    auto [filtered_left_indices, filtered_right_indices] = make_result_vectors(num_valid);

    auto input_iter =
      thrust::make_zip_iterator(cuda::std::tuple{left_indices.begin(), right_indices.begin()});
    auto output_iter = thrust::make_zip_iterator(
      cuda::std::tuple{filtered_left_indices->begin(), filtered_right_indices->begin()});

    thrust::copy_if(rmm::exec_policy_nosync(stream),
                    input_iter,
                    input_iter + left_indices.size(),
                    thrust::counting_iterator{0},
                    output_iter,
                    [valid_predicate] __device__(size_type idx) { return valid_predicate(idx); });

    return std::pair{std::move(filtered_left_indices), std::move(filtered_right_indices)};

  } else if (join_kind == join_kind::LEFT_JOIN) {
    // LEFT_JOIN: Keep all left rows, nullify right indices for failed predicates
    // Using same complex logic as AST version with hash set for unmatched left rows
    using SetType =
      cuco::static_set<size_type,
                       cuco::extent<std::size_t>,
                       cuda::thread_scope_device,
                       cuda::std::equal_to<size_type>,
                       cuco::double_hashing<1, cuco::default_hash_function<size_type>>,
                       rmm::mr::polymorphic_allocator<char>>;
    SetType filter_passing_indices{cuco::extent{static_cast<std::size_t>(left.num_rows())},
                                   cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
                                   cuco::empty_key{-1},
                                   {},
                                   {},
                                   {},
                                   {},
                                   {},
                                   stream.value()};

    auto predicate_func = [predicate_results_ptr] __device__(size_t idx) {
      return static_cast<bool>(predicate_results_ptr[idx]);
    };
    auto const num_filter_passing =
      filter_passing_indices.insert_if(left_ptr,
                                       left_ptr + left_indices.size(),
                                       cuda::counting_iterator<size_t>(0),
                                       predicate_func,
                                       stream.value());

    auto const num_invalid = left.num_rows() - num_filter_passing;

    // Rest of LEFT_JOIN logic follows AST implementation...
    cudf::detail::device_scalar<size_t> d_num_valid(stream);
    {
      auto const predicate_it =
        cuda::transform_iterator{predicate_results_ptr,
                                 cuda::proclaim_return_type<size_t>(
                                   [] __device__(auto val) -> size_t { return val ? 1 : 0; })};
      size_t temp_storage_bytes = 0;
      cub::DeviceReduce::Sum(nullptr,
                             temp_storage_bytes,
                             predicate_it,
                             d_num_valid.data(),
                             left_indices.size(),
                             stream.value());
      rmm::device_buffer temp_storage(temp_storage_bytes, stream);
      cub::DeviceReduce::Sum(temp_storage.data(),
                             temp_storage_bytes,
                             predicate_it,
                             d_num_valid.data(),
                             left_indices.size(),
                             stream.value());
    }
    auto const num_valid   = d_num_valid.value(stream);
    auto const output_size = num_valid + num_invalid;
    if (output_size == 0) { return make_empty_result(); }

    auto [filtered_left_indices, filtered_right_indices] = make_result_vectors(output_size);
    if (num_valid > 0) {
      auto input_iter =
        thrust::make_zip_iterator(cuda::std::tuple{left_indices.begin(), right_indices.begin()});
      auto output_iter = thrust::make_zip_iterator(
        cuda::std::tuple{filtered_left_indices->begin(), filtered_right_indices->begin()});
      auto valid_predicate = [predicate_results_ptr] __device__(auto i) -> bool {
        return predicate_results_ptr[i];
      };

      size_t temp_storage_bytes = 0;
      cub::DeviceSelect::FlaggedIf(nullptr,
                                   temp_storage_bytes,
                                   input_iter,
                                   cuda::counting_iterator<size_t>(0),
                                   output_iter,
                                   d_num_valid.data(),
                                   left_indices.size(),
                                   valid_predicate,
                                   stream.value());
      rmm::device_buffer temp_storage(temp_storage_bytes, stream);
      cub::DeviceSelect::FlaggedIf(temp_storage.data(),
                                   temp_storage_bytes,
                                   input_iter,
                                   cuda::counting_iterator<size_t>(0),
                                   output_iter,
                                   d_num_valid.data(),
                                   left_indices.size(),
                                   valid_predicate,
                                   stream.value());
    }
    if (num_invalid > 0) {
      size_t temp_storage_bytes       = 0;
      auto filter_passing_indices_ref = filter_passing_indices.ref(cuco::contains);
      auto is_unmatched_idx           = [filter_passing_indices_ref] __device__(size_type idx) {
        auto is_unmatched = !filter_passing_indices_ref.contains(idx);
        return is_unmatched;
      };
      cudf::detail::device_scalar<size_t> d_num_invalid(num_invalid, stream);
      cub::DeviceSelect::If(nullptr,
                            temp_storage_bytes,
                            cuda::counting_iterator<size_t>(0),
                            filtered_left_indices->begin() + num_valid,
                            d_num_invalid.data(),
                            left.num_rows(),
                            is_unmatched_idx,
                            stream.value());
      rmm::device_buffer temp_storage(temp_storage_bytes, stream);
      cub::DeviceSelect::If(temp_storage.data(),
                            temp_storage_bytes,
                            cuda::counting_iterator<size_t>(0),
                            filtered_left_indices->begin() + num_valid,
                            d_num_invalid.data(),
                            left.num_rows(),
                            is_unmatched_idx,
                            stream.value());

      cub::DeviceTransform::Fill(
        filtered_right_indices->begin() + num_valid, num_invalid, JoinNoMatch, stream.value());
    }

    return std::pair{std::move(filtered_left_indices), std::move(filtered_right_indices)};

  } else if (join_kind == join_kind::FULL_JOIN) {
    // FULL_JOIN: Preserve all rows, split failed matches - same as AST version
    auto is_failed_matched_pair = [=] __device__(size_type i) -> bool {
      return !predicate_results_ptr[i] && left_ptr[i] != JoinNoMatch && right_ptr[i] != JoinNoMatch;
    };

    auto const failed_matched_count =
      thrust::count_if(rmm::exec_policy_nosync(stream),
                       thrust::counting_iterator{0},
                       thrust::counting_iterator{static_cast<size_type>(left_indices.size())},
                       is_failed_matched_pair);
    auto const output_size = left_indices.size() + failed_matched_count;

    if (output_size == 0) { return make_empty_result(); }

    auto [filtered_left_indices, filtered_right_indices] = make_result_vectors(output_size);

    thrust::transform(rmm::exec_policy_nosync(stream),
                      thrust::counting_iterator{0},
                      thrust::counting_iterator{static_cast<size_type>(left_indices.size())},
                      thrust::make_zip_iterator(cuda::std::tuple{filtered_left_indices->begin(),
                                                                 filtered_right_indices->begin()}),
                      [=] __device__(size_type i) -> cuda::std::tuple<size_type, size_type> {
                        auto const left_idx  = left_ptr[i];
                        auto const right_idx = right_ptr[i];
                        auto const output_right_idx =
                          (predicate_results_ptr[i] || left_idx == JoinNoMatch) ? right_idx
                                                                                : JoinNoMatch;
                        return cuda::std::tuple{left_idx, output_right_idx};
                      });

    if (failed_matched_count > 0) {
      auto secondary_iter = thrust::make_zip_iterator(
        cuda::std::tuple{filtered_left_indices->begin() + left_indices.size(),
                         filtered_right_indices->begin() + left_indices.size()});

      auto failed_match_iter = cudf::detail::make_counting_transform_iterator(
        0, [=] __device__(size_type i) -> cuda::std::tuple<size_type, size_type> {
          return cuda::std::tuple{JoinNoMatch, right_ptr[i]};
        });
      thrust::copy_if(rmm::exec_policy_nosync(stream),
                      failed_match_iter,
                      failed_match_iter + left_indices.size(),
                      thrust::counting_iterator{0},
                      secondary_iter,
                      is_failed_matched_pair);
    }

    return std::pair{std::move(filtered_left_indices), std::move(filtered_right_indices)};

  } else {
    CUDF_FAIL("Unsupported join kind for jit_filter_join_indices");
  }
}

}  // anonymous namespace

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
jit_filter_join_indices(cudf::table_view const& left,
                        cudf::table_view const& right,
                        cudf::device_span<size_type const> left_indices,
                        cudf::device_span<size_type const> right_indices,
                        std::string const& predicate_code,
                        join_kind join_kind,
                        bool is_ptx,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  
  // Validate inputs - same as AST version
  CUDF_EXPECTS(left_indices.size() == right_indices.size(),
               "Left and right index arrays must have the same size",
               std::invalid_argument);

  CUDF_EXPECTS(join_kind == join_kind::INNER_JOIN || join_kind == join_kind::LEFT_JOIN ||
                 join_kind == join_kind::FULL_JOIN,
               "jit_filter_join_indices only supports INNER_JOIN, LEFT_JOIN, and FULL_JOIN.",
               std::invalid_argument);

  auto make_empty_result = [&]() {
    return std::pair{std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr)};
  };

  if (left_indices.empty()) { return make_empty_result(); }

  // Compile JIT kernel
  std::vector<column_view> left_cols(left.begin(), left.end());
  std::vector<column_view> right_cols(right.begin(), right.end());

  auto kernel = build_join_filter_kernel(predicate_code,
                                        left_cols,
                                        right_cols,
                                        is_ptx,
                                        false,  // has_user_data = false for now
                                        stream,
                                        mr);
  
  // Allocate predicate results
  auto predicate_results = rmm::device_uvector<bool>(left_indices.size(), stream);
  
  // Launch kernel
  launch_join_filter_kernel(kernel, 
                           left, 
                           right, 
                           left_indices, 
                           right_indices,
                           predicate_results.data(), 
                           std::nullopt, // no user data for now
                           stream, 
                           mr);
  
  // Apply same join semantics as AST version
  return apply_join_semantics(left, right, left_indices, right_indices,
                            predicate_results, join_kind, stream, mr);
}

}  // namespace detail

// Public API implementation
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
jit_filter_join_indices(cudf::table_view const& left,
                        cudf::table_view const& right,
                        cudf::device_span<size_type const> left_indices,
                        cudf::device_span<size_type const> right_indices,
                        std::string const& predicate_code,
                        cudf::join_kind join_kind,
                        bool is_ptx,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::jit_filter_join_indices(
    left, right, left_indices, right_indices, predicate_code, join_kind, is_ptx, stream, mr);
}

}  // namespace cudf
