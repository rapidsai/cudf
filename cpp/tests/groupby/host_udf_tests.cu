/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

/**
 * @brief A host-based UDF implementation.
 *
 * The aggregations perform the following computation:
 *  - For reduction: compute `sum(value^2, for value in group)` (this is sum of squared).
 *  - For segmented reduction: compute `segment_size * sum(value^2, for value in group)`.
 *  - For groupby: compute `(group_idx + 1) * sum(value^2, for value in group)`.
 *
 * In addition, for segmented reduction, if null_policy is set to `INCLUDE`, the null values are
 * replaced with an initial value if it is provided.
 */
template <typename cudf_aggregation>
struct test_udf_simple_type : cudf::host_udf_base {
  static_assert(std::is_same_v<cudf_aggregation, cudf::reduce_aggregation> ||
                std::is_same_v<cudf_aggregation, cudf::segmented_reduce_aggregation> ||
                std::is_same_v<cudf_aggregation, cudf::groupby_aggregation>);

  test_udf_simple_type() = default;

  [[nodiscard]] std::unordered_set<input_kind> const& get_required_data_kinds() const override
  {
    static std::unordered_set<input_kind> const required_data_kinds =
      [&]() -> std::unordered_set<input_kind> {
      if constexpr (std::is_same_v<cudf_aggregation, cudf::reduce_aggregation>) {
        return {input_kind::INPUT_VALUES, input_kind::OUTPUT_DTYPE, input_kind::INIT_VALUE};
      } else if constexpr (std::is_same_v<cudf_aggregation, cudf::segmented_reduce_aggregation>) {
        return {input_kind::INPUT_VALUES,
                input_kind::OUTPUT_DTYPE,
                input_kind::INIT_VALUE,
                input_kind::NULL_POLICY,
                input_kind::OFFSETS};
      } else {
        return {input_kind::OFFSETS, input_kind::GROUP_LABELS, input_kind::GROUPED_VALUES};
      }
    }();

    return required_data_kinds;
  }

  [[nodiscard]] output_type operator()(std::unordered_map<input_kind, input_data> const& input,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr) const override
  {
    if constexpr (std::is_same_v<cudf_aggregation, cudf::reduce_aggregation>) {
      auto const& values      = std::get<cudf::column_view>(input.at(input_kind::INPUT_VALUES));
      auto const output_dtype = std::get<cudf::data_type>(input.at(input_kind::OUTPUT_DTYPE));
      return cudf::double_type_dispatcher(
        values.type(), output_dtype, reduce_fn{this}, input, stream, mr);
    } else if constexpr (std::is_same_v<cudf_aggregation, cudf::segmented_reduce_aggregation>) {
      auto const& values      = std::get<cudf::column_view>(input.at(input_kind::INPUT_VALUES));
      auto const output_dtype = std::get<cudf::data_type>(input.at(input_kind::OUTPUT_DTYPE));
      return cudf::double_type_dispatcher(
        values.type(), output_dtype, segmented_reduce_fn{this}, input, stream, mr);
    } else {
      auto const& values = std::get<cudf::column_view>(input.at(input_kind::GROUPED_VALUES));
      return cudf::type_dispatcher(values.type(), groupby_fn{this}, input, stream, mr);
    }
  }

  [[nodiscard]] output_type get_empty_output(
    [[maybe_unused]] std::optional<cudf::data_type> output_dtype,
    [[maybe_unused]] rmm::cuda_stream_view stream,
    [[maybe_unused]] rmm::device_async_resource_ref mr) const override
  {
    if constexpr (std::is_same_v<cudf_aggregation, cudf::reduce_aggregation>) {
      CUDF_EXPECTS(output_dtype.has_value(),
                   "Data type for the reduction result must be specified.");
      return cudf::make_default_constructed_scalar(output_dtype.value(), stream, mr);
    } else if constexpr (std::is_same_v<cudf_aggregation, cudf::segmented_reduce_aggregation>) {
      CUDF_EXPECTS(output_dtype.has_value(),
                   "Data type for the reduction result must be specified.");
      return cudf::make_empty_column(output_dtype.value());
    } else {
      return cudf::make_empty_column(
        cudf::data_type{cudf::type_to_id<typename groupby_fn::OutputType>()});
    }
  }

  [[nodiscard]] bool is_equal(host_udf_base const& other) const override
  {
    // Just check if the other object is also instance of the same derived class.
    return dynamic_cast<test_udf_simple_type const*>(&other) != nullptr;
  }

  [[nodiscard]] std::size_t do_hash() const override
  {
    return std::hash<std::string>{}({"test_udf_simple_type"});
  }

  [[nodiscard]] std::unique_ptr<host_udf_base> clone() const override
  {
    return std::make_unique<test_udf_simple_type>();
  }

  // For faster compile times, we only support a few input/output types.
  template <typename T>
  static constexpr bool is_valid_input_t()
  {
    return std::is_same_v<T, double> || std::is_same_v<T, int32_t>;
  }

  // For faster compile times, we only support a few input/output types.
  template <typename T>
  static constexpr bool is_valid_output_t()
  {
    return std::is_same_v<T, int64_t>;
  }

  struct reduce_fn {
    // Store pointer to the parent class so we can call its functions.
    test_udf_simple_type const* parent;

    template <typename InputType,
              typename OutputType,
              typename... Args,
              CUDF_ENABLE_IF(!is_valid_input_t<InputType>() || !is_valid_output_t<OutputType>())>
    output_type operator()(Args...) const
    {
      CUDF_FAIL("Unsupported input type.");
    }

    template <typename InputType,
              typename OutputType,
              CUDF_ENABLE_IF(is_valid_input_t<InputType>() && is_valid_output_t<OutputType>())>
    output_type operator()(std::unordered_map<input_kind, input_data> const& input,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr) const
    {
      auto const& values      = std::get<cudf::column_view>(input.at(input_kind::INPUT_VALUES));
      auto const output_dtype = std::get<cudf::data_type>(input.at(input_kind::OUTPUT_DTYPE));
      auto const input_init_value =
        std::get<std::optional<std::reference_wrapper<cudf::scalar const>>>(
          input.at(input_kind::INIT_VALUE));

      if (values.size() == 0) { return parent->get_empty_output(output_dtype, stream, mr); }

      auto const init_value = [&]() -> InputType {
        if (input_init_value.has_value() && input_init_value.value().get().is_valid(stream)) {
          auto const numeric_init_scalar =
            dynamic_cast<cudf::numeric_scalar<InputType> const*>(&input_init_value.value().get());
          CUDF_EXPECTS(numeric_init_scalar != nullptr, "Invalid init scalar for reduction.");
          return numeric_init_scalar->value(stream);
        }
        return InputType{0};
      }();

      auto const values_dv_ptr = cudf::column_device_view::create(values, stream);
      auto const result =
        thrust::transform_reduce(rmm::exec_policy(stream),
                                 thrust::make_counting_iterator(0),
                                 thrust::make_counting_iterator(values.size()),
                                 transform_fn<InputType, OutputType>{*values_dv_ptr},
                                 static_cast<OutputType>(init_value),
                                 thrust::plus<>{});

      auto output = cudf::make_numeric_scalar(output_dtype, stream, mr);
      static_cast<cudf::scalar_type_t<OutputType>*>(output.get())->set_value(result, stream);
      return output;
    }

    template <typename InputType, typename OutputType>
    struct transform_fn {
      cudf::column_device_view values;
      OutputType __device__ operator()(cudf::size_type idx) const
      {
        if (values.is_null(idx)) { return OutputType{0}; }
        auto const val = static_cast<OutputType>(values.element<InputType>(idx));
        return val * val;
      }
    };
  };

  struct segmented_reduce_fn {
    // Store pointer to the parent class so we can call its functions.
    test_udf_simple_type const* parent;

    template <typename InputType,
              typename OutputType,
              typename... Args,
              CUDF_ENABLE_IF(!is_valid_input_t<InputType>() || !is_valid_output_t<OutputType>())>
    output_type operator()(Args...) const
    {
      CUDF_FAIL("Unsupported input type.");
    }

    template <typename InputType,
              typename OutputType,
              CUDF_ENABLE_IF(is_valid_input_t<InputType>() && is_valid_output_t<OutputType>())>
    output_type operator()(std::unordered_map<input_kind, input_data> const& input,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr) const
    {
      auto const& values      = std::get<cudf::column_view>(input.at(input_kind::INPUT_VALUES));
      auto const output_dtype = std::get<cudf::data_type>(input.at(input_kind::OUTPUT_DTYPE));
      auto const input_init_value =
        std::get<std::optional<std::reference_wrapper<cudf::scalar const>>>(
          input.at(input_kind::INIT_VALUE));

      if (values.size() == 0) { return parent->get_empty_output(output_dtype, stream, mr); }

      auto const init_value = [&]() -> InputType {
        if (input_init_value.has_value() && input_init_value.value().get().is_valid(stream)) {
          auto const numeric_init_scalar =
            dynamic_cast<cudf::numeric_scalar<InputType> const*>(&input_init_value.value().get());
          CUDF_EXPECTS(numeric_init_scalar != nullptr, "Invalid init scalar for reduction.");
          return numeric_init_scalar->value(stream);
        }
        return InputType{0};
      }();

      auto const null_handling = std::get<cudf::null_policy>(input.at(input_kind::NULL_POLICY));
      auto const offsets =
        std::get<cudf::device_span<cudf::size_type const>>(input.at(input_kind::OFFSETS));
      CUDF_EXPECTS(offsets.size() > 0, "Invalid offsets.");
      auto const num_segments = static_cast<cudf::size_type>(offsets.size()) - 1;

      auto const values_dv_ptr = cudf::column_device_view::create(values, stream);
      auto output              = cudf::make_numeric_column(
        output_dtype, num_segments, cudf::mask_state::UNALLOCATED, stream);
      rmm::device_uvector<bool> validity(num_segments, stream);

      thrust::transform(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_segments),
        thrust::make_zip_iterator(output->mutable_view().begin<OutputType>(), validity.begin()),
        transform_fn<InputType, OutputType>{
          *values_dv_ptr, offsets, static_cast<OutputType>(init_value), null_handling});
      auto [null_mask, null_count] =
        cudf::detail::valid_if(validity.begin(), validity.end(), thrust::identity<>{}, stream, mr);
      if (null_count > 0) { output->set_null_mask(std::move(null_mask), null_count); }
      return output;
    }

    template <typename InputType, typename OutputType>
    struct transform_fn {
      cudf::column_device_view values;
      cudf::device_span<cudf::size_type const> offsets;
      OutputType init_value;
      cudf::null_policy null_handling;

      thrust::tuple<OutputType, bool> __device__ operator()(cudf::size_type idx) const
      {
        auto const start = offsets[idx];
        auto const end   = offsets[idx + 1];
        if (start == end) { return {OutputType{0}, false}; }

        auto sum = init_value;
        for (auto i = start; i < end; ++i) {
          if (values.is_null(i)) {
            if (null_handling == cudf::null_policy::INCLUDE) { sum += init_value * init_value; }
            continue;
          }
          auto const val = static_cast<OutputType>(values.element<InputType>(i));
          sum += val * val;
        }
        auto const segment_size = end - start;
        return {static_cast<OutputType>(segment_size) * sum, true};
      }
    };
  };

  struct groupby_fn {
    // Store pointer to the parent class so we can call its functions.
    test_udf_simple_type const* parent;
    using OutputType = double;

    template <typename InputType, typename... Args, CUDF_ENABLE_IF(!cudf::is_numeric<InputType>())>
    output_type operator()(Args...) const
    {
      CUDF_FAIL("Unsupported input type.");
    }

    template <typename InputType, CUDF_ENABLE_IF(cudf::is_numeric<InputType>())>
    output_type operator()(std::unordered_map<input_kind, input_data> const& input,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr) const
    {
      auto const& values = std::get<cudf::column_view>(input.at(input_kind::GROUPED_VALUES));
      if (values.size() == 0) { return parent->get_empty_output(std::nullopt, stream, mr); }

      auto const offsets =
        std::get<cudf::device_span<cudf::size_type const>>(input.at(input_kind::OFFSETS));
      CUDF_EXPECTS(offsets.size() > 0, "Invalid offsets.");
      auto const num_groups = static_cast<int>(offsets.size()) - 1;
      auto const group_indices =
        std::get<cudf::device_span<cudf::size_type const>>(input.at(input_kind::GROUP_LABELS));

      auto const values_dv_ptr = cudf::column_device_view::create(values, stream);
      auto output = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<OutputType>()},
                                              num_groups,
                                              cudf::mask_state::UNALLOCATED,
                                              stream);
      rmm::device_uvector<bool> validity(num_groups, stream);

      thrust::transform(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_groups),
        thrust::make_zip_iterator(output->mutable_view().begin<OutputType>(), validity.begin()),
        transform_fn<InputType, OutputType>{*values_dv_ptr, offsets, group_indices});
      auto [null_mask, null_count] =
        cudf::detail::valid_if(validity.begin(), validity.end(), thrust::identity<>{}, stream, mr);
      if (null_count > 0) { output->set_null_mask(std::move(null_mask), null_count); }
      return output;
    }

    template <typename InputType, typename OutputType>
    struct transform_fn {
      cudf::column_device_view values;
      cudf::device_span<cudf::size_type const> offsets;
      cudf::device_span<cudf::size_type const> group_indices;

      thrust::tuple<OutputType, bool> __device__ operator()(cudf::size_type idx) const
      {
        auto const start = offsets[idx];
        auto const end   = offsets[idx + 1];
        if (start == end) { return {OutputType{0}, false}; }

        auto sum = OutputType{0};
        for (auto i = start; i < end; ++i) {
          if (values.is_null(i)) { continue; }
          auto const val = static_cast<OutputType>(values.element<InputType>(i));
          sum += val * val;
        }
        return {static_cast<OutputType>((group_indices[idx] + 1)) * sum, true};
      }
    };
  };
};

using doubles_col = cudf::test::fixed_width_column_wrapper<double>;
using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64s_col  = cudf::test::fixed_width_column_wrapper<int64_t>;

struct HostUDFImplementationTest : cudf::test::BaseFixture {};

TEST_F(HostUDFImplementationTest, ReductionSimpleInput)
{
  auto const vals = doubles_col{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  auto const agg  = cudf::make_host_udf_aggregation<cudf::reduce_aggregation>(
    std::make_unique<test_udf_simple_type<cudf::reduce_aggregation>>());
  auto const reduced = cudf::reduce(vals,
                                    *agg,
                                    cudf::data_type{cudf::type_id::INT64},
                                    cudf::get_default_stream(),
                                    cudf::get_current_device_resource_ref());
  EXPECT_TRUE(reduced->is_valid());
  EXPECT_EQ(cudf::type_id::INT64, reduced->type().id());
  auto const result =
    static_cast<cudf::scalar_type_t<int64_t>*>(reduced.get())->value(cudf::get_default_stream());
  auto constexpr expected = 55;  // 0^2 + 1^2 + 2^2 + 3^2 + 4^2 + 5^2 = 55
  EXPECT_EQ(expected, result);
}

TEST_F(HostUDFImplementationTest, ReductionEmptyInput)
{
  auto const vals = doubles_col{};
  auto const agg  = cudf::make_host_udf_aggregation<cudf::reduce_aggregation>(
    std::make_unique<test_udf_simple_type<cudf::reduce_aggregation>>());
  auto const reduced = cudf::reduce(vals,
                                    *agg,
                                    cudf::data_type{cudf::type_id::INT64},
                                    cudf::get_default_stream(),
                                    cudf::get_current_device_resource_ref());
  EXPECT_FALSE(reduced->is_valid());
  EXPECT_EQ(cudf::type_id::INT64, reduced->type().id());
}

TEST_F(HostUDFImplementationTest, SegmentedReductionSimpleInput)
{
  auto const vals = doubles_col{
    {0.0, 0.0 /*null*/, 2.0, 3.0, 0.0 /*null*/, 5.0, 0.0 /*null*/, 0.0 /*null*/, 8.0, 9.0},
    {true, false, true, true, false, true, false, false, true, true}};
  auto const offsets = int32s_col{0, 3, 5, 10}.release();
  auto const agg     = cudf::make_host_udf_aggregation<cudf::segmented_reduce_aggregation>(
    std::make_unique<test_udf_simple_type<cudf::segmented_reduce_aggregation>>());

  // Test without init_value.
  {
    auto const result = cudf::segmented_reduce(
      vals,
      cudf::device_span<int const>(offsets->view().begin<int>(), offsets->size()),
      *agg,
      cudf::data_type{cudf::type_id::INT64},
      cudf::null_policy::INCLUDE,
      std::nullopt,  // init_value
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref());

    // When null_policy is set to `INCLUDE`, the null values are replaced with the init_value.
    // Since init_value is not given, it is set to 0.
    // [ 3 * (0^2 + init^2 + 2^2), 2 * (3^2 + init^2), 5 * (5^2 + init^2 + init^2 + 8^2 + 9^2) ]
    auto const expected = int64s_col{12, 18, 850};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  // Test with init value, and include nulls.
  {
    auto const init_scalar = cudf::make_fixed_width_scalar<double>(3.0);
    auto const result      = cudf::segmented_reduce(
      vals,
      cudf::device_span<int const>(offsets->view().begin<int>(), offsets->size()),
      *agg,
      cudf::data_type{cudf::type_id::INT64},
      cudf::null_policy::INCLUDE,
      *init_scalar,
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref());

    // When null_policy is set to `INCLUDE`, the null values are replaced with the init_value.
    // [ 3 * (3 + 0^2 + 3^2 + 2^2), 2 * (3 + 3^2 + 3^2), 5 * (3 + 5^2 + 3^2 + 3^2 + 8^2 + 9^2) ]
    auto const expected = int64s_col{48, 42, 955};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  // Test with init value, and exclude nulls.
  {
    auto const init_scalar = cudf::make_fixed_width_scalar<double>(3.0);
    auto const result      = cudf::segmented_reduce(
      vals,
      cudf::device_span<int const>(offsets->view().begin<int>(), offsets->size()),
      *agg,
      cudf::data_type{cudf::type_id::INT64},
      cudf::null_policy::EXCLUDE,
      *init_scalar,
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref());

    // [ 3 * (3 + 0^2 + 2^2), 2 * (3 + 3^2), 5 * (3 + 5^2 + 8^2 + 9^2) ]
    auto const expected = int64s_col{21, 24, 865};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
}

TEST_F(HostUDFImplementationTest, SegmentedReductionEmptySegments)
{
  auto const vals    = int32s_col{};
  auto const offsets = int32s_col{0, 0, 0, 0}.release();
  auto const agg     = cudf::make_host_udf_aggregation<cudf::segmented_reduce_aggregation>(
    std::make_unique<test_udf_simple_type<cudf::segmented_reduce_aggregation>>());
  auto const result = cudf::segmented_reduce(
    vals,
    cudf::device_span<int const>(offsets->view().begin<int>(), offsets->size()),
    *agg,
    cudf::data_type{cudf::type_id::INT64},
    cudf::null_policy::INCLUDE,
    std::nullopt,  // init_value
    cudf::get_default_stream(),
    cudf::get_current_device_resource_ref());
  auto const expected = int64s_col{{0, 0, 0, 0}, {false, false, false, false}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(HostUDFImplementationTest, SegmentedReductionEmptyInput)
{
  auto const vals = int32s_col{};
  // Cannot be empty due to a bug in the libcudf: https://github.com/rapidsai/cudf/issues/17433.
  auto const offsets = int32s_col{0}.release();
  auto const agg     = cudf::make_host_udf_aggregation<cudf::segmented_reduce_aggregation>(
    std::make_unique<test_udf_simple_type<cudf::segmented_reduce_aggregation>>());
  auto const result = cudf::segmented_reduce(
    vals,
    cudf::device_span<int const>(offsets->view().begin<int>(), offsets->size()),
    *agg,
    cudf::data_type{cudf::type_id::INT64},
    cudf::null_policy::INCLUDE,
    std::nullopt,  // init_value
    cudf::get_default_stream(),
    cudf::get_current_device_resource_ref());
  auto const expected = int64s_col{};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}
