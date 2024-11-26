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

#include <cudf/column/column_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>

namespace {
/**
 * @brief A host-based UDF implementation used for unit tests.
 */
template <typename cudf_aggregation, int test_location>
struct host_udf_unit_test : cudf::host_udf_base {
  static_assert(std::is_same_v<cudf_aggregation, cudf::reduce_aggregation> ||
                std::is_same_v<cudf_aggregation, cudf::segmented_reduce_aggregation> ||
                std::is_same_v<cudf_aggregation, cudf::groupby_aggregation>);

  input_data_attributes input_attrs;
  host_udf_unit_test(input_data_attributes input_attrs_ = {}) : input_attrs(std::move(input_attrs_))
  {
  }

  [[nodiscard]] input_data_attributes get_required_data() const override { return input_attrs; }

  // This is the main testing function, which checks for the correctness of input data.
  // The rests are just to satisfy the interface.
  [[nodiscard]] output_type operator()(host_udf_input const& input,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr) const override
  {
    SCOPED_TRACE("Original line of failure: " + std::to_string(test_location));

    input_data_attributes check_attrs = input_attrs;
    if constexpr (std::is_same_v<cudf_aggregation, cudf::reduce_aggregation>) {
      if (check_attrs.empty()) {
        check_attrs = input_data_attributes{reduction_data_attribute::INPUT_VALUES,
                                            reduction_data_attribute::OUTPUT_DTYPE,
                                            reduction_data_attribute::INIT_VALUE};
      }
      EXPECT_EQ(input.size(), check_attrs.size());
      for (auto const attr : check_attrs) {
        EXPECT_TRUE(input.count(attr) > 0);
        EXPECT_TRUE(std::holds_alternative<reduction_data_attribute>(attr.value));
        switch (auto const attr_val = std::get<reduction_data_attribute>(attr.value)) {
          case reduction_data_attribute::INPUT_VALUES:
            EXPECT_TRUE(std::holds_alternative<cudf::column_view>(input.at(attr)));
            break;
          case reduction_data_attribute::OUTPUT_DTYPE:
            EXPECT_TRUE(std::holds_alternative<cudf::data_type>(input.at(attr)));
            break;
          case reduction_data_attribute::INIT_VALUE:
            EXPECT_TRUE(
              std::holds_alternative<std::optional<std::reference_wrapper<cudf::scalar const>>>(
                input.at(attr)));
            break;
          default:;
        }
      }
    } else if constexpr (std::is_same_v<cudf_aggregation, cudf::segmented_reduce_aggregation>) {
      if (check_attrs.empty()) {
        check_attrs = input_data_attributes{segmented_reduction_data_attribute::INPUT_VALUES,
                                            segmented_reduction_data_attribute::OUTPUT_DTYPE,
                                            segmented_reduction_data_attribute::INIT_VALUE,
                                            segmented_reduction_data_attribute::NULL_POLICY,
                                            segmented_reduction_data_attribute::OFFSETS};
      }
      EXPECT_EQ(input.size(), check_attrs.size());
      for (auto const attr : check_attrs) {
        EXPECT_TRUE(input.count(attr) > 0);
        EXPECT_TRUE(std::holds_alternative<segmented_reduction_data_attribute>(attr.value));
        switch (auto const attr_val = std::get<segmented_reduction_data_attribute>(attr.value)) {
          case segmented_reduction_data_attribute::INPUT_VALUES:
            EXPECT_TRUE(std::holds_alternative<cudf::column_view>(input.at(attr)));
            break;
          case segmented_reduction_data_attribute::OUTPUT_DTYPE:
            EXPECT_TRUE(std::holds_alternative<cudf::data_type>(input.at(attr)));
            break;
          case segmented_reduction_data_attribute::INIT_VALUE:
            EXPECT_TRUE(
              std::holds_alternative<std::optional<std::reference_wrapper<cudf::scalar const>>>(
                input.at(attr)));
            break;
          case segmented_reduction_data_attribute::NULL_POLICY:
            EXPECT_TRUE(std::holds_alternative<cudf::null_policy>(input.at(attr)));
            break;
          case segmented_reduction_data_attribute::OFFSETS:
            EXPECT_TRUE(
              std::holds_alternative<cudf::device_span<cudf::size_type const>>(input.at(attr)));
            break;
          default:;
        }
      }
    } else {
      if (check_attrs.empty()) {
        check_attrs = input_data_attributes{groupby_data_attribute::INPUT_VALUES,
                                            groupby_data_attribute::GROUPED_VALUES,
                                            groupby_data_attribute::SORTED_GROUPED_VALUES,
                                            groupby_data_attribute::GROUP_OFFSETS,
                                            groupby_data_attribute::GROUP_LABELS};
      }
      EXPECT_EQ(input.size(), check_attrs.size());
      for (auto const attr : check_attrs) {
        EXPECT_TRUE(input.count(attr) > 0);
        EXPECT_TRUE(std::holds_alternative<groupby_data_attribute>(attr.value) ||
                    std::holds_alternative<std::unique_ptr<cudf::aggregation>>(attr.value));
        if (std::holds_alternative<groupby_data_attribute>(attr.value)) {
          switch (auto const attr_val = std::get<groupby_data_attribute>(attr.value)) {
            case groupby_data_attribute::INPUT_VALUES:
              EXPECT_TRUE(std::holds_alternative<cudf::column_view>(input.at(attr)));
              break;
            case groupby_data_attribute::GROUPED_VALUES:
              EXPECT_TRUE(std::holds_alternative<cudf::column_view>(input.at(attr)));
              break;
            case groupby_data_attribute::SORTED_GROUPED_VALUES:
              EXPECT_TRUE(std::holds_alternative<cudf::column_view>(input.at(attr)));
              break;
            case groupby_data_attribute::GROUP_OFFSETS:
              EXPECT_TRUE(
                std::holds_alternative<cudf::device_span<cudf::size_type const>>(input.at(attr)));
              break;
            case groupby_data_attribute::GROUP_LABELS:
              EXPECT_TRUE(
                std::holds_alternative<cudf::device_span<cudf::size_type const>>(input.at(attr)));
              break;
            default:;
          }
        } else {  // std::holds_alternative<std::unique_ptr<cudf::aggregation>>(attr.value)
          EXPECT_TRUE(std::holds_alternative<std::unique_ptr<cudf::aggregation>>(input.at(attr)));
        }
      }
    }

    return get_empty_output(std::nullopt, stream, mr);
  }

  [[nodiscard]] output_type get_empty_output(
    [[maybe_unused]] std::optional<cudf::data_type> output_dtype,
    [[maybe_unused]] rmm::cuda_stream_view stream,
    [[maybe_unused]] rmm::device_async_resource_ref mr) const override
  {
    if constexpr (std::is_same_v<cudf_aggregation, cudf::reduce_aggregation>) {
      return cudf::make_fixed_width_scalar(0, stream, mr);
    } else if constexpr (std::is_same_v<cudf_aggregation, cudf::segmented_reduce_aggregation>) {
      return cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
    } else {
      return cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
    }
  }

  [[nodiscard]] bool is_equal(host_udf_base const& other) const override { return true; }
  [[nodiscard]] std::size_t do_hash() const override { return 0; }
  [[nodiscard]] std::unique_ptr<host_udf_base> clone() const override
  {
    return std::make_unique<host_udf_unit_test>();
  }
};

}  // namespace

using int32s_col = cudf::test::fixed_width_column_wrapper<int32_t>;

struct HostUDFTest : cudf::test::BaseFixture {};

TEST_F(HostUDFTest, ReductionAllInput)
{
  auto const vals = int32s_col{1, 2, 3};
  auto const agg  = cudf::make_host_udf_aggregation<cudf::reduce_aggregation>(
    std::make_unique<host_udf_unit_test<cudf::reduce_aggregation, __LINE__>>());
  [[maybe_unused]] auto const reduced = cudf::reduce(vals,
                                                     *agg,
                                                     cudf::data_type{cudf::type_id::INT64},
                                                     cudf::get_default_stream(),
                                                     cudf::get_current_device_resource_ref());
}

TEST_F(HostUDFTest, SegmentedReductionAllInput)
{
  auto const vals    = int32s_col{1, 2, 3};
  auto const offsets = int32s_col{0, 3, 5, 10}.release();
  auto const agg     = cudf::make_host_udf_aggregation<cudf::segmented_reduce_aggregation>(
    std::make_unique<host_udf_unit_test<cudf::segmented_reduce_aggregation, __LINE__>>());

  [[maybe_unused]] auto const result = cudf::segmented_reduce(
    vals,
    cudf::device_span<int const>(offsets->view().begin<int>(), offsets->size()),
    *agg,
    cudf::data_type{cudf::type_id::INT64},
    cudf::null_policy::INCLUDE,
    std::nullopt,  // init value
    cudf::get_default_stream(),
    cudf::get_current_device_resource_ref());
}

TEST_F(HostUDFTest, GroupbyAllInput)
{
  auto const keys = int32s_col{0, 1, 2};
  auto const vals = int32s_col{0, 1, 2};
  auto agg        = cudf::make_host_udf_aggregation<cudf::groupby_aggregation>(
    std::make_unique<host_udf_unit_test<cudf::groupby_aggregation, __LINE__>>());

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back();
  requests[0].values = vals;
  requests[0].aggregations.push_back(std::move(agg));
  cudf::groupby::groupby gb_obj(
    cudf::table_view({keys}), cudf::null_policy::INCLUDE, cudf::sorted::NO, {}, {});

  [[maybe_unused]] auto const grp_result =
    gb_obj.aggregate(requests, cudf::test::get_default_stream());
}
