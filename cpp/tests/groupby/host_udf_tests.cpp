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

#include <cudf/aggregation/host_udf.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include <random>
#include <vector>

namespace {
/**
 * @brief A host-based UDF implementation used for unit tests.
 */
struct host_udf_test_base : cudf::host_udf_base {
  int test_location_line;  // the location where testing is called
  bool* test_run;          // to check if the test is accidentally skipped
  data_attribute_set_t input_attrs;

  host_udf_test_base(int test_location_line_, bool* test_run_, data_attribute_set_t input_attrs_)
    : test_location_line{test_location_line_},
      test_run{test_run_},
      input_attrs(std::move(input_attrs_))
  {
  }

  [[nodiscard]] data_attribute_set_t get_required_data() const override { return input_attrs; }

  // This is the main testing function, which checks for the correctness of input data.
  // The rests are just to satisfy the interface.
  [[nodiscard]] output_t operator()(input_map_t const& input,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr) const override
  {
    SCOPED_TRACE("Test instance created at line: " + std::to_string(test_location_line));

    test_data_attributes(input, stream, mr);

    *test_run = true;  // test is run successfully
    return get_empty_output(std::nullopt, stream, mr);
  }

  [[nodiscard]] std::size_t do_hash() const override { return 0; }
  [[nodiscard]] bool is_equal(host_udf_base const& other) const override { return true; }

  // The main test function, which must be implemented for each kind of aggregations
  // (groupby/reduction/segmented_reduction).
  virtual void test_data_attributes(input_map_t const& input,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr) const = 0;
};

/**
 * @brief A host-based UDF implementation used for unit tests for reduction.
 */
struct host_udf_reduction_test : host_udf_test_base {
  host_udf_reduction_test(int test_location_line_,
                          bool* test_run_,
                          data_attribute_set_t input_attrs_ = {})
    : host_udf_test_base(test_location_line_, test_run_, std::move(input_attrs_))
  {
  }

  [[nodiscard]] std::unique_ptr<host_udf_base> clone() const override
  {
    return std::make_unique<host_udf_reduction_test>(test_location_line, test_run, input_attrs);
  }

  [[nodiscard]] output_t get_empty_output(
    [[maybe_unused]] std::optional<cudf::data_type> output_dtype,
    [[maybe_unused]] rmm::cuda_stream_view stream,
    [[maybe_unused]] rmm::device_async_resource_ref mr) const override
  {
    // Unused function - dummy output.
    return cudf::make_fixed_width_scalar(0, stream, mr);
  }

  void test_data_attributes(input_map_t const& input,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr) const override
  {
    data_attribute_set_t check_attrs = input_attrs;
    if (check_attrs.empty()) {
      check_attrs = data_attribute_set_t{reduction_data_attribute::INPUT_VALUES,
                                         reduction_data_attribute::OUTPUT_DTYPE,
                                         reduction_data_attribute::INIT_VALUE};
    }
    EXPECT_EQ(input.size(), check_attrs.size());
    for (auto const& attr : check_attrs) {
      EXPECT_TRUE(input.count(attr) > 0);
      EXPECT_TRUE(std::holds_alternative<reduction_data_attribute>(attr.value));
      switch (std::get<reduction_data_attribute>(attr.value)) {
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
        default: CUDF_UNREACHABLE("Invalid input data attribute for HOST_UDF reduction.");
      }
    }
  }
};

/**
 * @brief A host-based UDF implementation used for unit tests for segmented reduction.
 */
struct host_udf_segmented_reduction_test : host_udf_test_base {
  host_udf_segmented_reduction_test(int test_location_line_,
                                    bool* test_run_,
                                    data_attribute_set_t input_attrs_ = {})
    : host_udf_test_base(test_location_line_, test_run_, std::move(input_attrs_))
  {
  }

  [[nodiscard]] std::unique_ptr<host_udf_base> clone() const override
  {
    return std::make_unique<host_udf_segmented_reduction_test>(
      test_location_line, test_run, input_attrs);
  }

  [[nodiscard]] output_t get_empty_output(
    [[maybe_unused]] std::optional<cudf::data_type> output_dtype,
    [[maybe_unused]] rmm::cuda_stream_view stream,
    [[maybe_unused]] rmm::device_async_resource_ref mr) const override
  {
    // Unused function - dummy output.
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
  }

  void test_data_attributes(input_map_t const& input,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr) const override
  {
    data_attribute_set_t check_attrs = input_attrs;
    if (check_attrs.empty()) {
      check_attrs = data_attribute_set_t{segmented_reduction_data_attribute::INPUT_VALUES,
                                         segmented_reduction_data_attribute::OUTPUT_DTYPE,
                                         segmented_reduction_data_attribute::INIT_VALUE,
                                         segmented_reduction_data_attribute::NULL_POLICY,
                                         segmented_reduction_data_attribute::OFFSETS};
    }
    EXPECT_EQ(input.size(), check_attrs.size());
    for (auto const& attr : check_attrs) {
      EXPECT_TRUE(input.count(attr) > 0);
      EXPECT_TRUE(std::holds_alternative<segmented_reduction_data_attribute>(attr.value));
      switch (std::get<segmented_reduction_data_attribute>(attr.value)) {
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
        default: CUDF_UNREACHABLE("Invalid input data attribute for HOST_UDF reduction.");
      }
    }
  }
};

/**
 * @brief A host-based UDF implementation used for unit tests for groupby aggregation.
 */
struct host_udf_groupby_test : host_udf_test_base {
  host_udf_groupby_test(int test_location_line_,
                        bool* test_run_,
                        data_attribute_set_t input_attrs_ = {})
    : host_udf_test_base(test_location_line_, test_run_, std::move(input_attrs_))
  {
  }

  [[nodiscard]] std::unique_ptr<host_udf_base> clone() const override
  {
    return std::make_unique<host_udf_groupby_test>(test_location_line, test_run, input_attrs);
  }

  [[nodiscard]] output_t get_empty_output(
    [[maybe_unused]] std::optional<cudf::data_type> output_dtype,
    [[maybe_unused]] rmm::cuda_stream_view stream,
    [[maybe_unused]] rmm::device_async_resource_ref mr) const override
  {
    // Unused function - dummy output.
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
  }

  void test_data_attributes(input_map_t const& input,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr) const override
  {
    data_attribute_set_t check_attrs = input_attrs;
    if (check_attrs.empty()) {
      check_attrs = data_attribute_set_t{groupby_data_attribute::INPUT_VALUES,
                                         groupby_data_attribute::GROUPED_VALUES,
                                         groupby_data_attribute::SORTED_GROUPED_VALUES,
                                         groupby_data_attribute::NUM_GROUPS,
                                         groupby_data_attribute::GROUP_OFFSETS,
                                         groupby_data_attribute::GROUP_LABELS};
    }
    EXPECT_EQ(input.size(), check_attrs.size());
    for (auto const& attr : check_attrs) {
      EXPECT_TRUE(input.count(attr) > 0);
      EXPECT_TRUE(std::holds_alternative<groupby_data_attribute>(attr.value) ||
                  std::holds_alternative<std::unique_ptr<cudf::aggregation>>(attr.value));
      if (std::holds_alternative<groupby_data_attribute>(attr.value)) {
        switch (std::get<groupby_data_attribute>(attr.value)) {
          case groupby_data_attribute::INPUT_VALUES:
            EXPECT_TRUE(std::holds_alternative<cudf::column_view>(input.at(attr)));
            break;
          case groupby_data_attribute::GROUPED_VALUES:
            EXPECT_TRUE(std::holds_alternative<cudf::column_view>(input.at(attr)));
            break;
          case groupby_data_attribute::SORTED_GROUPED_VALUES:
            EXPECT_TRUE(std::holds_alternative<cudf::column_view>(input.at(attr)));
            break;
          case groupby_data_attribute::NUM_GROUPS:
            EXPECT_TRUE(std::holds_alternative<cudf::size_type>(input.at(attr)));
            break;
          case groupby_data_attribute::GROUP_OFFSETS:
            EXPECT_TRUE(
              std::holds_alternative<cudf::device_span<cudf::size_type const>>(input.at(attr)));
            break;
          case groupby_data_attribute::GROUP_LABELS:
            EXPECT_TRUE(
              std::holds_alternative<cudf::device_span<cudf::size_type const>>(input.at(attr)));
            break;
          default:
            CUDF_UNREACHABLE("Invalid input data attribute for HOST_UDF groupby aggregation.");
        }
      } else {  // std::holds_alternative<std::unique_ptr<cudf::aggregation>>(attr.value)
        EXPECT_TRUE(std::holds_alternative<cudf::column_view>(input.at(attr)));
      }
    }
  }
};

/**
 * @brief Get a random subset of input data attributes.
 */
cudf::host_udf_base::data_attribute_set_t get_subset(
  cudf::host_udf_base::data_attribute_set_t const& attrs)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<std::size_t> size_distr(1, attrs.size() - 1);
  auto const subset_size = size_distr(gen);
  auto const elements =
    std::vector<cudf::host_udf_base::data_attribute>(attrs.begin(), attrs.end());
  std::uniform_int_distribution<std::size_t> idx_distr(0, attrs.size() - 1);
  cudf::host_udf_base::data_attribute_set_t output;
  while (output.size() < subset_size) {
    output.insert(elements[idx_distr(gen)]);
  }
  return output;
}

/**
 * @brief Generate a random aggregation object from {min, max, sum, product}.
 */
std::unique_ptr<cudf::aggregation> get_random_agg()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> distr(1, 4);
  switch (distr(gen)) {
    case 1: return cudf::make_min_aggregation();
    case 2: return cudf::make_max_aggregation();
    case 3: return cudf::make_sum_aggregation();
    case 4: return cudf::make_product_aggregation();
    default: CUDF_UNREACHABLE("This should not be reached.");
  }
  return nullptr;
}

}  // namespace

using int32s_col = cudf::test::fixed_width_column_wrapper<int32_t>;

// Number of randomly testing on the input data attributes.
// For each test, a subset of data attributes will be randomly generated from all the possible input
// data attributes. The input data corresponding to that subset passed from libcudf will be tested
// for correctness.
constexpr int NUM_RANDOM_TESTS = 20;

struct HostUDFTest : cudf::test::BaseFixture {};

TEST_F(HostUDFTest, ReductionAllInput)
{
  bool test_run   = false;
  auto const vals = int32s_col{1, 2, 3};
  auto const agg  = cudf::make_host_udf_aggregation<cudf::reduce_aggregation>(
    std::make_unique<host_udf_reduction_test>(__LINE__, &test_run));
  [[maybe_unused]] auto const reduced = cudf::reduce(vals,
                                                     *agg,
                                                     cudf::data_type{cudf::type_id::INT32},
                                                     cudf::get_default_stream(),
                                                     cudf::get_current_device_resource_ref());
  EXPECT_TRUE(test_run);
}

TEST_F(HostUDFTest, ReductionSomeInput)
{
  auto const vals      = int32s_col{1, 2, 3};
  auto const all_attrs = cudf::host_udf_base::data_attribute_set_t{
    cudf::host_udf_base::reduction_data_attribute::INPUT_VALUES,
    cudf::host_udf_base::reduction_data_attribute::OUTPUT_DTYPE,
    cudf::host_udf_base::reduction_data_attribute::INIT_VALUE};
  for (int i = 0; i < NUM_RANDOM_TESTS; ++i) {
    bool test_run    = false;
    auto input_attrs = get_subset(all_attrs);
    auto const agg   = cudf::make_host_udf_aggregation<cudf::reduce_aggregation>(
      std::make_unique<host_udf_reduction_test>(__LINE__, &test_run, std::move(input_attrs)));
    [[maybe_unused]] auto const reduced = cudf::reduce(vals,
                                                       *agg,
                                                       cudf::data_type{cudf::type_id::INT32},
                                                       cudf::get_default_stream(),
                                                       cudf::get_current_device_resource_ref());
    EXPECT_TRUE(test_run);
  }
}

TEST_F(HostUDFTest, SegmentedReductionAllInput)
{
  bool test_run      = false;
  auto const vals    = int32s_col{1, 2, 3};
  auto const offsets = int32s_col{0, 3, 5, 10}.release();
  auto const agg     = cudf::make_host_udf_aggregation<cudf::segmented_reduce_aggregation>(
    std::make_unique<host_udf_segmented_reduction_test>(__LINE__, &test_run));
  [[maybe_unused]] auto const result = cudf::segmented_reduce(
    vals,
    cudf::device_span<int const>(offsets->view().begin<int>(), offsets->size()),
    *agg,
    cudf::data_type{cudf::type_id::INT32},
    cudf::null_policy::INCLUDE,
    std::nullopt,  // init value
    cudf::get_default_stream(),
    cudf::get_current_device_resource_ref());
  EXPECT_TRUE(test_run);
}

TEST_F(HostUDFTest, SegmentedReductionSomeInput)
{
  auto const vals      = int32s_col{1, 2, 3};
  auto const all_attrs = cudf::host_udf_base::data_attribute_set_t{
    cudf::host_udf_base::segmented_reduction_data_attribute::INPUT_VALUES,
    cudf::host_udf_base::segmented_reduction_data_attribute::OUTPUT_DTYPE,
    cudf::host_udf_base::segmented_reduction_data_attribute::INIT_VALUE,
    cudf::host_udf_base::segmented_reduction_data_attribute::NULL_POLICY,
    cudf::host_udf_base::segmented_reduction_data_attribute::OFFSETS};
  auto const offsets = int32s_col{0, 3, 5, 10}.release();
  for (int i = 0; i < NUM_RANDOM_TESTS; ++i) {
    bool test_run    = false;
    auto input_attrs = get_subset(all_attrs);
    auto const agg   = cudf::make_host_udf_aggregation<cudf::segmented_reduce_aggregation>(
      std::make_unique<host_udf_segmented_reduction_test>(
        __LINE__, &test_run, std::move(input_attrs)));
    [[maybe_unused]] auto const result = cudf::segmented_reduce(
      vals,
      cudf::device_span<int const>(offsets->view().begin<int>(), offsets->size()),
      *agg,
      cudf::data_type{cudf::type_id::INT64},
      cudf::null_policy::INCLUDE,
      std::nullopt,  // init value
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref());
    EXPECT_TRUE(test_run);
  }
}

TEST_F(HostUDFTest, GroupbyAllInput)
{
  bool test_run   = false;
  auto const keys = int32s_col{0, 1, 2};
  auto const vals = int32s_col{0, 1, 2};
  auto agg        = cudf::make_host_udf_aggregation<cudf::groupby_aggregation>(
    std::make_unique<host_udf_groupby_test>(__LINE__, &test_run));

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back();
  requests[0].values = vals;
  requests[0].aggregations.push_back(std::move(agg));
  cudf::groupby::groupby gb_obj(
    cudf::table_view({keys}), cudf::null_policy::INCLUDE, cudf::sorted::NO, {}, {});
  [[maybe_unused]] auto const grp_result = gb_obj.aggregate(
    requests, cudf::test::get_default_stream(), cudf::get_current_device_resource_ref());
  EXPECT_TRUE(test_run);
}

TEST_F(HostUDFTest, GroupbySomeInput)
{
  auto const keys      = int32s_col{0, 1, 2};
  auto const vals      = int32s_col{0, 1, 2};
  auto const all_attrs = cudf::host_udf_base::data_attribute_set_t{
    cudf::host_udf_base::groupby_data_attribute::INPUT_VALUES,
    cudf::host_udf_base::groupby_data_attribute::GROUPED_VALUES,
    cudf::host_udf_base::groupby_data_attribute::SORTED_GROUPED_VALUES,
    cudf::host_udf_base::groupby_data_attribute::NUM_GROUPS,
    cudf::host_udf_base::groupby_data_attribute::GROUP_OFFSETS,
    cudf::host_udf_base::groupby_data_attribute::GROUP_LABELS};
  for (int i = 0; i < NUM_RANDOM_TESTS; ++i) {
    bool test_run    = false;
    auto input_attrs = get_subset(all_attrs);
    input_attrs.insert(get_random_agg());
    auto agg = cudf::make_host_udf_aggregation<cudf::groupby_aggregation>(
      std::make_unique<host_udf_groupby_test>(__LINE__, &test_run, std::move(input_attrs)));

    std::vector<cudf::groupby::aggregation_request> requests;
    requests.emplace_back();
    requests[0].values = vals;
    requests[0].aggregations.push_back(std::move(agg));
    cudf::groupby::groupby gb_obj(
      cudf::table_view({keys}), cudf::null_policy::INCLUDE, cudf::sorted::NO, {}, {});
    [[maybe_unused]] auto const grp_result = gb_obj.aggregate(
      requests, cudf::test::get_default_stream(), cudf::get_current_device_resource_ref());
    EXPECT_TRUE(test_run);
  }
}
