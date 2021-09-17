#include <arrow/util/tdigest.h>

#include <cudf/detail/valid_if.cuh>
#include <cudf/groupby.hpp>
#include <cudf/quantiles.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <rmm/exec_policy.hpp>

#include <tests/groupby/groupby_test_util.hpp>

using namespace cudf;

struct PercentileApproxTest : public cudf::test::BaseFixture {
};

struct tdigest_gen {
  template <
    typename T,
    typename std::enable_if_t<cudf::is_numeric<T>() || cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& keys, column_view const& values, int delta)
  {
    cudf::table_view t({keys});
    cudf::groupby::groupby gb(t);
    std::vector<cudf::groupby::aggregation_request> requests;
    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
    aggregations.push_back(cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta));
    requests.push_back({values, std::move(aggregations)});
    auto result = gb.aggregate(requests);
    return std::move(result.second[0].results[0]);
  }

  template <
    typename T,
    typename std::enable_if_t<!cudf::is_numeric<T>() && !cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& keys, column_view const& values, int delta)
  {
    CUDF_FAIL("Invalid tdigest test type");
  }
};

std::unique_ptr<column> arrow_percentile_approx(column_view const& _values,
                                                int delta,
                                                std::vector<double> const& percentages)
{
  // sort the incoming values using the same settings that groupby does.
  // this is a little weak because null_order::AFTER is hardcoded internally to groupby.
  table_view t({_values});
  auto sorted_t      = cudf::sort(t, {}, {null_order::AFTER});
  auto sorted_values = sorted_t->get_column(0).view();

  std::vector<double> h_values(sorted_values.size());
  cudaMemcpy(h_values.data(),
             sorted_values.data<double>(),
             sizeof(double) * sorted_values.size(),
             cudaMemcpyDeviceToHost);
  std::vector<char> h_validity(sorted_values.size());
  if (sorted_values.null_mask() != nullptr) {
    auto validity = cudf::mask_to_bools(sorted_values.null_mask(), 0, sorted_values.size());
    cudaMemcpy(h_validity.data(),
               (validity->view().data<char>()),
               sizeof(char) * sorted_values.size(),
               cudaMemcpyDeviceToHost);
  }

  // generate the tdigest
  arrow::internal::TDigest atd(delta, sorted_values.size() * 2);
  for (size_t idx = 0; idx < h_values.size(); idx++) {
    if (sorted_values.null_mask() == nullptr || h_validity[idx]) { atd.Add(h_values[idx]); }
  }

  // generate the percentiles and stuff them into a list column
  std::vector<double> h_result;
  h_result.reserve(percentages.size());
  std::transform(
    percentages.begin(), percentages.end(), std::back_inserter(h_result), [&atd](double p) {
      return atd.Quantile(p);
    });
  cudf::test::fixed_width_column_wrapper<double> result(h_result.begin(), h_result.end());
  cudf::test::fixed_width_column_wrapper<size_type> offsets{
    0, static_cast<size_type>(percentages.size())};
  return cudf::make_lists_column(1, offsets.release(), result.release(), 0, {});
}

struct percentile_approx_dispatch {
  template <
    typename T,
    typename std::enable_if_t<cudf::is_numeric<T>() || cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& keys,
                                     column_view const& values,
                                     int delta,
                                     std::vector<double> const& percentages,
                                     size_type ulps)
  {
    // arrow implementation.
    auto expected = [&]() {
      // we're explicitly casting back to doubles here but this is ok because that is
      // exactly what happens inside of the cudf implementation as values are processed as well. so
      // this should not affect results.
      auto as_doubles = cudf::cast(values, data_type{type_id::FLOAT64});
      return arrow_percentile_approx(*as_doubles, delta, percentages);
    }();

    // gpu
    cudf::table_view t({keys});
    cudf::groupby::groupby gb(t);
    std::vector<cudf::groupby::aggregation_request> requests;
    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
    aggregations.push_back(cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta));
    requests.push_back({values, std::move(aggregations)});
    auto gb_result = gb.aggregate(requests);

    cudf::test::fixed_width_column_wrapper<double> g_percentages(percentages.begin(),
                                                                 percentages.end());
    structs_column_view scv(*(gb_result.second[0].results[0]));
    auto result = cudf::percentile_approx(scv, g_percentages);

    cudf::test::expect_columns_equivalent(
      *expected, *result, cudf::test::debug_output_level::FIRST_ERROR, ulps);

    return result;
  }

  template <
    typename T,
    typename std::enable_if_t<!cudf::is_numeric<T>() && !cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& keys,
                                     column_view const& values,
                                     int delta,
                                     std::vector<double> const& percentages,
                                     size_type ulps)
  {
    CUDF_FAIL("Invalid input type for percentile_approx test");
  }
};

void percentile_approx_test(column_view const& _keys,
                            column_view const& _values,
                            int delta,
                            std::vector<double> const& percentages,
                            data_type t    = data_type{type_id::FLOAT64},
                            size_type ulps = cudf::test::default_ulp)
{
  // first pass:  validate the actual percentages we get per group.

  // produce the groups
  cudf::table_view k({_keys});
  cudf::groupby::groupby pass1_gb(k);
  cudf::table_view v({_values});
  auto groups = pass1_gb.get_groups(v);
  // slice it all up so we have keys/columns for everything.
  std::vector<column_view> keys;
  std::vector<column_view> values;
  for (size_t idx = 0; idx < groups.offsets.size() - 1; idx++) {
    auto k =
      cudf::slice(groups.keys->get_column(0), {groups.offsets[idx], groups.offsets[idx + 1]});
    keys.push_back(k[0]);

    auto v =
      cudf::slice(groups.values->get_column(0), {groups.offsets[idx], groups.offsets[idx + 1]});
    values.push_back(v[0]);
  }

  std::vector<std::unique_ptr<column>> parts;
  for (size_t idx = 0; idx < values.size(); idx++) {
    // do any casting of the input
    auto cast_values = cudf::cast(values[idx], t);
    parts.push_back(cudf::type_dispatcher(cast_values->type(),
                                          percentile_approx_dispatch{},
                                          keys[idx],
                                          *cast_values,
                                          delta,
                                          percentages,
                                          ulps));
  }
  std::vector<column_view> part_views;
  std::transform(parts.begin(),
                 parts.end(),
                 std::back_inserter(part_views),
                 [](std::unique_ptr<column> const& c) { return c->view(); });
  auto expected = cudf::concatenate(part_views);

  // second pass. run the percentile_approx with all the keys in one pass and make sure we get the
  // same results as the concatenated by-key results above

  cudf::groupby::groupby gb(k);
  std::vector<cudf::groupby::aggregation_request> requests;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
  aggregations.push_back(cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta));
  requests.push_back({_values, std::move(aggregations)});
  auto gb_result = gb.aggregate(requests);

  cudf::test::fixed_width_column_wrapper<double> g_percentages(percentages.begin(),
                                                               percentages.end());
  structs_column_view scv(*(gb_result.second[0].results[0]));
  auto result = cudf::percentile_approx(scv, g_percentages);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *result);
}

TEST_F(PercentileApproxTest, Simple)
{
  auto values =
    cudf::test::generate_standardized_percentile_distribution(data_type{type_id::FLOAT64});
  // all in the same group
  auto keys = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, values->size(), mask_state::UNALLOCATED);
  thrust::fill(rmm::exec_policy(rmm::cuda_stream_default),
               keys->mutable_view().template begin<int>(),
               keys->mutable_view().template end<int>(),
               0);

  // delta 1000
  {
    int const delta = 1000;
    percentile_approx_test(*keys,
                           *values,
                           delta,
                           {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0},
                           data_type{type_id::FLOAT64},
                           cudf::test::default_ulp);
  }

  // delta 100
  {
    int const delta = 100;
    percentile_approx_test(*keys,
                           *values,
                           delta,
                           {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0},
                           data_type{type_id::FLOAT64},
                           cudf::test::default_ulp * 4);
  }

  // delta 10
  {
    int const delta = 10;
    percentile_approx_test(*keys,
                           *values,
                           delta,
                           {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0},
                           data_type{type_id::FLOAT64},
                           cudf::test::default_ulp * 11);
  }
}

struct group_index {
  __device__ int operator()(int i) { return i / 2000; }
};

TEST_F(PercentileApproxTest, Grouped)
{
  auto values =
    cudf::test::generate_standardized_percentile_distribution(data_type{type_id::FLOAT64});
  // all in the same group
  auto keys = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, values->size(), mask_state::UNALLOCATED);
  auto i = thrust::make_counting_iterator(0);
  thrust::transform(rmm::exec_policy(rmm::cuda_stream_default),
                    i,
                    i + values->size(),
                    keys->mutable_view().template begin<int>(),
                    group_index{});

  // delta 1000
  {
    int const delta = 1000;
    percentile_approx_test(*keys,
                           *values,
                           delta,
                           {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0},
                           data_type{type_id::FLOAT64},
                           cudf::test::default_ulp);
  }

  // delta 100
  {
    int const delta = 100;
    percentile_approx_test(*keys,
                           *values,
                           delta,
                           {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0},
                           data_type{type_id::FLOAT64},
                           cudf::test::default_ulp);
  }

  // delta 10
  {
    int const delta = 10;
    percentile_approx_test(*keys,
                           *values,
                           delta,
                           {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0},
                           data_type{type_id::FLOAT64},
                           cudf::test::default_ulp * 4);
  }
}

std::pair<rmm::device_buffer, size_type> make_null_mask(column_view const& col)
{
  return cudf::detail::valid_if(thrust::make_counting_iterator<size_type>(0),
                                thrust::make_counting_iterator<size_type>(col.size()),
                                [] __device__(size_type i) { return i % 2 == 0; });
}

TEST_F(PercentileApproxTest, SimpleWithNulls)
{
  auto values =
    cudf::test::generate_standardized_percentile_distribution(data_type{type_id::FLOAT64});
  // all in the same group
  auto keys = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, values->size(), mask_state::UNALLOCATED);
  thrust::fill(rmm::exec_policy(rmm::cuda_stream_default),
               keys->mutable_view().template begin<int>(),
               keys->mutable_view().template end<int>(),
               0);

  // add a null mask
  auto mask = make_null_mask(*values);
  values->set_null_mask(mask.first, mask.second);

  // delta 1000
  {
    int const delta = 1000;
    percentile_approx_test(*keys,
                           *values,
                           delta,
                           {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0},
                           data_type{type_id::FLOAT64},
                           cudf::test::default_ulp);
  }

  // delta 100
  {
    int const delta = 100;
    percentile_approx_test(*keys,
                           *values,
                           delta,
                           {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0},
                           data_type{type_id::FLOAT64},
                           cudf::test::default_ulp);
  }

  // delta 10
  {
    int const delta = 10;
    percentile_approx_test(*keys,
                           *values,
                           delta,
                           {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0},
                           data_type{type_id::FLOAT64},
                           cudf::test::default_ulp * 11);
  }
}

TEST_F(PercentileApproxTest, GroupedWithNulls)
{
  auto values =
    cudf::test::generate_standardized_percentile_distribution(data_type{type_id::FLOAT64});
  // all in the same group
  auto keys = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, values->size(), mask_state::UNALLOCATED);
  auto i = thrust::make_counting_iterator(0);
  thrust::transform(rmm::exec_policy(rmm::cuda_stream_default),
                    i,
                    i + values->size(),
                    keys->mutable_view().template begin<int>(),
                    group_index{});

  // add a null mask
  auto mask = make_null_mask(*values);
  values->set_null_mask(mask.first, mask.second);

  // delta 1000
  {
    int const delta = 1000;
    percentile_approx_test(*keys,
                           *values,
                           delta,
                           {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0},
                           data_type{type_id::FLOAT64},
                           cudf::test::default_ulp);
  }

  // delta 100
  {
    int const delta = 100;
    percentile_approx_test(*keys,
                           *values,
                           delta,
                           {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0},
                           data_type{type_id::FLOAT64},
                           cudf::test::default_ulp);
  }

  // delta 10
  {
    int const delta = 10;
    percentile_approx_test(*keys,
                           *values,
                           delta,
                           {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0},
                           data_type{type_id::FLOAT64},
                           cudf::test::default_ulp);
  }
}