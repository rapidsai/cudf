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

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/join.hpp>
#include <cudf/reduction.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <chrono>
#include <ctime>
#include <iostream>

// RMM memory resource creation utilities
inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }
inline auto make_pool()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
    make_cuda(), rmm::percent_of_free_device_memory(50));
}
inline auto make_managed() { return std::make_shared<rmm::mr::managed_memory_resource>(); }
inline auto make_managed_pool()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
    make_managed(), rmm::percent_of_free_device_memory(50));
}
inline std::shared_ptr<rmm::mr::device_memory_resource> create_memory_resource(
  std::string const& mode)
{
  if (mode == "cuda") return make_cuda();
  if (mode == "pool") return make_pool();
  if (mode == "managed") return make_managed();
  if (mode == "managed_pool") return make_managed_pool();
  CUDF_FAIL("Unknown rmm_mode parameter: " + mode +
            "\nExpecting: cuda, pool, managed, or managed_pool");
}

/**
 * @brief A class to represent a table with column names attached
 */
class table_with_cols {
 public:
  table_with_cols(std::unique_ptr<cudf::table> tbl, std::vector<std::string> col_names)
    : tbl(std::move(tbl)), col_names(col_names)
  {
  }
  /**
   * @brief Return the table view
   */
  cudf::table_view table() { return tbl->view(); }
  /**
   * @brief Return the column view for a given column name
   *
   * @param col_name The name of the column
   */
  cudf::column_view column(std::string col_name) { return tbl->view().column(col_id(col_name)); }
  /**
   * @param Return the column names of the table
   */
  std::vector<std::string> columns() { return col_names; }
  /**
   * @brief Translate a column name to a column index
   *
   * @param col_name The name of the column
   */
  cudf::size_type col_id(std::string col_name)
  {
    CUDF_FUNC_RANGE();
    auto it = std::find(col_names.begin(), col_names.end(), col_name);
    if (it == col_names.end()) { throw std::runtime_error("Column not found"); }
    return std::distance(col_names.begin(), it);
  }
  /**
   * @brief Append a column to the table
   *
   * @param col The column to append
   * @param col_name The name of the appended column
   */
  std::unique_ptr<table_with_cols> append(std::unique_ptr<cudf::column>& col, std::string col_name)
  {
    CUDF_FUNC_RANGE();
    auto cols = tbl->release();
    cols.push_back(std::move(col));
    col_names.push_back(col_name);
    auto appended_table = std::make_unique<cudf::table>(std::move(cols));
    return std::make_unique<table_with_cols>(std::move(appended_table), col_names);
  }
  /**
   * @brief Select a subset of columns from the table
   *
   * @param col_names The names of the columns to select
   */
  cudf::table_view select(std::vector<std::string> col_names)
  {
    CUDF_FUNC_RANGE();
    std::vector<cudf::size_type> col_indices;
    for (auto const& col_name : col_names) {
      col_indices.push_back(col_id(col_name));
    }
    return tbl->select(col_indices);
  }
  /**
   * @brief Write the table to a parquet file
   *
   * @param filepath The path to the parquet file
   */
  void to_parquet(std::string filepath)
  {
    CUDF_FUNC_RANGE();
    auto const sink_info = cudf::io::sink_info(filepath);
    cudf::io::table_metadata metadata;
    std::vector<cudf::io::column_name_info> col_name_infos;
    for (auto const& col_name : col_names) {
      col_name_infos.push_back(cudf::io::column_name_info(col_name));
    }
    metadata.schema_info            = col_name_infos;
    auto const table_input_metadata = cudf::io::table_input_metadata{metadata};
    auto builder = cudf::io::parquet_writer_options::builder(sink_info, tbl->view());
    builder.metadata(table_input_metadata);
    auto const options = builder.build();
    cudf::io::write_parquet(options);
  }

 private:
  std::unique_ptr<cudf::table> tbl;
  std::vector<std::string> col_names;
};

/**
 * @brief Concatenate two vectors
 *
 * @param lhs The left vector
 * @param rhs The right vector
 */
template <typename T>
std::vector<T> concat(std::vector<T> const& lhs, std::vector<T> const& rhs)
{
  std::vector<T> result;
  result.reserve(lhs.size() + rhs.size());
  std::copy(lhs.begin(), lhs.end(), std::back_inserter(result));
  std::copy(rhs.begin(), rhs.end(), std::back_inserter(result));
  return result;
}

/**
 * @brief Inner join two tables and gather the result
 *
 * @param left_input The left input table
 * @param right_input The right input table
 * @param left_on The columns to join on in the left table
 * @param right_on The columns to join on in the right table
 * @param compare_nulls The null equality policy
 */
std::unique_ptr<cudf::table> join_and_gather(cudf::table_view left_input,
                                             cudf::table_view right_input,
                                             std::vector<cudf::size_type> left_on,
                                             std::vector<cudf::size_type> right_on,
                                             cudf::null_equality compare_nulls)
{
  CUDF_FUNC_RANGE();
  auto oob_policy                                    = cudf::out_of_bounds_policy::DONT_CHECK;
  auto const left_selected                           = left_input.select(left_on);
  auto const right_selected                          = right_input.select(right_on);
  auto const [left_join_indices, right_join_indices] = cudf::inner_join(
    left_selected, right_selected, compare_nulls, rmm::mr::get_current_device_resource());

  auto const left_indices_span  = cudf::device_span<cudf::size_type const>{*left_join_indices};
  auto const right_indices_span = cudf::device_span<cudf::size_type const>{*right_join_indices};

  auto const left_indices_col  = cudf::column_view{left_indices_span};
  auto const right_indices_col = cudf::column_view{right_indices_span};

  auto const left_result  = cudf::gather(left_input, left_indices_col, oob_policy);
  auto const right_result = cudf::gather(right_input, right_indices_col, oob_policy);

  auto joined_cols = left_result->release();
  auto right_cols  = right_result->release();
  joined_cols.insert(joined_cols.end(),
                     std::make_move_iterator(right_cols.begin()),
                     std::make_move_iterator(right_cols.end()));
  return std::make_unique<cudf::table>(std::move(joined_cols));
}

/**
 * @brief Apply an inner join operation to two tables
 *
 * @param left_input The left input table
 * @param right_input The right input table
 * @param left_on The columns to join on in the left table
 * @param right_on The columns to join on in the right table
 * @param compare_nulls The null equality policy
 */
std::unique_ptr<table_with_cols> apply_inner_join(
  std::unique_ptr<table_with_cols>& left_input,
  std::unique_ptr<table_with_cols>& right_input,
  std::vector<std::string> left_on,
  std::vector<std::string> right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL)
{
  CUDF_FUNC_RANGE();
  std::vector<cudf::size_type> left_on_indices;
  std::vector<cudf::size_type> right_on_indices;
  for (auto& col_name : left_on) {
    left_on_indices.push_back(left_input->col_id(col_name));
  }
  for (auto& col_name : right_on) {
    right_on_indices.push_back(right_input->col_id(col_name));
  }
  auto table = join_and_gather(
    left_input->table(), right_input->table(), left_on_indices, right_on_indices, compare_nulls);
  return std::make_unique<table_with_cols>(std::move(table),
                                           concat(left_input->columns(), right_input->columns()));
}

/**
 * @brief Apply a filter predicated to a table
 *
 * @param table The input table
 * @param predicate The filter predicate
 */
std::unique_ptr<table_with_cols> apply_filter(std::unique_ptr<table_with_cols>& table,
                                              cudf::ast::operation& predicate)
{
  CUDF_FUNC_RANGE();
  auto const boolean_mask = cudf::compute_column(table->table(), predicate);
  auto result_table       = cudf::apply_boolean_mask(table->table(), boolean_mask->view());
  return std::make_unique<table_with_cols>(std::move(result_table), table->columns());
}

/**
 * @brief Apply a boolean mask to a table
 *
 * @param table The input table
 * @param mask The boolean mask
 */
std::unique_ptr<table_with_cols> apply_mask(std::unique_ptr<table_with_cols>& table,
                                            std::unique_ptr<cudf::column>& mask)
{
  CUDF_FUNC_RANGE();
  auto result_table = cudf::apply_boolean_mask(table->table(), mask->view());
  return std::make_unique<table_with_cols>(std::move(result_table), table->columns());
}

struct groupby_context_t {
  std::vector<std::string> keys;
  std::unordered_map<std::string, std::vector<std::pair<cudf::aggregation::Kind, std::string>>>
    values;
};

/**
 * @brief Apply a groupby operation to a table
 *
 * @param table The input table
 * @param ctx The groupby context
 */
std::unique_ptr<table_with_cols> apply_groupby(std::unique_ptr<table_with_cols>& table,
                                               groupby_context_t ctx)
{
  CUDF_FUNC_RANGE();
  auto const keys = table->select(ctx.keys);
  cudf::groupby::groupby groupby_obj(keys);
  std::vector<std::string> result_column_names;
  result_column_names.insert(result_column_names.end(), ctx.keys.begin(), ctx.keys.end());
  std::vector<cudf::groupby::aggregation_request> requests;
  for (auto& [value_col, aggregations] : ctx.values) {
    requests.emplace_back(cudf::groupby::aggregation_request());
    for (auto& agg : aggregations) {
      if (agg.first == cudf::aggregation::Kind::SUM) {
        requests.back().aggregations.push_back(
          cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      } else if (agg.first == cudf::aggregation::Kind::MEAN) {
        requests.back().aggregations.push_back(
          cudf::make_mean_aggregation<cudf::groupby_aggregation>());
      } else if (agg.first == cudf::aggregation::Kind::COUNT_ALL) {
        requests.back().aggregations.push_back(
          cudf::make_count_aggregation<cudf::groupby_aggregation>());
      } else {
        throw std::runtime_error("Unsupported aggregation");
      }
      result_column_names.push_back(agg.second);
    }
    requests.back().values = table->column(value_col);
  }
  auto agg_results = groupby_obj.aggregate(requests);
  std::vector<std::unique_ptr<cudf::column>> result_columns;
  for (size_t i = 0; i < agg_results.first->num_columns(); i++) {
    auto col = std::make_unique<cudf::column>(agg_results.first->get_column(i));
    result_columns.push_back(std::move(col));
  }
  for (size_t i = 0; i < agg_results.second.size(); i++) {
    for (size_t j = 0; j < agg_results.second[i].results.size(); j++) {
      result_columns.push_back(std::move(agg_results.second[i].results[j]));
    }
  }
  auto result_table = std::make_unique<cudf::table>(std::move(result_columns));
  return std::make_unique<table_with_cols>(std::move(result_table), result_column_names);
}

/**
 * @brief Apply an order by operation to a table
 *
 * @param table The input table
 * @param sort_keys The sort keys
 * @param sort_key_orders The sort key orders
 */
std::unique_ptr<table_with_cols> apply_orderby(std::unique_ptr<table_with_cols>& table,
                                               std::vector<std::string> sort_keys,
                                               std::vector<cudf::order> sort_key_orders)
{
  CUDF_FUNC_RANGE();
  std::vector<cudf::column_view> column_views;
  for (auto& key : sort_keys) {
    column_views.push_back(table->column(key));
  }
  auto result_table =
    cudf::sort_by_key(table->table(), cudf::table_view{column_views}, sort_key_orders);
  return std::make_unique<table_with_cols>(std::move(result_table), table->columns());
}

/**
 * @brief Apply a reduction operation to a column
 *
 * @param column The input column
 * @param agg_kind The aggregation kind
 * @param col_name The name of the output column
 */
std::unique_ptr<table_with_cols> apply_reduction(cudf::column_view& column,
                                                 cudf::aggregation::Kind agg_kind,
                                                 std::string col_name)
{
  CUDF_FUNC_RANGE();
  auto const agg            = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto const result         = cudf::reduce(column, *agg, column.type());
  cudf::size_type const len = 1;
  auto col                  = cudf::make_column_from_scalar(*result, len);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(col));
  auto result_table                  = std::make_unique<cudf::table>(std::move(columns));
  std::vector<std::string> col_names = {col_name};
  return std::make_unique<table_with_cols>(std::move(result_table), col_names);
}

/**
 * @brief Read a parquet file into a table
 *
 * @param filename The path to the parquet file
 * @param columns The columns to read
 * @param predicate The filter predicate to pushdown
 */
std::unique_ptr<table_with_cols> read_parquet(
  std::string filename,
  std::vector<std::string> columns                = {},
  std::unique_ptr<cudf::ast::operation> predicate = nullptr)
{
  CUDF_FUNC_RANGE();
  auto const source = cudf::io::source_info(filename);
  auto builder      = cudf::io::parquet_reader_options_builder(source);
  if (columns.size()) { builder.columns(columns); }
  if (predicate) { builder.filter(*predicate); }
  auto const options       = builder.build();
  auto table_with_metadata = cudf::io::read_parquet(options);
  auto const schema_info   = table_with_metadata.metadata.schema_info;
  std::vector<std::string> column_names;
  for (auto const& col_info : schema_info) {
    column_names.push_back(col_info.name);
  }
  return std::make_unique<table_with_cols>(std::move(table_with_metadata.tbl), column_names);
}

/**
 * @brief Generate the `std::tm` structure from year, month, and day
 *
 * @param year The year
 * @param month The month
 * @param day The day
 */
std::tm make_tm(int year, int month, int day)
{
  std::tm tm = {0};
  tm.tm_year = year - 1900;
  tm.tm_mon  = month - 1;
  tm.tm_mday = day;
  return tm;
}

/**
 * @brief Calculate the number of days since the UNIX epoch
 *
 * @param year The year
 * @param month The month
 * @param day The day
 */
int32_t days_since_epoch(int year, int month, int day)
{
  std::tm tm             = make_tm(year, month, day);
  std::tm epoch          = make_tm(1970, 1, 1);
  std::time_t time       = std::mktime(&tm);
  std::time_t epoch_time = std::mktime(&epoch);
  double diff            = std::difftime(time, epoch_time) / (60 * 60 * 24);
  return static_cast<int32_t>(diff);
}

struct tpch_args_t {
  std::string dataset_dir;
  std::string memory_resource_type;
};

/**
 * @brief Parse command line arguments into a struct
 *
 * @param argc The number of command line arguments
 * @param argv The command line arguments
 */
tpch_args_t parse_args(int argc, char const** argv)
{
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <dataset_dir> <memory_resource_type>" << std::endl;
    std::cerr << std::endl;
    std::cerr << "The query result will be saved to a parquet file named "
              << "q{query_no}.parquet in the current working directory." << std::endl;
    exit(1);
  }
  tpch_args_t args;
  args.dataset_dir          = argv[1];
  args.memory_resource_type = argv[2];
  return args;
}

/**
 * @brief Light-weight timer for parquet reader and writer instrumentation
 *
 * Timer object constructed from std::chrono, instrumenting at microseconds
 * precision. Can display elapsed durations at milli and micro second
 * scales. Timer starts at object construction.
 */
class Timer {
 public:
  using micros = std::chrono::microseconds;
  using millis = std::chrono::milliseconds;

  Timer() { reset(); }
  void reset() { start_time = std::chrono::high_resolution_clock::now(); }
  auto elapsed() { return (std::chrono::high_resolution_clock::now() - start_time); }
  void print_elapsed_micros()
  {
    std::cout << "Elapsed Time: " << std::chrono::duration_cast<micros>(elapsed()).count()
              << "us\n\n";
  }
  void print_elapsed_millis()
  {
    std::cout << "Elapsed Time: " << std::chrono::duration_cast<millis>(elapsed()).count()
              << "ms\n\n";
  }

 private:
  using time_point_t = std::chrono::time_point<std::chrono::high_resolution_clock>;
  time_point_t start_time;
};
