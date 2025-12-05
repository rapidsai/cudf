#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
// Include the correct header for mixed_join
#include <cudf/join/mixed_join.hpp>
#include <cudf/join/sort_merge_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <cudf_test/base_fixture.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <iostream>
#include <memory>
#include <vector>
#include <string>

struct MyTests : public cudf::test::BaseFixture {};

// Helper function to convert a vector of unique_ptr<column> to a vector of column_view
std::vector<cudf::column_view> get_column_views(
  std::vector<std::unique_ptr<cudf::column>> const& columns)
{
  std::vector<cudf::column_view> views;
  views.reserve(columns.size());
  for (auto const& col : columns) {
    views.push_back(col->view());
  }
  return views;
}

// Helper to create a strings column from a host vector of strings
std::unique_ptr<cudf::column> make_strings_column_from_host(std::vector<std::string> const& data)
{
  std::vector<cudf::size_type> offsets;
  offsets.push_back(0);
  std::string chars;
  for (auto const& s : data) {
    chars.append(s);
    offsets.push_back(static_cast<cudf::size_type>(chars.size()));
  }

  rmm::device_uvector<cudf::size_type> d_offsets(offsets.size(), rmm::cuda_stream_default);
  RMM_CUDA_TRY(cudaMemcpy(d_offsets.data(),
                          offsets.data(),
                          offsets.size() * sizeof(cudf::size_type),
                          cudaMemcpyHostToDevice));

  rmm::device_buffer d_chars(chars.data(), chars.size(), rmm::cuda_stream_default);

  auto offsets_col =
    std::make_unique<cudf::column>(std::move(d_offsets), rmm::device_buffer{}, 0);

  return cudf::make_strings_column(
    data.size(), std::move(offsets_col), std::move(d_chars), 0, rmm::device_buffer{});
}

TEST_F(MyTests, join_hang)
{
  // --- 1. Create the Left Table (Single Row) ---
  std::vector<std::unique_ptr<cudf::column>> left_columns;
  {
    std::vector<std::string> imsi_data{"310260250298289"};
    left_columns.push_back(make_strings_column_from_host(imsi_data));

    std::vector<int32_t> hour_data{0};
    rmm::device_buffer hour_buffer(hour_data.data(), hour_data.size() * sizeof(int32_t), rmm::cuda_stream_default);
    left_columns.push_back(cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, 1, std::move(hour_buffer), 0, {}));

    std::vector<int64_t> time_data{1759115400L};
    rmm::device_buffer time_buffer(time_data.data(), time_data.size() * sizeof(int64_t), rmm::cuda_stream_default);
    left_columns.push_back(cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64}, 1, std::move(time_buffer), 0, {}));
  }

  cudf::table_view left_table(get_column_views(left_columns));
  std::cout << "Left table created. Rows: " << left_table.num_rows()
            << ", Columns: " << left_table.num_columns() << std::endl;

  // --- 2. Create the Right Table (Many Rows) ---
  constexpr cudf::size_type num_rows = 25445819;
  std::string join_imsi              = "310260250298289";

  std::vector<std::unique_ptr<cudf::column>> right_columns;
  {
    std::vector<std::string> imsi_data(num_rows, join_imsi);
    right_columns.push_back(make_strings_column_from_host(imsi_data));

    std::vector<int32_t> hour_data(num_rows, 0); // Matching hour_part = 0
    rmm::device_buffer hour_buffer(hour_data.data(), hour_data.size() * sizeof(int32_t), rmm::cuda_stream_default);
    right_columns.push_back(cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, num_rows, std::move(hour_buffer), 0, {}));

    std::vector<int64_t> start_time_data(num_rows, 1759113600L); // 2025-09-29 00:00:00 UTC
    rmm::device_buffer start_time_buffer(start_time_data.data(), start_time_data.size() * sizeof(int64_t), rmm::cuda_stream_default);
    right_columns.push_back(cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64}, num_rows, std::move(start_time_buffer), 0, {}));

    std::vector<int64_t> end_time_data(num_rows, 1759117199L); // 2025-09-29 00:59:59 UTC
    rmm::device_buffer end_time_buffer(end_time_data.data(), end_time_data.size() * sizeof(int64_t), rmm::cuda_stream_default);
    right_columns.push_back(cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64}, num_rows, std::move(end_time_buffer), 0, {}));
  }

  cudf::table_view right_table(get_column_views(right_columns));
  std::cout << "Right table created. Rows: " << right_table.num_rows()
            << ", Columns: " << right_table.num_columns() << std::endl;

  // --- 3. Execute the Join ---
  cudf::ast::tree expr_tree;
  auto const& ge_expr = expr_tree.emplace<cudf::ast::operation>(
    cudf::ast::ast_operator::GREATER_EQUAL,
    expr_tree.emplace<cudf::ast::column_reference>(2, cudf::ast::table_reference::LEFT),
    expr_tree.emplace<cudf::ast::column_reference>(2, cudf::ast::table_reference::RIGHT));

  auto const& le_expr = expr_tree.emplace<cudf::ast::operation>(
    cudf::ast::ast_operator::LESS_EQUAL,
    expr_tree.emplace<cudf::ast::column_reference>(2, cudf::ast::table_reference::LEFT),
    expr_tree.emplace<cudf::ast::column_reference>(3, cudf::ast::table_reference::RIGHT));

  auto const& and_expr =
    expr_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::LOGICAL_AND, ge_expr, le_expr);

  std::cout << "Starting join... this is where it will hang." << std::endl;

  // Use mixed_left_join for joins with both equality and non-equality conditions.
  // This overload takes the full tables and vectors specifying the equality key columns.
#if 0
  std::cout << "Using mixed left join\n";
  auto filtered_join_result =
    cudf::mixed_left_join(left_table, right_table, left_table, right_table, and_expr);
#endif

#if 1
  cudf::sort_merge_join obj(right_table.select({0, 1}), cudf::sorted::NO, cudf::null_equality::EQUAL, rmm::cuda_stream_default);
  std::cout << "Created sort merge join object\n";
  auto const [left_join_indices, right_join_indices] = obj.left_join(left_table.select({0, 1}), cudf::sorted::NO, rmm::cuda_stream_default);
  std::cout << "Completed left sort merge join\n";
  auto filtered_join_result = cudf::filter_join_indices(
    left_table,
    right_table,
    cudf::device_span<cudf::size_type const>(*left_join_indices),
    cudf::device_span<cudf::size_type const>(*right_join_indices),
    and_expr,
    cudf::join_kind::LEFT_JOIN);
#endif

  std::cout << "Join finished. Result has " << filtered_join_result.first->size() <<  " rows." << std::endl;
}
