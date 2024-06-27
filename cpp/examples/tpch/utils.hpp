#include <iostream>
#include <chrono>
#include <ctime>

#include <cudf/table/table.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/join.hpp>
#include <cudf/copying.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>


std::unique_ptr<cudf::table> join_and_gather(
    cudf::table_view left_input,
    cudf::table_view right_input,
    std::vector<cudf::size_type> left_on,
    std::vector<cudf::size_type> right_on,
    cudf::null_equality compare_nulls,
    rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource()) {

    auto oob_policy = cudf::out_of_bounds_policy::DONT_CHECK;
    auto left_selected  = left_input.select(left_on);
    auto right_selected = right_input.select(right_on);
    auto const [left_join_indices, right_join_indices] = 
        cudf::inner_join(left_selected, right_selected, compare_nulls, mr);

    auto left_indices_span  = cudf::device_span<cudf::size_type const>{*left_join_indices};
    auto right_indices_span = cudf::device_span<cudf::size_type const>{*right_join_indices};

    auto left_indices_col  = cudf::column_view{left_indices_span};
    auto right_indices_col = cudf::column_view{right_indices_span};

    auto left_result  = cudf::gather(left_input, left_indices_col, oob_policy);
    auto right_result = cudf::gather(right_input, right_indices_col, oob_policy);

    auto joined_cols = left_result->release();
    auto right_cols  = right_result->release();
    joined_cols.insert(joined_cols.end(),
                        std::make_move_iterator(right_cols.begin()),
                        std::make_move_iterator(right_cols.end()));
    return std::make_unique<cudf::table>(std::move(joined_cols));
}

std::unique_ptr<cudf::table> inner_join(
  cudf::table_view left_input,
  cudf::table_view right_input,
  std::vector<cudf::size_type> left_on,
  std::vector<cudf::size_type> right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL)
{
  return join_and_gather(
    left_input, right_input, left_on, right_on, compare_nulls);
}

template <typename T>
std::vector<T> concat(const std::vector<T>& lhs, const std::vector<T>& rhs) {
    std::vector<T> result;
    result.reserve(lhs.size() + rhs.size());
    std::copy(lhs.begin(), lhs.end(), std::back_inserter(result));
    std::copy(rhs.begin(), rhs.end(), std::back_inserter(result));
    return result;
}

std::pair<std::unique_ptr<cudf::table>, std::vector<std::string>> read_parquet(std::string filename) {
    auto source = cudf::io::source_info(filename);
    auto builder = cudf::io::parquet_reader_options_builder(source);
    auto options = builder.build();
    auto table_with_metadata = cudf::io::read_parquet(options);
    auto schema_info = table_with_metadata.metadata.schema_info;
    std::vector<std::string> column_names;
    for (auto &col_info : schema_info) {
        column_names.push_back(col_info.name);
    }
    return std::make_pair(std::move(table_with_metadata.tbl), column_names);
}

std::tm make_tm(int year, int month, int day) {
    std::tm tm = {0};
    tm.tm_year = year - 1900;
    tm.tm_mon = month - 1;
    tm.tm_mday = day;
    return tm;
}

int32_t days_since_epoch(int year, int month, int day) {
    std::tm tm = make_tm(year, month, day);
    std::tm epoch = make_tm(1970, 1, 1);
    std::time_t time = std::mktime(&tm);
    std::time_t epoch_time = std::mktime(&epoch);
    double diff = std::difftime(time, epoch_time) / (60*60*24);
    return static_cast<int32_t>(diff);
}

cudf::io::table_metadata create_table_metadata(std::vector<std::string> column_names) {
    cudf::io::table_metadata metadata;
    std::vector<cudf::io::column_name_info> column_name_infos;
    for (auto &col_name : column_names) {
        column_name_infos.push_back(cudf::io::column_name_info(col_name));
    }
    metadata.schema_info = column_name_infos;
    return metadata;
}

std::unique_ptr<cudf::table> append_col_to_table(
    std::unique_ptr<cudf::table>& table, std::unique_ptr<cudf::column>& col) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    for (size_t i = 0; i < table->num_columns(); i++) {
        columns.push_back(std::make_unique<cudf::column>(table->get_column(i)));
    }
    columns.push_back(std::move(col));
    return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::table> order_by(
    std::unique_ptr<cudf::table>& table, std::vector<int32_t> keys) {
    auto table_view = table->view();
    std::vector<cudf::column_view> column_views;
    for (auto& key : keys) {
        column_views.push_back(table_view.column(key));
    }    
    return cudf::sort_by_key(
        table_view, 
        cudf::table_view{column_views},
        {cudf::order::DESCENDING}
    );
}

void write_parquet(std::unique_ptr<cudf::table>& table, cudf::io::table_metadata metadata, std::string filepath) {
    auto sink_info = cudf::io::sink_info(filepath);
    auto table_input_metadata = cudf::io::table_input_metadata{metadata};
    auto builder = cudf::io::parquet_writer_options::builder(sink_info, table->view());
    builder.metadata(table_input_metadata);
    auto options = builder.build();
    cudf::io::write_parquet(options);
}

template<typename T>
rmm::device_buffer get_device_buffer_from_value(T value) {
    auto stream = cudf::get_default_stream();    
    rmm::cuda_stream_view stream_view(stream);

    rmm::device_scalar<T> scalar(stream_view);
    scalar.set_value_async(value, stream_view);

    rmm::device_buffer buffer(scalar.data(), scalar.size(), stream_view);
    return buffer;
}

rmm::device_buffer get_empty_device_buffer() {
    auto stream = cudf::get_default_stream();    
    rmm::cuda_stream_view stream_view(stream);
    rmm::device_buffer buffer(0, stream_view);
    return buffer;
}
