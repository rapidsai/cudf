#include <iostream>
#include <chrono>
#include <ctime>

#include <cudf/table/table.hpp>
#include <cudf/io/parquet.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>


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

void write_parquet(std::unique_ptr<cudf::table>& table, cudf::io::table_metadata& metadata, std::string& filepath) {
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
