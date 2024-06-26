#include <iostream>
#include <chrono>
#include <ctime>

#include <cudf/table/table.hpp>
#include <cudf/io/parquet.hpp>

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

void write_parquet(std::unique_ptr<cudf::table>& table, cudf::io::table_metadata& metadata, std::string& filepath) {
    auto sink_info = cudf::io::sink_info(filepath);
    auto table_input_metadata = cudf::io::table_input_metadata{metadata};
    auto builder = cudf::io::parquet_writer_options::builder(sink_info, table->view());
    builder.metadata(table_input_metadata);
    auto options = builder.build();
    cudf::io::write_parquet(options);
}
