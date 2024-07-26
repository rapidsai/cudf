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
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>

#include <ctime>
#include <vector>

void write_parquet(cudf::table_view tbl,
                   std::string const& path,
                   std::vector<std::string> const& col_names)
{
  std::cout << "Writing to " << path << "\n";
  auto const sink_info = cudf::io::sink_info(path);
  cudf::io::table_metadata metadata;
  std::vector<cudf::io::column_name_info> col_name_infos;
  for (auto& col_name : col_names) {
    col_name_infos.push_back(cudf::io::column_name_info(col_name));
  }
  metadata.schema_info            = col_name_infos;
  auto const table_input_metadata = cudf::io::table_input_metadata{metadata};
  auto builder                    = cudf::io::parquet_writer_options::builder(sink_info, tbl);
  builder.metadata(table_input_metadata);
  auto const options = builder.build();
  cudf::io::write_parquet(options);
}

std::unique_ptr<cudf::table> perform_left_join(cudf::table_view const& left_input,
                                               cudf::table_view const& right_input,
                                               std::vector<cudf::size_type> const& left_on,
                                               std::vector<cudf::size_type> const& right_on,
                                               cudf::null_equality compare_nulls)
{
  constexpr auto oob_policy                          = cudf::out_of_bounds_policy::NULLIFY;
  auto const left_selected                           = left_input.select(left_on);
  auto const right_selected                          = right_input.select(right_on);
  auto const [left_join_indices, right_join_indices] = cudf::left_join(
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

struct groupby_context_t {
  std::vector<int64_t> keys;
  std::unordered_map<std::string, std::vector<std::pair<cudf::aggregation::Kind, std::string>>>
    values;
};

/**
 * @brief Generate the `std::tm` structure from year, month, and day
 *
 * @param year The year
 * @param month The month
 * @param day The day
 */
std::tm make_tm(int year, int month, int day)
{
  std::tm tm{};
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
