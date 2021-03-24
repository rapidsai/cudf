#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>

#include <cudf/io/csv.hpp>

std::unique_ptr<cudf::table> read_csv(std::string const& file_path)
{
  auto source_info    = cudf::io::source_info(file_path);
  auto builder        = cudf::io::csv_reader_options::builder(source_info);
  auto options        = builder.build();
  auto data_with_meta = cudf::io::read_csv(options);
  return data_with_meta.tbl;
}

void write_csv(cudf::table_view const& tbl_view, std::string const& file_path)
{
  auto sink_info = cudf::io::sink_info(file_path);
  auto builder   = cudf::io::csv_writer_options::builder(sink_info, tbl_view);
  auto options   = builder.build();
  cudf::io::write_csv(options);
}

std::unique_ptr<cudf::table> get_columns_from_table(cudf::table_view table,
                                                    std::vector<int> const& indices)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  for (auto i : indices) {
    cols.push_back(std::move(std::make_unique<cudf::column>(table.column(i))));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

int main(int argc, char** argv)
{
  // Init memory pool
  size_t available_mem, total_mem, prealloc_mem;
  cudaMemGetInfo(&available_mem, &total_mem);
  prealloc_mem = available_mem * 0.5;
  prealloc_mem = prealloc_mem - prealloc_mem % 256;

  auto cuda_mr = std::make_unique<rmm::mr::cuda_memory_resource>();
  auto pool_mr = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
    cuda_mr.get(), prealloc_mem);
  rmm::mr::set_current_device_resource(pool_mr.get());

  // Read data
  auto table = read_csv("4stock_5day.csv");

  // Schema: | Company | Date | Open | High | Low | Close | Volume |
  std::vector<int> key_cols{0};  // Company
  std::vector<int> val_cols{2, 3, 4, 5, 6};

  auto keys = get_columns_from_table(table, key_cols);
  auto vals = get_columns_from_table(table, val_cols);

  // Compute the average of each company's closing price across 5 day span
  cudf::groupby::groupby grpby_obj(keys);
  cudf::groupby::aggregation_request agg_req;
  agg_req.values = vals.column(3);
  agg_req.aggregations.push_back(cudf::make_mean_aggregation());
  std::vector<cudf::groupby::aggregation_request> reqs;
  reqs.push_back(std::move(agg_req));

  auto agg_res = grpby_obj.aggregate(reqs);

  // Assembling the result
  auto res_key = std::move(agg_res.first);
  auto res_val = std::move(agg_res.second[0].results[0]);

  std::vector<cudf::column_view> cols(res_key->view().begin(), res_key->view().end());
  cols.push_back(res_val->view());
  auto res = cudf::table_view(cols);

  // Write out result
  write_csv(res, "4stock_5day_avg.csv");

  return 0;
}
