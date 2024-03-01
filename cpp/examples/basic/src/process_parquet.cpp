/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <cuda/memory_resource>
#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>
#include <rmm/mr/host/host_memory_resource.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

cudf::io::table_with_metadata read_parquet(
  std::string const& file_path, 
  std::string const& col_name)
{
  std::cout << "reading parquet file: " << file_path << " column: " << col_name << std::endl;
  auto source_info = cudf::io::source_info(file_path);
  auto builder     = cudf::io::parquet_reader_options::builder(source_info);
  cudf::io::parquet_reader_options options;
  if (col_name == "ALL") {
    options = builder.build();
  } else {
    options = builder.columns({col_name}).build();
  }
  auto reader = cudf::io::chunked_parquet_reader(
    1L*1024*1024*1024, 
    4L*1024*1024*1024, options);
  cudf::io::table_with_metadata res;
  while (reader.has_next()) {
    res = reader.read_chunk();
    std::cout << "table of " << res.tbl->num_rows() << " rows scanned" << std::endl;
    for (int i = 0; i < res.tbl->num_columns(); ++i) { 
      std::cout << "Col " << i 
                << " num_rows: " << res.tbl->get_column(i).size() 
                << " num_nulls: " << res.tbl->get_column(i).null_count() << std::endl;
    }
  }
  return res;
}

void simple_int_column(int num_rows)
{  
  std::string filepath("/home/abellina/table_with_dict.parquet");

  [[maybe_unused]] auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return 1; }); //i % 2 == 0; });
    //0, [](auto i) { return i == 123 || i == 555 ? 0 : 1; });
    
  /// 0, [](auto i) { return 1; });
  //  0, [](auto i) { return i == 123 || i == 777 ? 0 : 1; });
  auto iter1 = cudf::detail::make_counting_transform_iterator(0, [](int i) { return i % 10; });
  cudf::test::fixed_width_column_wrapper<int> col1(iter1, iter1 + num_rows, valids);
  //cudf::test::fixed_width_column_wrapper<int> col1(iter1, iter1 + num_rows);
  auto tbl = cudf::table_view{{col1}}; 
  
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)    
    .dictionary_policy(cudf::io::dictionary_policy::ALWAYS);
  cudf::io::write_parquet(out_opts);
}

int main(int argc, char** argv)
{
  cudaSetDevice(0);

  //auto resource       = cudf::test::create_memory_resource("pool");
  //auto resource = cudf::test::create_memory_resource("cuda");
  auto resource       = cudf::test::create_memory_resource("cuda");
  rmm::mr::set_current_device_resource(resource.get());

  // Read data
  //auto store_sales = read_parquet("/home/abellina/part-00191-9dcfb50c-76b0-4dbf-882b-b60e7ad5b925.c000.snappy.parquet");
  //auto store_sales = read_parquet("/home/abellina/cudf/part-00000-f7d7d84a-8f00-4921-95d1-ef17638a1c83-c000.snappy.parquet");
  //auto store_sales = read_parquet("/home/abellina/cudf/second_mill.snappy.parquet");
  //auto store_sales = read_parquet("/home/abellina/cudf/s_1.5.snappy.parquet");
  //auto store_sales = read_parquet("/home/abellina/cudf/s_1.25.snappy.parquet");
  //auto store_sales = read_parquet("/home/abellina/cudf/s_1.125.snappy.parquet");
  //auto store_sales = read_parquet("/home/abellina/cudf/s_1.005.snappy.parquet");
  //auto store_sales = read_parquet("/home/abellina/cudf/part-00000-f7d7d84a-8f00-4921-95d1-ef17638a1c83-c000.snappy.parquet");
  ///cudaDeviceSynchronize();
// std::cout <<"done1"<<std::endl;
  //auto store_sales = read_parquet("/home/abellina/cudf/first_1m.snappy.parquet");

// ENABLE THIS
[[maybe_unused]]const char* name = nullptr;
name = argv[1];
std::string col_names[] = {
"ss_sold_time_sk", 
"ss_item_sk", 
"ss_customer_sk", 
"ss_cdemo_sk", 
"ss_hdemo_sk", 
"ss_addr_sk", 
"ss_store_sk", 
"ss_promo_sk", 
"ss_ticket_number", 
"ss_quantity", 
"ss_wholesale_cost", 
"ss_list_price", 
"ss_sales_price", 
"ss_ext_discount_amt", 
"ss_ext_sales_price", 
"ss_ext_wholesale_cost", 
"ss_ext_list_price", 
"ss_ext_tax", 
"ss_coupon_amt", 
"ss_net_paid", 
"ss_net_paid_inc_tax", 
"ss_net_profit"
};

std::string store_col_names[] = {
 "s_store_sk", //: integer (nullable = true)
 "s_store_id", //: string (nullable = true)
 "s_rec_start_date", //: date (nullable = true)
 "s_rec_end_date", //: date (nullable = true)
 "s_closed_date_sk", //: integer (nullable = true)
 "s_store_name", //: string (nullable = true)
 "s_number_employees", //: integer (nullable = true)
 "s_floor_space", //: integer (nullable = true)
 "s_hours", //: string (nullable = true)
 "s_manager", //: string (nullable = true)
 "s_market_id", //: integer (nullable = true)
 "s_geography_class", //: string (nullable = true)
 "s_market_desc", //: string (nullable = true)
 "s_market_manager", //: string (nullable = true)
 "s_division_id", //: integer (nullable = true)
 "s_division_name", //: string (nullable = true)
 "s_company_id", //: integer (nullable = true)
 "s_company_name", //: string (nullable = true)
 "s_street_number", //: string (nullable = true)
 "s_street_name", //: string (nullable = true)
 "s_street_type", //: string (nullable = true)
 "s_suite_number", //: string (nullable = true)
 "s_city", //: string (nullable = true)
 "s_county", //: string (nullable = true)
 "s_state", //: string (nullable = true)
 "s_zip", //: string (nullable = true)
 "s_country", //: string (nullable = true)
 "s_gmt_offset", //: decimal(5,2) (nullable = true)
 "s_tax_precentage" //: decimal(5,2) (nullable = true)
};

  std::string bad_file_col_names[] = {
    "ss_sold_time_sk",
    "ss_hdemo_sk",
    "ss_store_sk"
  };


//for (std::string col : col_names) {
//  // setenv("USE_FIXED_OP", "0", 1);
//  // auto expected = read_parquet(name, col);
//  // cudaDeviceSynchronize();

//  //setenv("USE_FIXED_OP", "2", 1);
//  read_parquet(name, col);
//  cudaDeviceSynchronize();
//  // CUDF_TEST_EXPECT_TABLES_EQUAL(expected.tbl->view(), actual.tbl->view());
//  std::cout << "done" << std::endl;
//}
for (int i  = 0; i < 1; ++i) {
   setenv("USE_FIXED_OP", "0", 1);
   auto expected = read_parquet(name, "ALL");
   cudaDeviceSynchronize();
   std::cout << "op0:" << cudf::test::to_string(expected.tbl->get_column(0).view(), std::string(",")) << std::endl;

   setenv("USE_FIXED_OP", "2", 1);
   auto actual = read_parquet(name, "ALL");
   //read_parquet(name, "ALL");
   cudaDeviceSynchronize();
   CUDF_TEST_EXPECT_TABLES_EQUAL(expected.tbl->view(), actual.tbl->view());
   std::cout << "op2:" << cudf::test::to_string(actual.tbl->get_column(0).view(), std::string(",")) << std::endl;
   std::cout << "done " << i << std::endl;
}


 //if (argc > 1) {
 //  num_rows = atoi(argv[1]);
 //}
 //simple_int_column(17);
 //read_parquet("/home/abellina/table_with_dict.parquet", "ALL");

// std::cout << "over here: " << cudf::test::to_string(simple.tbl->get_column(0).view(), std::string(",")) << std::endl;
 //std::cout << "done" << std::endl;

  return 0;
}
