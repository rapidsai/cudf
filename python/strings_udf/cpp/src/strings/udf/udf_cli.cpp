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

#include <cudf/strings/udf/udf_apis.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cuda_runtime.h>
#include <sys/time.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

double GetTime()
{
  timeval tv;
  gettimeofday(&tv, NULL);
  return (double)(tv.tv_sec * 1000000 + tv.tv_usec) / 1000000.0;
}

std::string load_udf(std::ifstream &input)
{
  std::stringstream udf;
  std::string line;
  while (std::getline(input, line)) udf << line << "\n";
  return udf.str();
}

void print_column(cudf::strings_column_view const &input)
{
  if (input.chars_size() == 0) {
    printf("empty\n");
    return;
  }

  auto offsets = input.offsets();
  std::vector<int32_t> h_offsets(offsets.size());
  auto chars = input.chars();
  std::vector<char> h_chars(chars.size());
  cudaMemcpy(h_offsets.data(),
             offsets.data<int32_t>(),
             offsets.size() * sizeof(int32_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_chars.data(), chars.data<char>(), chars.size(), cudaMemcpyDeviceToHost);

  for (int idx = 0; idx < input.size(); ++idx) {
    int offset      = h_offsets[idx];
    const char *str = h_chars.data() + offset;
    int length      = h_offsets[idx + 1] - offset;
    std::string output(str, length);
    std::cout << output << "\n";
  }
}

std::map<std::string, std::string> parse_cli_parms(int argc, const char **argv)
{
  std::map<std::string, std::string> parms;
  while (argc > 1) {
    const char *value = argv[argc - 1];
    const char *key   = (argv[argc - 2]) + 1;
    parms[key]        = value;
    argc -= 2;
  }
  return parms;
}

int main(int argc, const char **argv)
{
  if (argc < 3) {
    printf("parameters:\n");
    printf("-u UDF text file\n");
    printf("-n kernel name (default is 'udf_kernel')\n");
    printf("-i libcudf include dir (default is $CONDA_PREFIX/include)\n");
    printf("-t text/csv file\n");
    printf("-c 0-based column number if csv file (default is 0=first column)\n");
    printf("-r number of rows to read from file (default is 0=entire file)\n");
    printf("-f output file (default is stdout)\n");
    printf("-m malloc heap size (default is 1GB)\n");
    return 0;
  }

  std::map<std::string, std::string> parms = parse_cli_parms(argc, argv);

  std::string const udf_text = parms["u"];
  std::string const csv_file = parms["t"];
  if (udf_text.empty() || csv_file.empty()) {
    printf("UDF file (-u) and text file (-t) are required parameters.\n");
    return 0;
  }

  std::string const cudf_include = [parms] {
    if (parms.find("i") != parms.end()) return parms.at("i");
    std::string conda_prefix =
      std::getenv("CONDA_PREFIX") ? std::getenv("CONDA_PREFIX") : "/conda/envs/rapids";
    return conda_prefix + "/include";
  }();

  std::string const udf_name = parms.find("n") != parms.end() ? parms["n"] : "udf_kernel";

  int const column   = parms.find("c") != parms.end() ? std::atoi(parms["c"].c_str()) : 0;
  int const rows     = parms.find("r") != parms.end() ? std::atoi(parms["r"].c_str()) : 0;
  auto const verbose = parms.find("v") != parms.end();
  size_t const heap_size =
    (parms.find("m") != parms.end() ? std::atoi(parms["m"].c_str()) : 1024) * 1024 * 1024;

  // load the udf source code
  std::ifstream udf_stream(udf_text);
  if (!udf_stream.is_open()) {
    printf("could not open file [%s]\n", udf_text.c_str());
    return 0;
  }
  std::string udf_code = load_udf(udf_stream);
  // adding the filename to the top of the source code
  // helps with jitify displaying compile errors
  udf_code = udf_text + "\n" + udf_code;

  // load the text file using the CSV reader
  double st_load_data = GetTime();
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{csv_file}).header(-1);
  in_opts.set_use_cols_indexes({column});
  if (rows > 0) in_opts.set_nrows(rows);
  in_opts.set_dtypes({cudf::data_type{cudf::type_id::STRING}});
  auto csv_result = cudf::io::read_csv(in_opts);
  auto input      = cudf::strings_column_view(csv_result.tbl->view().column(0));

  double et_load_data = GetTime() - st_load_data;
  if (verbose) fprintf(stderr, "Load data: %g seconds\n", et_load_data);

  auto const strings_count = input.size();
  if (verbose) fprintf(stderr, "input strings count = %d\n", strings_count);

  // create UDF module
  std::vector<std::string> options;
  options.push_back("-I" + cudf_include);
  double st_compile = GetTime();
  auto module       = create_udf_module(udf_code, options);
  double et_compile = GetTime() - st_compile;
  if (module == nullptr || module->program == nullptr) {
    printf("compile error\n");
    return 0;
  }
  if (verbose) fprintf(stderr, "Compile UDF: %g seconds\n", et_compile);

  // run UDF module
  double st_run = GetTime();
  auto results  = call_udf(*module, udf_name, strings_count, {input.parent()}, heap_size);
  double et_run = GetTime() - st_run;
  if (verbose) fprintf(stderr, "Run UDF: %g seconds\n", et_run);

  auto scv = cudf::strings_column_view(results->view());
  // output results
  std::string out_filename = parms["f"];
  if (out_filename.empty()) {
    print_column(scv);
    return 0;
  }

  // write csv file
  double st_output_data = GetTime();
  auto output_table     = cudf::table_view{std::vector<cudf::column_view>{results->view()}};
  cudf::io::sink_info const sink{out_filename};
  cudf::io::csv_writer_options writer_options =
    cudf::io::csv_writer_options::builder(sink, output_table).include_header(false);
  cudf::io::write_csv(writer_options);
  double et_output_data = GetTime() - st_output_data;

  if (verbose) fprintf(stderr, "Output to file: %g seconds\n", et_output_data);

  return 0;
}
