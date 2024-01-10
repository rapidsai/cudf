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

#include <io/json/read_json.hpp>
#include <io/utilities/hostdevice_vector.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/span.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/io_metadata_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>

#include <string>

// Base test fixture for tests
struct JsonNormalizationTest : public cudf::test::BaseFixture {};

TEST_F(JsonNormalizationTest, Valid)
{
  // Test input
  std::string const input = R"({"A":'TEST"'})";
  auto device_input_ptr   = cudf::make_string_scalar(input, cudf::test::get_default_stream());
  auto& device_input      = static_cast<scalar_type_t<std::string>&>(*device_input_ptr);

  // RMM memory resource
  std::shared_ptr<rmm::mr::device_memory_resource> rsc =
    std::make_shared<rmm::mr::cuda_memory_resource>();

  auto device_fst_output_ptr =
    normalize_quotes(device_input.data(), cudf::test::get_default_stream(), rsc.get());
  // Initialize parsing options (reading json lines)
  cudf::io::json_reader_options input_options = cudf::io::json_reader_options::builder(
    cudf::io::source_info{device_span(*device_fst_output_ptr)});

  cudf::io::table_with_metadata processed_table =
    cudf::io::read_json(input_options, cudf::test::get_default_stream(), rsc);
}

CUDF_TEST_PROGRAM_MAIN()
