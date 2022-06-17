/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/json.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <src/io/json/data_casting.cuh>

#include <type_traits>

struct JSONTypeCastTest : public cudf::test::BaseFixture {
};

namespace {
using string_pair = thrust::pair<char const*, cudf::size_type>;
struct string_view_only {
  __device__ cudf::string_view operator()(thrust::pair<cudf::string_view, bool> const& p)
  {
    return p.first;
  }
};
}  // namespace

TEST_F(JSONTypeCastTest, RealBasic)
{
  auto const stream = rmm::cuda_stream_default;
  std::vector<cudf::data_type> types{cudf::data_type{cudf::type_id::INT32}};

  cudf::test::strings_column_wrapper data({"this", "is", "a", "column", "of", "strings"});
  auto d_column = cudf::column_device_view::create(data);
  rmm::device_uvector<cudf::string_view> svs(d_column->size(), rmm::cuda_stream_default);
  thrust::transform(thrust::device,
                    d_column->pair_begin<cudf::string_view, false>(),
                    d_column->pair_end<cudf::string_view, false>(),
                    svs.data(),
                    string_view_only{});

  std::vector<cudf::string_view*> str_spans{svs.data()};
  auto d_str_spans = cudf::detail::make_device_uvector_async(str_spans, stream);
  std::vector<cudf::size_type> col_size{(cudf::size_type)svs.size()};
  auto d_col_size = cudf::detail::make_device_uvector_async(col_size, stream);

  cudf::io::json::experimental::parse_data<cudf::string_view**, cudf::size_type*>(
    d_str_spans.data(), d_col_size.data(), types, rmm::cuda_stream_default);
}

CUDF_TEST_PROGRAM_MAIN()
