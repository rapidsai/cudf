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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/interop.hpp>
#include <cudf/table/table_view.hpp>

#include <dlpack/dlpack.h>

namespace {
struct dlpack_deleter {
  void operator()(DLManagedTensor* tensor) { tensor->deleter(tensor); }
};
}  // namespace

struct DLPackTest : public cudf::test::BaseFixture {};

TEST_F(DLPackTest, ToDLPack)
{
  cudf::table_view empty(std::vector<cudf::column_view>{});
  cudf::to_dlpack(empty, cudf::test::get_default_stream());
}

TEST_F(DLPackTest, FromDLPack)
{
  using unique_managed_tensor = std::unique_ptr<DLManagedTensor, dlpack_deleter>;
  cudf::test::fixed_width_column_wrapper<int32_t> col1({});
  cudf::test::fixed_width_column_wrapper<int32_t> col2({});
  cudf::table_view input({col1, col2});
  unique_managed_tensor tensor(cudf::to_dlpack(input, cudf::test::get_default_stream()));
  auto result = cudf::from_dlpack(tensor.get(), cudf::test::get_default_stream());
}
