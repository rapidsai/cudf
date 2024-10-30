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
#include <cudf_test/default_stream.hpp>

#include <cudf/io/text/data_chunk_source_factories.hpp>
#include <cudf/io/text/multibyte_split.hpp>

#include <string>

class MultibyteSplitTest : public cudf::test::BaseFixture {};

TEST_F(MultibyteSplitTest, Reader)
{
  auto delimiter  = std::string(":");
  auto host_input = std::string("abc:def");
  auto source     = cudf::io::text::make_source(host_input);
  cudf::io::text::parse_options options{};
  auto result =
    cudf::io::text::multibyte_split(*source, delimiter, options, cudf::test::get_default_stream());
}
