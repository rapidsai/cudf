/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cstddef>
#include <cstring>
#include <string>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/utilities/span.hpp>

using cudf::detail::span;

template <typename T>
class SpanTest : public cudf::test::BaseFixture {
 public:
  SpanTest() : _hello_world("hello world") {}

  std::vector<char> hello_world()
  {
    return std::vector<char>(_hello_world.begin(), _hello_world.end());
  }

 private:
  std::string const _hello_world;
};

TYPED_TEST_CASE(SpanTest, cudf::test::FloatingPointTypes);

TYPED_TEST(SpanTest, CanTakeFirst) { auto message = this->hello_world(); }
