/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/io/text/trie.hpp>

#include <sstream>

using namespace cudf;
using namespace test;

constexpr bool print_all{false};

struct TrieTest : public BaseFixture {
};

TEST_F(TrieTest, CanMatchSinglePattern)
{
  auto pattern = cudf::io::text::trie::create("abac", {});

  (void)pattern;
}

TEST_F(TrieTest, CanMatchMultiplePatterns)
{
  auto patterns = std::vector<std::string>{"abac", "abad"};
  auto pattern  = cudf::io::text::trie::create(patterns, {});

  (void)pattern;
}
