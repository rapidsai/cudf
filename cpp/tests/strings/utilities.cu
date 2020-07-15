/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <tests/strings/utilities.h>

#include <gmock/gmock.h>

namespace cudf {
namespace test {
void expect_strings_empty(cudf::column_view strings_column)
{
  EXPECT_EQ(type_id::STRING, strings_column.type().id());
  EXPECT_EQ(0, strings_column.size());
  EXPECT_EQ(0, strings_column.null_count());
  EXPECT_EQ(0, strings_column.num_children());
}

}  // namespace test
}  // namespace cudf
