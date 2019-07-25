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

#include "io_test_utils.hpp"

#include <gtest/gtest.h>

#include <nvstrings/NVStrings.h>

bool checkFile(std::string const &fname) {
  struct stat st;
  return (stat(fname.c_str(), &st) ? 0 : 1);
}

void checkStrColumn(gdf_column const *col, std::vector<std::string> const &refs) {
  ASSERT_EQ(col->dtype, GDF_STRING);

  const auto stringList = reinterpret_cast<NVStrings *>(col->data);
  ASSERT_NE(stringList, nullptr);

  const auto count = stringList->size();
  ASSERT_EQ(count, refs.size());

  std::vector<int> lengths(count);
  ASSERT_NE(stringList->byte_count(lengths.data(), false), 0u);

  // Check the actual strings themselves
  std::vector<char *> strings(count);
  for (size_t i = 0; i < count; ++i) {
    strings[i] = new char[lengths[i] + 1];
    strings[i][lengths[i]] = 0;
  }
  EXPECT_EQ(stringList->to_host(strings.data(), 0, count), 0);

  for (size_t i = 0; i < count; ++i) {
    EXPECT_STREQ(strings[i], refs[i].c_str());
  }
  for (size_t i = 0; i < count; ++i) {
    delete[] strings[i];
  }
}
