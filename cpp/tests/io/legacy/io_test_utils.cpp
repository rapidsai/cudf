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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <nvstrings/NVCategory.h>
#include <nvstrings/NVStrings.h>

bool checkFile(std::string const &fname) {
  struct stat st;
  return (stat(fname.c_str(), &st) ? 0 : 1);
}

void checkStrColumn(gdf_column const *col, std::vector<std::string> const &refs) {
  ASSERT_EQ(col->dtype, GDF_STRING);

  auto result = nvstrings_to_strings(static_cast<NVStrings *>(col->data));
  EXPECT_THAT(refs, ::testing::ContainerEq(result));
}

std::vector<std::string> nvstrings_to_strings(NVStrings *nvstr) {
  const auto num_strings = nvstr->size();

  // Allocate host buffer large enough for characters + null-terminator
  std::vector<int> lengths(num_strings);
  auto total_mem = nvstr->byte_count(lengths.data(), false);
  std::vector<char> buffer(total_mem + num_strings, 0);

  // Copy all strings to host memory
  std::vector<char *> strings(num_strings);
  size_t offset = 0;
  for (size_t i = 0; i < num_strings; ++i) {
    strings[i] = buffer.data() + offset;
    offset += lengths[i] + 1;
  }
  nvstr->to_host(strings.data(), 0, num_strings);

  return std::vector<std::string>(strings.data(), strings.data() + num_strings);
}

std::vector<std::string> nvcategory_to_strings(NVCategory *nvcat) {
  return nvstrings_to_strings(nvcat->to_strings());
}
