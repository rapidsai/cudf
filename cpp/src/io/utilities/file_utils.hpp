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

#pragma once

#include <stdlib.h>


// TODO merge with DataSource
/**
 * @brief Helper class for memory mapping a file source
 **/
class MappedFile {
  int fd_ = -1;
  size_t size_ = 0;
  void *map_data_ = nullptr;
  size_t map_size_ = 0;
  size_t map_offset_ = 0;

public:
  MappedFile(const char *path, int oflag);
  MappedFile() noexcept = default;
  ~MappedFile();

  auto size() { return size_; }
  auto data() { return map_data_; }

  void map(size_t size, off_t offset);
};
