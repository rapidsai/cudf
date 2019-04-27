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

#include "file_utils.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "cudf.h"
#include "utilities/error_utils.hpp"

MappedFile::MappedFile(const char *path, int oflag) {
  CUDF_EXPECTS((fd_ = open(path, oflag)) != -1, "Cannot open input file");

  struct stat st {};
  if (fstat(fd_, &st) == -1 || st.st_size < 0) {
    close(fd_);
    CUDF_FAIL("Cannot stat input file");
  }
  size_ = static_cast<size_t>(st.st_size);
}

void MappedFile::map(size_t size, off_t offset) {
  CUDF_EXPECTS(size > 0, "Cannot have zero size mapping");

  map_data_ = mmap(0, size, PROT_READ, MAP_PRIVATE, fd_, offset);
  CUDF_EXPECTS(map_data_ != MAP_FAILED, "Error mapping input file");
  map_offset_ = offset;
  map_size_ = size;
}

MappedFile::~MappedFile() {
  close(fd_);
  if (map_data_ != nullptr) {
    munmap(map_data_, map_size_);
  }
}
