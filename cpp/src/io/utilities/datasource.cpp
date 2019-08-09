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

#include "datasource.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cudf/cudf.h>
#include <utilities/error_utils.hpp>

namespace cudf {
namespace io {

/**
 * @brief Implementation class for reading from an Apache Arrow file. The file
 * could be a memory-mapped file or other implementation supported by Arrow.
 **/
class arrow_io_source : public datasource {
 public:
  explicit arrow_io_source(std::shared_ptr<arrow::io::RandomAccessFile> file)
      : arrow_file(file) {}

  const std::shared_ptr<arrow::Buffer> get_buffer(size_t position,
                                                  size_t length) override {
    std::shared_ptr<arrow::Buffer> out;
    CUDF_EXPECTS(arrow_file->ReadAt(position, length, &out).ok(),
                 "Cannot read file data");
    return out;
  }

  size_t size() const override {
    int64_t size;
    CUDF_EXPECTS(arrow_file->GetSize(&size).ok(), "Cannot get file size");
    return size;
  }

 private:
  std::shared_ptr<arrow::io::RandomAccessFile> arrow_file;
};

/**
 * @brief Implementation class for reading from a file or memory source using
 * memory mapped access.
 *
 * Unlike Arrow's memory mapped IO class, this implementation allows memory
 * mapping a subset of the file where the starting offset may not be zero.
 **/
class memory_mapped_source : public datasource {
  struct file_wrapper {
    const int fd = -1;
    explicit file_wrapper(const char *filepath)
        : fd(open(filepath, O_RDONLY)) {}
    ~file_wrapper() { close(fd); }
  };

 public:
  explicit memory_mapped_source(const char *filepath, size_t offset,
                                size_t size) {
    auto file = file_wrapper(filepath);
    CUDF_EXPECTS(file.fd != -1, "Cannot open file");

    struct stat st {};
    CUDF_EXPECTS(fstat(file.fd, &st) != -1, "Cannot query file size");
    file_size_ = static_cast<size_t>(st.st_size);

    if (file_size_ != 0) {
      map(file.fd, offset, size);
    }
  }

  virtual ~memory_mapped_source() {
    if (map_addr_ != nullptr) {
      munmap(map_addr_, map_size_);
    }
  }

  const std::shared_ptr<arrow::Buffer> get_buffer(size_t offset,
                                                  size_t size) override {
    // Clamp length to available data in the mapped region
    CUDF_EXPECTS(offset >= map_offset_, "Requested offset is outside mapping");
    size = std::min(size, map_size_ - (offset - map_offset_));

    return arrow::Buffer::Wrap(
        static_cast<uint8_t *>(map_addr_) + (offset - map_offset_), size);
  }

  size_t size() const override { return file_size_; }

 private:
  void map(int fd, size_t offset, size_t size) {
    // Offset for `mmap()` must be page aligned
    const auto map_offset = offset & ~(sysconf(_SC_PAGESIZE) - 1);
    CUDF_EXPECTS(offset < file_size_, "Offset is past end of file");

    // Clamp length to available data in the file
    if (size == 0) {
      size = file_size_ - offset;
    } else {
      if ((offset + size) > file_size_) {
        size = file_size_ - offset;
      }
    }

    // Size for `mmap()` needs to include the page padding
    const auto map_size = size + (offset - map_offset);

    // Check if accessing a region within already mapped area
    map_addr_ = mmap(NULL, map_size, PROT_READ, MAP_PRIVATE, fd, map_offset);
    CUDF_EXPECTS(map_addr_ != MAP_FAILED, "Cannot create memory mapping");
    map_offset_ = map_offset;
    map_size_ = map_size;
  }

 private:
  size_t file_size_ = 0;
  void *map_addr_ = nullptr;
  size_t map_size_ = 0;
  size_t map_offset_ = 0;
};

std::unique_ptr<datasource> datasource::create(const std::string filepath,
                                               size_t offset, size_t size) {
  // Use our own memory mapping implementation for direct file reads
  return std::make_unique<memory_mapped_source>(filepath.c_str(), offset, size);
}

std::unique_ptr<datasource> datasource::create(const char *data,
                                               size_t length) {
  // Use Arrow IO buffer class for zero-copy reads of host memory
  return std::make_unique<arrow_io_source>(
      std::make_shared<arrow::io::BufferReader>(
          reinterpret_cast<const uint8_t *>(data), length));
}

std::unique_ptr<datasource> datasource::create(
    std::shared_ptr<arrow::io::RandomAccessFile> file) {
  // Support derived classes of the top-level Arrow IO interface
  return std::make_unique<arrow_io_source>(file);
}

}  // namespace io
}  // namespace cudf
