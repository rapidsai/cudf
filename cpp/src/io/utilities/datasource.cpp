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
#include <cudf/utilities/error.hpp>

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

  virtual bool has_fixed_mappings() const override { return true; }

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

/**
 * @brief Implementation class for reading from a file optimized for reading
 * to device memory. Attempts to hide HtoD transfer latency with disk io by using
 * a double-buffer.
 *
 **/
class gpu_async_filereader : public datasource {
  static constexpr uint32_t default_buffer_size = 2 << 20; // 2x2MB

  struct pinned_filebuf {
    uint8_t *host_bfr = nullptr;
    cudaStream_t stream = 0;
    cudaEvent_t completion_event = 0;
    uint32_t position = 0;
    bool event_pending = false;

    ~pinned_filebuf() {
      if (completion_event) {
        cudaEventDestroy(completion_event);
      }
      if (host_bfr) {
        cudaFreeHost(host_bfr);
      }
    }

    void flush() {
      if (position != 0 && !event_pending) {
        if (!completion_event) {
          CUDA_TRY(cudaEventCreateWithFlags(&completion_event, cudaEventDisableTiming));
        }
        CUDA_TRY(cudaEventRecord(completion_event, stream));
        cudaStreamQuery(stream); // In case cudaEventRecord did not flush all the way to gpu
        event_pending = true;
      }
    }

    void init_or_wait(uint32_t buffer_size, cudaStream_t stream_ = 0) {
      if (!host_bfr) {
        CUDA_TRY(cudaMallocHost(reinterpret_cast<void **>(&host_bfr), buffer_size));
      }
      stream = stream_;
      position = 0;
      if (event_pending) {
        event_pending = false;
        cudaEventSynchronize(completion_event);
      }
    }

    void transfer_h2d(void *dst, size_t bytecnt) {
      CUDA_TRY(cudaMemcpyAsync(dst, host_bfr + position, bytecnt, cudaMemcpyHostToDevice, stream));
      position += static_cast<uint32_t>(bytecnt);
    }
  };

 public:
  explicit gpu_async_filereader(std::shared_ptr<arrow::io::RandomAccessFile> file, uint32_t buffer_size = default_buffer_size):
    file_(file), bufid_(0), buffer_size_(buffer_size) {
  }

  size_t size() const override {
    int64_t sz = 0;
    CUDF_EXPECTS(file_->GetSize(&sz).ok(), "Failed to get file size");
    return sz;
  }

  const std::shared_ptr<arrow::Buffer> get_buffer(size_t offset, size_t size) override {
    std::shared_ptr<arrow::Buffer> buffer;
    CUDF_EXPECTS(file_->ReadAt(offset, size, &buffer).ok(), "ReadAt failed");
    CUDF_EXPECTS(size == buffer->size(), "Failed to read all bytes");
    return buffer;
  }

  void device_read(size_t offset, size_t size, void *dst, cudaStream_t stream) override {
    // Break up larger reads in buffer_size chunks, trying to re-use space in the current buffer
    // as long as it doesn't result in too small chunks 
    while (size > 0) {
      uint32_t avail_bytes = buffer_size_ - dblbuf_[bufid_].position;
      bool merge_enable = ((size <= avail_bytes) || (size > buffer_size_ && size - buffer_size_ <= avail_bytes))
                       && (avail_bytes == buffer_size_ || (stream == dblbuf_[bufid_].stream && !dblbuf_[bufid_].event_pending));
      size_t bytecnt = std::min<size_t>(size, (merge_enable) ? avail_bytes : buffer_size_);
      if (!merge_enable) {
        dblbuf_[bufid_].flush();
        bufid_ = !bufid_;
      }
      if (!merge_enable || !dblbuf_[bufid_].host_bfr) {
        // If HtoD transfers are faster than disk reads, we should in theory never actually wait on the completion events
        dblbuf_[bufid_].init_or_wait(buffer_size_, stream);
      }
      if (bytecnt != 0) {
        auto& buffer = dblbuf_[bufid_];
        CUDF_EXPECTS(bytecnt == read_at(offset, buffer.host_bfr + buffer.position, bytecnt), // blocking
                     "File read failed");
        buffer.transfer_h2d(dst, bytecnt); // async
        dst = reinterpret_cast<uint8_t *>(dst) + bytecnt;
        offset += bytecnt;
        size -= bytecnt;
      }
    }
  }

  void flush() override {
    dblbuf_[bufid_].flush();
  }

 protected:
  size_t read_at(size_t offset, void *dst, size_t bytecnt) {
    int64_t bytes_read = 0;
    CUDF_EXPECTS(file_->ReadAt(offset, bytecnt, &bytes_read, dst).ok(), "ReadAt failed");
    return bytes_read;
  }

 private:
  std::shared_ptr<arrow::io::RandomAccessFile> file_;
  const uint32_t buffer_size_;
  int bufid_;
  pinned_filebuf dblbuf_[2];
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

std::unique_ptr<datasource> datasource::create_async_gpureader(const std::string filepath) {
  std::shared_ptr<arrow::io::ReadableFile> file;
  auto status = arrow::io::ReadableFile::Open(filepath, &file);
  CUDF_EXPECTS(status.ok(), "Failed to open file");
  return std::make_unique<gpu_async_filereader>(file);
}


}  // namespace io
}  // namespace cudf
