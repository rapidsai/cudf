#pragma once

#include <stdlib.h>

// DOXY
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
