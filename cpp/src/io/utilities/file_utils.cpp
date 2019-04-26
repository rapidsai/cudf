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
