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
#include <cstring>
#include <limits>

#include <cufile.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include <cudf/utilities/error.hpp>
#include <sys/stat.h>
#include <sys/types.h>

#include "jni_utils.hpp"

namespace {

/**
 * @brief Get the error description based on the CUDA driver error code.
 *
 * @param cu_result CUDA driver error code.
 * @return Description for the error.
 */
char const *GetCuErrorString(CUresult cu_result) {
  char const *description;
  if (cuGetErrorName(cu_result, &description) != CUDA_SUCCESS)
    description = "unknown cuda error";
  return description;
}

/**
 * @brief Get the error description based on the integer error code.
 *
 * cuFile APIs return both cuFile specific error codes as well as POSIX error codes for ease of use.
 *
 * @param error_code Integer error code.
 * @return Description of the error.
 */
std::string cuFileGetErrorString(int error_code) {
  return IS_CUFILE_ERR(error_code) ? std::string(CUFILE_ERRSTR(error_code)) :
                                     std::string(std::strerror(error_code));
}

/**
 * @brief Get the error description based on the cuFile return status.
 *
 * @param status cuFile return status.
 * @return Description of the error.
 */
std::string cuFileGetErrorString(CUfileError_t status) {
  std::string error = cuFileGetErrorString(status.err);
  if (IS_CUDA_ERR(status)) {
    error.append(".").append(GetCuErrorString(status.cu_err));
  }
  return error;
}

/**
 * @brief RAII wrapper for the cuFile driver.
 */
class cufile_driver {
public:
  /** @brief Construct a new driver instance by opening the cuFile driver. */
  cufile_driver() {
    auto const status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
      CUDF_FAIL("Failed to initialize cuFile driver: " + cuFileGetErrorString(status));
    }
  }

  // Disable copy (and move) semantics.
  cufile_driver(cufile_driver const &) = delete;
  cufile_driver &operator=(cufile_driver const &) = delete;

  /** @brief Destroy the driver instance by closing the cuFile driver. */
  ~cufile_driver() { cuFileDriverClose(); }
};

/** @brief RAII wrapper for a device buffer used by cuFile. */
class cufile_buffer {
public:
  /**
   * @brief Construct a new cuFile buffer.
   *
   * @param device_pointer Pointer to the device buffer.
   * @param size The size of the allocated device buffer.
   * @param register_buffer Whether to register the buffer with cuFile. This should only be set to
   * true if this buffer is being reused to fill a larger buffer.
   */
  cufile_buffer(void *device_pointer, std::size_t size, bool register_buffer = false)
      : device_pointer_{device_pointer}, size_{size}, register_buffer_{register_buffer} {
    if (register_buffer_) {
      auto const status = cuFileBufRegister(device_pointer_, size_, 0);
      if (status.err != CU_FILE_SUCCESS) {
        CUDF_FAIL("Failed to register cuFile buffer: " + cuFileGetErrorString(status));
      }
    }
  }

  // Disable copy (and move) semantics.
  cufile_buffer(cufile_buffer const &) = delete;
  cufile_buffer &operator=(cufile_buffer const &) = delete;

  /** @brief Destroy the buffer by de-registering it if necessary. */
  ~cufile_buffer() {
    if (register_buffer_) {
      cuFileBufDeregister(device_pointer_);
    }
  }

  /**
   * @brief Get the pointer to the underlying device buffer.
   *
   * @return Pointer to the device buffer.
   */
  void *device_pointer() const { return device_pointer_; }

  /**
   * @brief Get the size of the underlying device buffer.
   *
   * @return The size of the device buffer.
   */
  std::size_t size() const { return size_; }

private:
  /// Pointer to the device buffer.
  void *device_pointer_;
  /// Size of the device buffer.
  std::size_t size_;
  /// Whether to register the buffer with cuFile.
  bool register_buffer_;
};

/** @brief RAII wrapper for a file descriptor and the corresponding cuFile handle. */
class cufile_file {
public:
  /**
   * @brief Construct a file wrapper.
   *
   * Should not be called directly; use the following factory methods instead.
   *
   * @param file_descriptor A valid file descriptor.
   */
  explicit cufile_file(int file_descriptor) : file_descriptor_{file_descriptor} {
    CUfileDescr_t cufile_descriptor{CU_FILE_HANDLE_TYPE_OPAQUE_FD, file_descriptor_};
    auto const status = cuFileHandleRegister(&cufile_handle_, &cufile_descriptor);
    if (status.err != CU_FILE_SUCCESS) {
      close(file_descriptor_);
      CUDF_FAIL("Failed to register cuFile handle: " + cuFileGetErrorString(status));
    }
  }

  /**
   * @brief Read a file into a device buffer.
   *
   * @param path Absolute path of the file to read from.
   * @param buffer Device buffer to read the file content into.
   * @param file_offset Starting offset from which to read the file.
   */
  static void read(char const *path, cufile_buffer const &buffer, std::size_t file_offset) {
    auto const file_descriptor = open(path, O_RDONLY | O_DIRECT);
    if (file_descriptor < 0) {
      CUDF_FAIL("Failed to open file " + std::string(path) + ": " + cuFileGetErrorString(errno));
    }

    cufile_file file{file_descriptor};
    auto const status =
        cuFileRead(file.cufile_handle_, buffer.device_pointer(), buffer.size(), file_offset, 0);

    if (status < 0) {
      if (IS_CUFILE_ERR(status)) {
        CUDF_FAIL("Failed to read file " + std::string(path) +
                  " into buffer: " + cuFileGetErrorString(status));
      } else {
        CUDF_FAIL("Failed to read file " + std::string(path) +
                  " into buffer: " + cuFileGetErrorString(errno));
      }
    }

    CUDF_EXPECTS(status == buffer.size(), "Size of bytes read is different from buffer size");
  }

  /**
   * @brief Write a device buffer to a file.
   *
   * @param path Absolute path of the file to write to.
   * @param buffer The device buffer to write.
   * @param file_offset Starting offset from which to write the file.
   */
  static void write(char const *path, cufile_buffer const &buffer, std::size_t file_offset) {
    do_write(path, buffer, file_offset);
  }

  /**
   * @brief Append a device buffer to a file.
   *
   * @param path Absolute path of the file to append to.
   * @param buffer The device buffer to append.
   * @return The file offset from which the buffer was appended.
   */
  static std::size_t append(char const *path, cufile_buffer const &buffer) {
    return do_write(path, buffer);
  }

  // Disable copy (and move) semantics.
  cufile_file(cufile_file const &) = delete;
  cufile_file &operator=(cufile_file const &) = delete;

  /** @brief Destroy the file wrapper by de-registering the cuFile handle and closing the file. */
  ~cufile_file() {
    cuFileHandleDeregister(cufile_handle_);
    close(file_descriptor_);
  }

private:
  /**
   * @brief Write a device buffer to a file.
   *
   * @param path Absolute path of the file to write to.
   * @param buffer The device buffer to write.
   * @param file_offset Starting offset from which to write the file. If set to max std::size_t,
   * append to the file.
   * @return The file offset from which the buffer was appended.
   */
  static std::size_t do_write(char const *path, cufile_buffer const &buffer,
                              std::size_t file_offset = std::numeric_limits<std::size_t>::max()) {
    auto file_descriptor = open(path, O_CREAT | O_WRONLY | O_DIRECT, S_IRUSR | S_IWUSR);
    if (file_descriptor < 0) {
      CUDF_FAIL("Failed to open file " + std::string(path) + ": " + cuFileGetErrorString(errno));
    }

    while (true) {
      cufile_file file{file_descriptor};

      if (file_offset == std::numeric_limits<std::size_t>::max()) {
        struct stat stat_buffer;
        auto const stat_status = fstat(file_descriptor, &stat_buffer);
        if (stat_status < 0) {
          CUDF_FAIL("Failed to get file status for " + std::string(path) + ": " +
                    cuFileGetErrorString(errno));
        }
        file_offset = static_cast<std::size_t>(stat_buffer.st_size);
      }

      auto const status =
          cuFileWrite(file.cufile_handle_, buffer.device_pointer(), buffer.size(), file_offset, 0);

      if (status < 0) {
        if (errno == EEXIST) {
          file_descriptor = open(path, O_WRONLY | O_DIRECT);
          if (file_descriptor < 0) {
            CUDF_FAIL("Failed to open file " + std::string(path) + ": " +
                      cuFileGetErrorString(errno));
          }
          file_offset = std::numeric_limits<std::size_t>::max();
          continue;
        }
        if (IS_CUFILE_ERR(status)) {
          CUDF_FAIL("Failed to write buffer to file " + std::string(path) + ": " +
                    cuFileGetErrorString(status));
        } else {
          CUDF_FAIL("Failed to write buffer to file " + std::string(path) + ": " +
                    cuFileGetErrorString(errno));
        }
      }

      CUDF_EXPECTS(status == buffer.size(), "Size of bytes written is different from buffer size");

      return file_offset;
    }
  }

  /// The underlying file descriptor.
  int file_descriptor_;
  /// The registered cuFile handle.
  CUfileHandle_t cufile_handle_{};
};

} // anonymous namespace

extern "C" {

/**
 * @brief Create a new cuFile driver wrapper.
 *
 * @param env The JNI environment.
 * @return Pointer address to the new driver wrapper instance.
 */
JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_CuFile_createDriver(JNIEnv *env, jclass) {
  try {
    return reinterpret_cast<jlong>(new cufile_driver());
  }
  CATCH_STD(env, 0);
}

/**
 * @brief Destroy the given cuFile driver wrapper.
 *
 * @param env The JNI environment.
 * @param pointer Pointer address to the driver wrapper instance.
 */
JNIEXPORT void JNICALL Java_ai_rapids_cudf_CuFile_destroyDriver(JNIEnv *env, jclass,
                                                                jlong pointer) {
  try {
    if (pointer != 0) {
      auto *driver = reinterpret_cast<cufile_driver *>(pointer);
      delete driver;
    }
  }
  CATCH_STD(env, );
}

/**
 * @brief Write a device buffer into a given file path.
 *
 * @param env The JNI environment.
 * @param path Absolute path of the file to copy the buffer to.
 * @param file_offset The file offset from which the buffer was written.
 * @param device_pointer Pointer address to the device buffer.
 * @param size The size of the device buffer.
 */
JNIEXPORT void JNICALL Java_ai_rapids_cudf_CuFile_writeToFile(JNIEnv *env, jclass, jstring path,
                                                              jlong file_offset,
                                                              jlong device_pointer, jlong size) {
  try {
    cufile_buffer buffer{reinterpret_cast<void *>(device_pointer), static_cast<std::size_t>(size)};
    cufile_file::write(env->GetStringUTFChars(path, nullptr), buffer, file_offset);
  }
  CATCH_STD(env, );
}

/**
 * @brief Append a device buffer into a given file path.
 *
 * @param env The JNI environment.
 * @param path Absolute path of the file to copy the buffer to.
 * @param device_pointer Pointer address to the device buffer.
 * @param size The size of the device buffer.
 */
JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_CuFile_appendToFile(JNIEnv *env, jclass, jstring path,
                                                                jlong device_pointer, jlong size) {
  try {
    cufile_buffer buffer{reinterpret_cast<void *>(device_pointer), static_cast<std::size_t>(size)};
    return cufile_file::append(env->GetStringUTFChars(path, nullptr), buffer);
  }
  CATCH_STD(env, -1);
}

/**
 * @brief Read from a given file path into a device buffer.
 *
 * @param env The JNI environment.
 * @param device_pointer Pointer address to the device buffer.
 * @param size The size of the device buffer.
 * @param path Absolute path of the file to copy from.
 * @param file_offset The file offset from which to copy content.
 */
JNIEXPORT void JNICALL Java_ai_rapids_cudf_CuFile_readFromFile(JNIEnv *env, jclass,
                                                               jlong device_pointer, jlong size,
                                                               jstring path, jlong file_offset) {
  try {
    cufile_buffer buffer{reinterpret_cast<void *>(device_pointer), static_cast<std::size_t>(size)};
    cufile_file::read(env->GetStringUTFChars(path, nullptr), buffer, file_offset);
  }
  CATCH_STD(env, );
}

} // extern "C"
