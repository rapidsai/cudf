/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>

#include <cudf/utilities/error.hpp>

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
   * @brief Factory method to create a file wrapper for reading.
   *
   * @param path Absolute path of the file to read from.
   * @return std::unique_ptr<cufile_file> for reading.
   */
  static auto make_reader(char const *path) {
    auto const file_descriptor = open(path, O_RDONLY | O_DIRECT);
    if (file_descriptor < 0) {
      CUDF_FAIL("Failed to open file to read: " + cuFileGetErrorString(errno));
    }
    return std::make_unique<cufile_file>(file_descriptor);
  }

  /**
   * @brief Factory method to create a file wrapper for writing.
   *
   * @param path Absolute path of the file to write to.
   * @return std::unique_ptr<cufile_file> for writing.
   */
  static auto make_writer(char const *path) {
    auto const file_descriptor = open(path, O_CREAT | O_WRONLY | O_DIRECT, S_IRUSR | S_IWUSR);
    if (file_descriptor < 0) {
      CUDF_FAIL("Failed to open file to write: " + cuFileGetErrorString(errno));
    }
    return std::make_unique<cufile_file>(file_descriptor);
  }

  // Disable copy (and move) semantics.
  cufile_file(cufile_file const &) = delete;
  cufile_file &operator=(cufile_file const &) = delete;

  /** @brief Destroy the file wrapper by de-registering the cuFile handle and closing the file. */
  ~cufile_file() {
    cuFileHandleDeregister(cufile_handle_);
    close(file_descriptor_);
  }

  /**
   * @brief Read the file into a device buffer.
   *
   * @param buffer Device buffer to read the file content into.
   * @param file_offset Starting offset from which to read the file.
   */
  void read(cufile_buffer const &buffer, std::size_t file_offset) const {
    auto const status =
        cuFileRead(cufile_handle_, buffer.device_pointer(), buffer.size(), file_offset, 0);

    if (status < 0) {
      if (IS_CUFILE_ERR(status)) {
        CUDF_FAIL("Failed to read file into buffer: " + cuFileGetErrorString(status));
      } else {
        CUDF_FAIL("Failed to read file into buffer: " + cuFileGetErrorString(errno));
      }
    }

    CUDF_EXPECTS(status == buffer.size(), "Size of bytes read is different from buffer size");
  }

  /**
   * @brief Write a device buffer to the file.
   *
   * @param buffer The device buffer to write.
   * @param file_offset Starting offset from which to write the file.
   */
  void write(cufile_buffer const &buffer, std::size_t file_offset) {
    auto const status =
        cuFileWrite(cufile_handle_, buffer.device_pointer(), buffer.size(), file_offset, 0);

    if (status < 0) {
      if (IS_CUFILE_ERR(status)) {
        CUDF_FAIL("Failed to write buffer to file: " + cuFileGetErrorString(status));
      } else {
        CUDF_FAIL("Failed to write buffer to file: " + cuFileGetErrorString(errno));
      }
    }

    CUDF_EXPECTS(status == buffer.size(), "Size of bytes written is different from buffer size");
  }

  /**
   * @brief Append a device buffer to the file.
   *
   * @param buffer The device buffer to write.
   * @return The file offset from which the buffer was appended.
   */
  std::size_t append(cufile_buffer const &buffer) {
    auto const status = lseek(file_descriptor_, 0, SEEK_END);
    if (status < 0) {
      CUDF_FAIL("Failed to seek end of file: " + cuFileGetErrorString(errno));
    }

    auto const file_offset = static_cast<std::size_t>(status);
    write(buffer, file_offset);
    return file_offset;
  }

private:
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
    auto writer = cufile_file::make_writer(env->GetStringUTFChars(path, nullptr));
    writer->write(buffer, file_offset);
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
    auto writer = cufile_file::make_writer(env->GetStringUTFChars(path, nullptr));
    return writer->append(buffer);
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
    auto const reader = cufile_file::make_reader(env->GetStringUTFChars(path, nullptr));
    reader->read(buffer, file_offset);
  }
  CATCH_STD(env, );
}

} // extern "C"
