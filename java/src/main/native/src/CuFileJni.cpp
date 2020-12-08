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

//
// cuda driver error description
//
char const *GetCuErrorString(CUresult cu_result) {
  const char *description;
  if (cuGetErrorName(cu_result, &description) != CUDA_SUCCESS)
    description = "unknown cuda error";
  return description;
}

//
// cuFile APIs return both cuFile specific error codes as well as POSIX error codes
// for ease, the below template can be used for getting the error description depending
// on its type.

// POSIX
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, std::nullptr_t>::type = nullptr>
std::string cuFileGetErrorString(T status) {
  status = std::abs(status);
  return IS_CUFILE_ERR(status) ? std::string(CUFILE_ERRSTR(status)) :
                                 std::string(std::strerror(status));
}

// CUfileError_t
template <typename T,
          typename std::enable_if<!std::is_integral<T>::value, std::nullptr_t>::type = nullptr>
std::string cuFileGetErrorString(T status) {
  std::string error = cuFileGetErrorString(static_cast<int>(status.err));
  if (IS_CUDA_ERR(status)) {
    error.append(".").append(GetCuErrorString(status.cu_err));
  }
  return error;
}

class cufile_driver {
public:
  cufile_driver() {
    auto const status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
      CUDF_FAIL("Failed to initialize cuFile driver: " + cuFileGetErrorString(status));
    }
  }

  // Disable copy (and move) semantics.
  cufile_driver(cufile_driver const &) = delete;
  cufile_driver &operator=(cufile_driver const &) = delete;

  ~cufile_driver() { cuFileDriverClose(); }
};

class cufile_buffer {
public:
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

  ~cufile_buffer() {
    if (register_buffer_) {
      cuFileBufDeregister(device_pointer_);
    }
  }

  void *device_pointer() const { return device_pointer_; }
  std::size_t size() const { return size_; }

private:
  void *device_pointer_;
  std::size_t size_;
  bool register_buffer_;
};

class cufile_file {
public:
  cufile_file(char const *path) {
    file_descriptor_ = open(path, O_CREAT | O_RDWR | O_DIRECT, 0644);
    if (file_descriptor_ < 0) {
      CUDF_FAIL("Failed to open file: " + cuFileGetErrorString(errno));
    }
    CUfileDescr_t cufile_descriptor{CU_FILE_HANDLE_TYPE_OPAQUE_FD, file_descriptor_};
    auto const status = cuFileHandleRegister(&cufile_handle_, &cufile_descriptor);
    if (status.err != CU_FILE_SUCCESS) {
      close(file_descriptor_);
      CUDF_FAIL("Failed to register cuFile handle: " + cuFileGetErrorString(status));
    }
  }

  // Disable copy (and move) semantics.
  cufile_file(cufile_file const &) = delete;
  cufile_file &operator=(cufile_file const &) = delete;

  ~cufile_file() {
    cuFileHandleDeregister(cufile_handle_);
    close(file_descriptor_);
  }

  void read(cufile_buffer const &buffer, std::size_t file_offset) {
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

  void write(cufile_buffer const &buffer, std::size_t file_offset = 0) {
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

  std::size_t append(cufile_buffer const &buffer) {
    auto const status = lseek(file_descriptor_, 0, SEEK_END);
    if (status < 0) {
      CUDF_FAIL("Failed to seek end of file: " + cuFileGetErrorString(errno));
    }

    write(buffer, status);
    return status;
  }

private:
  int file_descriptor_;
  CUfileHandle_t cufile_handle_;
};

} // anonymous namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_CuFile_createDriver(JNIEnv *env, jclass) {
  try {
    return reinterpret_cast<jlong>(new cufile_driver());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_CuFile_destroyDriver(JNIEnv *env, jclass,
                                                                jlong pointer) {
  try {
    if (pointer != 0) {
      cufile_driver *driver = reinterpret_cast<cufile_driver *>(pointer);
      delete driver;
    }
  }
  CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_CuFile_copyToFile(JNIEnv *env, jclass, jstring path,
                                                              jlong device_pointer, jlong size,
                                                              jboolean append) {
  try {
    cufile_buffer buffer{reinterpret_cast<void *>(device_pointer), static_cast<std::size_t>(size)};
    cufile_file file{(env->GetStringUTFChars(path, nullptr))};
    if (append) {
      return file.append(buffer);
    } else {
      file.write(buffer);
      return 0;
    }
  }
  CATCH_STD(env, -1);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_CuFile_copyFromFile(JNIEnv *env, jclass,
                                                               jlong device_pointer, jlong size,
                                                               jstring path, jlong file_offset) {
  try {
    cufile_buffer buffer{reinterpret_cast<void *>(device_pointer), static_cast<std::size_t>(size)};
    cufile_file file{(env->GetStringUTFChars(path, nullptr))};
    file.read(buffer, file_offset);
  }
  CATCH_STD(env, );
}

} // extern "C"
