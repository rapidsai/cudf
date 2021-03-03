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
#include <io/utilities/file_utils.hpp>

#include <dlfcn.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <rmm/device_buffer.hpp>

namespace cudf {
namespace io {

file_wrapper::file_wrapper(std::string const &filepath, int flags)
  : fd(open(filepath.c_str(), flags))
{
  CUDF_EXPECTS(fd != -1, "Cannot open file " + filepath);
}

file_wrapper::file_wrapper(std::string const &filepath, int flags, mode_t mode)
  : fd(open(filepath.c_str(), flags, mode))
{
  CUDF_EXPECTS(fd != -1, "Cannot open file " + filepath);
}

/**
 * Returns the directory from which the libcudf.so is loaded.
 */
std::string get_libcudf_dir_path()
{
  Dl_info dl_info{};
  dladdr((void *)get_libcudf_dir_path, &dl_info);
  std::string full_path{dl_info.dli_fname};
  auto const dir_path = full_path.substr(0, full_path.find_last_of('/') + 1);
  return dir_path;
}

file_wrapper::~file_wrapper() { close(fd); }

long file_wrapper::size() const
{
  if (_size < 0) {
    struct stat st;
    CUDF_EXPECTS(fstat(fd, &st) != -1, "Cannot query file size");
    _size = static_cast<size_t>(st.st_size);
  }
  return _size;
}

#ifdef CUFILE_FOUND

class cufile_config {
  bool enabled = false;

  cufile_config()
  {
    auto const policy = std::getenv("LIBCUDF_CUFILE_POLICY");
    if (policy == nullptr) {
      enabled = false;
    } else {
      auto const policy_string = std::string(policy);
      enabled                  = (policy_string == "ALWAYS" || policy_string == "GDS");
    }
  }

 public:
  bool is_enabled() const { return enabled; }

  static cufile_config const *instance()
  {
    static cufile_config _instance;
    return &_instance;
  }
};

/**
 * @brief Class that dynamically loads the cuFile library and manages the cuFile driver.
 */
class cufile_shim {
 private:
  cufile_shim();

  std::unique_ptr<cudf::logic_error> init_error;
  auto is_valid() const noexcept { return init_error == nullptr; }

 public:
  cufile_shim(cufile_shim const &) = delete;
  cufile_shim &operator=(cufile_shim const &) = delete;

  static cufile_shim const *instance();

  ~cufile_shim()
  {
    driver_close();
    dlclose(cf_lib);
  }

  void *cf_lib                                        = nullptr;
  decltype(cuFileDriverOpen) *driver_open             = nullptr;
  decltype(cuFileDriverClose) *driver_close           = nullptr;
  decltype(cuFileHandleRegister) *handle_register     = nullptr;
  decltype(cuFileHandleDeregister) *handle_deregister = nullptr;
  decltype(cuFileRead) *read                          = nullptr;
  decltype(cuFileWrite) *write                        = nullptr;
};

cufile_shim::cufile_shim()
{
  try {
    cf_lib      = dlopen("libcufile.so", RTLD_NOW);
    driver_open = reinterpret_cast<decltype(driver_open)>(dlsym(cf_lib, "cuFileDriverOpen"));
    CUDF_EXPECTS(driver_open != nullptr, "could not find cuFile cuFileDriverOpen symbol");
    driver_close = reinterpret_cast<decltype(driver_close)>(dlsym(cf_lib, "cuFileDriverClose"));
    CUDF_EXPECTS(driver_close != nullptr, "could not find cuFile cuFileDriverClose symbol");
    handle_register =
      reinterpret_cast<decltype(handle_register)>(dlsym(cf_lib, "cuFileHandleRegister"));
    CUDF_EXPECTS(handle_register != nullptr, "could not find cuFile cuFileHandleRegister symbol");
    handle_deregister =
      reinterpret_cast<decltype(handle_deregister)>(dlsym(cf_lib, "cuFileHandleDeregister"));
    CUDF_EXPECTS(handle_deregister != nullptr,
                 "could not find cuFile cuFileHandleDeregister symbol");
    read = reinterpret_cast<decltype(read)>(dlsym(cf_lib, "cuFileRead"));
    CUDF_EXPECTS(read != nullptr, "could not find cuFile cuFileRead symbol");
    write = reinterpret_cast<decltype(write)>(dlsym(cf_lib, "cuFileWrite"));
    CUDF_EXPECTS(write != nullptr, "could not find cuFile cuFileWrite symbol");

    CUDF_EXPECTS(driver_open().err == CU_FILE_SUCCESS, "Failed to initialize cuFile driver");
  } catch (cudf::logic_error const &err) {
    init_error = std::make_unique<cudf::logic_error>(err);
  }
}

cufile_shim const *cufile_shim::instance()
{
  static cufile_shim _instance;
  // Defer throwing to avoid repeated attempts to load the library
  if (!_instance.is_valid()) CUDF_FAIL("" + std::string(_instance.init_error->what()));

  return &_instance;
}

void cufile_registered_file::register_handle()
{
  CUfileDescr_t cufile_desc{};
  cufile_desc.handle.fd = _file.desc();
  cufile_desc.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  CUDF_EXPECTS(shim->handle_register(&cf_handle, &cufile_desc).err == CU_FILE_SUCCESS,
               "Cannot register file handle with cuFile");
}

cufile_registered_file::~cufile_registered_file() { shim->handle_deregister(cf_handle); }

cufile_input_impl::cufile_input_impl(std::string const &filepath)
  : shim{cufile_shim::instance()}, cf_file(shim, filepath, O_RDONLY | O_DIRECT)
{
}

std::unique_ptr<datasource::buffer> cufile_input_impl::read(size_t offset,
                                                            size_t size,
                                                            rmm::cuda_stream_view stream)
{
  rmm::device_buffer out_data(size, stream);
  CUDF_EXPECTS(shim->read(cf_file.handle(), out_data.data(), size, offset, 0) != -1,
               "cuFile error reading from a file");

  return datasource::buffer::create(std::move(out_data));
}

size_t cufile_input_impl::read(size_t offset,
                               size_t size,
                               uint8_t *dst,
                               rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(shim->read(cf_file.handle(), dst, size, offset, 0) != -1,
               "cuFile error reading from a file");
  // have to read the requested size for now
  return size;
}

cufile_output_impl::cufile_output_impl(std::string const &filepath)
  : shim{cufile_shim::instance()}, cf_file(shim, filepath, O_CREAT | O_RDWR | O_DIRECT, 0664)
{
}

void cufile_output_impl::write(void const *data, size_t offset, size_t size)
{
  CUDF_EXPECTS(shim->write(cf_file.handle(), data, size, offset, 0) != -1,
               "cuFile error writing to a file");
}
#endif

std::unique_ptr<cufile_input_impl> make_cufile_input(std::string const &filepath)
{
#ifdef CUFILE_FOUND
  if (cufile_config::instance()->is_enabled()) {
    return std::make_unique<cufile_input_impl>(filepath);
  }
#endif
  return nullptr;
}

std::unique_ptr<cufile_output_impl> make_cufile_output(std::string const &filepath)
{
#ifdef CUFILE_FOUND
  if (cufile_config::instance()->is_enabled()) {
    return std::make_unique<cufile_output_impl>(filepath);
  }
#endif
  return nullptr;
}

};  // namespace io
};  // namespace cudf
