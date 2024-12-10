/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "file_io_utilities.hpp"

#include "getenv_or.hpp"

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/config_utils.hpp>
#include <cudf/logger.hpp>

#include <dlfcn.h>
#include <sys/stat.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>

namespace cudf {
namespace io {
namespace detail {
namespace {

[[nodiscard]] int open_file_checked(std::string const& filepath, int flags, mode_t mode)
{
  auto const fd = open(filepath.c_str(), flags, mode);
  if (fd == -1) { throw_on_file_open_failure(filepath, flags & O_CREAT); }

  return fd;
}

[[nodiscard]] size_t get_file_size(int file_descriptor)
{
  struct stat st {};
  CUDF_EXPECTS(fstat(file_descriptor, &st) != -1, "Cannot query file size");
  return static_cast<size_t>(st.st_size);
}

}  // namespace

void force_init_cuda_context()
{
  // Workaround for https://github.com/rapidsai/cudf/issues/14140, where cuFileDriverOpen errors
  // out if no CUDA calls have been made before it. This is a no-op if the CUDA context is already
  // initialized.
  cudaFree(nullptr);
}

[[noreturn]] void throw_on_file_open_failure(std::string const& filepath, bool is_create)
{
  // save errno because it may be overwritten by subsequent calls
  auto const err = errno;

  if (auto const path = std::filesystem::path(filepath); is_create) {
    CUDF_EXPECTS(std::filesystem::exists(path.parent_path()),
                 "Cannot create output file; directory does not exist");

  } else {
    CUDF_EXPECTS(std::filesystem::exists(path), "Cannot open file; it does not exist");
  }

  std::array<char, 1024> error_msg_buffer{};
  auto const error_msg = strerror_r(err, error_msg_buffer.data(), 1024);
  CUDF_FAIL("Cannot open file; failed with errno: " + std::string{error_msg});
}

file_wrapper::file_wrapper(std::string const& filepath, int flags, mode_t mode)
  : fd(open_file_checked(filepath.c_str(), flags, mode)), _size{get_file_size(fd)}
{
}

file_wrapper::~file_wrapper() { close(fd); }

#ifdef CUDF_CUFILE_FOUND

/**
 * @brief Class that dynamically loads the cuFile library and manages the cuFile driver.
 */
class cufile_shim {
 private:
  cufile_shim();
  void modify_cufile_json() const;
  void load_cufile_lib();

  void* cf_lib                              = nullptr;
  decltype(cuFileDriverOpen)* driver_open   = nullptr;
  decltype(cuFileDriverClose)* driver_close = nullptr;

  std::unique_ptr<cudf::logic_error> init_error;
  [[nodiscard]] auto is_valid() const noexcept { return init_error == nullptr; }

 public:
  cufile_shim(cufile_shim const&)            = delete;
  cufile_shim& operator=(cufile_shim const&) = delete;

  static cufile_shim const* instance();

  ~cufile_shim()
  {
    // Explicit cuFile driver close should not be performed here to avoid segfault. However, in the
    // absence of driver_close(), cuFile will implicitly do that, which in most cases causes
    // segfault anyway. TODO: Revisit this conundrum once cuFile is fixed.
    // https://github.com/rapidsai/cudf/issues/17121

    if (cf_lib != nullptr) dlclose(cf_lib);
  }

  decltype(cuFileHandleRegister)* handle_register     = nullptr;
  decltype(cuFileHandleDeregister)* handle_deregister = nullptr;
  decltype(cuFileRead)* read                          = nullptr;
  decltype(cuFileWrite)* write                        = nullptr;
};

void cufile_shim::modify_cufile_json() const
{
  std::string const json_path_env_var = "CUFILE_ENV_PATH_JSON";
  static temp_directory const tmp_config_dir{"cudf_cufile_config"};

  // Modify the config file based on the policy
  auto const config_file_path = getenv_or<std::string>(json_path_env_var, "/etc/cufile.json");
  std::ifstream user_config_file(config_file_path);
  // Modified config file is stored in a temporary directory
  auto const cudf_config_path = tmp_config_dir.path() + "cufile.json";
  std::ofstream cudf_config_file(cudf_config_path);

  std::string line;
  while (std::getline(user_config_file, line)) {
    std::string const tag = "\"allow_compat_mode\"";
    if (line.find(tag) != std::string::npos) {
      // TODO: only replace the true/false value instead of replacing the whole line
      // Enable compatibility mode when cuDF does not fall back to host path
      cudf_config_file << tag << ": "
                       << (cufile_integration::is_always_enabled() ? "true" : "false") << ",\n";
    } else {
      cudf_config_file << line << '\n';
    }

    // Point libcufile to the modified config file
    CUDF_EXPECTS(setenv(json_path_env_var.c_str(), cudf_config_path.c_str(), 0) == 0,
                 "Failed to set the cuFile config file environment variable.");
  }
}

void cufile_shim::load_cufile_lib()
{
  for (auto&& name : {"libcufile.so.0",
                      // Prior to CUDA 11.7.1, although ABI
                      // compatibility was maintained, some (at least
                      // Debian) packages do not have the .0 symlink,
                      // instead request the exact version.
                      "libcufile.so.1.3.0" /* 11.7.0 */,
                      "libcufile.so.1.2.1" /* 11.6.2, 11.6.1 */,
                      "libcufile.so.1.2.0" /* 11.6.0 */,
                      "libcufile.so.1.1.1" /* 11.5.1 */,
                      "libcufile.so.1.1.0" /* 11.5.0 */,
                      "libcufile.so.1.0.2" /* 11.4.4, 11.4.3, 11.4.2 */,
                      "libcufile.so.1.0.1" /* 11.4.1 */,
                      "libcufile.so.1.0.0" /* 11.4.0 */}) {
    cf_lib = dlopen(name, RTLD_LAZY | RTLD_LOCAL | RTLD_NODELETE);
    if (cf_lib != nullptr) break;
  }
  CUDF_EXPECTS(cf_lib != nullptr, "Failed to load cuFile library");
  driver_open = reinterpret_cast<decltype(driver_open)>(dlsym(cf_lib, "cuFileDriverOpen"));
  CUDF_EXPECTS(driver_open != nullptr, "could not find cuFile cuFileDriverOpen symbol");
  driver_close = reinterpret_cast<decltype(driver_close)>(dlsym(cf_lib, "cuFileDriverClose"));
  CUDF_EXPECTS(driver_close != nullptr, "could not find cuFile cuFileDriverClose symbol");
  handle_register =
    reinterpret_cast<decltype(handle_register)>(dlsym(cf_lib, "cuFileHandleRegister"));
  CUDF_EXPECTS(handle_register != nullptr, "could not find cuFile cuFileHandleRegister symbol");
  handle_deregister =
    reinterpret_cast<decltype(handle_deregister)>(dlsym(cf_lib, "cuFileHandleDeregister"));
  CUDF_EXPECTS(handle_deregister != nullptr, "could not find cuFile cuFileHandleDeregister symbol");
  read = reinterpret_cast<decltype(read)>(dlsym(cf_lib, "cuFileRead"));
  CUDF_EXPECTS(read != nullptr, "could not find cuFile cuFileRead symbol");
  write = reinterpret_cast<decltype(write)>(dlsym(cf_lib, "cuFileWrite"));
  CUDF_EXPECTS(write != nullptr, "could not find cuFile cuFileWrite symbol");
}

cufile_shim::cufile_shim()
{
  try {
    modify_cufile_json();
    load_cufile_lib();

    CUDF_EXPECTS(driver_open().err == CU_FILE_SUCCESS, "Failed to initialize cuFile driver");
  } catch (cudf::logic_error const& err) {
    init_error = std::make_unique<cudf::logic_error>(err);
  }
}

cufile_shim const* cufile_shim::instance()
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

cufile_input_impl::cufile_input_impl(std::string const& filepath)
  : shim{cufile_shim::instance()},
    cf_file(shim, filepath, O_RDONLY | O_DIRECT),
    // The benefit from multithreaded read plateaus around 16 threads
    pool(getenv_or("LIBCUDF_CUFILE_THREAD_COUNT", 16))
{
}

namespace {

template <typename DataT,
          typename F,
          typename ResultT = std::invoke_result_t<F, DataT*, size_t, size_t>>
std::vector<std::future<ResultT>> make_sliced_tasks(
  F function, DataT* ptr, size_t offset, size_t size, BS::thread_pool& pool)
{
  constexpr size_t default_max_slice_size = 4 * 1024 * 1024;
  static auto const max_slice_size = getenv_or("LIBCUDF_CUFILE_SLICE_SIZE", default_max_slice_size);
  auto const slices                = make_file_io_slices(size, max_slice_size);
  std::vector<std::future<ResultT>> slice_tasks;
  std::transform(slices.cbegin(), slices.cend(), std::back_inserter(slice_tasks), [&](auto& slice) {
    return pool.submit_task(
      [=] { return function(ptr + slice.offset, slice.size, offset + slice.offset); });
  });
  return slice_tasks;
}

}  // namespace

std::future<size_t> cufile_input_impl::read_async(size_t offset,
                                                  size_t size,
                                                  uint8_t* dst,
                                                  rmm::cuda_stream_view stream)
{
  int device = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&device));

  auto read_slice = [device, gds_read = shim->read, file_handle = cf_file.handle()](
                      void* dst, size_t size, size_t offset) -> ssize_t {
    CUDF_CUDA_TRY(cudaSetDevice(device));
    auto read_size = gds_read(file_handle, dst, size, offset, 0);
    CUDF_EXPECTS(read_size != -1, "cuFile error reading from a file");
    return read_size;
  };

  auto slice_tasks = make_sliced_tasks(read_slice, dst, offset, size, pool);

  auto waiter = [](auto slice_tasks) -> size_t {
    return std::accumulate(slice_tasks.begin(), slice_tasks.end(), 0, [](auto sum, auto& task) {
      return sum + task.get();
    });
  };
  // The future returned from this function is deferred, not async because we want to avoid creating
  // threads for each read_async call. This overhead is significant in case of multiple small reads.
  return std::async(std::launch::deferred, waiter, std::move(slice_tasks));
}

cufile_output_impl::cufile_output_impl(std::string const& filepath)
  : shim{cufile_shim::instance()},
    cf_file(shim, filepath, O_CREAT | O_RDWR | O_DIRECT, 0664),
    pool(getenv_or("LIBCUDF_CUFILE_THREAD_COUNT", 16))
{
}

std::future<void> cufile_output_impl::write_async(void const* data, size_t offset, size_t size)
{
  int device = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&device));

  auto write_slice = [device, gds_write = shim->write, file_handle = cf_file.handle()](
                       void const* src, size_t size, size_t offset) -> void {
    CUDF_CUDA_TRY(cudaSetDevice(device));
    auto write_size = gds_write(file_handle, src, size, offset, 0);
    CUDF_EXPECTS(write_size != -1 and write_size == static_cast<decltype(write_size)>(size),
                 "cuFile error writing to a file");
  };

  auto source      = static_cast<uint8_t const*>(data);
  auto slice_tasks = make_sliced_tasks(write_slice, source, offset, size, pool);

  auto waiter = [](auto slice_tasks) -> void {
    for (auto const& task : slice_tasks) {
      task.wait();
    }
  };
  // The future returned from this function is deferred, not async because we want to avoid creating
  // threads for each write_async call. This overhead is significant in case of multiple small
  // writes.
  return std::async(std::launch::deferred, waiter, std::move(slice_tasks));
}
#else
cufile_input_impl::cufile_input_impl(std::string const& filepath)
{
  CUDF_FAIL("Cannot create cuFile source, current build was compiled without cuFile headers");
}

cufile_output_impl::cufile_output_impl(std::string const& filepath)
{
  CUDF_FAIL("Cannot create cuFile sink, current build was compiled without cuFile headers");
}
#endif

std::unique_ptr<cufile_input_impl> make_cufile_input(std::string const& filepath)
{
  if (cufile_integration::is_gds_enabled()) {
    try {
      auto cufile_in = std::make_unique<cufile_input_impl>(filepath);
      CUDF_LOG_INFO("File successfully opened for reading with GDS.");
      return cufile_in;
    } catch (...) {
      if (cufile_integration::is_always_enabled()) {
        CUDF_LOG_ERROR(
          "Failed to open file for reading with GDS. Enable bounce buffer fallback to read this "
          "file.");
        throw;
      }
      CUDF_LOG_INFO(
        "Failed to open file for reading with GDS. Data will be read from the file using a bounce "
        "buffer (possible performance impact).");
    }
  }
  return {};
}

std::unique_ptr<cufile_output_impl> make_cufile_output(std::string const& filepath)
{
  if (cufile_integration::is_gds_enabled()) {
    try {
      auto cufile_out = std::make_unique<cufile_output_impl>(filepath);
      CUDF_LOG_INFO("File successfully opened for writing with GDS.");
      return cufile_out;
    } catch (...) {
      if (cufile_integration::is_always_enabled()) {
        CUDF_LOG_ERROR(
          "Failed to open file for writing with GDS. Enable bounce buffer fallback to write to "
          "this file.");
        throw;
      }
      CUDF_LOG_INFO(
        "Failed to open file for writing with GDS. Data will be written to the file using a bounce "
        "buffer (possible performance impact).");
    }
  }
  return {};
}

std::vector<file_io_slice> make_file_io_slices(size_t size, size_t max_slice_size)
{
  max_slice_size      = std::max(1024ul, max_slice_size);
  auto const n_slices = util::div_rounding_up_safe(size, max_slice_size);
  std::vector<file_io_slice> slices;
  slices.reserve(n_slices);
  std::generate_n(std::back_inserter(slices), n_slices, [&, idx = 0]() mutable {
    auto const slice_offset = idx++ * max_slice_size;
    auto const slice_size   = std::min(size - slice_offset, max_slice_size);
    return file_io_slice{slice_offset, slice_size};
  });

  return slices;
}

}  // namespace detail
}  // namespace io
}  // namespace cudf
