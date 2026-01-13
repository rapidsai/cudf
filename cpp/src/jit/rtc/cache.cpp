
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/defer.hpp>
#include <cudf/utilities/error.hpp>

#include <dirent.h>
#include <jit/rtc/cache.hpp>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <format>

namespace cudf {
namespace rtc {

namespace {

[[noreturn]] void throw_posix(std::string_view message, std::string_view syscall_name)
{
  auto errc = errno;
  auto err_str =
    std::format("{}. `{}` failed with {} ({})", message, syscall_name, errc, std::strerror(errc));
  CUDF_FAIL(+err_str, std::runtime_error);
}

}  // namespace

cache_t::cache_t(bool enabled, std::string cache_dir, cache_limits const& limits)
  : enabled_{enabled},
    cache_dir_{std::move(cache_dir)},
    limits_{limits},
    blobs_cache_{limits.num_blobs},
    fragments_cache_{limits.num_fragments},
    libraries_cache_{limits.num_libraries},
    tick_{0}
{
  CUDF_EXPECTS(limits.num_blobs >= 2, "Blob cache limit must be at least 2");
  CUDF_EXPECTS(limits.num_fragments >= 2, "Fragment cache limit must be at least 2");
  CUDF_EXPECTS(limits.num_libraries >= 2, "Library cache limit must be at least 2");
  // Create cache directory if it doesn't exist
  if (mkdir(cache_dir_.c_str(), S_IRWXU | S_IRWXG | S_IRWXO) == -1) {
    if (errno != EEXIST) { throw_posix("Failed to create RTC cache directory", "mkdir"); }
  }
}

bool cache_t::is_enabled()
{
  std::atomic_ref c{enabled_};
  return c.load(std::memory_order_relaxed);
}

void cache_t::enable()
{
  std::atomic_ref c{enabled_};
  c.store(true, std::memory_order_relaxed);
}

void cache_t::disable()
{
  std::atomic_ref c{enabled_};
  c.store(false, std::memory_order_relaxed);
}

std::string const& cache_t::get_cache_dir() { return cache_dir_; }

void cache_t::store_blob_to_memory(sha256_hash const& sha, std::shared_future<blob> binary)
{
  CUDF_FUNC_RANGE();

  if (!enabled_) { return; }

  std::atomic_ref tick{tick_};
  auto current_tick = tick.fetch_add(1, std::memory_order_relaxed);

  detail::write_guard guard{blobs_cache_.lock_};

  blobs_cache_.insert(sha, std::move(binary), current_tick);
}

void cache_t::store_blob_to_disk(sha256_hash const& sha, blob_view binary)
{
  CUDF_FUNC_RANGE();

  if (!enabled_) { return; }

  char temp_path[] = "/tmp/blob-XXXXXX";

  {
    int fd = mkstemp(temp_path);
    if (fd == -1) { throw_posix("Failed to create temporary file for RTC cache", "mkstemp"); }

    CUDF_DEFER([&] {
      if (close(fd) == -1) { throw_posix("Failed to close temporary RTC cache file", "close"); }
    });

    if (write(fd, binary.data(), binary.size()) == -1) {
      throw_posix("Failed to write RTC cache to temporary file", "write");
    }
  }

  auto hex        = sha.to_hex_string();
  auto final_path = std::format("{}/{}.blob", cache_dir_, hex.view());

  std::filesystem::create_directories(std::filesystem::path{final_path}.parent_path());

  // rename is atomic, even if another process is performing the same operation
  if (rename(temp_path, final_path.c_str()) == -1) {
    auto errc = errno;

    if (errc == EEXIST) {
      // another process has already created the file, so just remove our temp file
      if (remove(temp_path) == -1) {
        throw_posix("Failed to remove temporary RTC cache file", "remove");
      }
      return;
    } else {
      throw_posix(
        std::format("Failed to move temporary RTC cache file to final location ({})", final_path),
        "rename");
    }
  }
}

std::optional<std::shared_future<blob>> cache_t::query_blob_from_memory(sha256_hash const& sha)
{
  CUDF_FUNC_RANGE();

  if (!enabled_) { return std::nullopt; }

  std::atomic_ref tick{tick_};
  auto current_tick = tick.fetch_add(1, std::memory_order_relaxed);

  {
    detail::read_guard guard{blobs_cache_.lock_};
    auto const it = blobs_cache_.entries_.find(sha);

    if (it != blobs_cache_.entries_.end()) {
      counter_.hit_memory_blob();
      it->second.hit(current_tick);
      return it->second.value;
    } else {
      counter_.miss_memory_blob();
      return std::nullopt;
    }
  }
}

std::optional<blob> cache_t::query_blob_from_disk(sha256_hash const& sha)
{
  CUDF_FUNC_RANGE();

  if (!enabled_) { return std::nullopt; }

  auto hex  = sha.to_hex_string();
  auto path = std::format("{}/{}.blob", cache_dir_, hex.view());

  int fd = open(path.c_str(), O_RDONLY);

  if (fd == -1) {
    if (errno == ENOENT) {
      counter_.miss_disk_blob();
      return std::nullopt;
    } else {
      throw_posix("Failed to open RTC cache file from disk", "open");
    }
  }

  auto file_size = lseek(fd, 0, SEEK_END);
  if (file_size == -1) { throw_posix("Failed to determine size of RTC cache file", "lseek"); }

  void* map = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);

  if (map == MAP_FAILED) { throw_posix("Failed to memory-map RTC cache file", "mmap"); }

  if (close(fd) == -1) {
    throw_posix("Failed to close RTC cache file after memory-mapping", "close");
  }

  auto deleter = +[](void*, uint8_t const* buffer, size_t size) {
    if (munmap(static_cast<void*>(const_cast<uint8_t*>(buffer)), size) == -1) {
      throw_posix("Failed to unmap RTC cache file from memory", "munmap");
    }
  };

  counter_.hit_disk_blob();

  return std::make_shared<blob_t>(
    blob_t::from_parts(static_cast<uint8_t const*>(map), file_size, nullptr, deleter));
}

void cache_t::store_fragment(sha256_hash const& sha, std::shared_future<fragment> frag)
{
  CUDF_FUNC_RANGE();

  if (!enabled_) { return; }

  std::atomic_ref tick{tick_};
  auto current_tick = tick.fetch_add(1, std::memory_order_relaxed);

  {
    detail::write_guard guard{fragments_cache_.lock_};

    fragments_cache_.insert(sha, std::move(frag), current_tick);
  }
}

std::optional<std::shared_future<fragment>> cache_t::query_fragment(sha256_hash const& sha)
{
  CUDF_FUNC_RANGE();

  if (!enabled_) { return std::nullopt; }

  std::atomic_ref tick{tick_};
  auto current_tick = tick.fetch_add(1, std::memory_order_relaxed);

  {
    detail::read_guard guard{fragments_cache_.lock_};

    auto const it = fragments_cache_.entries_.find(sha);
    if (it != fragments_cache_.entries_.end()) {
      counter_.hit_memory_fragment();
      it->second.hit(current_tick);
      return it->second.value;
    } else {
      counter_.miss_memory_fragment();
      return std::nullopt;
    }
  }
}

void cache_t::store_library(sha256_hash const& sha, std::shared_future<library> mod)
{
  CUDF_FUNC_RANGE();

  if (!enabled_) { return; }

  std::atomic_ref tick{tick_};
  auto current_tick = tick.fetch_add(1, std::memory_order_relaxed);

  {
    detail::write_guard guard{libraries_cache_.lock_};

    libraries_cache_.insert(sha, std::move(mod), current_tick);
  }
}

std::optional<std::shared_future<library>> cache_t::query_library(sha256_hash const& sha)
{
  CUDF_FUNC_RANGE();

  if (!enabled_) { return std::nullopt; }

  std::atomic_ref tick{tick_};
  auto current_tick = tick.fetch_add(1, std::memory_order_relaxed);

  {
    detail::read_guard guard{libraries_cache_.lock_};

    auto const it = libraries_cache_.entries_.find(sha);
    if (it != libraries_cache_.entries_.end()) {
      counter_.hit_memory_library();
      it->second.hit(current_tick);
      return it->second.value;
    } else {
      counter_.miss_memory_library();
      return std::nullopt;
    }
  }
}

cache_statistics cache_t::get_statistics() { return counter_.get_statistics(); }

void cache_t::clear_statistics() { counter_.clear(); }

cache_limits cache_t::get_limits() { return limits_; }

size_t cache_t::get_blob_count()
{
  CUDF_FUNC_RANGE();

  {
    detail::read_guard guard{blobs_cache_.lock_};
    return blobs_cache_.entries_.size();
  }
}

size_t cache_t::get_fragment_count()
{
  CUDF_FUNC_RANGE();

  {
    detail::read_guard guard{fragments_cache_.lock_};
    return fragments_cache_.entries_.size();
  }
}

size_t cache_t::get_library_count()
{
  CUDF_FUNC_RANGE();

  {
    detail::read_guard guard{libraries_cache_.lock_};
    return libraries_cache_.entries_.size();
  }
}

void cache_t::clear_memory_store()
{
  CUDF_FUNC_RANGE();

  {
    detail::write_guard guard{blobs_cache_.lock_};
    blobs_cache_.entries_.clear();
  }

  {
    detail::write_guard guard{fragments_cache_.lock_};
    fragments_cache_.entries_.clear();
  }

  {
    detail::write_guard guard{libraries_cache_.lock_};
    libraries_cache_.entries_.clear();
  }
}

void cache_t::clear_disk_store()
{
  CUDF_FUNC_RANGE();

  DIR* dir = opendir(cache_dir_.c_str());

  if (dir == nullptr) { throw_posix("Failed to open RTC cache directory for clearing", "opendir"); }

  CUDF_DEFER([&] { closedir(dir); });

  errno = 0;  // reset errno before reading

  struct dirent* entry_iter = nullptr;
  std::vector<char> entry_path;
  entry_path.resize(PATH_MAX + 1);

  while (true) {
    entry_iter = readdir(dir);

    if (entry_iter == nullptr) {
      if (errno != 0) {
        throw_posix("Failed to read RTC cache directory for clearing", "readdir");
      } else {
        break;
      }
    }

    struct stat entry_stat;

    if (lstat(entry_path.data(), &entry_stat) == -1) {
      throw_posix("Failed to get file status for RTC cache clearing", "lstat");
    }

    if (S_ISREG(entry_stat.st_mode)) {
      if (unlink(entry_path.data()) == -1) {
        throw_posix("Failed to unlink RTC cache file during clearing", "unlink");
      }
    }

    // reset errno for next iteration
    errno = 0;
  }

  return;
}

}  // namespace rtc
}  // namespace cudf
