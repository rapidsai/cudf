
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <format>

namespace cudf {
namespace rtc {

namespace {

[[noreturn]] void throw_posix(std::string_view message, std::string_view syscall_name)
{
  auto error_code = errno;
  auto error_str  = std::format(
    "{}. `{}` failed with {} ({})", message, syscall_name, error_code, std::strerror(error_code));
  CUDF_FAIL(+error_str, std::runtime_error);
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
  // Create cache directory if it doesn't exist
  if (mkdir(cache_dir_.c_str(), S_IRWXU | S_IRWXG | S_IRWXO) == -1) {
    if (errno != EEXIST) { throw_posix("Failed to create RTC cache directory", "mkdir"); }
  }
}

bool cache_t::is_enabled() const { return enabled_; }

void cache_t::enable() { enabled_ = true; }

void cache_t::disable() { enabled_ = false; }

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
      if (remove(temp_path) == -1) {
        throw_posix("Failed to remove temporary RTC cache file", "remove");
      }
    });

    if (write(fd, binary.data(), binary.size()) == -1) {
      throw_posix("Failed to write RTC cache to temporary file", "write");
    }
  }

  auto hex = sha.to_hex_string();
  char final_path[PATH_MAX + 1];
  auto result = std::format_to_n(final_path, PATH_MAX, "{}/{}.blob", cache_dir_, hex.view());
  CUDF_EXPECTS(
    result.out != (final_path + PATH_MAX), "Path length exceeded PATH_MAX", std::runtime_error);
  *result.out = '\0';

  // rename is atomic, even if another process is performing the same operation
  if (rename(temp_path, final_path) == -1) {
    auto error_code = errno;

    if (error_code == EEXIST) {
      // another process has already created the file, so just remove our temp file
      if (remove(temp_path) == -1) {
        throw_posix("Failed to remove temporary RTC cache file", "remove");
      }
      return;
    }

    throw_posix("Failed to move temporary RTC cache file to final location", "rename");
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

cache_statistics cache_t::get_statistics() const { return counter_.get_statistics(); }

void cache_t::clear_statistics() { counter_.clear(); }

cache_limits cache_t::get_limits() const { return limits_; }

size_t cache_t::get_blob_count() const
{
  CUDF_FUNC_RANGE();

  {
    detail::read_guard guard{blobs_cache_.lock_};
    return blobs_cache_.entries_.size();
  }
}

size_t cache_t::get_fragment_count() const
{
  CUDF_FUNC_RANGE();

  {
    detail::read_guard guard{fragments_cache_.lock_};
    return fragments_cache_.entries_.size();
  }
}

size_t cache_t::get_library_count() const
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

  struct dirent* entry_iter;

  while (true) {
    entry_iter = readdir(dir);

    if (entry_iter == nullptr) {
      if (errno != 0) {
        throw_posix("Failed to read RTC cache directory for clearing", "readdir");
      } else {
        break;
      }
    }

    struct stat st;
    char path[PATH_MAX + 1];

    if (lstat(path, &st) == -1) {
      throw_posix("Failed to get file status for RTC cache clearing", "lstat");
    }

    if (S_ISREG(st.st_mode)) {
      if (unlink(path) == -1) {
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
