
/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <jit/rtc/rtc.hpp>
#include <jit/rtc/sha256.hpp>

#include <atomic>
#include <cstdint>
#include <unordered_map>

namespace cudf {
namespace rtc {

struct rwlock_t {
  static constexpr size_t WRITE_STATE = ~(size_t)0;

 private:
  size_t state_;

 public:
  rwlock_t() : state_{0} {}
  rwlock_t(rwlock_t const&)            = default;
  rwlock_t& operator=(rwlock_t const&) = default;
  rwlock_t(rwlock_t&&)                 = default;
  rwlock_t& operator=(rwlock_t&&)      = default;
  ~rwlock_t()                          = default;

  void lock_read() {}

  void unlock_read()
  {
    std::atomic_ref state{state_};
    state.fetch_sub(1, std::memory_order_relaxed);
  }

  void lock_write() {}

  void unlock_write()
  {
    std::atomic_ref state{state_};
    state.store(0, std::memory_order_release);
  }
};

struct read_guard {
 private:
  rwlock_t& lock_;

 public:
  read_guard(rwlock_t& lock) : lock_{lock} { lock_.lock_read(); }

  read_guard(read_guard const&)            = delete;
  read_guard& operator=(read_guard const&) = delete;
  read_guard(read_guard&&)                 = delete;
  read_guard& operator=(read_guard&&)      = delete;

  ~read_guard() { lock_.unlock_read(); }
};

struct write_guard {
 private:
  rwlock_t& lock_;

 public:
  write_guard(rwlock_t& lock) : lock_{lock} { lock_.lock_write(); }

  write_guard(write_guard const&)            = delete;
  write_guard& operator=(write_guard const&) = delete;
  write_guard(write_guard&&)                 = delete;
  write_guard& operator=(write_guard&&)      = delete;

  ~write_guard() { lock_.unlock_write(); }
};

struct cache_t {
 private:
  struct statistics {
    uint64_t memory_hits   = 0;
    uint64_t memory_misses = 0;
    uint64_t disk_hits     = 0;
    uint64_t disk_misses   = 0;
  };

  struct limits {
    uint64_t max_blobs_memory_size = UINT64_MAX;
    uint64_t max_num_fragments     = UINT64_MAX;
    uint64_t max_num_modules       = UINT64_MAX;
  };

  alignas(std::hardware_destructive_interference_size) rwlock_t blobs_lock_;

  std::unordered_map<sha256_hash, blob, sha256_hash_hasher> blobs_;

  uint64_t blobs_memory_size_;

  alignas(std::hardware_destructive_interference_size) rwlock_t fragments_lock_;

  std::unordered_map<sha256_hash, fragment, sha256_hash_hasher> fragments_;

  alignas(std::hardware_destructive_interference_size) rwlock_t modules_lock_;

  std::unordered_map<sha256_hash, module, sha256_hash_hasher> modules_;

  alignas(std::hardware_destructive_interference_size) statistics stats_;

 public:
  // [ ] memory usage
  // [ ] the object or blob being cached
  // [ ] best we can do is LRU?

  // [ ] when writing to path; store at target+".tmp${consistent_rand}"; then rename to target
  // [ ] local in-memory cache with memory limit & LRU eviction policy
  //
  // [ ] disk cache
  //
  // [ ] to use key as file id, convert to hex string
  cache_t()                          = default;
  cache_t(cache_t const&)            = delete;
  cache_t& operator=(cache_t const&) = delete;
  cache_t(cache_t&&)                 = delete;
  cache_t& operator=(cache_t&&)      = delete;
  ~cache_t();

  // [ ] file cache path?
  void store_blob(sha256_hash const& sha, blob binary);

  blob query_blob(sha256_hash const& sha);

  void store_fragment(sha256_hash const& sha, fragment frag);

  fragment query_fragment(sha256_hash const& sha);

  void store_module(sha256_hash const& sha, module mod);

  module query_module(sha256_hash const& sha);
};

// [ ] environment variables to control:
// [ ] cache path
// [ ] cache entries limit
// [ ] disable caching
// [ ] cache statistics: hits, misses, etc.
// [ ] cache tree pre-loading at startup
// [ ] on startup, log cache path, loading information, etc.

cache_t& global_cache();

void preload_cache();

// [ ] pre-load cuda
// [ ] if a user provides a key, use: USER_KEY+${key}, otherwise use sha256 of contents
// [ ] use resource type in key to avoid collisions
void compile_operator();

void link_operator();

}  // namespace rtc
}  // namespace cudf
