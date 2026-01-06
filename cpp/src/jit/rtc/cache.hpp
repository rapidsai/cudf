
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <jit/rtc/rtc.hpp>
#include <jit/rtc/sha256.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <future>
#include <optional>
#include <unordered_map>

namespace cudf {
namespace rtc {

struct [[nodiscard]] cache_statistics {
  uint64_t blob_memory_hits       = 0;
  uint64_t blob_memory_misses     = 0;
  uint64_t fragment_memory_hits   = 0;
  uint64_t fragment_memory_misses = 0;
  uint64_t library_memory_hits    = 0;
  uint64_t library_memory_misses  = 0;
  uint64_t blob_disk_hits         = 0;
  uint64_t blob_disk_misses       = 0;
};

struct [[nodiscard]] cache_limits {
  uint32_t num_blobs     = 1024;
  uint32_t num_fragments = 1024;
  uint32_t num_libraries = 1024;
};

namespace detail {

struct rw_spinlock_t {
 private:
  static constexpr size_t WRITE_STATE = ~size_t{0};
  static constexpr size_t IDLE_STATE  = 0;

  size_t state_;

 public:
  rw_spinlock_t() : state_{IDLE_STATE} {}
  rw_spinlock_t(rw_spinlock_t const&)            = default;
  rw_spinlock_t& operator=(rw_spinlock_t const&) = default;
  rw_spinlock_t(rw_spinlock_t&&)                 = default;
  rw_spinlock_t& operator=(rw_spinlock_t&&)      = default;
  ~rw_spinlock_t()                               = default;

  void lock_read()
  {
    std::atomic_ref state{state_};

    auto expected = IDLE_STATE;
    auto target   = size_t{1};

    while (!state.compare_exchange_weak(
      expected, target, std::memory_order_acquire, std::memory_order_relaxed)) {
      if (expected == WRITE_STATE) {
        expected = IDLE_STATE;
        target   = 1;
      } else {
        target = expected + 1;
      }
    }
  }

  void unlock_read()
  {
    std::atomic_ref state{state_};
    state.fetch_sub(1, std::memory_order_relaxed);
  }

  void lock_write()
  {
    std::atomic_ref state{state_};

    auto expected = IDLE_STATE;

    while (!state.compare_exchange_weak(
      expected, WRITE_STATE, std::memory_order_acquire, std::memory_order_relaxed)) {
      expected = IDLE_STATE;
    }
  }

  void unlock_write()
  {
    std::atomic_ref state{state_};
    state.store(IDLE_STATE, std::memory_order_release);
  }
};

template <typename Lock>
struct read_guard {
 private:
  Lock& lock_;

 public:
  read_guard(Lock& lock) : lock_{lock} { lock_.lock_read(); }

  read_guard(read_guard const&)            = delete;
  read_guard& operator=(read_guard const&) = delete;
  read_guard(read_guard&&)                 = delete;
  read_guard& operator=(read_guard&&)      = delete;

  ~read_guard() { lock_.unlock_read(); }
};

template <typename Lock>
struct write_guard {
 private:
  Lock& lock_;

 public:
  write_guard(Lock& lock) : lock_{lock} { lock_.lock_write(); }

  write_guard(write_guard const&)            = delete;
  write_guard& operator=(write_guard const&) = delete;
  write_guard(write_guard&&)                 = delete;
  write_guard& operator=(write_guard&&)      = delete;

  ~write_guard() { lock_.unlock_write(); }
};

inline constexpr size_t CACHELINE_ALIGNMENT =
  64;  // = std::hardware_destructive_interference_size */

template <typename T>
struct alignas(CACHELINE_ALIGNMENT) lru_memory_cache {
  struct entry {
    uint64_t last_touched_tick;
    T value;

    void hit(uint64_t tick) { last_touched_tick = tick; }
  };

  std::unordered_map<sha256_hash, entry, sha256_hash_hasher> entries_;
  rw_spinlock_t lock_;
  size_t limit_;

  explicit lru_memory_cache(size_t limit) : entries_{}, lock_{}, limit_{limit}
  {
    // reserve space to avoid rehashing
    entries_.reserve(limit * 2);
  }

  void insert(sha256_hash const& sha, T&& value, uint64_t tick)
  {
    if ((entries_.size() + 1) > limit_) {
      std::vector<std::pair<sha256_hash, uint64_t>> rankings;
      rankings.reserve(entries_.size());

      for (auto const& [key, ent] : entries_) {
        rankings.emplace_back(key, ent.last_touched_tick);
      }

      std::sort(rankings.begin(), rankings.end(), [](auto const& a, auto const& b) {
        return a.second < b.second;
      });

      // purge least recently used half

      auto num_to_purge = rankings.size() / 2;

      for (size_t i = 0; i < num_to_purge; ++i) {
        entries_.erase(rankings[i].first);
      }
    }

    entries_.emplace(sha, entry{tick, std::move(value)});
  }
};

struct alignas(CACHELINE_ALIGNMENT) counter {
  uint64_t value_ = 0;

  void increment()
  {
    std::atomic_ref c{value_};
    c.fetch_add(1, std::memory_order_relaxed);
  }

  [[nodiscard]] uint64_t get() const
  {
    std::atomic_ref c{value_};
    return c.load(std::memory_order_relaxed);
  }

  void reset()
  {
    std::atomic_ref c{value_};
    c.store(0, std::memory_order_relaxed);
  }
};

struct cache_statistics_counter {
  counter blob_memory_hits;
  counter blob_memory_misses;
  counter fragment_memory_hits;
  counter fragment_memory_misses;
  counter library_memory_hits;
  counter library_memory_misses;
  counter blob_disk_hits;
  counter blob_disk_misses;

  void clear()
  {
    blob_memory_hits.reset();
    blob_memory_misses.reset();
    fragment_memory_hits.reset();
    fragment_memory_misses.reset();
    library_memory_hits.reset();
    library_memory_misses.reset();
    blob_disk_hits.reset();
    blob_disk_misses.reset();
  }

  void hit_memory_blob() { blob_memory_hits.increment(); }

  void miss_memory_blob() { blob_memory_misses.increment(); }

  void hit_memory_fragment() { fragment_memory_hits.increment(); }

  void miss_memory_fragment() { fragment_memory_misses.increment(); }

  void hit_memory_library() { library_memory_hits.increment(); }

  void miss_memory_library() { library_memory_misses.increment(); }

  void hit_disk_blob() { blob_disk_hits.increment(); }

  void miss_disk_blob() { blob_disk_misses.increment(); }

  cache_statistics get_statistics() const
  {
    return cache_statistics{.blob_memory_hits       = blob_memory_hits.get(),
                            .blob_memory_misses     = blob_memory_misses.get(),
                            .fragment_memory_hits   = fragment_memory_hits.get(),
                            .fragment_memory_misses = fragment_memory_misses.get(),
                            .library_memory_hits    = library_memory_hits.get(),
                            .library_memory_misses  = library_memory_misses.get(),
                            .blob_disk_hits         = blob_disk_hits.get(),
                            .blob_disk_misses       = blob_disk_misses.get()};
  }
};

}  // namespace detail

/// @brief Thread-safe compile cache for compiled blobs, fragments, and libraries
/// @details Provides in-memory and on-disk caching of compiled RTC artifacts.
/// The cache uses an LRU eviction policy when the number of cached items
/// exceeds user-defined limits.
/// In-memory cache is implemented using a thread-safe LRU cache that supports concurrent reads.
/// The on-disk cache also allows concurrent access and stores cached items in files
/// within a specified directory. Writing to disk is atomic to prevent corruption from
/// concurrent writes or process interruptions.
/// In addition, the cache maintains statistics on cache hits and misses for both
/// in-memory and on-disk caches to help monitor cache performance in benchmarking and debugging.
/// The interface is zero-copy throughout, using shared pointers and spans to avoid unnecessary data
/// copying across threads and disk.
struct cache_t {
 private:
  bool enabled_;

  std::string cache_dir_;

  cache_limits limits_;

  detail::lru_memory_cache<std::shared_future<blob>> blobs_cache_;

  detail::lru_memory_cache<std::shared_future<fragment>> fragments_cache_;

  detail::lru_memory_cache<std::shared_future<library>> libraries_cache_;

  detail::cache_statistics_counter counter_;

  alignas(detail::CACHELINE_ALIGNMENT) uint64_t tick_;

 public:
  cache_t(bool enabled, std::string cache_dir, cache_limits const& limits);
  cache_t(cache_t const&)            = delete;
  cache_t& operator=(cache_t const&) = delete;
  cache_t(cache_t&&)                 = delete;
  cache_t& operator=(cache_t&&)      = delete;
  ~cache_t()                         = default;

  [[nodiscard]] bool is_enabled();

  void enable();

  void disable();

  void store_blob_to_memory(sha256_hash const& sha, std::shared_future<blob> binary);

  void store_blob_to_disk(sha256_hash const& sha, blob_view binary);

  std::optional<std::shared_future<blob>> query_blob_from_memory(sha256_hash const& sha);

  std::optional<blob> query_blob_from_disk(sha256_hash const& sha);

  void store_fragment(sha256_hash const& sha, std::shared_future<fragment> frag);

  std::optional<std::shared_future<fragment>> query_fragment(sha256_hash const& sha);

  void store_library(sha256_hash const& sha, std::shared_future<library> mod);

  std::optional<std::shared_future<library>> query_library(sha256_hash const& sha);

  cache_statistics get_statistics();

  void clear_statistics();

  cache_limits get_limits();

  [[nodiscard]] size_t get_blob_count();

  [[nodiscard]] size_t get_fragment_count();

  [[nodiscard]] size_t get_library_count();

  void clear_memory_store();

  void clear_disk_store();
};

}  // namespace rtc
}  // namespace cudf
