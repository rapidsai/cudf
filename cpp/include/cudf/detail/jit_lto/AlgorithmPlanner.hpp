/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/jit_lto/AlgorithmLauncher.hpp>
#include <cudf/detail/jit_lto/FragmentEntry.hpp>

#include <memory>
#include <shared_mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cudf::detail::jit_lto {

struct LauncherJitCache {
  std::shared_mutex mutex;
  std::unordered_map<std::string, std::shared_ptr<AlgorithmLauncher>> launchers;
};

struct AlgorithmPlanner {
  AlgorithmPlanner(std::string entrypoint, LauncherJitCache& jit_cache)
    : entrypoint(std::move(entrypoint)), jit_cache_(jit_cache)
  {
  }

  std::shared_ptr<AlgorithmLauncher> get_launcher();

  std::string entrypoint;
  std::vector<std::unique_ptr<FragmentEntry>> fragments;

  template <typename T, typename = std::enable_if_t<std::is_convertible_v<T*, FragmentEntry*>>>
  void add_fragment(std::unique_ptr<T> fragment)
  {
    fragments.push_back(std::unique_ptr<FragmentEntry>(std::move(fragment)));
  }

  template <typename FragmentTag>
  void add_static_fragment()
  {
    add_fragment(std::make_unique<StaticFatbinFragmentEntry<FragmentTag>>());
  }

 private:
  std::string get_fragments_key() const;
  std::shared_ptr<AlgorithmLauncher> build();

  std::shared_ptr<AlgorithmLauncher> read_cache(std::string const& launch_key) const;

  LauncherJitCache& jit_cache_;
};

}  // namespace cudf::detail::jit_lto
