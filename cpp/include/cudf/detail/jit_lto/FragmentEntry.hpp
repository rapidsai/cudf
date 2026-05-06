/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/jit_lto/nvjitlink_checker.hpp>

#include <nvJitLink.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <typeinfo>
#include <vector>

namespace cudf::detail::jit_lto {

struct FragmentEntry {
  virtual ~FragmentEntry() = default;

  virtual bool add_to(nvJitLinkHandle& handle) const = 0;

  virtual const char* get_key() const = 0;
};

struct FatbinFragmentEntry : FragmentEntry {
  virtual const uint8_t* get_data() const = 0;

  virtual size_t get_length() const = 0;

  bool add_to(nvJitLinkHandle& handle) const override final;
};

template <typename FragmentTag>
struct StaticFatbinFragmentEntry final : FatbinFragmentEntry {
  const uint8_t* get_data() const override { return StaticFatbinFragmentEntry<FragmentTag>::data; }

  size_t get_length() const override { return StaticFatbinFragmentEntry<FragmentTag>::length; }

  const char* get_key() const override
  {
    return typeid(StaticFatbinFragmentEntry<FragmentTag>).name();
  }

  static const uint8_t* const data;
  static const size_t length;
};

}  // namespace cudf::detail::jit_lto
