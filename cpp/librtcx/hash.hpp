/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace rtcx {

struct [[nodiscard]] hash128_hex_string {
  static constexpr std::size_t NUM_HEX_DIGITS = 32;
  static constexpr std::size_t NUM_HEX_BYTES  = NUM_HEX_DIGITS / 2;

  char data_[NUM_HEX_DIGITS + 1];  // NOLINT(modernize-avoid-c-arrays)

  [[nodiscard]] constexpr std::string_view view() const
  {
    return std::string_view{data_, NUM_HEX_DIGITS};
  }

  [[nodiscard]] constexpr operator std::string_view() const { return view(); }

  [[nodiscard]] char const* data() const;

  [[nodiscard]] char const* c_str() const;

  [[nodiscard]] static constexpr std::size_t size() { return NUM_HEX_DIGITS; }

  static hash128_hex_string make(std::span<std::uint8_t const, NUM_HEX_BYTES> input);

  static hash128_hex_string make(__uint128_t hash);
};

struct hash128 {
  __uint128_t value;

  constexpr hash128(__uint128_t v = 0) : value(v) {}

  constexpr hash128(std::uint64_t high, std::uint64_t low)
    : value((static_cast<__uint128_t>(high) << 64) | low)
  {
  }

  [[nodiscard]] constexpr bool operator==(hash128 const&) const = default;

  [[nodiscard]] std::uint8_t operator[](std::size_t index) const;

  [[nodiscard]] std::size_t size() const;

  [[nodiscard]] std::uint8_t const* data() const;

  hash128_hex_string to_hex_string() const;

  static hash128 parse(std::string_view hex);
};

}  // namespace rtcx
