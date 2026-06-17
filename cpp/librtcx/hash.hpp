/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <bit>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <format>
#include <span>
#include <stdexcept>
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

  [[nodiscard]] char const* data() const { return data_; }

  [[nodiscard]] char const* c_str() const { return data_; }

  [[nodiscard]] static constexpr std::size_t size() { return NUM_HEX_DIGITS; }

  static hash128_hex_string make(std::span<std::uint8_t const, NUM_HEX_BYTES> input)
  {
    constexpr char const HEX_CHARS[] = "0123456789abcdef";  // NOLINT(modernize-avoid-c-arrays)
    hash128_hex_string hex;
    for (std::size_t i = 0; i < NUM_HEX_BYTES; ++i) {
      hex.data_[i * 2]     = HEX_CHARS[(input[i] >> 4) & 0x0F];
      hex.data_[i * 2 + 1] = HEX_CHARS[input[i] & 0x0F];
    }
    hex.data_[NUM_HEX_DIGITS] = '\0';
    return hex;
  }

  static hash128_hex_string make(__uint128_t hash)
  {
    auto array = std::bit_cast<std::array<std::uint8_t, NUM_HEX_BYTES>>(hash);
    return make(array);
  }
};

struct hash128 {
  __uint128_t value;

  constexpr hash128(__uint128_t v = 0) : value(v) {}

  constexpr hash128(std::uint64_t high, std::uint64_t low)
    : value((static_cast<__uint128_t>(high) << 64) | low)
  {
  }

  [[nodiscard]] constexpr bool operator==(hash128 const&) const = default;

  [[nodiscard]] std::uint8_t operator[](std::size_t index) const
  {
    return reinterpret_cast<std::uint8_t const*>(&value)[16 - index];
  }

  [[nodiscard]] std::size_t size() const { return 16; }

  [[nodiscard]] std::uint8_t const* data() const
  {
    return reinterpret_cast<std::uint8_t const*>(&value);
  }

  hash128_hex_string to_hex_string() const { return hash128_hex_string::make(value); }

  static hash128 parse(std::string_view hex)
  {
    if (hex.size() != hash128_hex_string::NUM_HEX_DIGITS) {
      throw std::invalid_argument(
        std::format("Invalid hash128 hex string length, expected {} got {} (hash: `{}`)",
                    hash128_hex_string::NUM_HEX_DIGITS,
                    hex.size(),
                    hex));
    }
    std::array<std::uint8_t, hash128_hex_string::NUM_HEX_BYTES> data{};
    for (std::size_t i = 0; i < hash128_hex_string::NUM_HEX_BYTES; ++i) {
      auto hex_byte  = hex.substr(i * 2, 2);
      auto [ptr, ec] = std::from_chars(hex_byte.begin(), hex_byte.end(), data[i], 16);
      if (ec != std::errc()) {
        throw std::invalid_argument(
          std::format("Invalid hex character {} in HEX string: `{}`", hex_byte, hex));
      }
    }
    return hash128{std::bit_cast<__uint128_t>(data)};
  }
};

}  // namespace rtcx
