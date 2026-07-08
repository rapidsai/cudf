/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hash.hpp"

#include <array>
#include <bit>
#include <charconv>
#include <format>
#include <stdexcept>

namespace rtcx {

char const* hash128_hex_string::data() const { return data_; }

char const* hash128_hex_string::c_str() const { return data_; }

hash128_hex_string hash128_hex_string::make(std::span<std::uint8_t const, NUM_HEX_BYTES> input)
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

hash128_hex_string hash128_hex_string::make(__uint128_t hash)
{
  auto array = std::bit_cast<std::array<std::uint8_t, NUM_HEX_BYTES>>(hash);
  return make(array);
}

std::uint8_t hash128::operator[](std::size_t index) const
{
  return reinterpret_cast<std::uint8_t const*>(&value)[15 - index];
}

std::size_t hash128::size() const { return 16; }

std::uint8_t const* hash128::data() const { return reinterpret_cast<std::uint8_t const*>(&value); }

hash128_hex_string hash128::to_hex_string() const { return hash128_hex_string::make(value); }

hash128 hash128::parse(std::string_view hex)
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

}  // namespace rtcx
