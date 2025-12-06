/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/utilities/export.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

extern "C" {
typedef struct evp_md_ctx_st EVP_MD_CTX;
}

namespace CUDF_EXPORT cudf {
namespace rtc {

struct sha256_hex_string {
  char data_[65];

  constexpr operator std::string_view() const { return std::string_view{data_, 64}; }

  char const* data() const { return data_; }

  char const* c_str() const { return data_; }

  static constexpr size_t size() { return 64; }
};

struct sha256_hash {
  alignas(16) uint8_t data_[32];

  constexpr bool operator==(sha256_hash const& hash) const
  {
    return std::equal(std::begin(data_), std::end(data_), std::begin(hash.data_));
  }

  constexpr bool operator!=(sha256_hash const& hash) const { return !(*this == hash); }

  constexpr sha256_hex_string to_hex() const
  {
    static constexpr char const HEX_CHARS[] = "0123456789abcdef";
    sha256_hex_string hex;
    for (size_t i = 0; i < 32; ++i) {
      hex.data_[i * 2]     = HEX_CHARS[(data_[i] >> 4) & 0x0F];
      hex.data_[i * 2 + 1] = HEX_CHARS[data_[i] & 0x0F];
    }
    hex.data_[64] = '\0';
    return hex;
  }
};

struct sha256_hash_hasher {
  constexpr uint64_t operator()(sha256_hash const& obj) const
  {
    struct u64x4 {
      uint64_t v[4];
    };

    auto value    = std::bit_cast<u64x4>(obj);
    auto const h0 = value.v[0];
    auto const h1 = value.v[1];
    auto const h2 = value.v[2];
    auto const h3 = value.v[3];

    auto mix = [](uint64_t seed, uint64_t v) {
      seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
      return seed;
    };

    return mix(mix(mix(h0, h1), h2), h3);
  }
};

struct sha256_context {
 private:
  EVP_MD_CTX* ectx_;

 public:
  sha256_context();
  sha256_context(sha256_context const& other)            = delete;
  sha256_context& operator=(sha256_context const& other) = delete;
  sha256_context(sha256_context&& other) : ectx_(other.ectx_) { other.ectx_ = nullptr; }

  sha256_context& operator=(sha256_context&& other)
  {
    if (this == &other) [[unlikely]] { return *this; }
    this->~sha256_context();
    new (this) sha256_context(std::move(other));
    return *this;
  }

  ~sha256_context();

  void update(std::span<uint8_t const> data);

  sha256_hash finalize();
};

}  // namespace rtc
}  // namespace CUDF_EXPORT cudf
