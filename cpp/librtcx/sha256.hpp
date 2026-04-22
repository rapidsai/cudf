

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <charconv>
#include <cstdint>
#include <cstring>
#include <format>
#include <span>
#include <stdexcept>
#include <string_view>

namespace rtcx {

namespace {

std::uint32_t ror(std::uint32_t x, std::uint32_t n) { return (x >> n) | (x << (32 - n)); }

std::uint32_t ch(std::uint32_t x, std::uint32_t y, std::uint32_t z) { return z ^ (x & (y ^ z)); }

std::uint32_t maj(std::uint32_t x, std::uint32_t y, std::uint32_t z)
{
  return ((x | y) & z) | (x & y);
}

std::uint32_t sigma0(std::uint32_t x) { return ror(x, 2) ^ ror(x, 13) ^ ror(x, 22); }

std::uint32_t sigma1(std::uint32_t x) { return ror(x, 6) ^ ror(x, 11) ^ ror(x, 25); }

std::uint32_t gamma0(std::uint32_t x) { return ror(x, 7) ^ ror(x, 18) ^ (x >> 3); }

std::uint32_t gamma1(std::uint32_t x) { return ror(x, 17) ^ ror(x, 19) ^ (x >> 10); }

void put_be32(void* ptr, std::uint32_t value)
{
  auto* p = (std::uint8_t*)ptr;
  p[0]    = (value >> 24) & 0xff;
  p[1]    = (value >> 16) & 0xff;
  p[2]    = (value >> 8) & 0xff;
  p[3]    = (value >> 0) & 0xff;
}

std::uint32_t get_be32(void const* ptr)
{
  auto const* p = (std::uint8_t const*)ptr;
  return (std::uint32_t)p[0] << 24 | (std::uint32_t)p[1] << 16 | (std::uint32_t)p[2] << 8 |
         (std::uint32_t)p[3] << 0;
}

}  // namespace

struct [[nodiscard]] sha256_hex_string {
  char data_[65];  // NOLINT(modernize-avoid-c-arrays)

  [[nodiscard]] constexpr std::string_view view() const { return std::string_view{data_, 64}; }

  [[nodiscard]] constexpr operator std::string_view() const { return view(); }

  [[nodiscard]] char const* data() const { return data_; }

  [[nodiscard]] char const* c_str() const { return data_; }

  [[nodiscard]] static constexpr std::size_t size() { return 64; }

  static sha256_hex_string make(std::span<std::uint8_t const, 32> input)
  {
    constexpr char const HEX_CHARS[] = "0123456789abcdef";  // NOLINT(modernize-avoid-c-arrays)
    sha256_hex_string hex;
    for (std::size_t i = 0; i < 32; ++i) {
      hex.data_[i * 2]     = HEX_CHARS[(input[i] >> 4) & 0x0F];
      hex.data_[i * 2 + 1] = HEX_CHARS[input[i] & 0x0F];
    }
    hex.data_[64] = '\0';
    return hex;
  }
};

struct [[nodiscard]] sha256 {
  alignas(16) std::uint8_t data_[32];  // NOLINT(modernize-avoid-c-arrays)

  [[nodiscard]] std::uint8_t operator[](std::size_t index) const { return data_[index]; }

  [[nodiscard]] std::size_t size() const { return 32; }

  [[nodiscard]] std::uint8_t const* data() const { return data_; }

  [[nodiscard]] constexpr bool operator==(sha256 const&) const = default;

  sha256_hex_string to_hex_string() const { return sha256_hex_string::make(data_); }

  static sha256 parse(std::string_view hex)
  {
    if (hex.size() != 64) {
      throw std::invalid_argument(std::format(
        "Invalid SHA256 hex string length, expected 64 got {} (sha: `{}`)", hex.size(), hex));
    }
    sha256 hash;
    for (std::size_t i = 0; i < 32; ++i) {
      auto hex_byte  = hex.substr(i * 2, 2);
      auto [ptr, ec] = std::from_chars(hex_byte.begin(), hex_byte.end(), hash.data_[i], 16);
      if (ec != std::errc()) {
        throw std::invalid_argument(
          std::format("Invalid hex character in SHA256 string: `{}`", hex_byte));
      }
    }
    return hash;
  }
};

struct sha256_context {
 private:
  static constexpr size_t BLOCK_SIZE = 64;
  std::uint32_t state_[8]            =  // NOLINT(modernize-avoid-c-arrays)
    {0x6a09'e667ul,
     0xbb67'ae85ul,
     0x3c6e'f372ul,
     0xa54f'f53aul,
     0x510e'527ful,
     0x9b05'688cul,
     0x1f83'd9abul,
     0x5be0'cd19ul};
  std::uint64_t size_           = 0;
  std::uint8_t buf_[BLOCK_SIZE] = {};  // NOLINT(modernize-avoid-c-arrays)

 public:
  sha256_context()                                 = default;
  sha256_context(sha256_context const&)            = delete;
  sha256_context& operator=(sha256_context const&) = delete;
  sha256_context(sha256_context&&)                 = delete;
  sha256_context& operator=(sha256_context&&)      = delete;
  ~sha256_context()                                = default;

 private:
  void transform(std::uint8_t const* buf)
  {
    std::uint32_t S[8], W[64], t0, t1;  // NOLINT(modernize-avoid-c-arrays)
    int i;

    /* copy state into S */
    for (i = 0; i < 8; i++)
      S[i] = state_[i];

    /* copy the state into 512-bits into W[0..15] */
    for (i = 0; i < 16; i++, buf += sizeof(std::uint32_t))
      W[i] = get_be32(buf);

    /* fill W[16..63] */
    for (i = 16; i < 64; i++)
      W[i] = gamma1(W[i - 2]) + W[i - 7] + gamma0(W[i - 15]) + W[i - 16];

#define RND(a, b, c, d, e, f, g, h, i, ki)      \
  t0 = h + sigma1(e) + ch(e, f, g) + ki + W[i]; \
  t1 = sigma0(a) + maj(a, b, c);                \
  d += t0;                                      \
  h = t0 + t1;

    RND(S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7], 0, 0x428a'2f98);
    RND(S[7], S[0], S[1], S[2], S[3], S[4], S[5], S[6], 1, 0x7137'4491);
    RND(S[6], S[7], S[0], S[1], S[2], S[3], S[4], S[5], 2, 0xb5c0'fbcf);
    RND(S[5], S[6], S[7], S[0], S[1], S[2], S[3], S[4], 3, 0xe9b5'dba5);
    RND(S[4], S[5], S[6], S[7], S[0], S[1], S[2], S[3], 4, 0x3956'c25b);
    RND(S[3], S[4], S[5], S[6], S[7], S[0], S[1], S[2], 5, 0x59f1'11f1);
    RND(S[2], S[3], S[4], S[5], S[6], S[7], S[0], S[1], 6, 0x923f'82a4);
    RND(S[1], S[2], S[3], S[4], S[5], S[6], S[7], S[0], 7, 0xab1c'5ed5);
    RND(S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7], 8, 0xd807'aa98);
    RND(S[7], S[0], S[1], S[2], S[3], S[4], S[5], S[6], 9, 0x1283'5b01);
    RND(S[6], S[7], S[0], S[1], S[2], S[3], S[4], S[5], 10, 0x2431'85be);
    RND(S[5], S[6], S[7], S[0], S[1], S[2], S[3], S[4], 11, 0x550c'7dc3);
    RND(S[4], S[5], S[6], S[7], S[0], S[1], S[2], S[3], 12, 0x72be'5d74);
    RND(S[3], S[4], S[5], S[6], S[7], S[0], S[1], S[2], 13, 0x80de'b1fe);
    RND(S[2], S[3], S[4], S[5], S[6], S[7], S[0], S[1], 14, 0x9bdc'06a7);
    RND(S[1], S[2], S[3], S[4], S[5], S[6], S[7], S[0], 15, 0xc19b'f174);
    RND(S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7], 16, 0xe49b'69c1);
    RND(S[7], S[0], S[1], S[2], S[3], S[4], S[5], S[6], 17, 0xefbe'4786);
    RND(S[6], S[7], S[0], S[1], S[2], S[3], S[4], S[5], 18, 0x0fc1'9dc6);
    RND(S[5], S[6], S[7], S[0], S[1], S[2], S[3], S[4], 19, 0x240c'a1cc);
    RND(S[4], S[5], S[6], S[7], S[0], S[1], S[2], S[3], 20, 0x2de9'2c6f);
    RND(S[3], S[4], S[5], S[6], S[7], S[0], S[1], S[2], 21, 0x4a74'84aa);
    RND(S[2], S[3], S[4], S[5], S[6], S[7], S[0], S[1], 22, 0x5cb0'a9dc);
    RND(S[1], S[2], S[3], S[4], S[5], S[6], S[7], S[0], 23, 0x76f9'88da);
    RND(S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7], 24, 0x983e'5152);
    RND(S[7], S[0], S[1], S[2], S[3], S[4], S[5], S[6], 25, 0xa831'c66d);
    RND(S[6], S[7], S[0], S[1], S[2], S[3], S[4], S[5], 26, 0xb003'27c8);
    RND(S[5], S[6], S[7], S[0], S[1], S[2], S[3], S[4], 27, 0xbf59'7fc7);
    RND(S[4], S[5], S[6], S[7], S[0], S[1], S[2], S[3], 28, 0xc6e0'0bf3);
    RND(S[3], S[4], S[5], S[6], S[7], S[0], S[1], S[2], 29, 0xd5a7'9147);
    RND(S[2], S[3], S[4], S[5], S[6], S[7], S[0], S[1], 30, 0x06ca'6351);
    RND(S[1], S[2], S[3], S[4], S[5], S[6], S[7], S[0], 31, 0x1429'2967);
    RND(S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7], 32, 0x27b7'0a85);
    RND(S[7], S[0], S[1], S[2], S[3], S[4], S[5], S[6], 33, 0x2e1b'2138);
    RND(S[6], S[7], S[0], S[1], S[2], S[3], S[4], S[5], 34, 0x4d2c'6dfc);
    RND(S[5], S[6], S[7], S[0], S[1], S[2], S[3], S[4], 35, 0x5338'0d13);
    RND(S[4], S[5], S[6], S[7], S[0], S[1], S[2], S[3], 36, 0x650a'7354);
    RND(S[3], S[4], S[5], S[6], S[7], S[0], S[1], S[2], 37, 0x766a'0abb);
    RND(S[2], S[3], S[4], S[5], S[6], S[7], S[0], S[1], 38, 0x81c2'c92e);
    RND(S[1], S[2], S[3], S[4], S[5], S[6], S[7], S[0], 39, 0x9272'2c85);
    RND(S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7], 40, 0xa2bf'e8a1);
    RND(S[7], S[0], S[1], S[2], S[3], S[4], S[5], S[6], 41, 0xa81a'664b);
    RND(S[6], S[7], S[0], S[1], S[2], S[3], S[4], S[5], 42, 0xc24b'8b70);
    RND(S[5], S[6], S[7], S[0], S[1], S[2], S[3], S[4], 43, 0xc76c'51a3);
    RND(S[4], S[5], S[6], S[7], S[0], S[1], S[2], S[3], 44, 0xd192'e819);
    RND(S[3], S[4], S[5], S[6], S[7], S[0], S[1], S[2], 45, 0xd699'0624);
    RND(S[2], S[3], S[4], S[5], S[6], S[7], S[0], S[1], 46, 0xf40e'3585);
    RND(S[1], S[2], S[3], S[4], S[5], S[6], S[7], S[0], 47, 0x106a'a070);
    RND(S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7], 48, 0x19a4'c116);
    RND(S[7], S[0], S[1], S[2], S[3], S[4], S[5], S[6], 49, 0x1e37'6c08);
    RND(S[6], S[7], S[0], S[1], S[2], S[3], S[4], S[5], 50, 0x2748'774c);
    RND(S[5], S[6], S[7], S[0], S[1], S[2], S[3], S[4], 51, 0x34b0'bcb5);
    RND(S[4], S[5], S[6], S[7], S[0], S[1], S[2], S[3], 52, 0x391c'0cb3);
    RND(S[3], S[4], S[5], S[6], S[7], S[0], S[1], S[2], 53, 0x4ed8'aa4a);
    RND(S[2], S[3], S[4], S[5], S[6], S[7], S[0], S[1], 54, 0x5b9c'ca4f);
    RND(S[1], S[2], S[3], S[4], S[5], S[6], S[7], S[0], 55, 0x682e'6ff3);
    RND(S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7], 56, 0x748f'82ee);
    RND(S[7], S[0], S[1], S[2], S[3], S[4], S[5], S[6], 57, 0x78a5'636f);
    RND(S[6], S[7], S[0], S[1], S[2], S[3], S[4], S[5], 58, 0x84c8'7814);
    RND(S[5], S[6], S[7], S[0], S[1], S[2], S[3], S[4], 59, 0x8cc7'0208);
    RND(S[4], S[5], S[6], S[7], S[0], S[1], S[2], S[3], 60, 0x90be'fffa);
    RND(S[3], S[4], S[5], S[6], S[7], S[0], S[1], S[2], 61, 0xa450'6ceb);
    RND(S[2], S[3], S[4], S[5], S[6], S[7], S[0], S[1], 62, 0xbef9'a3f7);
    RND(S[1], S[2], S[3], S[4], S[5], S[6], S[7], S[0], 63, 0xc671'78f2);

#undef RND

    for (i = 0; i < 8; i++) {
      state_[i] += S[i];
    }
  }

 public:
  void update(std::span<std::uint8_t const> span)
  {
    std::uint32_t len_buf = size_ & 63;
    auto data             = span.data();
    auto len              = span.size();

    size_ += len;

    /* Read the data into buf and process blocks as they get full */
    if (len_buf) {
      std::uint32_t left = 64 - len_buf;
      if (len < left) left = len;
      memcpy(len_buf + buf_, data, left);
      len_buf = (len_buf + left) & 63;
      len -= left;
      data = (data + left);
      if (len_buf) return;
      transform(buf_);
    }

    while (len >= 64) {
      transform(data);
      data = data + 64;
      len -= 64;
    }

    if (len) memcpy(buf_, data, len);
  }

  sha256 finalize()
  {
    static std::uint8_t const pad[64] = {0x80};  // NOLINT(modernize-avoid-c-arrays)
    std::uint32_t padlen[2];                     // NOLINT(modernize-avoid-c-arrays)
    int i;

    /* Pad with a binary 1 (ie 0x80), then zeroes, then length */
    padlen[0] = __builtin_bswap32((std::uint32_t)(size_ >> 29));
    padlen[1] = __builtin_bswap32((std::uint32_t)(size_ << 3));

    i = size_ & 63;
    update(std::span{pad, (std::size_t)(1 + (63 & (55 - i)))});
    update(std::span{reinterpret_cast<std::uint8_t*>(padlen), sizeof(padlen)});

    sha256 out;
    std::uint8_t* digest = out.data_;

    /* copy output */
    for (i = 0; i < 8; i++, digest += sizeof(std::uint32_t)) {
      put_be32(digest, state_[i]);
    }
    return out;
  }
};

}  // namespace rtcx
