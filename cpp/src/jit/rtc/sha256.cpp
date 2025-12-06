/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/utilities/error.hpp>

#include <jit/rtc/sha256.hpp>

extern "C" {
#include <openssl/evp.h>
}

namespace CUDF_EXPORT cudf {
namespace rtc {

sha256_context::sha256_context() : ectx_(nullptr)
{
  const EVP_MD* type = EVP_sha256();
  ectx_              = EVP_MD_CTX_new();
  CUDF_EXPECTS(ectx_ != nullptr, "EVP_MD_CTX_new failed");
  CUDF_EXPECTS(EVP_DigestInit_ex(ectx_, type, nullptr) == 1, "EVP_DigestInit_ex failed");
}

sha256_context::~sha256_context()
{
  if (ectx_ != nullptr) { EVP_MD_CTX_free(ectx_); }
}

void sha256_context::update(std::span<uint8_t const> data)
{
  CUDF_EXPECTS(EVP_DigestUpdate(ectx_, data.data(), data.size()) == 1, "EVP_DigestUpdate failed");
}

sha256_hash sha256_context::finalize()
{
  sha256_hash hash;
  unsigned int length = 0;
  CUDF_EXPECTS(EVP_DigestFinal_ex(ectx_, hash.data_, &length) == 1, "EVP_DigestFinal_ex failed");
  CUDF_EXPECTS(length == 64, "Unexpected SHA256 length");
  EVP_MD const* type = EVP_sha256();
  CUDF_EXPECTS(EVP_DigestInit_ex(ectx_, type, nullptr) == 1, "EVP_DigestInit_ex failed");
  return hash;
}

}  // namespace rtc
}  // namespace CUDF_EXPORT cudf
