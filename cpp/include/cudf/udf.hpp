/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace CUDF_EXPORT cudf {

/**
 * @brief Binary Fragment Types for compiled CUDA programs.
 */
enum class lto_fragment_type : uint8_t {
  LTO_IR,  //< LTO-IR binary
  FATBIN,  //< FATBIN binary
  PTX      //< PTX source
};

/**
 * @brief A pre-compiled LTO binary fragment.
 */
struct lto_fragment {
  lto_fragment_type type = lto_fragment_type::LTO_IR;  ///< Type of the LTO fragment

  std::span<uint8_t const> data{};  ///< Binary data of the fragment
};

struct udf_module {
  std::span<char const*>
    include_names{};  ///< Null-terminated strings containing the names of the included files to be
                      ///< provided for the UDF compilation

  std::span<char const*> includes{};  ///< Null-terminated strings containing the contents of the
  ///< included files to be provided for the UDF compilation

  std::span<lto_fragment const> lto_fragments{};  ///< Pre-compiled LTO binary fragments

  char const* cuda_source = "";  ///< Null-terminated string-view containing the body of the UDF
};

/**
 * @brief A CUDA UDF containing the source code and the expression name to be used as the UDF.
 */
struct udf {
  udf_module module{};  ///< The module containing the UDF source and LTO fragments

  std::string function{};  ///< symbol name of the UDF to be used

  bool is_cuda_function =
    false;  ///< Indicates whether the function_name represents a CUDA function

  /**
   * @brief Factory function to create a CUDA UDF.
   *
   * @param expression The expression name to be used as the UDF.
   * @param cuda_source The CUDA source code for the UDF.
   * @param include_names Names of the included files for compilation.
   * @param includes Contents of the included files for compilation.
   * @param lto_fragments Pre-compiled LTO binary fragments.
   * @return udf The constructed CUDA UDF.
   */
  static udf cuda(std::string_view expression,
                  char const* cuda_source,
                  std::span<char const*> include_names        = {},
                  std::span<char const*> includes             = {},
                  std::span<lto_fragment const> lto_fragments = {});

  /**
   * @brief Factory function to create an LTO-based UDF.
   *
   * @param symbol The symbol name of the UDF.
   * @param lto_fragments Pre-compiled LTO binary fragments.
   * @param cuda_source The CUDA source code for the UDF.
   * @param include_names Names of the included files for compilation.
   * @param includes Contents of the included files for compilation.
   * @return udf The constructed LTO-based UDF.
   */
  static udf lto(std::string_view symbol,
                 std::span<lto_fragment const> lto_fragments = {},
                 char const* cuda_source                     = "",
                 std::span<char const*> include_names        = {},
                 std::span<char const*> includes             = {});
};

}  // namespace CUDF_EXPORT cudf