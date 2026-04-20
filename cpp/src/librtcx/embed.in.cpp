/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "embed.hpp"

int main()
{
  std::string_view id          = "@EMBED_SCRIPT__ID@";
  auto file_paths              = split_string("@EMBED_SCRIPT__FILE_PATHS@", ';');
  auto file_dests              = split_string("@EMBED_SCRIPT__FILE_DESTS@", ';');
  auto include_directories     = split_string("@EMBED_SCRIPT__INCLUDE_DIRS@", ';');
  std::string_view compression = "@EMBED_SCRIPT__COMPRESSION@";
  std::string_view output_dir  = "@EMBED_SCRIPT__OUTPUT_DIR@";

  generate_embed(id, file_paths, file_dests, include_directories, compression, output_dir);
  return EXIT_SUCCESS;
}
