/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "embed.hpp"

int main()
{
  std::string_view id          = "@EMBED_SCRIPT__ID@";
  auto array_ids               = rtcx_embed::split_string("@EMBED_SCRIPT__ARRAY_IDS@", ';');
  auto array_values            = rtcx_embed::split_string("@EMBED_SCRIPT__ARRAY_VALUES@", ';');
  auto file_ids                = rtcx_embed::split_string("@EMBED_SCRIPT__FILE_IDS@", ';');
  auto file_paths              = rtcx_embed::split_string("@EMBED_SCRIPT__FILE_PATHS@", ';');
  auto file_dests              = rtcx_embed::split_string("@EMBED_SCRIPT__FILE_DESTS@", ';');
  auto include_directories     = rtcx_embed::split_string("@EMBED_SCRIPT__INCLUDE_DIRS@", ';');
  std::string_view compression = "@EMBED_SCRIPT__COMPRESSION@";
  std::string_view output_dir  = "@EMBED_SCRIPT__OUTPUT_DIR@";

  rtcx_embed::generate_embed(id,
                             array_ids,
                             array_values,
                             file_ids,
                             file_paths,
                             file_dests,
                             include_directories,
                             compression,
                             output_dir);
  return EXIT_SUCCESS;
}
