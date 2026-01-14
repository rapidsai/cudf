/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2022, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * Enumeration of compression formats.
 */
public enum CompressionType {
  /** No compression */
  NONE(0),

  /** Automatically detect or select the compression codec */
  AUTO(1),

  /** Snappy format using byte-oriented LZ77 */
  SNAPPY(2),

  /** GZIP format using the DEFLATE algorithm */
  GZIP(3),

  /** BZIP2 format using Burrows-Wheeler transform */
  BZIP2(4),

  /** BROTLI format using LZ77 + Huffman + 2nd order context modeling */
  BROTLI(5),

  /** ZIP format using DEFLATE algorithm */
  ZIP(6),

  /** XZ format using LZMA(2) algorithm */
  XZ(7),

  /** ZLIB format, using DEFLATE algorithm */
  ZLIB(8),

  /** LZ4 format, using LZ77 */
  LZ4(9),

  /** Lempel–Ziv–Oberhumer format */
  LZO(10),

  /** Zstandard format */
  ZSTD(11);

  final int nativeId;

  CompressionType(int nativeId) { this.nativeId = nativeId; }
}
