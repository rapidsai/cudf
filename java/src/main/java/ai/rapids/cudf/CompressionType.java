/*
 *
 *  Copyright (c) 2019, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
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
  XZ(7);

  final int nativeId;

  CompressionType(int nativeId) {
    this.nativeId = nativeId;
  }
}
