/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.nvcomp;

/** ZSTD decompressor that operates on multiple input buffers in a batch */
public class BatchedZstdDecompressor extends BatchedDecompressor {
  public BatchedZstdDecompressor(long chunkSize) {
    super(chunkSize);
  }

  @Override
  protected long batchedDecompressGetTempSize(long numChunks, long maxUncompressedChunkBytes,
      long maxTotalSize) {
    return NvcompJni.batchedZstdDecompressGetTempSize(numChunks, maxUncompressedChunkBytes,
      maxTotalSize);
  }

  @Override
  protected void batchedDecompressAsync(long devInPtrs, long devInSizes, long devOutSizes,
      long batchSize, long tempPtr, long tempSize, long devOutPtrs, long stream) {
    NvcompJni.batchedZstdDecompressAsync(devInPtrs, devInSizes, devOutSizes, batchSize, tempPtr,
        tempSize, devOutPtrs, stream);
  }

}
