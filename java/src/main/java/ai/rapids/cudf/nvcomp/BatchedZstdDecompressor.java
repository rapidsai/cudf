/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.rapids.cudf.nvcomp;

/** ZSTD decompressor that operates on multiple input buffers in a batch */
public class BatchedZstdDecompressor extends BatchedDecompressor {
  public BatchedZstdDecompressor(long chunkSize) {
    super(chunkSize);
  }

  @Override
  protected long batchedDecompressGetTempSize(long numChunks, long maxUncompressedChunkBytes) {
    return NvcompJni.batchedZstdDecompressGetTempSize(numChunks, maxUncompressedChunkBytes);
  }

  @Override
  protected void batchedDecompressAsync(long devInPtrs, long devInSizes, long devOutSizes,
      long batchSize, long tempPtr, long tempSize, long devOutPtrs, long stream) {
    NvcompJni.batchedZstdDecompressAsync(devInPtrs, devInSizes, devOutSizes, batchSize, tempPtr,
        tempSize, devOutPtrs, stream);
  }

}
