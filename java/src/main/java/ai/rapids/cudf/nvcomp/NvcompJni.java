/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

import ai.rapids.cudf.NativeDepsLoader;

/** Raw JNI interface to the nvcomp library. */
class NvcompJni {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Get the temporary workspace size required to perform compression of entire LZ4 batch.
   * @param batchSize number of chunks in the batch
   * @param maxChunkSize maximum size of an uncompressed chunk in bytes
   * @return The size of required temporary workspace in bytes to compress the batch.
   */
  static native long batchedLZ4CompressGetTempSize(long batchSize, long maxChunkSize);

  /**
   * Get the maximum size any chunk could compress to in a LZ4 batch. This is the minimum amount of
   * output memory to allocate per chunk when batch compressing.
   * @param maxChunkSize maximum size of an uncompressed chunk size in bytes
   * @return maximum compressed output size of a chunk
   */
  static native long batchedLZ4CompressGetMaxOutputChunkSize(long maxChunkSize);

  /**
   * Asynchronously compress a batch of buffers with LZ4. Note that
   * compressedSizesOutPtr must point to pinned memory for this operation
   * to be asynchronous.
   * @param devInPtrs device address of uncompressed buffer addresses vector
   * @param devInSizes device address of uncompressed buffer sizes vector
   * @param chunkSize maximum size of an uncompressed chunk in bytes
   * @param batchSize number of chunks in the batch
   * @param tempPtr device address of the temporary workspace buffer
   * @param tempSize size of the temporary workspace buffer in bytes
   * @param devOutPtrs device address of output buffer addresses vector
   * @param compressedSizesOutPtr device address where to write the sizes of the
   *                              compressed data written to the corresponding
   *                              output buffers. Must point to a buffer with
   *                              at least 8 bytes of memory per output buffer
   *                              in the batch.
   * @param stream CUDA stream to use
   */
  static native void batchedLZ4CompressAsync(
      long devInPtrs,
      long devInSizes,
      long chunkSize,
      long batchSize,
      long tempPtr,
      long tempSize,
      long devOutPtrs,
      long compressedSizesOutPtr,
      long stream);

  /**
   * Computes the temporary storage size in bytes needed to decompress a LZ4-compressed batch.
   * @param numChunks number of chunks in the batch
   * @param maxUncompressedChunkBytes maximum uncompressed size of any chunk in bytes
   * @return number of temporary storage bytes needed to decompress the batch
   */
  static native long batchedLZ4DecompressGetTempSize(
      long numChunks,
      long maxUncompressedChunkBytes);

  /**
   * Asynchronously decompress a batch of LZ4-compressed data buffers.
   * @param devInPtrs device address of compressed input buffer addresses vector
   * @param devInSizes device address of compressed input buffer sizes vector
   * @param devOutSizes device address of uncompressed buffer sizes vector
   * @param batchSize number of buffers in the batch
   * @param tempPtr device address of the temporary decompression space
   * @param tempSize size of the temporary decompression space in bytes
   * @param devOutPtrs device address of uncompressed output buffer addresses vector
   * @param stream CUDA stream to use
   */
  static native void batchedLZ4DecompressAsync(
      long devInPtrs,
      long devInSizes,
      long devOutSizes,
      long batchSize,
      long tempPtr,
      long tempSize,
      long devOutPtrs,
      long stream);

  /**
   * Asynchronously calculates the decompressed size needed for each chunk.
   * @param devInPtrs device address of compressed input buffer addresses vector
   * @param devInSizes device address of compressed input buffer sizes vector
   * @param devOutSizes device address of calculated decompress sizes vector
   * @param batchSize number of buffers in the batch
   * @param stream CUDA stream to use
   */
  static native void batchedLZ4GetDecompressSizeAsync(
      long devInPtrs,
      long devInSizes,
      long devOutSizes,
      long batchSize,
      long stream);
}
