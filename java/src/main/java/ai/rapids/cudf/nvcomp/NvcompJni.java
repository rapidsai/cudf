/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
   * Determine if data is compressed with the nvcomp LZ4 compressor.
   * @param inPtr device address of the compressed data
   * @param inSize size of the compressed data in bytes
   * @param stream CUDA stream to use
   * @return true if the data is compressed with the nvcomp LZ4 compressor
   */
  static native boolean isLZ4Data(long inPtr, long inSize, long stream);

  /**
   * Determine if the metadata corresponds to data compressed with the nvcomp LZ4 compressor.
   * @param metadataPtr address of the metadata object
   * @return true if the metadata describes data compressed with the nvcomp LZ4 compressor.
   */
  static native boolean isLZ4Metadata(long metadataPtr);

  /**
   * Return the LZ4 compression configuration necessary for a particular chunk size.
   * @param chunkSize maximum size of an uncompressed chunk in bytes
   * @param uncompressedSize total size of the uncompressed data
   * @return array of three longs containing metadata size, temp storage size,
   *         and output buffer size
   */
  static native long[] lz4CompressConfigure(long chunkSize, long uncompressedSize);

  /**
   * Perform LZ4 compression asynchronously using the specified CUDA stream.
   * @param compressedSizeOutputPtr host address of a 64-bit integer to update
   *                                with the resulting compressed size of the
   *                                data. For the operation to be truly
   *                                asynchronous this should point to pinned
   *                                host memory.
   * @param inPtr device address of the uncompressed data
   * @param inSize size of the uncompressed data in bytes
   * @param inputType type of uncompressed data
   * @param chunkSize size of an LZ4 chunk in bytes
   * @param tempPtr device address of the temporary compression storage buffer
   * @param tempSize size of the temporary storage buffer in bytes
   * @param outPtr device address of the output buffer
   * @param outSize size of the output buffer in bytes
   * @param stream CUDA stream to use
   */
  static native void lz4CompressAsync(
      long compressedSizeOutputPtr,
      long inPtr,
      long inSize,
      int inputType,
      long chunkSize,
      long tempPtr,
      long tempSize,
      long outPtr,
      long outSize,
      long stream);

  /**
   * Return the decompression configuration for a compressed input.
   * NOTE: The resulting configuration object must be closed to destroy the corresponding
   * host-side metadata created by this method to avoid a native memory leak.
   * @param inPtr device address of the compressed data
   * @param inSize size of the compressed data
   * @return array of four longs containing metadata address, metadata size, temp storage size,
   *         and output buffer size
   */
  static native long[] lz4DecompressConfigure(long inPtr, long inSize, long stream);

  /**
   * Perform LZ4 decompression asynchronously using the specified CUDA stream.
   * @param inPtr device address of the uncompressed data
   * @param inSize size of the uncompressed data in bytes
   * @param metadataPtr host address of the metadata
   * @param metadataSize size of the metadata in bytes
   * @param tempPtr device address of the temporary compression storage buffer
   * @param tempSize size of the temporary storage buffer in bytes
   * @param outPtr device address of the output buffer
   * @param outSize size of the output buffer in bytes
   * @param stream CUDA stream to use
   */
  static native void lz4DecompressAsync(
      long inPtr,
      long inSize,
      long metadataPtr,
      long metadataSize,
      long tempPtr,
      long tempSize,
      long outPtr,
      long outSize,
      long stream);

  /**
   * Destroy host-side metadata created by {@link NvcompJni#lz4DecompressConfigure(long, long, long)}
   * @param metadataPtr host address of metadata
   */
  static native void lz4DestroyMetadata(long metadataPtr);

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
