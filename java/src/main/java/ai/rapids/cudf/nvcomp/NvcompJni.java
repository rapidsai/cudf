/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
   * Extracts the metadata from the input on the device and copies
   * it to the host. Note that the result must be released with a
   * call to decompressDestroyMetadata
   * @param inPtr device address of the compressed data
   * @param inSize size of the compressed data in bytes
   * @param stream address of CUDA stream that will be used for synchronization
   * @return address of the metadata on the host
   */
  static native long decompressGetMetadata(long inPtr, long inSize, long stream);

  /**
   * Destroys the metadata object and frees the associated memory.
   * @param metadataPtr address of the metadata object
   */
  static native void decompressDestroyMetadata(long metadataPtr);

  /**
   * Computes the temporary storage size needed to decompress.
   * This over-estimates the needed storage considerably.
   * @param metadataPtr address of the metadata object
   * @return the number of temporary storage bytes needed to decompress
   */
  static native long decompressGetTempSize(long metadataPtr);

  /**
   * Computes the decompressed size of the data.  Gets this from the
   * metadata contained in the compressed data.
   * @param metadataPtr address of the metadata object
   * @return the size of the decompressed data in bytes
   */
  static native long decompressGetOutputSize(long metadataPtr);

  /**
   * Perform asynchronous decompression using the specified CUDA stream.
   * The input, temporary, and output buffers must all be in GPU-accessible
   * memory.
   * @param inPtr device address of the compressed buffer
   * @param inSize size of the compressed data in bytes
   * @param tempPtr device address of the temporary decompression storage buffer
   * @param tempSize size of the temporary decompression storage buffer
   * @param metadataPtr address of the metadata object
   * @param outPtr device address of the buffer to use for uncompressed output
   * @param outSize size of the uncompressed output buffer in bytes
   * @param stream CUDA stream to use
   */
  static native void decompressAsync(
      long inPtr,
      long inSize,
      long tempPtr,
      long tempSize,
      long metadataPtr,
      long outPtr,
      long outSize,
      long stream);

  /**
   * Determine if data is compressed with the nvcomp LZ4 compressor.
   * @param inPtr device address of the compressed data
   * @param inSize size of the compressed data in bytes
   * @return true if the data is compressed with the nvcomp LZ4 compressor
   */
  static native boolean isLZ4Data(long inPtr, long inSize);

  /**
   * Determine if the metadata corresponds to data compressed with the nvcomp LZ4 compressor.
   * @param metadataPtr address of the metadata object
   * @return true if the metadata describes data compressed with the nvcomp LZ4 compressor.
   */
  static native boolean isLZ4Metadata(long metadataPtr);

  /**
   * Calculate the temporary buffer size needed for LZ4 compression.
   * @param inPtr device address of the uncompressed data
   * @param inSize size of the uncompressed data in bytes
   * @param inputType type of uncompressed data
   * @param chunkSize size of an LZ4 chunk in bytes
   * @return number of temporary storage bytes needed to compress
   */
  static native long lz4CompressGetTempSize(
      long inPtr,
      long inSize,
      int inputType,
      long chunkSize);

  /**
   * Calculate the output buffer size for LZ4 compression. The output
   * size can be an estimated upper bound or the exact value.
   * @param inPtr device address of the uncompressed data
   * @param inSize size of the uncompressed data in bytes
   * @param inputType type of uncompressed data
   * @param chunkSize size of an LZ4 chunk in bytes
   * @param tempPtr device address of the temporary storage buffer
   * @param tempSize size of the temporary storage buffer in bytes
   * @param computeExactSize set to true to compute the exact output size
   * @return output buffer size in bytes. If computeExactSize is true then
   * this is an exact size otherwise it is a maximum, worst-case size of the
   * compressed data.
   */
  static native long lz4CompressGetOutputSize(
      long inPtr,
      long inSize,
      int inputType,
      long chunkSize,
      long tempPtr,
      long tempSize,
      boolean computeExactSize);

  /**
   * Perform LZ4 compression synchronously using the specified CUDA
   * stream.
   * @param inPtr device address of the uncompressed data
   * @param inSize size of the uncompressed data in bytes
   * @param inputType type of uncompressed data
   * @param chunkSize size of an LZ4 chunk in bytes
   * @param tempPtr device address of the temporary compression storage buffer
   * @param tempSize size of the temporary storage buffer in bytes
   * @param outPtr device address of the output buffer
   * @param outSize size of the output buffer in bytes
   * @param stream CUDA stream to use
   * @return size of the compressed output in bytes
   */
  static native long lz4Compress(
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
   * Perform LZ4 compression synchronously using the specified CUDA
   * stream.
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
   * Extracts the metadata from the batch of inputs on the device and
   * copies them to the host. This synchronizes on the stream.
   * @param inPtrs device addresses of the compressed buffers in the batch
   * @param inSizes corresponding byte sizes of the compressed buffers
   * @param stream CUDA stream to use
   * @return handle to the batched decompress metadata on the host
   */
  static native long batchedLZ4DecompressGetMetadata(
      long[] inPtrs,
      long[] inSizes,
      long stream);

  /**
   * Destroys batched metadata and frees the underlying host memory.
   * @param batchedMetadata handle to the batched decompress metadata
   */
  static native void batchedLZ4DecompressDestroyMetadata(long batchedMetadata);

  /**
   * Computes the temporary storage size in bytes needed to decompress
   * the compressed batch.
   * @param batchedMetadata handle to the batched metadata
   * @return number of temporary storage bytes needed to decompress the batch
   */
  static native long batchedLZ4DecompressGetTempSize(long batchedMetadata);

  /**
   * Computes the decompressed size of each chunk in the batch.
   * @param batchedMetadata handle to the batched metadata
   * @param numOutputs number of output buffers in the batch
   * @return Array of corresponding output sizes needed to decompress
   * each buffer in the batch.
   */
  static native long[] batchedLZ4DecompressGetOutputSize(
      long batchedMetadata,
      long numOutputs);

  /**
   * Asynchronously decompress a batch of compressed data buffers.
   * @param inPtrs device addresses of the compressed buffers
   * @param inSizes corresponding byte sizes of the compressed buffers
   * @param tempPtr device address of the temporary decompression space
   * @param tempSize size of the temporary decompression space in bytes
   * @param batchedMetadata handle to the batched metadata
   * @param outPtrs device addresses of the uncompressed output buffers
   * @param outSizes corresponding byte sizes of the uncompressed output buffers
   * @param stream CUDA stream to use
   */
  static native void batchedLZ4DecompressAsync(
      long[] inPtrs,
      long[] inSizes,
      long tempPtr,
      long tempSize,
      long batchedMetadata,
      long[] outPtrs,
      long[] outSizes,
      long stream);

  /**
   * Get the temporary workspace size required to perform compression of entire batch.
   * @param inPtrs device addresses of the uncompressed buffers
   * @param inSizes corresponding byte sizes of the uncompressed buffers
   * @param chunkSize size of an LZ4 chunk in bytes
   * @return The size of required temporary workspace in bytes to compress the batch.
   */
  static native long batchedLZ4CompressGetTempSize(
      long[] inPtrs,
      long[] inSizes,
      long chunkSize);

  /**
   * Get the required output sizes of each chunk to perform compression.
   * @param inPtrs device addresses of the uncompressed buffers
   * @param inSizes corresponding byte sizes of the uncompressed buffers
   * @param chunkSize size of an LZ4 chunk in bytes
   * @param tempPtr device address of the temporary workspace buffer
   * @param tempSize size of the temporary workspace buffer in bytes
   * @return array of corresponding sizes in bytes of the output buffers needed
   * to compress the buffers in the batch.
   */
  static native long[] batchedLZ4CompressGetOutputSize(
      long[] inPtrs,
      long[] inSizes,
      long chunkSize,
      long tempPtr,
      long tempSize);

  /**
   * Asynchronously compress a batch of buffers. Note that
   * compressedSizesOutPtr must point to pinned memory for this operation
   * to be asynchronous.
   * @param compressedSizesOutPtr host address where to write the sizes of the
   *                              compressed data written to the corresponding
   *                              output buffers. Must point to a buffer with
   *                              at least 8 bytes of memory per output buffer
   *                              in the batch. For asynchronous operation
   *                              this must point to pinned host memory.
   * @param inPtrs device addresses of the uncompressed buffers
   * @param inSizes corresponding byte sizes of the uncompressed buffers
   * @param chunkSize size of an LZ4 chunk in bytes
   * @param tempPtr device address of the temporary workspace buffer
   * @param tempSize size of the temporary workspace buffer in bytes
   * @param outPtrs device addresses of the output compressed buffers
   * @param outSizes corresponding sizes in bytes of the output buffers
   * @param stream CUDA stream to use
   */
  static native void batchedLZ4CompressAsync(
      long compressedSizesOutPtr,
      long[] inPtrs,
      long[] inSizes,
      long chunkSize,
      long tempPtr,
      long tempSize,
      long[] outPtrs,
      long[] outSizes,
      long stream);

  /**
   * Calculate the temporary buffer size needed for cascaded compression.
   * @param inPtr device address of the uncompressed data
   * @param inSize size of the uncompressed data in bytes
   * @param inputType type of uncompressed data
   * @param numRLEs number of Run Length Encoding layers to use
   * @param numDeltas number of delta layers to use
   * @param useBitPacking set to true if bit-packing should be used
   * @return number of temporary storage bytes needed to compress
   */
  static native long cascadedCompressGetTempSize(
      long inPtr,
      long inSize,
      int inputType,
      int numRLEs,
      int numDeltas,
      boolean useBitPacking);

  /**
   * Calculate the output buffer size for cascaded compression. The output
   * size can be an estimated upper bound or the exact value.
   * @param inPtr device address of the uncompressed data
   * @param inSize size of the uncompressed data in bytes
   * @param inputType type of uncompressed data
   * @param numRLEs number of Run Length Encoding layers to use
   * @param numDeltas number of delta layers to use
   * @param useBitPacking set to true if bit-packing should be used
   * @param tempPtr device address of the temporary compression storage buffer
   * @param tempSize size of the temporary storage buffer in bytes
   * @param computeExactSize set to true to compute the exact output size
   * @return output buffer size in bytes. If computeExactSize is true then
   * this is an exact size otherwise it is a maximum, worst-case size of the
   * compressed data.
   */
  static native long cascadedCompressGetOutputSize(
      long inPtr,
      long inSize,
      int inputType,
      int numRLEs,
      int numDeltas,
      boolean useBitPacking,
      long tempPtr,
      long tempSize,
      boolean computeExactSize);

  /**
   * Perform cascaded compression synchronously using the specified CUDA
   * stream.
   * @param inPtr device address of the uncompressed data
   * @param inSize size of the uncompressed data in bytes
   * @param inputType type of uncompressed data
   * @param numRLEs number of Run Length Encoding layers to use
   * @param numDeltas number of delta layers to use
   * @param useBitPacking set to true if bit-packing should be used
   * @param tempPtr device address of the temporary compression storage buffer
   * @param tempSize size of the temporary storage buffer in bytes
   * @param outPtr device address of the output buffer
   * @param outSize size of the output buffer in bytes
   * @param stream CUDA stream to use
   * @return size of the compressed output in bytes
   */
  static native long cascadedCompress(
      long inPtr,
      long inSize,
      int inputType,
      int numRLEs,
      int numDeltas,
      boolean useBitPacking,
      long tempPtr,
      long tempSize,
      long outPtr,
      long outSize,
      long stream);

  /**
   * Perform cascaded compression asynchronously using the specified CUDA
   * stream. Note if the compressedSizeOutputPtr argument points to paged
   * memory then this may synchronize in practice due to limitations of
   * copying from the device to paged memory.
   * @param compressedSizeOutputPtr address of a 64-bit integer to update with
   *                                the resulting compressed size of the data.
   *                                For the operation to be truly asynchronous
   *                                this should point to pinned host memory.
   * @param inPtr device address of the uncompressed data
   * @param inSize size of the uncompressed data in bytes
   * @param inputType type of uncompressed data
   * @param numRLEs number of Run Length Encoding layers to use
   * @param numDeltas number of delta layers to use
   * @param useBitPacking set to true if bit-packing should be used
   * @param tempPtr device address of the temporary compression storage buffer
   * @param tempSize size of the temporary storage buffer in bytes
   * @param outPtr device address of the output buffer
   * @param outSize size of the output buffer in bytes
   * @param stream CUDA stream to use
   */
  static native void cascadedCompressAsync(
      long compressedSizeOutputPtr,
      long inPtr,
      long inSize,
      int inputType,
      int numRLEs,
      int numDeltas,
      boolean useBitPacking,
      long tempPtr,
      long tempSize,
      long outPtr,
      long outSize,
      long stream);
}
