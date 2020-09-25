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

import ai.rapids.cudf.*;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Optional;

public class NvcompTest {
  private static final Logger log = LoggerFactory.getLogger(ColumnVector.class);

  @Test
  void testLZ4RoundTripSync() {
    lz4RoundTrip(false);
  }

  @Test
  void testLZ4RoundTripAsync() {
    lz4RoundTrip(true);
  }

  @Test
  void testBatchedLZ4RoundTripAsync() {
    final long chunkSize = 64 * 1024;
    final int maxElements = 1024 * 1024 + 1;
    final int numBuffers = 200;
    long[] data = new long[maxElements];
    for (int i = 0; i < maxElements; ++i) {
      data[i] = i;
    }

    DeviceMemoryBuffer[] originalBuffers = new DeviceMemoryBuffer[numBuffers];
    DeviceMemoryBuffer[] uncompressedBuffers = new DeviceMemoryBuffer[numBuffers];

    // compressed data in buffers that are likely oversized
    DeviceMemoryBuffer[] compressedBuffers = new DeviceMemoryBuffer[numBuffers];

    // compressed data in right-sized buffers
    DeviceMemoryBuffer[] compressedInputs = new DeviceMemoryBuffer[numBuffers];

    try {
      // create the batched buffers to compress
      for (int i = 0; i < numBuffers; ++i) {
        originalBuffers[i] = initBatchBuffer(data, i);
      }

      // compress the buffers
      long[] outputSizes;
      long[] compressedSizes;
      long tempSize = BatchedLZ4Compressor.getTempSize(originalBuffers, chunkSize);
      try (DeviceMemoryBuffer tempBuffer = DeviceMemoryBuffer.allocate(tempSize)) {
        outputSizes = BatchedLZ4Compressor.getOutputSizes(originalBuffers, chunkSize, tempBuffer);
        for (int i = 0; i < numBuffers; ++i) {
          compressedBuffers[i] = DeviceMemoryBuffer.allocate(outputSizes[i]);
        }
        long sizesBufferSize = BatchedLZ4Compressor.getCompressedSizesBufferSize(numBuffers);
        try (HostMemoryBuffer compressedSizesBuffer = HostMemoryBuffer.allocate(sizesBufferSize)) {
          BatchedLZ4Compressor.compressAsync(compressedSizesBuffer, originalBuffers, chunkSize,
              tempBuffer, compressedBuffers, Cuda.DEFAULT_STREAM);
          Cuda.DEFAULT_STREAM.sync();
          compressedSizes = new long[numBuffers];
          for (int i = 0; i < numBuffers; ++i) {
            compressedSizes[i] = compressedSizesBuffer.getLong(i * 8);
          }
        }
      }

      // right-size the compressed buffers based on reported compressed sizes
      for (int i = 0; i < numBuffers; ++i) {
        compressedInputs[i] = compressedBuffers[i].slice(0, compressedSizes[i]);
      }

      // decompress the buffers
      try (BatchedLZ4Decompressor.BatchedMetadata metadata =
               BatchedLZ4Decompressor.getMetadata(compressedInputs, Cuda.DEFAULT_STREAM)) {
        outputSizes = BatchedLZ4Decompressor.getOutputSizes(metadata, numBuffers);
        for (int i = 0; i < numBuffers; ++i) {
          uncompressedBuffers[i] = DeviceMemoryBuffer.allocate(outputSizes[i]);
        }
        tempSize = BatchedLZ4Decompressor.getTempSize(metadata);
        try (DeviceMemoryBuffer tempBuffer = DeviceMemoryBuffer.allocate(tempSize)) {
          BatchedLZ4Decompressor.decompressAsync(compressedInputs, tempBuffer, metadata,
              uncompressedBuffers, Cuda.DEFAULT_STREAM);
        }
      }

      // check the decompressed results against the original
      for (int i = 0; i < numBuffers; ++i) {
        try (HostMemoryBuffer expected = HostMemoryBuffer.allocate(originalBuffers[i].getLength());
             HostMemoryBuffer actual = HostMemoryBuffer.allocate(outputSizes[i])) {
          Assertions.assertTrue(expected.getLength() <= Integer.MAX_VALUE);
          Assertions.assertTrue(actual.getLength() <= Integer.MAX_VALUE);
          Assertions.assertEquals(originalBuffers[i].getLength(), uncompressedBuffers[i].getLength(),
              "uncompressed size mismatch at buffer " + i);
          expected.copyFromDeviceBuffer(originalBuffers[i]);
          actual.copyFromDeviceBuffer(uncompressedBuffers[i]);
          byte[] expectedBytes = new byte[(int) expected.getLength()];
          expected.getBytes(expectedBytes, 0, 0, expected.getLength());
          byte[] actualBytes = new byte[(int) actual.getLength()];
          actual.getBytes(actualBytes, 0, 0, actual.getLength());
          Assertions.assertArrayEquals(expectedBytes, actualBytes,
              "mismatch in batch buffer " + i);
        }
      }
    } finally {
      closeBufferArray(originalBuffers);
      closeBufferArray(uncompressedBuffers);
      closeBufferArray(compressedBuffers);
      closeBufferArray(compressedInputs);
    }
  }

  @Test
  void testBatchedLZ4CompressRoundTrip() {
    final long chunkSize = 64 * 1024;
    final int maxElements = 1024 * 1024 + 1;
    final int numBuffers = 200;
    long[] data = new long[maxElements];
    for (int i = 0; i < maxElements; ++i) {
      data[i] = i;
    }

    DeviceMemoryBuffer[] originalBuffers = new DeviceMemoryBuffer[numBuffers];
    DeviceMemoryBuffer[] uncompressedBuffers = new DeviceMemoryBuffer[numBuffers];
    BatchedLZ4Compressor.BatchedCompressionResult compResult = null;

    // compressed data in right-sized buffers
    DeviceMemoryBuffer[] compressedInputs = new DeviceMemoryBuffer[numBuffers];

    try {
      // create the batched buffers to compress
      for (int i = 0; i < numBuffers; ++i) {
        originalBuffers[i] = initBatchBuffer(data, i);
      }

      // compress the buffers
      compResult = BatchedLZ4Compressor.compress(originalBuffers, chunkSize, Cuda.DEFAULT_STREAM);

      // right-size the compressed buffers based on reported compressed sizes
      DeviceMemoryBuffer[] compressedBuffers = compResult.getCompressedBuffers();
      long[] compressedSizes = compResult.getCompressedSizes();
      for (int i = 0; i < numBuffers; ++i) {
        compressedInputs[i] = compressedBuffers[i].slice(0, compressedSizes[i]);
      }

      // decompress the buffers
      uncompressedBuffers = BatchedLZ4Decompressor.decompressAsync(compressedInputs,
              Cuda.DEFAULT_STREAM);

      // check the decompressed results against the original
      for (int i = 0; i < numBuffers; ++i) {
        try (HostMemoryBuffer expected = HostMemoryBuffer.allocate(originalBuffers[i].getLength());
             HostMemoryBuffer actual = HostMemoryBuffer.allocate(uncompressedBuffers[i].getLength())) {
          Assertions.assertTrue(expected.getLength() <= Integer.MAX_VALUE);
          Assertions.assertTrue(actual.getLength() <= Integer.MAX_VALUE);
          Assertions.assertEquals(originalBuffers[i].getLength(), uncompressedBuffers[i].getLength(),
                  "uncompressed size mismatch at buffer " + i);
          expected.copyFromDeviceBuffer(originalBuffers[i]);
          actual.copyFromDeviceBuffer(uncompressedBuffers[i]);
          byte[] expectedBytes = new byte[(int) expected.getLength()];
          expected.getBytes(expectedBytes, 0, 0, expected.getLength());
          byte[] actualBytes = new byte[(int) actual.getLength()];
          actual.getBytes(actualBytes, 0, 0, actual.getLength());
          Assertions.assertArrayEquals(expectedBytes, actualBytes,
                  "mismatch in batch buffer " + i);
        }
      }
    } finally {
      closeBufferArray(originalBuffers);
      closeBufferArray(uncompressedBuffers);
      closeBufferArray(compressedInputs);
      if (compResult != null) {
        closeBufferArray(compResult.getCompressedBuffers());
      }
    }
  }

  private void closeBuffer(MemoryBuffer buffer) {
    if (buffer != null) {
      buffer.close();
    }
  }

  private void closeBufferArray(MemoryBuffer[] buffers) {
    for (MemoryBuffer buffer : buffers) {
      if (buffer != null) {
        buffer.close();
      }
    }
  }

  private DeviceMemoryBuffer initBatchBuffer(long[] data, int bufferId) {
    // grab a subsection of the data based on buffer ID
    int dataStart = 0;
    int dataLength = data.length / (bufferId + 1);
    switch (bufferId % 3) {
      case 0:
        // take a portion of the first half
        dataLength /= 2;
        break;
      case 1:
        // take a portion of the last half
        dataStart = data.length / 2;
        dataLength /= 2;
        break;
      default:
        break;
    }
    long[] bufferData = Arrays.copyOfRange(data, dataStart, dataStart + dataLength + 1);
    DeviceMemoryBuffer devBuffer = null;
    try (HostMemoryBuffer hmb = HostMemoryBuffer.allocate(bufferData.length * 8)) {
      hmb.setLongs(0, bufferData, 0, bufferData.length);
      devBuffer = DeviceMemoryBuffer.allocate(hmb.getLength());
      devBuffer.copyFromHostBuffer(hmb);
      return devBuffer;
    } catch (Throwable t) {
      closeBuffer(devBuffer);
      throw new RuntimeException(t);
    }
  }

  private void lz4RoundTrip(boolean useAsync) {
    final long chunkSize = 64 * 1024;
    final int numElements = 10 * 1024 * 1024 + 1;
    long[] data = new long[numElements];
    for (int i = 0; i < numElements; ++i) {
      data[i] = i;
    }

    DeviceMemoryBuffer tempBuffer = null;
    DeviceMemoryBuffer compressedBuffer = null;
    DeviceMemoryBuffer uncompressedBuffer = null;
    try (ColumnVector v = ColumnVector.fromLongs(data)) {
      BaseDeviceMemoryBuffer inputBuffer = v.getDeviceBufferFor(BufferType.DATA);
      log.debug("Uncompressed size is {}", inputBuffer.getLength());

      long tempSize = LZ4Compressor.getTempSize(inputBuffer, CompressionType.CHAR, chunkSize);

      log.debug("Using {} temporary space for lz4 compression", tempSize);
      tempBuffer = DeviceMemoryBuffer.allocate(tempSize);

      long outSize = LZ4Compressor.getOutputSize(inputBuffer, CompressionType.CHAR, chunkSize,
          tempBuffer);
      log.debug("lz4 compressed size estimate is {}", outSize);

      compressedBuffer = DeviceMemoryBuffer.allocate(outSize);

      long startTime = System.nanoTime();
      long compressedSize;
      if (useAsync) {
        try (HostMemoryBuffer tempHostBuffer = HostMemoryBuffer.allocate(8)) {
          LZ4Compressor.compressAsync(tempHostBuffer, inputBuffer, CompressionType.CHAR, chunkSize,
              tempBuffer, compressedBuffer, Cuda.DEFAULT_STREAM);
          Cuda.DEFAULT_STREAM.sync();
          compressedSize = tempHostBuffer.getLong(0);
        }
      } else {
        compressedSize = LZ4Compressor.compress(inputBuffer, CompressionType.CHAR, chunkSize,
            tempBuffer, compressedBuffer, Cuda.DEFAULT_STREAM);
      }
      double duration = (System.nanoTime() - startTime) / 1000.0;
      log.info("Compressed with lz4 to {} in {} us", compressedSize, duration);

      tempBuffer.close();
      tempBuffer = null;

      Assertions.assertTrue(Decompressor.isLZ4Data(compressedBuffer));

      try (Decompressor.Metadata metadata =
               Decompressor.getMetadata(compressedBuffer, Cuda.DEFAULT_STREAM)) {
        Assertions.assertTrue(metadata.isLZ4Metadata());
        tempSize = Decompressor.getTempSize(metadata);

        log.debug("Using {} temporary space for lz4 compression", tempSize);
        tempBuffer = DeviceMemoryBuffer.allocate(tempSize);

        outSize = Decompressor.getOutputSize(metadata);
        Assertions.assertEquals(inputBuffer.getLength(), outSize);

        uncompressedBuffer = DeviceMemoryBuffer.allocate(outSize);

        Decompressor.decompressAsync(compressedBuffer, tempBuffer, metadata, uncompressedBuffer,
            Cuda.DEFAULT_STREAM);

        try (ColumnVector v2 = new ColumnVector(
            DType.INT64,
            numElements,
            Optional.empty(),
            uncompressedBuffer,
            null,
            null);
             HostColumnVector hv2 = v2.copyToHost()) {
          uncompressedBuffer = null;
          for (int i = 0; i < numElements; ++i) {
            long val = hv2.getLong(i);
            if (val != i) {
              Assertions.fail("Expected " + i + " at " + i + " found " + val);
            }
          }
        }
      }
    } finally {
      closeBuffer(tempBuffer);
      closeBuffer(compressedBuffer);
      closeBuffer(uncompressedBuffer);
    }
  }

  @Test
  void testCascadedRoundTripSync() {
    cascadedRoundTrip(false);
  }

  @Test
  void testCascadedRoundTripAsync() {
    cascadedRoundTrip(true);
  }

  private void cascadedRoundTrip(boolean useAsync) {
    final int numElements = 10 * 1024 * 1024 + 1;
    final int numRunLengthEncodings = 2;
    final int numDeltas = 1;
    final boolean useBitPacking = true;
    int[] data = new int[numElements];
    for (int i = 0; i < numElements; ++i) {
      data[i] = i;
    }

    DeviceMemoryBuffer tempBuffer = null;
    DeviceMemoryBuffer compressedBuffer = null;
    DeviceMemoryBuffer uncompressedBuffer = null;
    try (ColumnVector v = ColumnVector.fromInts(data)) {
      BaseDeviceMemoryBuffer inputBuffer = v.getDeviceBufferFor(BufferType.DATA);
      log.debug("Uncompressed size is " + inputBuffer.getLength());

      long tempSize = NvcompJni.cascadedCompressGetTempSize(
          inputBuffer.getAddress(),
          inputBuffer.getLength(),
          CompressionType.INT.nativeId,
          numRunLengthEncodings,
          numDeltas,
          useBitPacking);

      log.debug("Using {} temporary space for cascaded compression", tempSize);
      tempBuffer = DeviceMemoryBuffer.allocate(tempSize);

      long outSize = NvcompJni.cascadedCompressGetOutputSize(
          inputBuffer.getAddress(),
          inputBuffer.getLength(),
          CompressionType.INT.nativeId,
          numRunLengthEncodings,
          numDeltas,
          useBitPacking,
          tempBuffer.getAddress(),
          tempBuffer.getLength(),
          false);
      log.debug("Inexact cascaded compressed size estimate is {}", outSize);

      compressedBuffer = DeviceMemoryBuffer.allocate(outSize);

      long startTime = System.nanoTime();
      long compressedSize;
      if (useAsync) {
        try (HostMemoryBuffer tempHostBuffer = HostMemoryBuffer.allocate(8)) {
          NvcompJni.cascadedCompressAsync(
              tempHostBuffer.getAddress(),
              inputBuffer.getAddress(),
              inputBuffer.getLength(),
              CompressionType.INT.nativeId,
              numRunLengthEncodings,
              numDeltas,
              useBitPacking,
              tempBuffer.getAddress(),
              tempBuffer.getLength(),
              compressedBuffer.getAddress(),
              compressedBuffer.getLength(),
              0);
          Cuda.DEFAULT_STREAM.sync();
          compressedSize = tempHostBuffer.getLong(0);
        }
      } else {
        compressedSize = NvcompJni.cascadedCompress(
            inputBuffer.getAddress(),
            inputBuffer.getLength(),
            CompressionType.INT.nativeId,
            numRunLengthEncodings,
            numDeltas,
            useBitPacking,
            tempBuffer.getAddress(),
            tempBuffer.getLength(),
            compressedBuffer.getAddress(),
            compressedBuffer.getLength(),
            0);
      }

      double duration = (System.nanoTime() - startTime) / 1000.0;
      log.debug("Compressed with cascaded to {} in {} us", compressedSize, duration);

      tempBuffer.close();
      tempBuffer = null;

      try (Decompressor.Metadata metadata =
               Decompressor.getMetadata(compressedBuffer, Cuda.DEFAULT_STREAM)) {
        tempSize = Decompressor.getTempSize(metadata);

        log.debug("Using {} temporary space for cascaded compression", tempSize);
        tempBuffer = DeviceMemoryBuffer.allocate(tempSize);

        outSize = Decompressor.getOutputSize(metadata);
        Assertions.assertEquals(inputBuffer.getLength(), outSize);

        uncompressedBuffer = DeviceMemoryBuffer.allocate(outSize);

        Decompressor.decompressAsync(compressedBuffer, tempBuffer, metadata, uncompressedBuffer,
            Cuda.DEFAULT_STREAM);

        try (ColumnVector v2 = new ColumnVector(
            DType.INT32,
            numElements,
            Optional.empty(),
            uncompressedBuffer,
            null,
            null)) {
          uncompressedBuffer = null;
          try (ColumnVector compare = v2.equalTo(v);
               Scalar compareAll = compare.all()) {
            Assertions.assertTrue(compareAll.getBoolean());
          }
        }
      }
    } finally {
      closeBuffer(tempBuffer);
      closeBuffer(compressedBuffer);
      closeBuffer(uncompressedBuffer);
    }
  }
}
