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
    final int maxElements = 1 * 1024 * 1024 + 1;
    final int numBuffers = 200;
    long[] data = new long[maxElements];
    for (int i = 0; i < maxElements; ++i) {
      data[i] = i;
    }

    DeviceMemoryBuffer[] originalBuffers = new DeviceMemoryBuffer[numBuffers];
    DeviceMemoryBuffer[] uncompressedBuffers = new DeviceMemoryBuffer[numBuffers];
    DeviceMemoryBuffer[] compressedBuffers = new DeviceMemoryBuffer[numBuffers];
    try {
      // create the batched buffers to compress
      for (int i = 0; i < numBuffers; ++i) {
        originalBuffers[i] = initBatchBuffer(data, i);
      }

      // compress the buffers
      long[] inputAddrs = new long[numBuffers];
      long[] inputSizes = new long[numBuffers];
      for (int i = 0; i < numBuffers; ++i) {
        inputAddrs[i] = originalBuffers[i].getAddress();
        inputSizes[i] = originalBuffers[i].getLength();
      }
      long[] outputAddrs = new long[numBuffers];
      long[] outputSizes = null;
      long[] compressedSizes = new long[numBuffers];
      long tempSize = Nvcomp.batchedLZ4CompressGetTempSize(inputAddrs, inputSizes, chunkSize);
      try (DeviceMemoryBuffer tempBuffer = DeviceMemoryBuffer.allocate(tempSize)) {
        outputSizes = Nvcomp.batchedLZ4CompressGetOutputSize(inputAddrs, inputSizes, chunkSize,
                tempBuffer.getAddress(), tempBuffer.getLength());
        for (int i = 0; i < numBuffers; ++i) {
          compressedBuffers[i] = DeviceMemoryBuffer.allocate(outputSizes[i]);
          outputAddrs[i] = compressedBuffers[i].getAddress();
        }
        try (HostMemoryBuffer compressedSizesBuffer = HostMemoryBuffer.allocate(8 * numBuffers)) {
          Nvcomp.batchedLZ4CompressAsync(
              compressedSizesBuffer.getAddress(),
              inputAddrs,
              inputSizes,
              chunkSize,
              tempBuffer.getAddress(),
              tempBuffer.getLength(),
              outputAddrs,
              outputSizes,
              0);
          Cuda.DEFAULT_STREAM.sync();
          for (int i = 0; i < numBuffers; ++i) {
            compressedSizes[i] = compressedSizesBuffer.getLong(i * 8);
          }
        }
      }

      // decompress the buffers
      for (int i = 0; i < numBuffers; ++i) {
        inputAddrs[i] = compressedBuffers[i].getAddress();
      }
      inputSizes = compressedSizes;
      long metadata = Nvcomp.batchedLZ4DecompressGetMetadata(inputAddrs, inputSizes, 0);
      try {
        outputSizes = Nvcomp.batchedLZ4DecompressGetOutputSize(metadata, numBuffers);
        for (int i = 0; i < numBuffers; ++i) {
          uncompressedBuffers[i] = DeviceMemoryBuffer.allocate(outputSizes[i]);
          outputAddrs[i] = uncompressedBuffers[i].getAddress();
        }
        tempSize = Nvcomp.batchedLZ4DecompressGetTempSize(metadata);
        try (DeviceMemoryBuffer tempBuffer = DeviceMemoryBuffer.allocate(tempSize)) {
          Nvcomp.batchedLZ4DecompressAsync(
              inputAddrs,
              inputSizes,
              tempBuffer.getAddress(),
              tempBuffer.getLength(),
              metadata,
              outputAddrs,
              outputSizes,
              0);
        }
      } finally {
        Nvcomp.batchedLZ4DecompressDestroyMetadata(metadata);
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
    final long chunk_size = 64 * 1024;
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

      long tempSize = Nvcomp.lz4CompressGetTempSize(
          inputBuffer.getAddress(),
          inputBuffer.getLength(),
          CompressionType.CHAR.nativeId,
          chunk_size);

      log.debug("Using {} temporary space for lz4 compression", tempSize);
      tempBuffer = DeviceMemoryBuffer.allocate(tempSize);

      long outSize = Nvcomp.lz4CompressGetOutputSize(
          inputBuffer.getAddress(),
          inputBuffer.getLength(),
          CompressionType.CHAR.nativeId,
          chunk_size,
          tempBuffer.getAddress(),
          tempBuffer.getLength(),
          false);
      log.debug("Inexact lz4 compressed size estimate is {}", outSize);

      compressedBuffer = DeviceMemoryBuffer.allocate(outSize);

      long startTime = System.nanoTime();
      long compressedSize;
      if (useAsync) {
        try (HostMemoryBuffer tempHostBuffer = HostMemoryBuffer.allocate(8)) {
          Nvcomp.lz4CompressAsync(
              tempHostBuffer.getAddress(),
              inputBuffer.getAddress(),
              inputBuffer.getLength(),
              CompressionType.CHAR.nativeId,
              chunk_size,
              tempBuffer.getAddress(),
              tempBuffer.getLength(),
              compressedBuffer.getAddress(),
              compressedBuffer.getLength(),
              0);
          Cuda.DEFAULT_STREAM.sync();
          compressedSize = tempHostBuffer.getLong(0);
        }
      } else {
        compressedSize = Nvcomp.lz4Compress(
            inputBuffer.getAddress(),
            inputBuffer.getLength(),
            CompressionType.CHAR.nativeId,
            chunk_size,
            tempBuffer.getAddress(),
            tempBuffer.getLength(),
            compressedBuffer.getAddress(),
            compressedBuffer.getLength(),
            0);
      }
      double duration = (System.nanoTime() - startTime) / 1000.0;
      log.info("Compressed with lz4 to {} in {} us", compressedSize, duration);

      tempBuffer.close();
      tempBuffer = null;

      Assertions.assertTrue(Nvcomp.isLZ4Data(compressedBuffer.getAddress(), compressedSize));

      long metadata = Nvcomp.decompressGetMetadata(
          compressedBuffer.getAddress(),
          compressedBuffer.getLength(),
          0);
      try {
        Assertions.assertTrue(Nvcomp.isLZ4Metadata(metadata));
        tempSize = Nvcomp.decompressGetTempSize(metadata);

        log.debug("Using {} temporary space for lz4 compression", tempSize);
        tempBuffer = DeviceMemoryBuffer.allocate(tempSize);

        outSize = Nvcomp.decompressGetOutputSize(metadata);
        Assertions.assertEquals(inputBuffer.getLength(), outSize);

        uncompressedBuffer = DeviceMemoryBuffer.allocate(outSize);

        Nvcomp.decompressAsync(
            compressedBuffer.getAddress(),
            compressedSize,
            tempBuffer.getAddress(),
            tempBuffer.getLength(),
            metadata,
            uncompressedBuffer.getAddress(),
            uncompressedBuffer.getLength(),
            0);

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
      } finally {
        Nvcomp.decompressDestroyMetadata(metadata);
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

      long tempSize = Nvcomp.cascadedCompressGetTempSize(
          inputBuffer.getAddress(),
          inputBuffer.getLength(),
          CompressionType.INT.nativeId,
          numRunLengthEncodings,
          numDeltas,
          useBitPacking);

      log.debug("Using {} temporary space for cascaded compression", tempSize);
      tempBuffer = DeviceMemoryBuffer.allocate(tempSize);

      long outSize = Nvcomp.cascadedCompressGetOutputSize(
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
          Nvcomp.cascadedCompressAsync(
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
        compressedSize = Nvcomp.cascadedCompress(
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

      long metadata = Nvcomp.decompressGetMetadata(
          compressedBuffer.getAddress(),
          compressedBuffer.getLength(),
          0);
      try {
        tempSize = Nvcomp.decompressGetTempSize(metadata);

        log.debug("Using {} temporary space for cascaded compression", tempSize);
        tempBuffer = DeviceMemoryBuffer.allocate(tempSize);

        outSize = Nvcomp.decompressGetOutputSize(metadata);
        Assertions.assertEquals(inputBuffer.getLength(), outSize);

        uncompressedBuffer = DeviceMemoryBuffer.allocate(outSize);

        Nvcomp.decompressAsync(
            compressedBuffer.getAddress(),
            compressedSize,
            tempBuffer.getAddress(),
            tempBuffer.getLength(),
            metadata,
            uncompressedBuffer.getAddress(),
            uncompressedBuffer.getLength(),
            0);

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
      } finally {
        Nvcomp.decompressDestroyMetadata(metadata);
      }
    } finally {
      closeBuffer(tempBuffer);
      closeBuffer(compressedBuffer);
      closeBuffer(uncompressedBuffer);
    }
  }
}
