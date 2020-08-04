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
      if (tempBuffer != null) {
        tempBuffer.close();
      }
      if (compressedBuffer != null) {
        compressedBuffer.close();
      }
      if (uncompressedBuffer != null) {
        uncompressedBuffer.close();
      }
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
      if (tempBuffer != null) {
        tempBuffer.close();
      }
      if (compressedBuffer != null) {
        compressedBuffer.close();
      }
      if (uncompressedBuffer != null) {
        uncompressedBuffer.close();
      }
    }
  }
}
