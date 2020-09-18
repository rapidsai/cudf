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

import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.BaseDeviceMemoryBuffer;
import ai.rapids.cudf.MemoryCleaner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Generic single-buffer decompressor interface */
public class Decompressor {
  private static final Logger log = LoggerFactory.getLogger(Decompressor.class);

  public static Metadata getMetadata(BaseDeviceMemoryBuffer buffer, Cuda.Stream stream) {
    long metadata = NvcompJni.decompressGetMetadata(buffer.getAddress(), buffer.getLength(),
        stream.getStream());
    return new Metadata(metadata);
  }

  public static long getTempSize(Metadata metadata) {
    return NvcompJni.decompressGetTempSize(metadata.getMetadata());
  }

  public static long getOutputSize(Metadata metadata) {
    return NvcompJni.decompressGetOutputSize(metadata.getMetadata());
  }

  public static void decompressAsync(BaseDeviceMemoryBuffer input, BaseDeviceMemoryBuffer tempBuffer,
      Metadata metadata, BaseDeviceMemoryBuffer output, Cuda.Stream stream) {
    NvcompJni.decompressAsync(
        input.getAddress(), input.getLength(),
        tempBuffer.getAddress(), tempBuffer.getLength(),
        metadata.getMetadata(),
        output.getAddress(), output.getLength(),
        stream.getStream());
  }

  public static boolean isLZ4Data(BaseDeviceMemoryBuffer buffer) {
    return NvcompJni.isLZ4Data(buffer.getAddress(), buffer.getLength());
  }


  /** Opaque metadata object for single-buffer decompression */
  public static class Metadata implements AutoCloseable {
    private final MetadataCleaner cleaner;
    private final long id;
    private boolean closed = false;

    Metadata(long metadata) {
      this.cleaner = new MetadataCleaner(metadata);
      this.id = cleaner.id;
      MemoryCleaner.register(this, cleaner);
      cleaner.addRef();
    }

    long getMetadata() {
      return cleaner.metadata;
    }

    public boolean isLZ4Metadata() {
      return NvcompJni.isLZ4Metadata(getMetadata());
    }

    @Override
    public void close() {
      if (!closed) {
        cleaner.delRef();
      } else {
        cleaner.logRefCountDebug("double free " + this);
        throw new IllegalStateException("Close called too many times " + this);
      }
    }

    @Override
    public String toString() {
      return "DECOMPRESSOR METADATA (ID: " + id + " " +
          Long.toHexString(cleaner.metadata) + ")";
    }

    private static class MetadataCleaner extends MemoryCleaner.Cleaner {
      private long metadata;

      MetadataCleaner(long metadata) {
        this.metadata = metadata;
      }

      @Override
      protected boolean cleanImpl(boolean logErrorIfNotClean) {
        boolean neededCleanup = false;
        long address = metadata;
        if (metadata != 0) {
          try {
            NvcompJni.decompressDestroyMetadata(metadata);
          } finally {
            // Always mark the resource as freed even if an exception is thrown.
            // We cannot know how far it progressed before the exception, and
            // therefore it is unsafe to retry.
            metadata = 0;
          }
          neededCleanup = true;
        }
        if (neededCleanup && logErrorIfNotClean) {
          log.error("DECOMPRESSOR METADATA WAS LEAKED (Address: " +
              Long.toHexString(address) + ")");
          logRefCountDebug("Leaked event");
        }
        return neededCleanup;
      }

      @Override
      public boolean isClean() {
        return metadata != 0;
      }
    }
  }
}
