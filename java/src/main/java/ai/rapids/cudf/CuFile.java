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

package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * JNI wrapper for accessing the cuFile API.
 * <p>
 * Using this wrapper requires GPUDirect Storage (GDS)/cuFile to be installed in the target
 * environment, and the jar to be built with `USE_GDS=ON`. Otherwise it will throw an exception when
 * loading.
 * <p>
 * The Java APIs are experimental and subject to change.
 *
 * @see <a href="https://docs.nvidia.com/gpudirect-storage/">GDS documentation</a>
 */
public class CuFile {
  private static final Logger log = LoggerFactory.getLogger(CuFile.class);
  private static boolean initialized = false;
  private static long driverPointer = 0;

  static {
    initialize();
  }

  /**
   * Load the native libraries needed for libcufilejni, if not loaded already; open the cuFile
   * driver, and add a shutdown hook to close it.
   */
  private static synchronized void initialize() {
    if (!initialized) {
      try {
        NativeDepsLoader.loadNativeDeps(new String[]{"cufilejni"});
        driverPointer = createDriver();
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
          destroyDriver(driverPointer);
        }));
        initialized = true;
      } catch (Throwable t) {
        log.error("Could not load cuFile jni library...", t);
      }
    }
  }

  private static native long createDriver();

  private static native void destroyDriver(long pointer);

  /**
   * Check if the libcufilejni library is loaded.
   *
   * @return true if the libcufilejni library has been successfully loaded.
   */
  public static boolean libraryLoaded() {
    return initialized;
  }

  /**
   * Write a device buffer to a given file path synchronously.
   * <p>
   * This method is NOT thread safe if the path points to the same file on disk.
   *
   * @param path        The file path to copy to.
   * @param file_offset The file offset from which to write the buffer.
   * @param buffer      The device buffer to copy from.
   * @return The file offset from which the buffer was appended.
   */
  public static void writeDeviceBufferToFile(File path, long file_offset,
                                             BaseDeviceMemoryBuffer buffer) {
    writeToFile(path.getAbsolutePath(), file_offset, buffer.getAddress(), buffer.getLength());
  }

  /**
   * Append a device buffer to a given file path synchronously.
   * <p>
   * This method is NOT thread safe if the path points to the same file on disk.
   *
   * @param path   The file path to copy to.
   * @param buffer The device buffer to copy from.
   * @return The file offset from which the buffer was appended.
   */
  public static long appendDeviceBufferToFile(File path, BaseDeviceMemoryBuffer buffer) {
    return appendToFile(path.getAbsolutePath(), buffer.getAddress(), buffer.getLength());
  }

  /**
   * Read a file into a device buffer synchronously.
   * <p>
   * This method is NOT thread safe if the path points to the same file on disk.
   *
   * @param buffer     The device buffer to copy into.
   * @param path       The file path to copy from.
   * @param fileOffset The file offset from which to copy the content.
   */
  public static void readFileToDeviceBuffer(BaseDeviceMemoryBuffer buffer, File path,
                                            long fileOffset) {
    readFromFile(buffer.getAddress(), buffer.getLength(), path.getAbsolutePath(), fileOffset);
  }

  private static native void writeToFile(String path, long file_offset, long address, long length);

  private static native long appendToFile(String path, long address, long length);

  private static native void readFromFile(long address, long length, String path, long fileOffset);
}
