/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
  private static CuFileDriver driver;

  static {
    initialize();
  }

  /**
   * Load the native libraries needed for libcufilejni, if not loaded already; open the cuFile
   * driver, and add a shutdown hook to close it.
   */
  static synchronized void initialize() {
    if (!initialized) {
      try {
        NativeDepsLoader.loadNativeDeps(new String[]{"cufilejni"});
        driver = new CuFileDriver();
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
          driver.close();
        }));
        initialized = true;
      } catch (Throwable t) {
        // Cannot throw an exception here as the CI/CD machine may not have GDS installed.
        log.error("Could not load cuFile jni library...", t);
      }
    }
  }

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
   */
  public static void writeDeviceBufferToFile(File path, long file_offset,
                                             BaseDeviceMemoryBuffer buffer) {
    writeDeviceMemoryToFile(path, file_offset, buffer.getAddress(), buffer.getLength());
  }

  /**
   * Write device memory to a given file path synchronously.
   * <p>
   * This method is NOT thread safe if the path points to the same file on disk.
   *
   * @param path        The file path to copy to.
   * @param file_offset The file offset from which to write the buffer.
   * @param address     The device memory address to copy from.
   * @param length      The length to copy.
   */
  public static void writeDeviceMemoryToFile(File path, long file_offset, long address,
                                             long length) {
    writeToFile(path.getAbsolutePath(), file_offset, address, length);
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
    return appendDeviceMemoryToFile(path, buffer.getAddress(), buffer.getLength());
  }

  /**
   * Append device memory to a given file path synchronously.
   * <p>
   * This method is NOT thread safe if the path points to the same file on disk.
   *
   * @param path    The file path to copy to.
   * @param address The device memory address to copy from.
   * @param length  The length to copy.
   * @return The file offset from which the buffer was appended.
   */
  public static long appendDeviceMemoryToFile(File path, long address, long length) {
    return appendToFile(path.getAbsolutePath(), address, length);
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
    readFileToDeviceMemory(buffer.getAddress(), buffer.getLength(), path, fileOffset);
  }

  /**
   * Read a file into device memory synchronously.
   * <p>
   * This method is NOT thread safe if the path points to the same file on disk.
   *
   * @param address The device memory address to read into.
   * @param length  The length to read.
   * @param path       The file path to copy from.
   * @param fileOffset The file offset from which to copy the content.
   */
  public static void readFileToDeviceMemory(long address, long length, File path, long fileOffset) {
    readFromFile(address, length, path.getAbsolutePath(), fileOffset);
  }

  private static native void writeToFile(String path, long file_offset, long address, long length);

  private static native long appendToFile(String path, long address, long length);

  private static native void readFromFile(long address, long length, String path, long fileOffset);
}
