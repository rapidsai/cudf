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

  public static long copyDeviceBufferToFile(File path, DeviceMemoryBuffer buffer, boolean append) {
    return copyToFile(path.getAbsolutePath(), buffer.getAddress(), buffer.getLength(), append);
  }

  public static void copyFileToDeviceBuffer(DeviceMemoryBuffer buffer, File path, long fileOffset) {
    copyFromFile(buffer.getAddress(), buffer.getLength(), path.getAbsolutePath(), fileOffset);
  }

  private static native long copyToFile(String path, long address, long length, boolean append);

  private static native void copyFromFile(long address, long length, String path, long fileOffset);
}
