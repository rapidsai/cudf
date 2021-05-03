/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

/**
 * Represents a cuFile file handle for reading.
 */
public final class CuFileReadHandle extends CuFileHandle {

  /**
   * Construct a reader using the specified file path.
   *
   * @param path The file path for reading.
   */
  public CuFileReadHandle(String path) {
    super(create(path));
  }

  /**
   * Read the file content into the specified cuFile buffer.
   *
   * @param buffer The cuFile buffer to store the content.
   * @param fileOffset The file offset from which to read.
   */
  public void read(CuFileBuffer buffer, long fileOffset) {
    readIntoBuffer(getPointer(), fileOffset, buffer.getPointer());
  }

  private static native long create(String path);

  private static native void readIntoBuffer(long file, long fileOffset, long buffer);
}
