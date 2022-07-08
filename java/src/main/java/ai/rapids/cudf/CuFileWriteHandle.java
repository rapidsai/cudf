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
public final class CuFileWriteHandle extends CuFileHandle {

  /**
   * Construct a writer using the specified file path.
   *
   * @param path The file path for writing.
   */
  public CuFileWriteHandle(String path) {
    super(create(path));
  }

  /**
   * Write the specified cuFile buffer into the file.
   *
   * @param buffer The cuFile buffer to write from.
   * @param length The number of bytes to write.
   * @param fileOffset The starting file offset from which to write.
   */
  public void write(CuFileBuffer buffer, long length, long fileOffset) {
    writeFromBuffer(getPointer(), fileOffset, buffer.getPointer(), length);
  }

  /**
   * Append the specified cuFile buffer to the file.
   *
   * @param buffer The cuFile buffer to append from.
   * @param length The number of bytes to append.
   * @return The file offset from which the buffer was appended.
   */
  public long append(CuFileBuffer buffer, long length) {
    return appendFromBuffer(getPointer(), buffer.getPointer(), length);
  }

  private static native long create(String path);

  private static native void writeFromBuffer(long file, long fileOffset, long buffer, long length);

  private static native long appendFromBuffer(long file, long buffer, long length);
}
