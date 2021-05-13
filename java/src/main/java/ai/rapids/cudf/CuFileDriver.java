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
 * Represents a cuFile driver.
 */
final class CuFileDriver implements AutoCloseable {
  private final CuFileResourceCleaner cleaner;

  CuFileDriver() {
    cleaner = new CuFileResourceCleaner(create(), CuFileDriver::destroy);
    MemoryCleaner.register(this, cleaner);
  }

  @Override
  public void close() {
    cleaner.close(this);
  }

  private static native long create();

  private static native void destroy(long pointer);
}
