/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
 * A device memory resource that will log interactions.
 */
public class RmmLoggingResourceAdaptor<C extends RmmDeviceMemoryResource>
    extends RmmWrappingDeviceMemoryResource<C> {
  private long handle = 0;

  /**
   * Create a new logging resource adaptor.
   * @param wrapped the memory resource to log interactions with. This should not be reused.
   * @param conf the config of where this should be logged to
   * @param autoFlush should the results be flushed after each entry or not.
   */
  public RmmLoggingResourceAdaptor(C wrapped, Rmm.LogConf conf, boolean autoFlush) {
    super(wrapped);
    if (conf.loc == Rmm.LogLoc.NONE) {
      throw new RmmException("Cannot initialize RmmLoggingResourceAdaptor with no logging");
    }
    handle = Rmm.newLoggingResourceAdaptor(wrapped.getHandle(), conf.loc.internalId,
        conf.file == null ? null : conf.file.getAbsolutePath(), autoFlush);
  }

  @Override
  public long getHandle() {
    return handle;
  }

  @Override
  public void close() {
    if (handle != 0) {
      Rmm.releaseLoggingResourceAdaptor(handle);
      handle = 0;
    }
    super.close();
  }

  @Override
  public String toString() {
    return Long.toHexString(getHandle()) + "/LOG(" + wrapped + ")";
  }
}
