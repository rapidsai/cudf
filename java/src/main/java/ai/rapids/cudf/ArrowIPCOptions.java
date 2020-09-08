/*
 *
 *  Copyright (c) 2020, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package ai.rapids.cudf;

/**
 * Options for reading data in Arrow IPC format
 */
public class ArrowIPCOptions {

  public interface NeedGpu {
    /**
     * A callback to indicate that we are about to start putting data on the GPU.
     */
    void needTheGpu();
  }

  public static ArrowIPCOptions DEFAULT = new ArrowIPCOptions(new Builder());

  private final NeedGpu callback;

  private ArrowIPCOptions(Builder builder) {
    this.callback = builder.callback;
  }

  public NeedGpu getCallback() {
    return callback;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {
    private NeedGpu callback = () -> {};

    public Builder withCallback(NeedGpu callback) {
      if (callback == null) {
        this.callback = () -> {};
      } else {
        this.callback = callback;
      }
      return this;
    }

    public ArrowIPCOptions build() {
      return new ArrowIPCOptions(this);
    }
  }
}
