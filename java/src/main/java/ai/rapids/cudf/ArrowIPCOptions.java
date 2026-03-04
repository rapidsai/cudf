/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
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
