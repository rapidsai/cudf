/*
 *
 *  Copyright (c) 2022, NVIDIA CORPORATION.
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
 * Settings for export IPC message
 */
public class IPCWriterOptions extends WriterOptions {
  private IPCWriterOptions(Builder builder) {
    super(builder);
  }

  public static class Builder extends WriterBuilder<Builder> {
    @Override
    public Builder withColumnNames(String... columnNames) {
      return super.withColumnNames(columnNames);
    }

    @Override
    public Builder withNotNullableColumnNames(String... columnNames) {
      return super.withNotNullableColumnNames(columnNames);
    }

    public IPCWriterOptions build() {
      return new IPCWriterOptions(this);
    }
  }

  public static Builder builder() {
    return new Builder();
  }
};
