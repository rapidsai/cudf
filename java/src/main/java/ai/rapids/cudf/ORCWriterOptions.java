/*
 *
 *  Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
 * This class represents settings for writing ORC files. It includes meta data information
 * that will be used by the ORC writer to write the file.
 */
public class ORCWriterOptions extends CompressionMetadataWriterOptions {

  private ORCWriterOptions(Builder builder) {
    super(builder);
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder extends CompressionMetadataWriterOptions.Builder
          <Builder, ORCWriterOptions> {

    public ORCWriterOptions build() {
      return new ORCWriterOptions(this);
    }
  }
}
