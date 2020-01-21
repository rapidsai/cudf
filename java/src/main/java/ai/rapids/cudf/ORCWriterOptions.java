/*
 *
 *  Copyright (c) 2019, NVIDIA CORPORATION.
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

public class ORCWriterOptions {

  enum CompressionType {
    NONE(0),
    SNAPPY(1);

    public final int nativeId;

    CompressionType(int nativeId) {
      this.nativeId = nativeId;
    }
  }

  public static ORCWriterOptions DEFAULT = new ORCWriterOptions(new Builder());

  private final CompressionType compressionType;

  private ORCWriterOptions(Builder builder) {
    compressionType = builder.compressionType;
  }

  public CompressionType getCompressionType() {
    return compressionType;
  }

  public static Builder builder() {
    return new Builder();
  }
  public static class Builder {
    private CompressionType compressionType = CompressionType.SNAPPY;

    /**
     * Specify the compression type to use with this file
     * @return
     */
    public ORCWriterOptions.Builder withCompression(CompressionType compressionType) {
      this.compressionType = compressionType;
      return this;
    }

    public ORCWriterOptions build() {
      return new ORCWriterOptions(this);
    }
  }

}
