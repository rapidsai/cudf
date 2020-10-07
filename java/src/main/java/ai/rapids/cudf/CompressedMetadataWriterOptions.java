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

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

class CompressedMetadataWriterOptions extends WriterOptions {

  private final CompressionType compressionType;
  private final Map<String, String> metadata;

  <T extends CMWriterBuilder> CompressedMetadataWriterOptions(T builder) {
    super(builder);
    compressionType = builder.compressionType;
    metadata = Collections.unmodifiableMap(builder.metadata);
  }

  public CompressionType getCompressionType() {
    return compressionType;
  }

  public Map<String, String> getMetadata() {
    return metadata;
  }

  String[] getMetadataKeys() {
    return metadata.keySet().toArray(new String[metadata.size()]);
  }

  String[] getMetadataValues() {
    return metadata.values().toArray(new String[metadata.size()]);
  }

  protected static class CMWriterBuilder<T extends CMWriterBuilder> extends WriterBuilder<T> {
    final Map<String, String> metadata = new LinkedHashMap<>();
    CompressionType compressionType = CompressionType.AUTO;

    /**
     * Add a metadata key and a value
     * @param key
     * @param value
     */
    public T withMetadata(String key, String value) {
      this.metadata.put(key, value);
      return (T) this;
    }

    /**
     * Add a map of metadata keys and values
     * @param metadata
     */
    public T withMetadata(Map<String, String> metadata) {
      this.metadata.putAll(metadata);
      return (T) this;
    }

    /**
     * Set the compression type to use for writing
     * @param compression
     */
    public T withCompressionType(CompressionType compression) {
      this.compressionType = compression;
      return (T) this;
    }
  }
}
