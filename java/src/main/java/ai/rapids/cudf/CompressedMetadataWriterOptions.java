/*
 * SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
