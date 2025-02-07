/*
 *
 *  Copyright (c) 2021, NVIDIA CORPORATION.
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

import java.util.LinkedHashMap;
import java.util.Map;

public class CompressionMetadataWriterOptions extends ColumnWriterOptions.StructColumnWriterOptions {
  private final CompressionType compressionType;
  private final Map<String, String> metadata;

  protected CompressionMetadataWriterOptions(Builder builder) {
    super(builder);
    this.compressionType = builder.compressionType;
    this.metadata = builder.metadata;
  }

  @Override
  boolean[] getFlatIsTimeTypeInt96() {
    return super.getFlatBooleans(new boolean[]{}, (opt) -> opt.getFlatIsTimeTypeInt96());
  }

  @Override
  int[] getFlatPrecision() {
    return super.getFlatInts(new int[]{}, (opt) -> opt.getFlatPrecision());
  }

  @Override
  boolean[] getFlatHasParquetFieldId() {
    return super.getFlatBooleans(new boolean[]{}, (opt) -> opt.getFlatHasParquetFieldId());
  }

  @Override
  int[] getFlatParquetFieldId() {
    return super.getFlatInts(new int[]{}, (opt) -> opt.getFlatParquetFieldId());
  }

  @Override
  int[] getFlatNumChildren() {
    return super.getFlatInts(new int[]{}, (opt) -> opt.getFlatNumChildren());
  }

  @Override
  boolean[] getFlatIsNullable() {
    return super.getFlatBooleans(new boolean[]{}, (opt) -> opt.getFlatIsNullable());
  }

  @Override
  boolean[] getFlatIsMap() {
    return super.getFlatBooleans(new boolean[]{}, (opt) -> opt.getFlatIsMap());
  }

  @Override
  boolean[] getFlatIsBinary() {
    return super.getFlatBooleans(new boolean[]{}, (opt) -> opt.getFlatIsBinary());
  }

  @Override
  String[] getFlatColumnNames() {
    return super.getFlatColumnNames(new String[]{});
  }

  String[] getMetadataKeys() {
    return metadata.keySet().toArray(new String[metadata.size()]);
  }

  String[] getMetadataValues() {
    return metadata.values().toArray(new String[metadata.size()]);
  }

  public CompressionType getCompressionType() {
    return compressionType;
  }

  public Map<String, String> getMetadata() {
    return metadata;
  }

  public int getTopLevelChildren() {
    return childColumnOptions.length;
  }

  public abstract static class Builder<T extends Builder,
        V extends CompressionMetadataWriterOptions> extends AbstractStructBuilder<T, V> {
    final Map<String, String> metadata = new LinkedHashMap<>();
    CompressionType compressionType = CompressionType.AUTO;

    /**
     * Add a metadata key and a value
     */
    public T withMetadata(String key, String value) {
      this.metadata.put(key, value);
      return (T) this;
    }

    /**
     * Add a map of metadata keys and values
     */
    public T withMetadata(Map<String, String> metadata) {
      this.metadata.putAll(metadata);
      return (T) this;
    }

    /**
     * Set the compression type to use for writing
     */
    public T withCompressionType(CompressionType compression) {
      this.compressionType = compression;
      return (T) this;
    }
  }
}
