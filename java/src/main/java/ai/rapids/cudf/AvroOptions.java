/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * Options for reading an Avro file
 */
public class AvroOptions extends ColumnFilterOptions {

    public static AvroOptions DEFAULT = new AvroOptions(new Builder());

    private AvroOptions(Builder builder) {
        super(builder);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder extends ColumnFilterOptions.Builder<Builder> {
        public AvroOptions build() {
            return new AvroOptions(this);
        }
    }
}
