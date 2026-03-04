/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */
package ai.rapids.cudf;

/**
 * Quote style for CSV records, closely following cudf::io::quote_style.
 */
public enum QuoteStyle {
    MINIMAL(0),    // Quote only fields which contain special characters
    ALL(1),        // Quote all fields
    NONNUMERIC(2), // Quote all non-numeric fields
    NONE(3);       // Never quote fields; disable quotation parsing

    final int nativeId; // Native id, for use with libcudf.
    QuoteStyle(int nativeId) {
        this.nativeId = nativeId;
    }
}
