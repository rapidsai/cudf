/*
 *
 *  Copyright (c) 2023, NVIDIA CORPORATION.
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
