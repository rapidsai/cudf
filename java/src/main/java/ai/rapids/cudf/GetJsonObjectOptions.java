/*
 *
 *  Copyright (c) 2024, NVIDIA CORPORATION.
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

public class GetJsonObjectOptions {
    private boolean allowSingleQuotes;
    private boolean stripQuotesFromSingleStrings;
    private boolean missingFieldsAsNulls;

    public GetJsonObjectOptions(boolean allowSingleQuotes, boolean stripQuotesFromSingleStrings, boolean missingFieldsAsNulls) {
        this.allowSingleQuotes = allowSingleQuotes;
        this.stripQuotesFromSingleStrings = stripQuotesFromSingleStrings;
        this.missingFieldsAsNulls = missingFieldsAsNulls;
    }

    public boolean isAllowSingleQuotes() {
        return allowSingleQuotes;
    }

    public void setAllowSingleQuotes(boolean allowSingleQuotes) {
        this.allowSingleQuotes = allowSingleQuotes;
    }

    public boolean isStripQuotesFromSingleStrings() {
        return stripQuotesFromSingleStrings;
    }

    public void setStripQuotesFromSingleStrings(boolean stripQuotesFromSingleStrings) {
        this.stripQuotesFromSingleStrings = stripQuotesFromSingleStrings;
    }

    public boolean isMissingFieldsAsNulls() {
        return missingFieldsAsNulls;
    }

    public void setMissingFieldsAsNulls(boolean missingFieldsAsNulls) {
        this.missingFieldsAsNulls = missingFieldsAsNulls;
    }
}
