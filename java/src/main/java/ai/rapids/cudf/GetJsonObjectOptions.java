package ai.rapids.cudf;

public class GetJsonObjectOptions {
    private boolean allowSingleQuotes;
    private boolean stripQuotesFromSingleStrings;
    private boolean missingFieldsAsNulls;

    // Constructor with parameters to set boolean values
    public GetJsonObjectOptions(boolean allowSingleQuotes, boolean stripQuotesFromSingleStrings, boolean missingFieldsAsNulls) {
        this.allowSingleQuotes = allowSingleQuotes;
        this.stripQuotesFromSingleStrings = stripQuotesFromSingleStrings;
        this.missingFieldsAsNulls = missingFieldsAsNulls;
    }

    public GetJsonObjectOptions() {
        this(false, true, false); // Calls parameterized constructor with default values
    }

    // Getter and setter methods for allowSingleQuotes
    public boolean isAllowSingleQuotes() {
        return allowSingleQuotes;
    }

    public void setAllowSingleQuotes(boolean allowSingleQuotes) {
        this.allowSingleQuotes = allowSingleQuotes;
    }

    // Getter and setter methods for stripQuotesFromSingleStrings
    public boolean isStripQuotesFromSingleStrings() {
        return stripQuotesFromSingleStrings;
    }

    public void setStripQuotesFromSingleStrings(boolean stripQuotesFromSingleStrings) {
        this.stripQuotesFromSingleStrings = stripQuotesFromSingleStrings;
    }

    // Getter and setter methods for missingFieldsAsNulls
    public boolean isMissingFieldsAsNulls() {
        return missingFieldsAsNulls;
    }

    public void setMissingFieldsAsNulls(boolean missingFieldsAsNulls) {
        this.missingFieldsAsNulls = missingFieldsAsNulls;
    }
}
