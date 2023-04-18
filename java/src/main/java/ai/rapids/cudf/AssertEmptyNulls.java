package ai.rapids.cudf;

public class AssertEmptyNulls {
  // This is the recommended way to check at runtime that assertions are enabled. 
  // https://docs.oracle.com/javase/8/docs/technotes/guides/language/assert.html
  private static boolean ALLOW_NON_EMPTY_NULLS = true;
  static {
    try {
      assert (false); // Intentional side effect
    } catch (AssertionError ae) {
      ALLOW_NON_EMPTY_NULLS = false;
    }
  }

  public static void assertNullsAreEmpty(ColumnView cv) {
    if (cv.type.isNestedType() || cv.type.hasOffsets() && !ALLOW_NON_EMPTY_NULLS) {
      assert !cv.hasNonEmptyNulls() : "Column has non-empty nulls";
    }
  }
}
