package ai.rapids.cudf;

public class AssertEmptyNulls {
  /*
  We have to read the system property to see if we should by-pass this assert. Ideally we should
  just use `-da:AssertNonEmptyNulls` as per Java standard, but the reason for doing this is because
  surefire-plugin doesn't expose anyway to disable assertions at the class level or package level
  instead we can only disable asserts at the ClassLoader level. Therefore, in order for the tests to
  pass we need to set a flag from the pom to know we are running unit-tests to allow non-empty nulls
 */
  private static final boolean ALLOW_NON_EMPTY_NULLS =
      Boolean.getBoolean("ai.rapids.allow.nonempty.nulls");
  public static void assertHasEmptyNulls(ColumnView cv) {
    if (!ALLOW_NON_EMPTY_NULLS) {
      assert !cv.hasNonEmptyNulls() : "Column has non-empty nulls";
    }
  }
}
