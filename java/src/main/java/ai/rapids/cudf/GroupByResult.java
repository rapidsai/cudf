package ai.rapids.cudf;

/**
 * Used by JNI
 * Used to save groups and keys for `contiguousSplitGroupsAndGenUniqKeys`
 * Each row in uniq keys table is corresponding to a group
 */
public class GroupByResult {
    // set by JNI cpp code
    // should be closed by caller
    private ContiguousTable[] groups;

    // set by JNI cpp code, used to construct an uniq key Table
    private long[] uniqKeyColumns;

    // Each row in uniq keys table is corresponding to a group
    // should be closed by caller
    private Table uniqKeysTable;

    // generate uniq keys table
    void genUniqKeysTable() {
        if (uniqKeysTable == null && uniqKeyColumns != null && uniqKeyColumns.length > 0) {
            // new `Table` asserts uniqKeyColumns.length > 0
            uniqKeysTable = new Table(uniqKeyColumns);
        }
    }

    public Table getUniqKeyTable() {
        return uniqKeysTable;
    }

    public ContiguousTable[] getGroups() {
        return groups;
    }
}
