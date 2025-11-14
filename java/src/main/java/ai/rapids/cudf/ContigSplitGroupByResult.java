/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

/**
 * Used to save groups and uniq key table for `Table.contiguousSplitGroupsAndGenUniqKeys`
 * Each row in uniq key table is corresponding to a group
 * Resource management note:
 * This class is the owner of `groups` and
 * `uniqKeysTable`(or uniqKeyColumns if table is not constructed)
 * 1: Use `closeGroups` and `closeUniqKeyTable` to close the resources separately
 * if you want to close eagerly.
 * 2: Or auto close them by `AutoCloseable`
 * Use `releaseGroups` to release the ownership of the `groups` to the caller,
 * then the caller is responsible to close the `groups`
 */
public class ContigSplitGroupByResult implements AutoCloseable {
  // set by JNI cpp code
  private ContiguousTable[] groups;

  // set by JNI cpp code, used to construct an uniq key Table
  private long[] uniqKeyColumns;

  // An additional table is introduced to store the group keys,
  // and each key is corresponding to a group.
  private Table uniqKeysTable;

  /**
   * Get the key table, each row in the key table is corresponding to a group.
   * Note: Close the key table by `closeUniqKeyTable`
   *
   * @return the key table, it could be null if invoking native method `Table.contiguousSplitGroups`
   * with `genUniqKeys` as false
   */
  public Table getUniqKeyTable() {
    if (uniqKeysTable == null && uniqKeyColumns != null && uniqKeyColumns.length > 0) {
      // new `Table` asserts uniqKeyColumns.length > 0
      uniqKeysTable = new Table(uniqKeyColumns);
      uniqKeyColumns = null;
    }
    return uniqKeysTable;
  }

  /**
   * Close the key table or key columns
   */
  public void closeUniqKeyTable() {
    if (uniqKeysTable != null) {
      uniqKeysTable.close();
      uniqKeysTable = null;
    } else if (uniqKeyColumns != null) {
      for (long handle : uniqKeyColumns) {
        ColumnVector.deleteCudfColumn(handle);
      }
      uniqKeyColumns = null;
    }
  }

  /**
   * Get the split group tables.
   * Note: Close the group tables by `closeGroups`
   *
   * @return the split group tables
   */
  public ContiguousTable[] getGroups() {
    return groups;
  }

  /**
   * Release the ownership of the `groups`
   * The caller is responsible to close the returned groups.
   *
   * @return split group tables
   */
  ContiguousTable[] releaseGroups() {
    ContiguousTable[] copy = groups;
    groups = null;
    return copy;
  }

  /**
   * Close the split group tables
   */
  public void closeGroups() {
    if (groups != null) {
      for (ContiguousTable contig : groups) {
        contig.close();
      }
      groups = null;
    }
  }

  @Override
  public void close() {
    try {
      closeUniqKeyTable();
    } finally {
      closeGroups();
    }
  }
}
