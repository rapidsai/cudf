/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Provides JNI wrappers for reading Parquet files with deletion vector support.
 *
 * Deletion vectors are used in Delta Lake and other table formats to track deleted rows
 * without physically rewriting data files. This class provides APIs to read Parquet files
 * while applying deletion vectors using 64-bit roaring bitmap serialization format.
 */
public class DeltaLake {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private static long[] getAddrsAndSizes(HostMemoryBuffer... buffers) {
    assert buffers.length > 0;
    long[] addrsSizes = new long[buffers.length * 2];
    for (int i = 0; i < buffers.length; i++) {
      addrsSizes[i * 2] = buffers[i].getAddress();
      addrsSizes[(i * 2) + 1] = buffers[i].getLength();
    }
    return addrsSizes;
  }

  /**
   * Information holder for a deletion vector associated with a Parquet file to read.
   */
  public static class DeletionVectorInfo {
    /**
     * Serialized 64-bit roaring bitmap in portable format representing the deletion vector.
     */
    public final HostMemoryBuffer serializedBitmap;

    /**
     * Row index offsets for each row group to read. Can be null to read all row groups.
     */
    public final long[] rowGroupOffsets;

    /**
     * Number of rows in each row group to read. Can be null to read all row groups.
     */
    public final int[] rowGroupNumRows;

    /**
     * Total number of rows to read from the Parquet file associated with this deletion vector.
     */
    public final int totalNumRows;

    public DeletionVectorInfo(HostMemoryBuffer serializedBitmap, long[] rowGroupOffsets, int[] rowGroupNumRows) {
      this.serializedBitmap = serializedBitmap;
      this.rowGroupOffsets = rowGroupOffsets;
      this.rowGroupNumRows = rowGroupNumRows;
      this.totalNumRows = computeTotalNumRows();

      if (rowGroupOffsets == null && rowGroupNumRows != null || rowGroupOffsets != null && rowGroupNumRows == null) {
        throw new IllegalArgumentException("rowGroupOffsets and rowGroupNumRows must both be null or non-null.");
      }
      if (rowGroupOffsets != null && rowGroupOffsets.length != rowGroupNumRows.length) {
        throw new IllegalArgumentException("rowGroupOffsets and rowGroupNumRows must have the same length.");
      }
    }

    private int computeTotalNumRows() {
      if (rowGroupNumRows == null || rowGroupNumRows.length == 0) {
        return Integer.MAX_VALUE;
      }
      final int sum = Arrays.stream(rowGroupNumRows).sum();
      if (sum < 0) {
        throw new IllegalStateException("Integer overflow when summing row group num rows.");
      }
      return sum;
    }
  }

  /**
   * Reads a Parquet file with deletion vector support.
   *
   * Reads a Parquet file, prepends an index column to the table, and applies the deletion vector
   * filter. If row group metadata is not provided, the index column will be a simple sequence
   * from 0 to the number of rows. If the deletion vector is null or empty, the table with the
   * prepended index column is returned as-is without filtering.
   *
   * @param opts ParquetOptions
   * @param dataBuffers Array of HostMemoryBuffers containing the Parquet file data.
   * @param deletionVectorInfos Array of DeletionVectorInfo objects representing deletion vectors
   *                            for each Parquet file to read.
   * @return A Table containing the filtered data with a prepended UINT64 index column.
   */
  public static Table readDeltaParquet(ParquetOptions opts,
                                  HostMemoryBuffer[] dataBuffers,
                                  DeletionVectorInfo[] deletionVectorInfos) {
    long[] dataBufferAddrsSizes = getAddrsAndSizes(dataBuffers);
    List<HostMemoryBuffer> serializedBitmapList = new ArrayList<>(deletionVectorInfos.length);
    List<Integer> deletionVectorRowCountsList = new ArrayList<>(deletionVectorInfos.length);
    List<Long> rowGroupOffsetsList = new ArrayList<>(deletionVectorInfos.length);
    List<Integer> rowGroupNumRowsList = new ArrayList<>(deletionVectorInfos.length);
    if (deletionVectorInfos != null) {
      for (DeletionVectorInfo info : deletionVectorInfos) {
        serializedBitmapList.add(info.serializedBitmap);
        deletionVectorRowCountsList.add(info.totalNumRows);
        if (info.rowGroupOffsets != null) {
          // rowGroupNumRows should also be non-null as per DeletionVectorInfo constructor check
          for (long offset : info.rowGroupOffsets) {
            rowGroupOffsetsList.add(offset);
          }
          for (int numRows : info.rowGroupNumRows) {
            rowGroupNumRowsList.add(numRows);
          }
        }
      }
    }
    long[] bitmapAddrsSizes = getAddrsAndSizes(serializedBitmapList.toArray(new HostMemoryBuffer[0]));
    int[] deletionVectorRowCounts = deletionVectorRowCountsList.stream().mapToInt(Integer::intValue).toArray();
    long[] rowGroupOffsets = rowGroupOffsetsList.stream().mapToLong(Long::longValue).toArray();
    int[] rowGroupNumRows = rowGroupNumRowsList.stream().mapToInt(Integer::intValue).toArray();
    long[] columnHandles = readDeltaParquet(opts.getIncludeColumnNames(),
                                            opts.getReadBinaryAsString(),
                                            null,
                                            dataBufferAddrsSizes,
                                            opts.timeUnit().typeId.getNativeId(),
                                            bitmapAddrsSizes,
                                            deletionVectorRowCounts,
                                            rowGroupOffsets,
                                            rowGroupNumRows);
    return new Table(columnHandles);
  }

  public static class ParquetChunkedReader implements AutoCloseable {

    /**
     * Handle to the native Parquet chunked reader.
     */
    private long readerHandle;

    /**
     * Handle to the native multi-host buffer source.
     */
    private long multiHostBufferSourceHandle;

    /**
     * Handle to the native deletion vector param.
     */
    private long deletionVectorParamHandle;

    /**
     * Auxiliary variable to help {@link #hasNext()} returning true at least once.
     */
    private boolean firstCall = true;

    /**
     * Construct the reader instance from a read limit and data in host memory buffers.
     *
     * @param chunkSizeByteLimit Limit on total number of bytes to be returned per read,
     *                           or 0 if there is no limit.
     * @param passReadLimit Limit on the amount of memory used for reading and decompressing data or
     *                      0 if there is no limit
     * @param opts The options for Parquet reading.
     * @param dataBuffers Array of HostMemoryBuffers containing the Parquet file data.
     * @param deletionVectorInfos Array of DeletionVectorInfo objects representing deletion vectors
     *                            for each Parquet file to read.
     */
    public ParquetChunkedReader(long chunkSizeByteLimit, long passReadLimit,
                                ParquetOptions opts,
                                HostMemoryBuffer[] dataBuffers,
                                DeletionVectorInfo[] deletionVectorInfos) {
      long[] addrsSizes = getAddrsAndSizes(dataBuffers);
      List<HostMemoryBuffer> serializedBitmapList = new ArrayList<>(deletionVectorInfos.length);
      List<Integer> deletionVectorRowCountsList = new ArrayList<>(deletionVectorInfos.length);
      List<Long> rowGroupOffsetsList = new ArrayList<>(deletionVectorInfos.length);
      List<Integer> rowGroupNumRowsList = new ArrayList<>(deletionVectorInfos.length);
      if (deletionVectorInfos != null) {
        for (DeletionVectorInfo info : deletionVectorInfos) {
          serializedBitmapList.add(info.serializedBitmap);
          deletionVectorRowCountsList.add(info.totalNumRows);
          if (info.rowGroupOffsets != null) {
            // rowGroupNumRows should also be non-null as per DeletionVectorInfo constructor check
            for (long offset : info.rowGroupOffsets) {
              rowGroupOffsetsList.add(offset);
            }
            for (int numRows : info.rowGroupNumRows) {
              rowGroupNumRowsList.add(numRows);
            }
          }
        }
      }
      long[] bitmapAddrsSizes = getAddrsAndSizes(serializedBitmapList.toArray(new HostMemoryBuffer[0]));
      int[] deletionVectorRowCounts = deletionVectorRowCountsList.stream().mapToInt(Integer::intValue).toArray();
      long[] rowGroupOffsets = rowGroupOffsetsList.stream().mapToLong(Long::longValue).toArray();
      int[] rowGroupNumRows = rowGroupNumRowsList.stream().mapToInt(Integer::intValue).toArray();
      long[] handles = createDeltaParquetChunkedReader(chunkSizeByteLimit, passReadLimit,
        opts.getIncludeColumnNames(), opts.getReadBinaryAsString(), null,
          addrsSizes, opts.timeUnit().typeId.getNativeId(),
          bitmapAddrsSizes, deletionVectorRowCounts, rowGroupOffsets, rowGroupNumRows);
      readerHandle = handles[0];
      if (readerHandle == 0) {
        throw new IllegalStateException("Cannot create native chunked Parquet reader object.");
      }
      multiHostBufferSourceHandle = handles[1];
      if (multiHostBufferSourceHandle == 0) {
        throw new IllegalStateException("Cannot create native multi-host buffer source object.");
      }
      deletionVectorParamHandle = handles[2];
      if (deletionVectorParamHandle == 0) {
        throw new IllegalStateException("Cannot create native deletion vector param object.");
      }
    }

    /**
     * Check if the given file has anything left to read.
     *
     * @return A boolean value indicating if there is more data to read from file.
     */
    public boolean hasNext() {
      if (readerHandle == 0) {
        throw new IllegalStateException("Native chunked Parquet reader object may have been closed.");
      }

      if (firstCall) {
        // This function needs to return true at least once, so an empty table
        // (but having empty columns instead of no column) can be returned by readChunk()
        // if the input file has no row.
        firstCall = false;
        return true;
      }
      return deltaParquetChunkedReaderHasNext(readerHandle);
    }

    /**
     * Read a chunk of rows in the given Parquet file such that the returning data has total size
     * does not exceed the given read limit. If the given file has no data, or all data has been read
     * before by previous calls to this function, a null Table will be returned.
     *
     * @return A table of new rows reading from the given file.
     */
    public Table readChunk() {
      if (readerHandle == 0) {
        throw new IllegalStateException("Native chunked Parquet reader object may have been closed.");
      }

      long[] columnPtrs = deltaParquetChunkedReaderReadChunk(readerHandle);
      return columnPtrs != null ? new Table(columnPtrs) : null;
    }

    @Override
    public void close() throws Exception {
      try (
        AutoCloseable closeable = () -> {
          if (readerHandle != 0) {
            closeDeltaParquetChunkedReader(readerHandle);
            readerHandle = 0;
          }
        };
        AutoCloseable closeable2 = () -> {
          if (multiHostBufferSourceHandle != 0) {
            destroyMultiHostBufferSource(multiHostBufferSourceHandle);
            multiHostBufferSourceHandle = 0;
          }
        };
        AutoCloseable closeable3 = () -> {
          if (deletionVectorParamHandle != 0) {
            destroyDeletionVectorParam(deletionVectorParamHandle);
            deletionVectorParamHandle = 0;
          }
        };) {
      }
    }
  }

  // Native methods

  private static native long[] readDeltaParquet(String[] filterColumnNames,
                                                boolean[] binaryToString,
                                                String filePath,
                                                long[] addrsAndSizes,
                                                int timeUnit,
                                                long[] serializedRoaring64,
                                                int[] deletionVectorRowCounts,
                                                long[] rowGroupOffsets,
                                                int[] rowGroupNumRows)
    throws CudfException;

  private static native long[] createDeltaParquetChunkedReader(long chunkReadLimit,
                                                               long passReadLimit,
                                                               String[] filterColumnNames,
                                                               boolean[] binaryToString,
                                                               String filePath,
                                                               long[] addrsAndSizes,
                                                               int timeUnit,
                                                               long[] serializedRoaringBitmaps,
                                                               int[] deletionVectorRowCounts,
                                                               long[] rowGroupOffsets,
                                                               int[] rowGroupNumRows)
    throws CudfException;

  private static native boolean deltaParquetChunkedReaderHasNext(long readerHandle) throws CudfException;

  private static native long[] deltaParquetChunkedReaderReadChunk(long readerHandle) throws CudfException;

  private static native void closeDeltaParquetChunkedReader(long readerHandle) throws CudfException;

  private static native void destroyMultiHostBufferSource(long handle);

  private static native void destroyDeletionVectorParam(long handle);
}
