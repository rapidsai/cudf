/*
 *
 *  Copyright (c) 2019, NVIDIA CORPORATION.
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

import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Serialize and deserialize CUDF tables and columns using a custom format.  The goal of this is
 * to provide a way to efficiently serialize and deserialize cudf data for distributed
 * processing within a single application. Typically after a partition like operation has happened.
 * It is not intended for inter-application communication or for long term storage of data, there
 * are much better standards based formats for all of that.
 * <p>
 * The goal is to transfer data from a local GPU to a remote GPU as quickly and efficiently as
 * possible using build in java communication channels.  There is no guarantee of compatibility
 * between different releases of CUDF.  This is to allow us to adapt if internal memory layouts
 * and formats change.
 * <p>
 * This version optimizes for reduced memory transfers, and as such will try to do the fewest number
 * of transfers possible when putting the data back onto the GPU.  This means that it will slice
 * a single large memory buffer into smaller buffers used by the resulting ColumnVectors.  The
 * downside of this is that generally none of the memory can be released until all of the
 * ColumnVectors are closed.  It is assumed that this will not be a problem because for processing
 * efficiency after the data is transferred it will likely be combined with other similar batches
 * from other processes into a single larger buffer.
 * <p>
 * There is a known bug in this where the null count of a range of values is lost, and replaced with
 * a 0 if it is known that there can be no nulls in the data, or a 1 if there is the possibility of
 * a null being in the data.  This is not likely to cause issues if the data is processed using cudf
 * as the null count is only used as a flag to check if a validity buffer is needed or not.
 * Processing outside of cudf should be careful.
 */
public class JCudfSerialization {
  /**
   * Magic number "CUDF" in ASCII, which is 1178883395 if read in LE from big endian, which is
   * too large for any reasonable metadata for arrow, so we should probably be okay detecting
   * this, and switching back/forth at a later time.
   */
  private static final int SER_FORMAT_MAGIC_NUMBER = 0x43554446;
  private static final short VERSION_NUMBER = 0x0000;

  private static final class ColumnOffsets {
    private final long validity;
    private final long validityLen;
    private final long offsets;
    private final long offsetsLen;
    private final long data;
    private final long dataLen;

    public ColumnOffsets(long validity, long validityLen,
                         long offsets, long offsetsLen,
                         long data, long dataLen) {
      this.validity = validity;
      this.validityLen = validityLen;
      this.offsets = offsets;
      this.offsetsLen = offsetsLen;
      this.data = data;
      this.dataLen = dataLen;
    }
  }

  /**
   * Holds the metadata about a serialized table. If this is being read from a stream
   * isInitialized will return true if the metadata was read correctly from the stream.
   * It will return false if an EOF was encountered at the beginning indicating that
   * there was no data to be read.
   */
  public static final class SerializedTableHeader {
    private int numColumns;
    int numRows;

    private DType[] types;
    private long[] nullCounts;
    private TimeUnit[] tu;
    long dataLen;

    private boolean initialized = false;
    private boolean dataRead = false;

    public SerializedTableHeader(DataInputStream din) throws IOException {
      readFrom(din);
    }

    SerializedTableHeader(int numRows, DType[] types, long[] nullCounts, TimeUnit[] tu, long dataLen) {
      this.numRows = numRows;
      if (types != null) {
        numColumns = types.length;
      } else {
        numColumns = 0;
      }
      this.types = types;
      this.nullCounts = nullCounts;
      this.tu = tu;
      this.dataLen = dataLen;
      initialized = true;
      dataRead = true;
    }

    public DType getColumnType(int columnIndex) {
      return types[columnIndex];
    }

    /**
     * Set to true once data is successfully read from a stream by readTableIntoBuffer.
     * @return true if data was read, else false.
     */
    public boolean wasDataRead() {
      return dataRead;
    }

    /**
     * Returns the size of a buffer needed to read data into the stream.
     */
    public long getDataLen() {
      return dataLen;
    }

    /**
     * Returns the number of rows stored in this table.
     */
    public int getNumRows() {
      return numRows;
    }

    /**
     * Returns the number of columns stored in this table
     */
    public int getNumColumns() {
      return numColumns;
    }

    /**
     * Returns true if the metadata for this table was read, else false indicating an EOF was
     * encountered.
     */
    public boolean wasInitialized() {
      return initialized;
    }

    private void readFrom(DataInputStream din) throws IOException {
      try {
        int num = din.readInt();
        if (num != SER_FORMAT_MAGIC_NUMBER) {
          throw new IllegalStateException("THIS DOES NOT LOOK LIKE CUDF SERIALIZED DATA. " +
              "Expected magic number " + SER_FORMAT_MAGIC_NUMBER + " Found " + num);
        }
      } catch (EOFException e) {
        // If we get an EOF at the very beginning don't treat it as an error because we may
        // have finished reading everything...
        return;
      }
      short version = din.readShort();
      if (version != VERSION_NUMBER) {
        throw new IllegalStateException("READING THE WRONG SERIALIZATION FORMAT VERSION FOUND "
            + version + " EXPECTED " + VERSION_NUMBER);
      }
      numColumns = din.readInt();
      numRows = din.readInt();

      types = new DType[numColumns];
      nullCounts = new long[numColumns];
      tu = new TimeUnit[numColumns];
      for (int i = 0; i < numColumns; i++) {
        types[i] = DType.fromNative(din.readInt());
        nullCounts[i] = din.readInt();
        tu[i] = TimeUnit.fromNative(din.readInt());
      }

      dataLen = din.readLong();
      initialized = true;
    }

    public void writeTo(DataWriter dout) throws IOException {
      // Now write out the data
      dout.writeInt(SER_FORMAT_MAGIC_NUMBER);
      dout.writeShort(VERSION_NUMBER);
      dout.writeInt(numColumns);
      dout.writeInt(numRows);

      // Header for each column...
      for (int i = 0; i < numColumns; i++) {
        dout.writeInt(types[i].nativeId);
        dout.writeInt((int) nullCounts[i]);
        dout.writeInt(tu[i].getNativeId());
      }
      dout.writeLong(dataLen);
    }
  }

  /**
   * Visible for testing
   */
  static abstract class ColumnBufferProvider {

    public abstract DType getType();

    public abstract long getNullCount();

    public abstract long getStartStringOffset(long index);

    public abstract long getEndStringOffset(long index);

    public abstract long getRowCount();

    public abstract TimeUnit getTimeUnit();

    public void ensureOnHost() {
      // Noop by default
    }

    public abstract HostMemoryBuffer getHostBufferFor(ColumnVector.BufferType buffType);

    public abstract long getBufferStartOffset(ColumnVector.BufferType buffType);

    public void copyBytesToArray(byte[] dest, int destOffset, ColumnVector.BufferType srcType, long srcOffset, int length) {
      HostMemoryBuffer buff = getHostBufferFor(srcType);
      srcOffset = srcOffset + getBufferStartOffset(srcType);
      buff.getBytes(dest, destOffset, srcOffset, length);
    }
  }

  /**
   * Visible for testing
   */
  static class ColumnProvider extends ColumnBufferProvider {
    private final ColumnVector column;

    ColumnProvider(ColumnVector column) {
      this.column = column;
    }

    @Override
    public DType getType() {
      return column.getType();
    }

    @Override
    public long getNullCount() {
      return column.getNullCount();
    }

    @Override
    public long getStartStringOffset(long index) {
      return column.getStartStringOffset(index);
    }

    @Override
    public long getEndStringOffset(long index) {
      return column.getEndStringOffset(index);
    }

    @Override
    public long getRowCount() {
      return column.getRowCount();
    }

    @Override
    public TimeUnit getTimeUnit() {
      return column.getTimeUnit();
    }

    @Override
    public void ensureOnHost() {
      column.ensureOnHost();
    }

    @Override
    public HostMemoryBuffer getHostBufferFor(ColumnVector.BufferType buffType) {
      return column.getHostBufferFor(buffType);
    }

    @Override
    public long getBufferStartOffset(ColumnVector.BufferType buffType) {
      // All of the buffers start at 0 for this.
      return 0;
    }
  }

  private static class BufferOffsetProvider extends ColumnBufferProvider {
    private final SerializedTableHeader header;
    private final int columnIndex;
    private final ColumnOffsets offsets;
    private final HostMemoryBuffer buffer;

    private BufferOffsetProvider(SerializedTableHeader header,
                                 int columnIndex,
                                 ColumnOffsets offsets,
                                 HostMemoryBuffer buffer) {
      this.header = header;
      this.columnIndex = columnIndex;
      this.offsets = offsets;
      this.buffer = buffer;
    }

    @Override
    public DType getType() {
      return header.types[columnIndex];
    }

    @Override
    public long getNullCount() {
      return header.nullCounts[columnIndex];
    }

    @Override
    public long getRowCount() {
      return header.numRows;
    }

    @Override
    public TimeUnit getTimeUnit() {
      return header.tu[columnIndex];
    }

    @Override
    public HostMemoryBuffer getHostBufferFor(ColumnVector.BufferType buffType) {
      return buffer;
    }

    @Override
    public long getBufferStartOffset(ColumnVector.BufferType buffType) {
      switch (buffType) {
        case DATA:
          return offsets.data;
        case OFFSET:
          return offsets.offsets;
        case VALIDITY:
          return offsets.validity;
        default:
          throw new IllegalArgumentException("Buffer type " + buffType + " is not supported");
      }
    }

    @Override
    public long getStartStringOffset(long index) {
      assert getType() == DType.STRING_CATEGORY || getType() == DType.STRING;
      assert (index >= 0 && index < getRowCount()) : "index is out of range 0 <= " + index + " < " + getRowCount();
      return buffer.getInt(offsets.offsets + (index * 4));
    }

    @Override
    public long getEndStringOffset(long index) {
      assert getType() == DType.STRING_CATEGORY || getType() == DType.STRING;
      assert (index >= 0 && index < getRowCount()) : "index is out of range 0 <= " + index + " < " + getRowCount();
      // The offsets has one more entry than there are rows.
      return buffer.getInt(offsets.offsets + ((index + 1) * 4));
    }
  }

  /**
   * Visible for testing
   */
  static abstract class DataWriter {

    public abstract void writeByte(byte b) throws IOException;

    public abstract void writeShort(short s) throws IOException;

    public abstract void writeInt(int i) throws IOException;

    public abstract void writeLong(long val) throws IOException;

    /**
     * Copy data from src starting at srcOffset and going for len bytes.
     * @param src where to copy from.
     * @param srcOffset offset to start at.
     * @param len amount to copy.
     */
    public abstract void copyDataFrom(HostMemoryBuffer src, long srcOffset, long len)
        throws IOException;

    public void copyDataFrom(ColumnBufferProvider column, ColumnVector.BufferType buffType,
                             long offset, long length) throws IOException {
      HostMemoryBuffer buff = column.getHostBufferFor(buffType);
      long startOffset = column.getBufferStartOffset(buffType);
      copyDataFrom(buff, startOffset + offset, length);
    }

    public void flush() throws IOException {
      // NOOP by default
    }

    public abstract void write(byte[] arr, int offset, int length) throws IOException;
  }

  /**
   * Visible for testing
   */
  static final class DataOutputStreamWriter extends DataWriter {
    private final byte[] arrayBuffer = new byte[1024 * 128];
    private final DataOutputStream dout;

    public DataOutputStreamWriter(DataOutputStream dout) {
      this.dout = dout;
    }

    @Override
    public void writeByte(byte b) throws IOException {
      dout.writeByte(b);
    }

    @Override
    public void writeShort(short s) throws IOException {
      dout.writeShort(s);
    }

    @Override
    public void writeInt(int i) throws IOException {
      dout.writeInt(i);
    }

    @Override
    public void writeLong(long val) throws IOException {
      dout.writeLong(val);
    }

    @Override
    public void copyDataFrom(HostMemoryBuffer src, long srcOffset, long len) throws IOException {
      long dataLeft = len;
      while (dataLeft > 0) {
        int amountToCopy = (int)Math.min(arrayBuffer.length, dataLeft);
        src.getBytes(arrayBuffer, 0, srcOffset, amountToCopy);
        dout.write(arrayBuffer, 0, amountToCopy);
        srcOffset += amountToCopy;
        dataLeft -= amountToCopy;
      }
    }

    @Override
    public void flush() throws IOException {
      dout.flush();
    }

    @Override
    public void write(byte[] arr, int offset, int length) throws IOException {
      dout.write(arr, offset, length);
    }
  }

  private static final class HostDataWriter extends DataWriter {
    private final HostMemoryBuffer buffer;
    private long offset = 0;

    public HostDataWriter(HostMemoryBuffer buffer) {
      this.buffer = buffer;
    }

    @Override
    public void writeByte(byte b) {
      buffer.setByte(offset, b);
      offset += 1;
    }

    @Override
    public void writeShort(short s) {
      buffer.setShort(offset, s);
      offset += 2;
    }

    @Override
    public void writeInt(int i) {
      buffer.setInt(offset, i);
      offset += 4;
    }

    @Override
    public void writeLong(long val) {
      buffer.setLong(offset, val);
      offset += 8;
    }

    @Override
    public void copyDataFrom(HostMemoryBuffer src, long srcOffset, long len) {
      buffer.copyFromHostBuffer(offset, src, srcOffset, len);
      offset += len;
    }

    @Override
    public void write(byte[] arr, int srcOffset, int length) {
      buffer.setBytes(offset, arr, srcOffset, length);
      offset += length;
    }
  }

  /////////////////////////////////////////////
  // METHODS
  /////////////////////////////////////////////


  /////////////////////////////////////////////
  // PADDING FOR ALIGNMENT
  /////////////////////////////////////////////
  private static long padFor64byteAlignment(long orig) {
    return ((orig + 63) / 64) * 64;
  }

  private static long padFor64byteAlignment(DataWriter out, long bytes) throws IOException {
    final long paddedBytes = padFor64byteAlignment(bytes);
    while (paddedBytes > bytes) {
      out.writeByte((byte)0);
      bytes++;
    }
    return paddedBytes;
  }

  /////////////////////////////////////////////
  // SERIALIZED SIZE
  /////////////////////////////////////////////

  private static long getRawStringDataLength(ColumnBufferProvider column, long rowOffset, long numRows) {
    if (numRows <= 0) {
      return 0;
    }
    long start = column.getStartStringOffset(rowOffset);
    long end = column.getEndStringOffset(rowOffset + numRows - 1);
    return end - start;
  }

  private static long getSlicedSerializedDataSizeInBytes(ColumnBufferProvider[] columns, long rowOffset, long numRows) {
    long totalDataSize = 0;
    for (ColumnBufferProvider column: columns) {
      DType type = column.getType();
      if (column.getNullCount() > 0) {
        totalDataSize += padFor64byteAlignment(BitVectorHelper.getValidityLengthInBytes(numRows));
      }
      if (type == DType.STRING || type == DType.STRING_CATEGORY) {
        // offsets
        totalDataSize += padFor64byteAlignment((numRows + 1) * 4);

        // data
        if (numRows > 0) {
          totalDataSize += padFor64byteAlignment(getRawStringDataLength(column, rowOffset, numRows));
        }
      } else {
        totalDataSize += padFor64byteAlignment(column.getType().sizeInBytes * numRows);
      }
    }
    return totalDataSize;
  }

  private static long getConcatedSerializedDataSizeInBytes(int numColumns, long[] nullCounts,
                                                           int numRows, DType[] types,
                                                           ColumnBufferProvider[][] columnsForEachBatch) {
    long totalDataSize = 0;
    for (int col = 0; col < numColumns; col++) {
      DType type = types[col];
      if (nullCounts[col] > 0) {
        totalDataSize += padFor64byteAlignment(BitVectorHelper.getValidityLengthInBytes(numRows));
      }
      if (type == DType.STRING || type == DType.STRING_CATEGORY) {
        // offsets
        totalDataSize += padFor64byteAlignment((numRows + 1) * 4);

        long stringDataLen = 0;
        for (int batchNumber = 0; batchNumber < columnsForEachBatch.length; batchNumber++) {
          ColumnBufferProvider provider = columnsForEachBatch[batchNumber][col];
          long numRowsInSubColumn = provider.getRowCount();
          stringDataLen += getRawStringDataLength(provider, 0, numRowsInSubColumn);
        }
        totalDataSize += padFor64byteAlignment(stringDataLen);
      } else {
        totalDataSize += padFor64byteAlignment(types[col].sizeInBytes * numRows);
      }
    }
    return totalDataSize;
  }

  /**
   * Get the size in bytes needed to serialize the given data.  The columns should be in host memory
   * before calling this.
   * @param columns columns to be serialized.
   * @param rowOffset the first row to serialize.
   * @param numRows the number of rows to serialize.
   * @return the size in bytes needed to serialize the data including the header.
   */
  public static long getSerializedSizeInBytes(ColumnVector[] columns, long rowOffset, long numRows) {
    ColumnBufferProvider[] providers = providersFrom(columns);
    return getSlicedSerializedDataSizeInBytes(providers, rowOffset, numRows) + (4 * 3) + 2; // The header size
  }

  /////////////////////////////////////////////
  // HELPER METHODS buildIndex
  /////////////////////////////////////////////

  static ColumnOffsets[] buildIndex(SerializedTableHeader header,
                                    HostMemoryBuffer buffer) {
    long bufferOffset = 0;
    DType[] dataTypes = header.types;
    int numColumns = dataTypes.length;
    long[] nullCounts = header.nullCounts;
    long numRows = header.getNumRows();
    ColumnOffsets[] ret = new ColumnOffsets[numColumns];
    for (int column = 0; column < numColumns; column++) {
      DType type = dataTypes[column];
      long nullCount = nullCounts[column];

      long validity = -1;
      long validityLen = 0;
      long offsets = -1;
      long offsetsLen = 0;
      long data;
      long dataLen;
      if (nullCount > 0) {
        validityLen = padFor64byteAlignment(BitVectorHelper.getValidityLengthInBytes(numRows));
        validity = bufferOffset;
        bufferOffset += validityLen;
      }

      if (type == DType.STRING || type == DType.STRING_CATEGORY) {
        offsetsLen = padFor64byteAlignment((numRows + 1) * 4);
        offsets = bufferOffset;
        int startStringOffset = buffer.getInt(bufferOffset);
        int endStringOffset = buffer.getInt(bufferOffset + (numRows * 4));
        bufferOffset += offsetsLen;

        dataLen = padFor64byteAlignment(endStringOffset - startStringOffset);
        data = bufferOffset;
        bufferOffset += dataLen;
      } else {
        dataLen = padFor64byteAlignment(type.sizeInBytes * numRows);
        data = bufferOffset;
        bufferOffset += dataLen;
      }
      ret[column] = new ColumnOffsets(validity, validityLen,
          offsets, offsetsLen,
          data, dataLen);
    }
    return ret;
  }

  /////////////////////////////////////////////
  // HELPER METHODS FOR PROVIDERS
  /////////////////////////////////////////////

  private static ColumnBufferProvider[] providersFrom(ColumnVector[] columns) {
    ColumnBufferProvider[] providers = new ColumnBufferProvider[columns.length];
    for (int i = 0; i < columns.length; i++) {
      providers[i] = new ColumnProvider(columns[i]);
    }
    return providers;
  }

  private static ColumnBufferProvider[][] providersFrom(SerializedTableHeader[] headers,
                                                        HostMemoryBuffer[] dataBuffers) {
    // Filter out empty tables, to make things simpler...
    int validCount = 0;
    for (int i = 0; i < headers.length; i++) {
      if (headers[i].numRows > 0) {
        validCount++;
      } else {
        assert headers[i].dataLen == 0;
      }
    }

    if (validCount < headers.length) {
      SerializedTableHeader[] filteredHeaders = new SerializedTableHeader[validCount];
      HostMemoryBuffer[] filteredBuffers = new HostMemoryBuffer[validCount];
      int at = 0;
      for (int i = 0; i < headers.length; i++) {
        if (headers[i].numRows > 0) {
          filteredHeaders[at] = headers[i];
          filteredBuffers[at] = dataBuffers[i];
          at++;
        }
      }
      headers = filteredHeaders;
      dataBuffers = filteredBuffers;
    }

    ColumnBufferProvider [][] ret = new ColumnBufferProvider[headers.length][];
    for (int batchNum = 0; batchNum < headers.length; batchNum++) {
      SerializedTableHeader header = headers[batchNum];
      HostMemoryBuffer dataBuffer = dataBuffers[batchNum];
      ColumnOffsets[] offsets = buildIndex(header, dataBuffer);

      ColumnBufferProvider [] parts = new ColumnBufferProvider[offsets.length];
      for (int columnIndex = 0; columnIndex < offsets.length; columnIndex++) {
        parts[columnIndex] = new BufferOffsetProvider(header, columnIndex,
            offsets[columnIndex], dataBuffer);
      }
      ret[batchNum] = parts;
    }

    return ret;
  }

  /////////////////////////////////////////////
  // HELPER METHODS FOR SerializedTableHeader
  /////////////////////////////////////////////

  private static SerializedTableHeader calcHeader(ColumnBufferProvider[] columns,
                                                  long rowOffset,
                                                  int numRows) {
    DType[] types = new DType[columns.length];
    long[] nullCount = new long[columns.length];
    TimeUnit[] tu = new TimeUnit[columns.length];
    for (int i = 0; i < columns.length; i++) {
      types[i] = columns[i].getType();
      nullCount[i] = columns[i].getNullCount();
      tu[i] = columns[i].getTimeUnit();
    }

    long dataLength = getSlicedSerializedDataSizeInBytes(columns, rowOffset, numRows);
    return new SerializedTableHeader(numRows, types, nullCount, tu, dataLength);
  }

  /**
   * Calculate the new header for a concatenated set of columns.
   * @param columnsForEachBatch first index is the batch, second index is the column.
   * @return the new header.
   */
  private static SerializedTableHeader calcConcatedHeader(ColumnBufferProvider[][] columnsForEachBatch) {

    // verify that all of the columns can be concated, we also need to verify that the sizes are going to work....
    int numColumns = 0;
    DType[] types;
    TimeUnit[] tu;
    long[] nullCounts;
    long numRows = 0;
    if (columnsForEachBatch.length > 0) {
      ColumnBufferProvider[] providers = columnsForEachBatch[0];
      numColumns = providers.length;
      types = new DType[numColumns];
      tu = new TimeUnit[numColumns];
      nullCounts = new long[numColumns];
      for (int i = 0; i < providers.length; i++) {
        types[i] = providers[i].getType();
        tu[i] = providers[i].getTimeUnit();
        nullCounts[i] = providers[i].getNullCount();
      }
      if (numColumns > 0) {
        numRows = providers[0].getRowCount();
      }
    } else {
      types = new DType[0];
      tu = new TimeUnit[0];
      nullCounts = new long[0];
    }

    for (int batchNum = 1; batchNum < columnsForEachBatch.length; batchNum++) {
      ColumnBufferProvider[] providers = columnsForEachBatch[batchNum];
      if (providers.length != numColumns) {
        throw new IllegalArgumentException("The number of columns did not match " + batchNum
            + " " + providers.length + " != " + numColumns);
      }
      for (int col = 0; col < numColumns; col++) {
        if (providers[col].getType() != types[col]) {
          throw new IllegalArgumentException("Type mismatch for column " + col);
        }

        if (providers[col].getTimeUnit() != tu[col]) {
          throw new IllegalArgumentException("TimeUnit mismatch for column " + col);
        }
        nullCounts[col] += providers[col].getNullCount();
      }
      if (numColumns > 0) {
        numRows += providers[0].getRowCount();
      }
    }

    if (numRows > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("CANNOT BUILD A BATCH LARGER THAN " + Integer.MAX_VALUE + " rows");
    }

    long totalDataSize = getConcatedSerializedDataSizeInBytes(numColumns, nullCounts, (int)numRows, types,
        columnsForEachBatch);
    return new SerializedTableHeader((int)numRows, types, nullCounts, tu, totalDataSize);
  }

  /////////////////////////////////////////////
  // HELPER METHODS FOR DataWriters
  /////////////////////////////////////////////

  private static DataWriter writerFrom(OutputStream out) {
    if (!(out instanceof DataOutputStream)) {
      out = new DataOutputStream(new BufferedOutputStream(out));
    }
    return new DataOutputStreamWriter((DataOutputStream) out);
  }

  private static DataWriter writerFrom(HostMemoryBuffer buffer) {
    return new HostDataWriter(buffer);
  }

  /////////////////////////////////////////////
  // Serialize Data Methods
  /////////////////////////////////////////////

  private static long copySlicedAndPad(DataWriter out,
                                       ColumnBufferProvider column,
                                       ColumnVector.BufferType buffer,
                                       long offset,
                                       long length) throws IOException {
    out.copyDataFrom(column, buffer, offset, length);
    return padFor64byteAlignment(out, length);
  }

  /////////////////////////////////////////////
  // VALIDITY
  /////////////////////////////////////////////

  private static int copyPartialValidity(byte[] dest,
                                         int destBitOffset,
                                         ColumnBufferProvider provider,
                                         int srcBitOffset,
                                         int lengthBits) {
    HostMemoryBuffer src = provider.getHostBufferFor(ColumnVector.BufferType.VALIDITY);
    long baseSrcByteOffset = provider.getBufferStartOffset(ColumnVector.BufferType.VALIDITY);

    int destStartBytes = destBitOffset / 8;
    int destStartBitOffset = destBitOffset % 8;
    long srcStartBytes = baseSrcByteOffset + (srcBitOffset / 8);
    int srcStartBitOffset = srcBitOffset % 8;
    int availableDestBits = (dest.length * 8) - destBitOffset;
    int bitsToCopy = Math.min(lengthBits, availableDestBits);

    int lastIndex = (bitsToCopy + destStartBitOffset + 7) / 8;

    byte allBitsSet = ~0;
    byte firstSrcMask = (byte)(allBitsSet << destStartBitOffset);

    int srcShift = destStartBitOffset - srcStartBitOffset;
    if (srcShift > 0) {
      // Shift left. If we are going to shift this is the path typically taken.

      byte current = src.getByte(srcStartBytes);
      byte result = (byte)(current << srcShift);
      // The first time we need to include any data already in dest.
      result |= dest[destStartBytes] & ~firstSrcMask;
      dest[destStartBytes] = result;

      // Keep the previous bytes around so we don't have to keep reading from src, which is not free
      byte previous = current;

      for (int index = 1; index < lastIndex; index++) {
        current = src.getByte(index + srcStartBytes);
        result = (byte)(current << srcShift);
        result |= (previous & 0xFF) >>> (8 - srcShift);
        dest[index + destStartBytes] = result;
        previous = current;
      }
      return bitsToCopy;
    } else if (srcShift < 0) {
      srcShift = -srcShift;

      // shifting right only happens when the buffer runs out of space.

      byte result = src.getByte(srcStartBytes);
      result = (byte)((result & 0xFF) >>> srcShift);
      byte next = 0;
      if (srcStartBytes + 1 < src.length) {
        next = src.getByte(srcStartBytes + 1);
      }
      result |= (byte)(next << 8 - srcShift);
      result &= firstSrcMask;

      // The first time through we need to include the data already in dest.
      result |= dest[destStartBytes] & ~firstSrcMask;
      dest[destStartBytes] = result;

      for (int index = 1; index < lastIndex - 1; index++) {
        result = next;
        result = (byte)((result & 0xFF) >>> srcShift);
        next = src.getByte(srcStartBytes + index + 1);
        result |= (byte)(next << 8 - srcShift);
        dest[index + destStartBytes] = result;
      }

      int idx = lastIndex - 1;
      if (idx > 0) {
        result = next;
        result = (byte) ((result & 0xFF) >>> srcShift);
        next = 0;
        if (srcStartBytes + idx + 1 < src.length) {
          next = src.getByte(srcStartBytes + idx + 1);
        }
        result |= (byte) (next << 8 - srcShift);
        dest[idx + destStartBytes] = result;
      }
      return bitsToCopy;
    } else {
      src.getBytes(dest, destStartBytes, srcStartBytes, (bitsToCopy + 7) / 8);
      return bitsToCopy;
    }
  }

  // package-private for testing
  static long copySlicedValidity(DataWriter out,
                                 ColumnBufferProvider column,
                                 long rowOffset,
                                 long numRows) throws IOException {
    long validityLen = BitVectorHelper.getValidityLengthInBytes(numRows);
    long byteOffset = (rowOffset / 8);
    long bytesLeft = validityLen;

    int lshift = (int) rowOffset % 8;
    if (lshift == 0) {
      out.copyDataFrom(column, ColumnVector.BufferType.VALIDITY, byteOffset, bytesLeft);
    } else {
      byte[] arrayBuffer = new byte[128 * 1024];
      int rowsStoredInArray = 0;
      int rowsLeftInBatch = (int) numRows;
      int validityBitOffset = (int) rowOffset;
      while(rowsLeftInBatch > 0) {
        int rowsStoredJustNow = copyPartialValidity(arrayBuffer, rowsStoredInArray, column, validityBitOffset, rowsLeftInBatch);
        assert rowsStoredJustNow > 0;
        rowsLeftInBatch -= rowsStoredJustNow;
        rowsStoredInArray += rowsStoredJustNow;
        validityBitOffset += rowsStoredJustNow;
        if (rowsStoredInArray == arrayBuffer.length * 8) {
          out.write(arrayBuffer, 0, arrayBuffer.length);
          rowsStoredInArray = 0;
        }
      }
      if (rowsStoredInArray > 0) {
        out.write(arrayBuffer, 0, (rowsStoredInArray + 7) / 8);
      }
    }
    return padFor64byteAlignment(out, validityLen);
  }

  // Package private for testing
  static int fillValidity(byte[] dest, int destBitOffset, int lengthBits) {
    int destStartBytes = destBitOffset / 8;
    int destStartBits = destBitOffset % 8;

    long lengthBytes = BitVectorHelper.getValidityLengthInBytes(lengthBits);
    int rshift = destStartBits;
    int totalCopied = 0;
    if (rshift != 0) {
      // Fill in what we need to make it copyable
      dest[destStartBytes] |= (0xFF << destStartBits);
      destStartBytes += 1;
      totalCopied = (8 - destStartBits);
      // Not used again, but just to be safe
      destStartBits = 0;
    }
    int amountToCopyBytes = (int) Math.min(lengthBytes, dest.length - destStartBytes);
    for (int i = 0; i < amountToCopyBytes; i++) {
      dest[i + destStartBytes] = (byte) 0xFF;
    }
    totalCopied += amountToCopyBytes * 8;
    return Math.min(totalCopied, lengthBits);
  }

  private static long concatValidity(DataWriter out,
                                     int columnIndex,
                                     int numRows,
                                     ColumnBufferProvider[][] providers) throws IOException {
    long validityLen = BitVectorHelper.getValidityLengthInBytes(numRows);
    byte[] arrayBuffer = new byte[128 * 1024];
    int rowsStoredInArray = 0;
    for (int batchIndex = 0; batchIndex < providers.length; batchIndex++) {
      ColumnBufferProvider provider = providers[batchIndex][columnIndex];
      int rowsLeftInBatch = (int) provider.getRowCount();
      int validityBitOffset = 0;
      while(rowsLeftInBatch > 0) {
        int rowsStoredJustNow;
        if (provider.getNullCount() > 0) {
          rowsStoredJustNow = copyPartialValidity(arrayBuffer, rowsStoredInArray, provider, validityBitOffset, rowsLeftInBatch);
        } else {
          rowsStoredJustNow = fillValidity(arrayBuffer, rowsStoredInArray, rowsLeftInBatch);
        }
        assert rowsStoredJustNow > 0;
        assert rowsStoredJustNow <= rowsLeftInBatch;
        rowsLeftInBatch -= rowsStoredJustNow;
        rowsStoredInArray += rowsStoredJustNow;
        validityBitOffset += rowsStoredJustNow;
        if (rowsStoredInArray == arrayBuffer.length * 8) {
          out.write(arrayBuffer, 0, arrayBuffer.length);
          rowsStoredInArray = 0;
        }
      }
    }

    if (rowsStoredInArray > 0) {
      int len = (rowsStoredInArray + 7) / 8;
      out.write(arrayBuffer, 0, len);
    }
    return padFor64byteAlignment(out, validityLen);
  }

  /////////////////////////////////////////////
  // STRING
  /////////////////////////////////////////////

  private static long copySlicedStringData(DataWriter out, ColumnBufferProvider column, long rowOffset,
                                           long numRows) throws IOException {
    if (numRows > 0) {
      long startByteOffset = column.getStartStringOffset(rowOffset);
      long endByteOffset = column.getEndStringOffset(rowOffset + numRows - 1);
      long bytesToCopy = endByteOffset - startByteOffset;
      long srcOffset = startByteOffset;
      return copySlicedAndPad(out, column, ColumnVector.BufferType.DATA, srcOffset, bytesToCopy);
    }
    return 0;
  }

  private static void copyConcateStringData(DataWriter out,
                                            int columnIndex,
                                            int[] dataLengths,
                                            ColumnBufferProvider[][] providers) throws IOException {
    long totalCopied = 0;

    for (int batchIndex = 0; batchIndex < providers.length; batchIndex++) {
      ColumnBufferProvider provider = providers[batchIndex][columnIndex];
      HostMemoryBuffer dataBuffer = provider.getHostBufferFor(ColumnVector.BufferType.DATA);
      long currentOffset = provider.getBufferStartOffset(ColumnVector.BufferType.DATA);
      int dataLeft = dataLengths[batchIndex];
      out.copyDataFrom(dataBuffer, currentOffset, dataLeft);
      totalCopied += dataLeft;
    }
    padFor64byteAlignment(out, totalCopied);
  }

  private static long copySlicedOffsets(DataWriter out, ColumnBufferProvider column, long rowOffset,
                                        long numRows) throws IOException {
    // If an offset is copied over as a part of a slice the first entry may be non-zero.  This is
    // okay because we fix them up when they are deserialized
    long bytesToCopy = (numRows + 1) * 4;
    long srcOffset = rowOffset * 4;
    return copySlicedAndPad(out, column, ColumnVector.BufferType.OFFSET, srcOffset, bytesToCopy);
  }

  private static int[] copyConcateOffsets(DataWriter out,
                                          int columnIndex,
                                          ColumnBufferProvider[][] providers) throws IOException {
    int dataLens[] = new int[providers.length];
    long totalCopied = 0;
    int offsetToAdd = 0;

    // First offset is always 0
    out.writeInt(0);
    totalCopied += 4;

    for (int batchIndex = 0; batchIndex < providers.length; batchIndex++) {
      ColumnBufferProvider provider = providers[batchIndex][columnIndex];
      HostMemoryBuffer dataBuffer = provider.getHostBufferFor(ColumnVector.BufferType.OFFSET);
      long currentOffset = provider.getBufferStartOffset(ColumnVector.BufferType.OFFSET);
      int numRowsForHeader = (int) provider.getRowCount();

      // We already output the first row
      int dataLeft = numRowsForHeader * 4;
      // fix up the offsets for the data
      int startStringOffset = dataBuffer.getInt(currentOffset);
      int endStringOffset = dataBuffer.getInt(currentOffset + (numRowsForHeader * 4));
      dataLens[batchIndex] = endStringOffset - startStringOffset;
      // The first index should always be 0, but that is not always true because we fix it up
      // on the receiving side, which is here...
      // But if this ever is written out twice we need to make sure
      // we fix up the 0 entry too.
      dataBuffer.setInt(currentOffset, offsetToAdd);
      for (int i = 1; i < (numRowsForHeader + 1); i++) {
        long at = currentOffset + (i * 4);
        int orig = dataBuffer.getInt(at);
        int o = orig + offsetToAdd - startStringOffset;
        dataBuffer.setInt(at, o);
      }
      offsetToAdd += dataLens[batchIndex];

      currentOffset += 4; // Skip the first entry that is always 0
      out.copyDataFrom(dataBuffer, currentOffset, dataLeft);
      totalCopied += dataLeft;
    }
    padFor64byteAlignment(out, totalCopied);
    return dataLens;
  }

  /////////////////////////////////////////////
  // BASIC DATA
  /////////////////////////////////////////////

  private static long sliceBasicData(DataWriter out,
                                     ColumnBufferProvider column,
                                     long rowOffset,
                                     long numRows) throws IOException {
    DType type = column.getType();
    long bytesToCopy = numRows * type.sizeInBytes;
    long srcOffset = rowOffset * type.sizeInBytes;
    return copySlicedAndPad(out, column, ColumnVector.BufferType.DATA, srcOffset, bytesToCopy);
  }

  private static void concatBasicData(DataWriter out,
                                      int columnIndex,
                                      DType type,
                                      ColumnBufferProvider[][] providers) throws IOException {
    long totalCopied = 0;
    for (int batchIndex = 0; batchIndex < providers.length; batchIndex++) {
      ColumnBufferProvider provider = providers[batchIndex][columnIndex];
      HostMemoryBuffer dataBuffer = provider.getHostBufferFor(ColumnVector.BufferType.DATA);
      long currentOffset = provider.getBufferStartOffset(ColumnVector.BufferType.DATA);
      int numRowsForBatch = (int) provider.getRowCount();

      int dataLeft = numRowsForBatch * type.sizeInBytes;
      out.copyDataFrom(dataBuffer, currentOffset, dataLeft);
      totalCopied += dataLeft;
    }
    padFor64byteAlignment(out, totalCopied);
  }

  /////////////////////////////////////////////
  // COLUMN AND TABLE WRITE
  /////////////////////////////////////////////

  private static void writeConcat(DataWriter out,
                                  int columnIndex,
                                  SerializedTableHeader combinedHeader,
                                  ColumnBufferProvider[][] providers) throws IOException {
    long nullCount = combinedHeader.nullCounts[columnIndex];
    if (nullCount > 0) {
      concatValidity(out, columnIndex, combinedHeader.numRows, providers);
    }

    DType type = combinedHeader.types[columnIndex];
    if (type == DType.STRING || type == DType.STRING_CATEGORY) {
      // Get the actual lengths for each section...
      int dataLens[] = copyConcateOffsets(out, columnIndex, providers);
      copyConcateStringData(out, columnIndex, dataLens, providers);
    } else {
      concatBasicData(out, columnIndex, type, providers);
    }
  }

  private static void writeSliced(DataWriter out,
                                  ColumnBufferProvider column,
                                  long rowOffset,
                                  long numRows) throws IOException {
    column.ensureOnHost();

    if (column.getNullCount() > 0) {
      try (NvtxRange range = new NvtxRange("Write Validity", NvtxColor.DARK_GREEN)) {
        copySlicedValidity(out, column, rowOffset, numRows);
      }
    }

    DType type = column.getType();
    if (type == DType.STRING || type == DType.STRING_CATEGORY) {
      try (NvtxRange range = new NvtxRange("Write String Data", NvtxColor.RED)) {
        copySlicedOffsets(out, column, rowOffset, numRows);
        copySlicedStringData(out, column, rowOffset, numRows);
      }
    } else {
      try (NvtxRange range = new NvtxRange("Write Data", NvtxColor.BLUE)) {
        sliceBasicData(out, column, rowOffset, numRows);
      }
    }
  }

  private static void writeSliced(ColumnBufferProvider[] columns,
                                  DataWriter out,
                                  long rowOffset,
                                  long numRows) throws IOException {
    assert rowOffset >= 0;
    assert numRows >= 0;
    for (int i = 0; i < columns.length; i++) {
      long rows = columns[i].getRowCount();
      assert rowOffset + numRows <= rows;
      long nullCount = columns[i].getNullCount();
      assert nullCount == (int) nullCount : "can only support an int for indexes";
      assert rows == (int) rows : "can only support an int for indexes";
    }

    SerializedTableHeader header = calcHeader(columns, rowOffset, (int) numRows);
    header.writeTo(out);

    try (NvtxRange range = new NvtxRange("Write Sliced", NvtxColor.GREEN)) {
      for (int i = 0; i < columns.length; i++) {
        writeSliced(out, columns[i], rowOffset, numRows);
      }
    }
    out.flush();
  }

  /**
   * Write all or part of a table out in an internal format.
   * @param t the table to be written.
   * @param out the stream to write the serialized table out to.
   * @param rowOffset the first row to write out.
   * @param numRows the number of rows to write out.
   */
  public static void writeToStream(Table t, OutputStream out, long rowOffset, long numRows)
      throws IOException {
    writeToStream(t.getColumns(), out, rowOffset, numRows);
  }

  /**
   * Write all or part of a set of columns out in an internal format.
   * @param columns the columns to be written.
   * @param out the stream to write the serialized table out to.
   * @param rowOffset the first row to write out.
   * @param numRows the number of rows to write out.
   */
  public static void writeToStream(ColumnVector[] columns, OutputStream out, long rowOffset,
                                   long numRows) throws IOException {

    ColumnBufferProvider[] providers = providersFrom(columns);
    DataWriter writer = writerFrom(out);
    writeSliced(providers, writer, rowOffset, numRows);
  }

  /**
   * Take the data from multiple batches stored in the parsed headers and the dataBuffer and write
   * it out to out as if it were a single buffer.
   * @param headers the headers parsed from multiple streams.
   * @param dataBuffers an array of buffers that hold the data, one per header.
   * @param out what to write the data out to.
   * @throws IOException on any error.
   */
  public static void writeConcatedStream(SerializedTableHeader[] headers,
                                         HostMemoryBuffer[] dataBuffers,
                                         OutputStream out) throws IOException {
    ColumnBufferProvider[][] providers = providersFrom(headers, dataBuffers);
    SerializedTableHeader combined = calcConcatedHeader(providers);
    DataWriter writer = writerFrom(out);
    combined.writeTo(writer);

    try (NvtxRange range = new NvtxRange("Concat Host Side", NvtxColor.GREEN)) {
      for (int columnIndex = 0; columnIndex < combined.numColumns; columnIndex++) {
        writeConcat(writer, columnIndex, combined, providers);
      }
    }
    writer.flush();
  }

  /////////////////////////////////////////////
  // COLUMN AND TABLE READ
  /////////////////////////////////////////////

  private static Table sliceUpColumnVectors(SerializedTableHeader header,
                                            DeviceMemoryBuffer combinedBuffer,
                                            HostMemoryBuffer combinedBufferOnHost) {
    try (NvtxRange range = new NvtxRange("bufferToTable", NvtxColor.PURPLE)) {
      ColumnOffsets[] columnOffsets = buildIndex(header, combinedBufferOnHost);
      DType[] dataTypes = header.types;
      long[] nullCounts = header.nullCounts;
      TimeUnit[] timeUnits = header.tu;
      long numRows = header.getNumRows();
      int numColumns = dataTypes.length;
      ColumnVector[] vectors = new ColumnVector[numColumns];
      DeviceMemoryBuffer validity = null;
      DeviceMemoryBuffer data = null;
      HostMemoryBuffer offsets = null;
      try {
        for (int column = 0; column < numColumns; column++) {
          DType type = dataTypes[column];
          long nullCount = nullCounts[column];
          TimeUnit tu = timeUnits[column];
          ColumnOffsets offsetInfo = columnOffsets[column];

          if (nullCount > 0) {
            validity = combinedBuffer.slice(offsetInfo.validity, offsetInfo.validityLen);
          }

          if (type == DType.STRING || type == DType.STRING_CATEGORY) {
            offsets = combinedBufferOnHost.slice(offsetInfo.offsets, offsetInfo.offsetsLen);
          }

          if (offsetInfo.dataLen == 0) {
            // The vector is possibly full of null strings. This is a rare corner case, but here is the
            // simplest place to work around it.
            data = DeviceMemoryBuffer.allocate(1);
          } else {
            data = combinedBuffer.slice(offsetInfo.data, offsetInfo.dataLen);
          }

          vectors[column] = new ColumnVector(type, tu, numRows, nullCount, data, validity, offsets, true);
          validity = null;
          data = null;
          offsets = null;
        }
        return new Table(vectors);
      } finally {
        if (validity != null) {
          validity.close();
        }

        if (data != null) {
          data.close();
        }

        if (offsets != null) {
          offsets.close();
        }

        for (ColumnVector cv: vectors) {
          if (cv != null) {
            cv.close();
          }
        }
      }
    }
  }

  public static Table readAndConcat(SerializedTableHeader[] headers,
                                    HostMemoryBuffer[] dataBuffers) throws IOException {

    ColumnBufferProvider[][] providers = providersFrom(headers, dataBuffers);
    SerializedTableHeader combined = calcConcatedHeader(providers);

    try (DevicePrediction prediction = new DevicePrediction(combined.dataLen, "readAndConcat");
         HostMemoryBuffer hostBuffer = HostMemoryBuffer.allocate(combined.dataLen);
         DeviceMemoryBuffer devBuffer = DeviceMemoryBuffer.allocate(hostBuffer.length)) {
      try (NvtxRange range = new NvtxRange("Concat Host Side", NvtxColor.GREEN)) {
        DataWriter writer = writerFrom(hostBuffer);
        for (int columnIndex = 0; columnIndex < combined.numColumns; columnIndex++) {
          writeConcat(writer, columnIndex, combined, providers);
        }
      }

      if (hostBuffer.length > 0) {
        try (NvtxRange range = new NvtxRange("Copy Data To Device", NvtxColor.WHITE)) {
          devBuffer.copyFromHostBuffer(hostBuffer);
        }
      }
      return sliceUpColumnVectors(combined, devBuffer, hostBuffer);
    }
  }

  /**
   * After reading a header for a table read the data portion into a host side buffer.
   * @param in the stream to read the data from.
   * @param header the header that finished just moments ago.
   * @param buffer the buffer to write the data into.  If there is not enough room to store
   *               the data in buffer it will not be read and header will still have dataRead
   *               set to false.
   * @throws IOException
   */
  public static void readTableIntoBuffer(InputStream in,
                                         SerializedTableHeader header,
                                         HostMemoryBuffer buffer) throws IOException {
    if (header.initialized &&
        (buffer.length >= header.dataLen)) {
      try (NvtxRange range = new NvtxRange("Read Data", NvtxColor.RED)) {
        buffer.copyFromStream(0, in, header.dataLen);
      }
      header.dataRead = true;
    }
  }

  public static Table readTableFrom(SerializedTableHeader header,
                                    HostMemoryBuffer hostBuffer) {
    try (DevicePrediction prediction = new DevicePrediction(hostBuffer.length, "readTableFrom");
         DeviceMemoryBuffer devBuffer = DeviceMemoryBuffer.allocate(hostBuffer.length)) {
      if (hostBuffer.length > 0) {
        try (NvtxRange range = new NvtxRange("Copy Data To Device", NvtxColor.WHITE)) {
          devBuffer.copyFromHostBuffer(hostBuffer);
        }
      }
      return sliceUpColumnVectors(header, devBuffer, hostBuffer);
    }
  }

  /**
   * Read a serialize table from the given InputStream.
   * @param in the stream to read the table data from.
   * @return the deserialized table in device memory, or null if the stream has no table to read
   * from, an end of the stream at the very beginning.
   * @throws IOException on any error.
   * @throws EOFException if the data stream ended unexpectedly in the middle of processing.
   */
  public static Table readTableFrom(InputStream in) throws IOException {
    DataInputStream din;
    if (in instanceof DataInputStream) {
      din = (DataInputStream) in;
    } else {
      din = new DataInputStream(in);
    }

    SerializedTableHeader header = new SerializedTableHeader(din);
    if (!header.initialized) {
      return null;
    }

    try (HostMemoryBuffer hostBuffer = HostMemoryBuffer.allocate(header.dataLen)) {
      if (header.dataLen > 0) {
        readTableIntoBuffer(din, header, hostBuffer);
      }
      return readTableFrom(header, hostBuffer);
    }
  }
}