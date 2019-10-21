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
import java.util.Arrays;

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

  private static long padFor64bitAlignment(long orig) {
    return ((orig + 7) / 8) * 8;
  }

  private static long padFor64bitAlignment(DataOutputStream out, long bytes) throws IOException {
    final long paddedBytes = padFor64bitAlignment(bytes);
    while (paddedBytes > bytes) {
      out.writeByte(0);
      bytes++;
    }
    return paddedBytes;
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
    return getSerializedDataSizeInBytes(columns, rowOffset, numRows) + (4 * 3) + 2; // The header size
  }

  private static long getSerializedDataSizeInBytes(ColumnVector[] columns, long rowOffset, long numRows) {
    long total = 0;
    for (int i = 0; i < columns.length; i++) {
      total += getSerializedDataSizeInBytes(columns[i], rowOffset, numRows);
    }
    return total;
  }

  private static long getSerializedDataSizeInBytes(ColumnVector column, long rowOffset, long numRows) {
    DType type = column.getType();
    long total = 0;
    if (column.getNullCount() > 0) {
      total += padFor64bitAlignment(BitVectorHelper.getValidityLengthInBytes(numRows));
    }
    if (type == DType.STRING || type == DType.STRING_CATEGORY) {
      // offsets
      total += padFor64bitAlignment((numRows + 1) * 4);

      // data
      if (numRows > 0) {
        long start = column.getStartStringOffset(rowOffset);
        long end = column.getEndStringOffset(rowOffset + numRows - 1);
        total += padFor64bitAlignment(end - start);
      }
    } else {
      total += padFor64bitAlignment(column.getType().sizeInBytes * numRows);
    }
    return total;
  }

  /**
   * Calculate the size of the buffer needed to concat multiple batches together on the CPU.
   * @param numRows the total number of rows being concated.
   * @param nullCounts the null count for each column.
   * @param types the type for each column
   * @param headers the serialized header for each batch being concated.
   * @param offsetsForEachHeader The offsets into the data buffer for all of the columns in all
   *                            of the batches. The outer array is indexed by batch, and the
   *                            inner array is indexed by column.
   * @param dataBuffers the buffers that stores all of the data for the batches, one per table.
   * @return how much data is needed to write out all of this as a single batch.
   */
  private static long getConcatedDataSizeInBytes(int numRows,
                                                 long[] nullCounts,
                                                 DType[] types,
                                                 SerializedTableHeader[] headers,
                                                 ColumnOffsets[][] offsetsForEachHeader,
                                                 HostMemoryBuffer[] dataBuffers) {
    long total = 0;
    // This is almost going to be like writing out the row itself...
    for (int col = 0; col < types.length; col++) {
      if (nullCounts[col] > 0) {
        total += padFor64bitAlignment(BitVectorHelper.getValidityLengthInBytes(numRows));
      }
      if (types[col] == DType.STRING || types[col] == DType.STRING_CATEGORY) {
        // offsets
        total += padFor64bitAlignment((numRows + 1) * 4);

        long subTotal = 0;
        for (int i = 0; i < offsetsForEachHeader.length; i++) {
          int numRowsInSubColumn = headers[i].numRows;
          subTotal += getStringDataLength(numRowsInSubColumn, offsetsForEachHeader[i][col], dataBuffers[i]);
        }
        total += padFor64bitAlignment(subTotal);
      } else {
        total += padFor64bitAlignment(types[col].sizeInBytes * numRows);
      }
    }
    return total;
  }

  private static long getStringDataLength(int numRows,
                                          ColumnOffsets columnOffsets,
                                          HostMemoryBuffer dataBuffer) {
    assert columnOffsets.offsets >= 0;
    int start = dataBuffer.getInt(columnOffsets.offsets);
    int end = dataBuffer.getInt(columnOffsets.offsets + (numRows * 4));
    return end - start;
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
    if (!(out instanceof DataOutputStream)) {
      out = new DataOutputStream(new BufferedOutputStream(out));
    }
    writeToDataStream(columns, (DataOutputStream) out, rowOffset, numRows);
  }

  private static void writeToDataStream(ColumnVector[] columns, DataOutputStream out, long rowOffset,
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

    out.writeInt(SER_FORMAT_MAGIC_NUMBER);
    out.writeShort(VERSION_NUMBER);
    out.writeInt(columns.length);
    // TODO this should really be a long eventually...
    out.writeInt((int) numRows);

    // Header for each column...
    for (int i = 0; i < columns.length; i++) {
      out.writeInt(columns[i].getType().nativeId);
      long nullCount = columns[i].getNullCount();
      if (nullCount != 0 && (rowOffset != 0 || numRows != columns[i].getRowCount())) {
        // TODO This is a hack.  We need to get partition to calculate this for us properly in the
        // future.
        nullCount = 1;
      }
      // TODO this should be a long eventually.
      out.writeInt((int) nullCount);
      out.writeInt(columns[i].getTimeUnit().getNativeId());
    }
    out.writeLong(getSerializedDataSizeInBytes(columns, rowOffset, numRows));
    for (int i = 0; i < columns.length; i++) {
      writeColumnToDataStream(out, columns[i], rowOffset, numRows);
    }
    out.flush();
  }

  private static void writeColumnToDataStream(DataOutputStream out, ColumnVector column,
                                              long rowOffset, long numRows) throws IOException {
    // We are using host based data, because reducing the number of data transfers
    // had a bigger impact on performance than reducing the computation/memory usage on the host.
    column.ensureOnHost();

    byte[] arrayBuffer = new byte[1024 * 128];
    if (column.getNullCount() > 0) {
      copyValidityData(out, column, rowOffset, numRows, arrayBuffer);
    }

    DType type = column.getType();
    if (type == DType.STRING || type == DType.STRING_CATEGORY) {
      copyStringOffsets(out, column, rowOffset, numRows, arrayBuffer);
      copyStringData(out, column, rowOffset, numRows, arrayBuffer);
    } else {
      copyBasicData(out, column, rowOffset, numRows, arrayBuffer);
    }
  }

  private static void writeAndConcatColumnToDataStream(DataOutputStream out,
                                                       DType type,
                                                       int columnIndex,
                                                       long nullCount,
                                                       int numRowsTotal,
                                                       SerializedTableHeader[] headers,
                                                       ColumnOffsets[][] offsetsForEachHeader,
                                                       HostMemoryBuffer[] dataBuffers) throws IOException {
    byte[] arrayBuffer = new byte[1024 * 128];
    if (nullCount > 0) {
      concatValidityData(out, columnIndex, numRowsTotal, headers,
          offsetsForEachHeader, dataBuffers, arrayBuffer);
    }
    if (type == DType.STRING || type == DType.STRING_CATEGORY) {
      // Get the actual lengths for each section...
      int dataLens[] = new int[headers.length];
      // OFFSETS:
      long totalCopied = 0;
      int offsetToAdd = 0;

      // First offset is always 0
      out.writeInt(0);
      totalCopied += 4;

      for (int headerIndex = 0; headerIndex < offsetsForEachHeader.length; headerIndex++) {
        ColumnOffsets[] offsets = offsetsForEachHeader[headerIndex];
        SerializedTableHeader header = headers[headerIndex];
        HostMemoryBuffer dataBuffer = dataBuffers[headerIndex];
        int numRowsForHeader = header.numRows;
        ColumnOffsets co = offsets[columnIndex];
        long currentOffset = co.offsets;
        // We already output the first row
        int dataLeft = numRowsForHeader * 4;
        // fix up the offsets for the data
        int startStringOffset = dataBuffer.getInt(currentOffset);
        int endStringOffset = dataBuffer.getInt(currentOffset + (numRowsForHeader * 4));
        dataLens[headerIndex] = endStringOffset - startStringOffset;
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
        offsetToAdd += dataLens[headerIndex];

        currentOffset += 4; // Skip the first entry that is always 0
        while (dataLeft > 0) {
          int amountToCopy = Math.min(arrayBuffer.length, dataLeft);
          dataBuffer.getBytes(arrayBuffer, 0, currentOffset, amountToCopy);
          out.write(arrayBuffer, 0, amountToCopy);
          totalCopied += amountToCopy;
          currentOffset += amountToCopy;
          dataLeft -= amountToCopy;
        }
      }
      padFor64bitAlignment(out, totalCopied);

      // STRING DATA
      totalCopied = 0;

      for (int headerIndex = 0; headerIndex < offsetsForEachHeader.length; headerIndex++) {
        HostMemoryBuffer dataBuffer = dataBuffers[headerIndex];
        ColumnOffsets[] offsets = offsetsForEachHeader[headerIndex];
        ColumnOffsets co = offsets[columnIndex];
        long currentOffset = co.data;
        // We already output the first row
        int dataLeft = dataLens[headerIndex];

        while (dataLeft > 0) {
          int amountToCopy = Math.min(arrayBuffer.length, dataLeft);
          dataBuffer.getBytes(arrayBuffer, 0, currentOffset, amountToCopy);
          out.write(arrayBuffer, 0, amountToCopy);
          totalCopied += amountToCopy;
          currentOffset += amountToCopy;
          dataLeft -= amountToCopy;
        }
      }
      padFor64bitAlignment(out, totalCopied);
    } else {
      long totalCopied = 0;
      for (int headerIndex = 0; headerIndex < offsetsForEachHeader.length; headerIndex++) {
        HostMemoryBuffer dataBuffer = dataBuffers[headerIndex];
        ColumnOffsets[] offsets = offsetsForEachHeader[headerIndex];
        SerializedTableHeader header = headers[headerIndex];
        int numRowsForHeader = header.numRows;
        ColumnOffsets co = offsets[columnIndex];
        long currentOffset = co.data;
        int dataLeft = numRowsForHeader * type.sizeInBytes;
        while (dataLeft > 0) {
          int amountToCopy = Math.min(arrayBuffer.length, dataLeft);
          dataBuffer.getBytes(arrayBuffer, 0, currentOffset, amountToCopy);
          out.write(arrayBuffer, 0, amountToCopy);
          totalCopied += amountToCopy;
          currentOffset += amountToCopy;
          dataLeft -= amountToCopy;
        }
      }
      padFor64bitAlignment(out, totalCopied);
    }
  }

  // Package private for testing
  static int fillValidityData(byte[] dest, int destBitOffset, int lengthBits) {
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
    return totalCopied;
  }

  private static int copyValidityDataInto(byte[] dest, int destBitOffset,
                                          HostMemoryBuffer src, long baseSrcByteOffset,
                                          int srcBitOffset, int lengthBits) {
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

      // shifting right only happens when the buffer runs out of space. This is not likely to
      // happen.

      byte result = src.getByte(srcStartBytes);
      result = (byte)((result & 0xFF) >>> srcShift);
      byte next =  src.getByte(srcStartBytes + 1);
      result |= (byte)(next << 8 - srcShift);
      result &= firstSrcMask;

      // The first time through we need to include the data already in dest.
      result |= dest[destStartBytes] & ~firstSrcMask;
      dest[destStartBytes] = result;

      for (int index = 1; index < lastIndex; index++) {
        result = next;
        result = (byte)((result & 0xFF) >>> srcShift);
        next = src.getByte(srcStartBytes + index + 1);
        result |= (byte)(next << 8 - srcShift);
        dest[index + destStartBytes] = result;
      }

      return bitsToCopy;
    } else {
      src.getBytes(dest, destStartBytes, srcStartBytes, (bitsToCopy + 7) / 8);
      return bitsToCopy;
    }
  }

  private static long concatValidityData(DataOutputStream out,
                                       int columnIndex,
                                       int numRows,
                                       SerializedTableHeader[] headers,
                                       ColumnOffsets[][] offsetsForEachHeader,
                                       HostMemoryBuffer[] dataBuffers,
                                       byte[] arrayBuffer) throws IOException {
    long validityLen = BitVectorHelper.getValidityLengthInBytes(numRows);

    int rowsStoredInArray = 0;
    for (int headerIndex = 0; headerIndex < headers.length; headerIndex++) {
      ColumnOffsets offsets = offsetsForEachHeader[headerIndex][columnIndex];
      HostMemoryBuffer dataBuffer = dataBuffers[headerIndex];
      SerializedTableHeader header = headers[headerIndex];
      int rowsLeftInHeader = header.numRows;
      int validityBitOffset = 0;
      while(rowsLeftInHeader > 0) {
        int rowsStoredJustNow;
        if (offsets.validityLen > 0) {
          rowsStoredJustNow = copyValidityDataInto(arrayBuffer, rowsStoredInArray, dataBuffer, offsets.validity, validityBitOffset, rowsLeftInHeader);
        } else {
          rowsStoredJustNow = fillValidityData(arrayBuffer, rowsStoredInArray, rowsLeftInHeader);
        }
        assert rowsStoredJustNow > 0;
        rowsLeftInHeader -= rowsStoredJustNow;
        rowsStoredInArray += rowsStoredJustNow;
        validityBitOffset += rowsStoredJustNow;
        if (rowsStoredInArray == arrayBuffer.length * 8) {
          out.write(arrayBuffer, 0, arrayBuffer.length);
          rowsStoredInArray = 0;
        }
      }
    }

    if (rowsStoredInArray > 0) {
      out.write(arrayBuffer, 0, (rowsStoredInArray + 7) / 8);
    }
    return padFor64bitAlignment(out, validityLen);
  }

  // package-private for testing
  static long copyValidityData(DataOutputStream out, ColumnVector column, long rowOffset,
                               long numRows, byte[] arrayBuffer) throws IOException {
    assert arrayBuffer.length > 1;
    long validityLen = BitVectorHelper.getValidityLengthInBytes(numRows);
    long maxValidityLen = BitVectorHelper.getValidityLengthInBytes(column.getRowCount());
    long byteOffset = (rowOffset / 8);
    long bytesLeft = validityLen;

    int lshift = (int) rowOffset % 8;
    if (lshift == 0) {
      while (bytesLeft > 0) {
        int amountToCopy = (int) Math.min(bytesLeft, arrayBuffer.length);
        column.copyHostBufferBytes(arrayBuffer, 0, ColumnVector.BufferType.VALIDITY, byteOffset, amountToCopy);
        out.write(arrayBuffer, 0, amountToCopy);
        bytesLeft -= amountToCopy;
        byteOffset += amountToCopy;
      }
    } else {
      int rshift = 8 - lshift;
      while (bytesLeft > 0) {
        int amountToCopy = (int) Math.min(bytesLeft, arrayBuffer.length - 1);
        // Need to read at least 1 more byte to be sure we get any spill over.
        int amountToCopyWithSpill = amountToCopy + 1;
        // if we are at the last byte of the validity vector, we need to stop at the end
        if (amountToCopyWithSpill + byteOffset > maxValidityLen) {
          // don't try to copy outside of the column's validity buffer
          amountToCopyWithSpill = amountToCopy;

          // set the byte after the last one we will copy to 0x00
          // s.t. we don't copy garbage bits to the output stream
          arrayBuffer[amountToCopy] = 0x00;
        }
        column.copyHostBufferBytes(arrayBuffer, 0, ColumnVector.BufferType.VALIDITY, byteOffset, amountToCopyWithSpill);

        byte currentByte = arrayBuffer[0];
        for (int byteIndex = 0; byteIndex < amountToCopy; byteIndex++) {
          byte nextByte = arrayBuffer[byteIndex + 1];
          arrayBuffer[byteIndex] = (byte) ((nextByte << rshift) | ((0xFF & currentByte) >> lshift));
          currentByte = nextByte;
        }

        out.write(arrayBuffer, 0, amountToCopy);
        bytesLeft -= amountToCopy;
        byteOffset += amountToCopy;
      }
    }
    return padFor64bitAlignment(out, validityLen);
  }

  private static long copyAndPad(DataOutputStream out, ColumnVector column,
                                 ColumnVector.BufferType buffer, long offset,
                                 long length, byte[] tmpBuffer) throws IOException {
    long left = length;
    long at = offset;
    while (left > 0) {
      int amountToCopy = (int) Math.min(left, tmpBuffer.length);
      column.copyHostBufferBytes(tmpBuffer, 0, buffer, at, amountToCopy);
      out.write(tmpBuffer, 0, amountToCopy);
      left -= amountToCopy;
      at += amountToCopy;
    }
    return padFor64bitAlignment(out, length);
  }

  private static long copyBasicData(DataOutputStream out, ColumnVector column, long rowOffset,
                                    long numRows, byte[] arrayBuffer) throws IOException {
    DType type = column.getType();
    long bytesToCopy = numRows * type.sizeInBytes;
    long srcOffset = rowOffset * type.sizeInBytes;
    return copyAndPad(out, column, ColumnVector.BufferType.DATA, srcOffset, bytesToCopy, arrayBuffer);
  }

  private static long copyStringData(DataOutputStream out, ColumnVector column, long rowOffset,
                                     long numRows, byte[] arrayBuffer) throws IOException {
    if (numRows > 0) {
      long startByteOffset = column.getStartStringOffset(rowOffset);
      long endByteOffset = column.getEndStringOffset(rowOffset + numRows - 1);
      long bytesToCopy = endByteOffset - startByteOffset;
      long srcOffset = startByteOffset;
      return copyAndPad(out, column, ColumnVector.BufferType.DATA, srcOffset, bytesToCopy, arrayBuffer);
    }
    return 0;
  }

  private static long copyStringOffsets(DataOutputStream out, ColumnVector column, long rowOffset,
                                        long numRows, byte[] arrayBuffer) throws IOException {
    // If an offset is copied over as a part of a slice the first entry may be non-zero.  This is
    // okay because we fix them up when they are deserialized
    long bytesToCopy = (numRows + 1) * 4;
    long srcOffset = rowOffset * 4;
    return copyAndPad(out, column, ColumnVector.BufferType.OFFSET, srcOffset, bytesToCopy, arrayBuffer);
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

    DataOutputStream dout;
    if (out instanceof DataOutputStream) {
      dout = (DataOutputStream) out;
    } else {
      dout = new DataOutputStream(new BufferedOutputStream(out));
    }
    // verify that all of the columns can be concated, we also need to verify that the sizes are going to work....
    int numColumns = 0;
    DType[] types;
    TimeUnit[] tu;
    long[] nullCounts;
    long numRows = 0;
    if (headers.length > 0) {
      numColumns = headers[0].numColumns;
      types = headers[0].types;
      tu = headers[0].tu;
      nullCounts = Arrays.copyOf(headers[0].nullCounts, numColumns);
      numRows = headers[0].numRows;
    } else {
      types = new DType[0];
      tu = new TimeUnit[0];
      nullCounts = new long[0];
    }

    for (int i = 1; i < headers.length; i++) {
      SerializedTableHeader other = headers[i];
      if (other.numColumns != numColumns) {
        throw new IllegalArgumentException("The number of columns did not match " + i + " " + other.numColumns + " != " + numColumns);
      }
      for (int col = 0; col < numColumns; col++) {
        if (other.types[col] != types[col]) {
          throw new IllegalArgumentException("Type mismatch for column " + col);
        }

        if (other.tu[col] != tu[col]) {
          throw new IllegalArgumentException("TimeUnit mismatch for column " + col);
        }
        nullCounts[col] += other.nullCounts[col];
      }
      numRows += other.numRows;
    }

    if (numRows > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("CANNOT BUILD A BATCH LARGER THAN " + Integer.MAX_VALUE + " rows");
    }

    ColumnOffsets [][] offsetsForEachHeader = new ColumnOffsets[headers.length][];
    for (int i = 0; i < headers.length; i++) {
      SerializedTableHeader header = headers[i];
      HostMemoryBuffer dataBuffer = dataBuffers[i];
      offsetsForEachHeader[i] = buildIndex(header, dataBuffer);
    }

    // Now write out the data
    dout.writeInt(SER_FORMAT_MAGIC_NUMBER);
    dout.writeShort(VERSION_NUMBER);
    dout.writeInt(numColumns);
    // TODO this should really be a long eventually...
    dout.writeInt((int) numRows);

    // Header for each column...
    for (int i = 0; i < numColumns; i++) {
      dout.writeInt(types[i].nativeId);
      dout.writeInt((int) nullCounts[i]);
      dout.writeInt(tu[i].getNativeId());
    }
    dout.writeLong(getConcatedDataSizeInBytes((int)numRows, nullCounts, types, headers,
        offsetsForEachHeader, dataBuffers));

    for (int columnIndex = 0; columnIndex < numColumns; columnIndex++) {
      writeAndConcatColumnToDataStream(dout, types[columnIndex],
          columnIndex, nullCounts[columnIndex], (int)numRows, headers, offsetsForEachHeader, dataBuffers);
    }
    dout.flush();
  }

  private static ColumnVector[] sliceOffColumnVectors(SerializedTableHeader header,
                                                      DeviceMemoryBuffer combinedBuffer,
                                                      HostMemoryBuffer combinedBufferOnHost) {
    ColumnOffsets[] columnOffsets = buildIndex(header, combinedBufferOnHost);
    DType[] dataTypes = header.types;
    long[] nullCounts = header.nullCounts;
    TimeUnit[] timeUnits = header.tu;
    long numRows = header.getNumRows();
    int numColumns = dataTypes.length;
    ColumnVector[] vectors = new ColumnVector[numColumns];
    boolean tableSuccess = false;
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
      tableSuccess = true;
      return vectors;
    } finally {
      if (validity != null) {
        validity.close();
      }

      if (data != null) {
        data.close();
      }

      if(offsets != null) {
        offsets.close();
      }

      if (!tableSuccess) {
        for (ColumnVector cv : vectors) {
          if (cv != null) {
            cv.close();
          }
        }
      }
    }
  }

  private static ColumnOffsets[] buildIndex(SerializedTableHeader header,
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
        validityLen = padFor64bitAlignment(BitVectorHelper.getValidityLengthInBytes(numRows));
        validity = bufferOffset;
        bufferOffset += validityLen;
      }

      if (type == DType.STRING || type == DType.STRING_CATEGORY) {
        offsetsLen = padFor64bitAlignment((numRows + 1) * 4);
        offsets = bufferOffset;
        int startStringOffset = buffer.getInt(bufferOffset);
        int endStringOffset = buffer.getInt(bufferOffset + (numRows * 4));
        bufferOffset += offsetsLen;

        dataLen = padFor64bitAlignment(endStringOffset - startStringOffset);
        data = bufferOffset;
        bufferOffset += dataLen;
      } else {
        dataLen = padFor64bitAlignment(type.sizeInBytes * numRows);
        data = bufferOffset;
        bufferOffset += dataLen;
      }
      ret[column] = new ColumnOffsets(validity, validityLen,
          offsets, offsetsLen,
          data, dataLen);
    }
    return ret;
  }

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
    // TODO this should really be a long eventually...
    private int numRows;

    private DType[] types;
    private long[] nullCounts;
    private TimeUnit[] tu;
    private long dataLen;

    private boolean initialized = false;
    private boolean dataRead = false;

    public SerializedTableHeader(DataInputStream din) throws IOException {
      readFrom(din);
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
      if (types == null) {
        return 0;
      }
      return types.length;
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
      // TODO this should really be a long eventually...
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
    DataInputStream din;
    if (in instanceof DataInputStream) {
      din = (DataInputStream) in;
    } else {
      din = new DataInputStream(in);
    }

    if (header.initialized &&
        (buffer.length >= header.dataLen)) {
      buffer.copyFromStream(0, din, header.dataLen);
      header.dataRead = true;
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

    ColumnVector[] vectors = null;
    try (HostMemoryBuffer hostBuffer = HostMemoryBuffer.allocate(header.dataLen);
         DeviceMemoryBuffer deviceFullBuffer = DeviceMemoryBuffer.allocate(header.dataLen)) {
      if (header.dataLen > 0) {
        hostBuffer.copyFromStream(0, in, header.dataLen);
        deviceFullBuffer.copyFromHostBuffer(hostBuffer);
      }
      vectors = sliceOffColumnVectors(header, deviceFullBuffer, hostBuffer);
      return new Table(vectors);
    } finally {
      if (vectors != null) {
        // The vectors are reference counted, Putting them in the table it will inc the ref count
        // so we should always close the vectors that we created.
        for (ColumnVector cv : vectors) {
          if (cv != null) {
            cv.close();
          }
        }
      }
    }
  }
}