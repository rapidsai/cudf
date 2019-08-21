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
      long start = column.getStartStringOffset(rowOffset);
      long end = column.getEndStringOffset(rowOffset + numRows - 1);
      total += padFor64bitAlignment(end - start);
    } else {
      total += padFor64bitAlignment(column.getType().sizeInBytes * numRows);
    }
    return total;
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
      long nc = columns[i].getNullCount();
      assert nc == (int) nc : "can only support an int for indexes";
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
      long nc = columns[i].getNullCount();
      if (nc != 0 && (rowOffset != 0 || numRows != columns[i].getRowCount())) {
        // TODO This is a hack.  We need to get partition to calculate this for us properly in the
        // future.
        nc = 1;
      }
      // TODO this should be a long eventually.
      out.writeInt((int) nc);
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
    long startByteOffset = column.getStartStringOffset(rowOffset);
    long endByteOffset = column.getEndStringOffset(rowOffset + numRows - 1);
    long bytesToCopy = endByteOffset - startByteOffset;
    long srcOffset = startByteOffset;
    return copyAndPad(out, column, ColumnVector.BufferType.DATA, srcOffset, bytesToCopy, arrayBuffer);
  }

  private static long copyStringOffsets(DataOutputStream out, ColumnVector column, long rowOffset,
                                        long numRows, byte[] arrayBuffer) throws IOException {
    // If an offset is copied over as a part of a slice the first entry may be non-zero.  This is
    // okay because we fix them up when they are deserialized
    long bytesToCopy = (numRows + 1) * 4;
    long srcOffset = rowOffset * 4;
    return copyAndPad(out, column, ColumnVector.BufferType.OFFSET, srcOffset, bytesToCopy, arrayBuffer);
  }

  private static ColumnVector[] sliceOffColumnVectors(DType[] dataTypes, long[] nullCounts,
                                                      TimeUnit[] timeUnits, long numRows,
                                                      DeviceMemoryBuffer combinedBuffer,
                                                      HostMemoryBuffer combinedBufferOnHost) {
    int numColumns = dataTypes.length;
    long combinedBufferOffset = 0;
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

        if (nullCount > 0) {
          long len = padFor64bitAlignment(BitVectorHelper.getValidityLengthInBytes(numRows));
          validity = combinedBuffer.slice(combinedBufferOffset, len);
          combinedBufferOffset += len;
        }

        if (type == DType.STRING || type == DType.STRING_CATEGORY) {
          long offsetsLen = padFor64bitAlignment((numRows + 1) * 4);
          offsets = combinedBufferOnHost.slice(combinedBufferOffset, offsetsLen);
          combinedBufferOffset += offsetsLen;
          int startStringOffset = offsets.getInt(0);
          int endStringOffset = offsets.getInt(numRows * 4);
          long deviceDataLen = padFor64bitAlignment(endStringOffset - startStringOffset);

          if (deviceDataLen == 0) {
            // The vector is full of null strings, this is a rare corner case, but here is the
            // simplest place to work around it
            data = DeviceMemoryBuffer.allocate(1);
          } else {
            data = combinedBuffer.slice(combinedBufferOffset, deviceDataLen);
          }
          combinedBufferOffset += deviceDataLen;
        } else {
          long deviceDataLen = padFor64bitAlignment(type.sizeInBytes * numRows);
          data = combinedBuffer.slice(combinedBufferOffset, deviceDataLen);
          combinedBufferOffset += deviceDataLen;
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

    try {
      int num = din.readInt();
      if (num != SER_FORMAT_MAGIC_NUMBER) {
        throw new IllegalStateException("THIS DOES NOT LOOK LIKE CUDF SERIALIZED DATA. " +
            "Expected magic number " + SER_FORMAT_MAGIC_NUMBER + " Found " + num);
      }
    } catch (EOFException e) {
      // If we get an EOF at the very beginning don't treat it as an error because we may
      // have finished reading everything...
      return null;
    }
    short version = din.readShort();
    if (version != VERSION_NUMBER) {
      throw new IllegalStateException("READING THE WRONG SERIALIZATION FORMAT VERSION FOUND "
          + version + " EXPECTED " + VERSION_NUMBER);
    }
    int numColumns = din.readInt();
    // TODO this should really be a long eventually...
    int numRows = din.readInt();

    DType[] types = new DType[numColumns];
    long[] nc = new long[numColumns];
    TimeUnit[] tu = new TimeUnit[numColumns];
    for (int i = 0; i < numColumns; i++) {
      types[i] = DType.fromNative(din.readInt());
      nc[i] = din.readInt();
      tu[i] = TimeUnit.fromNative(din.readInt());
    }

    long dataLen = din.readLong();
    ColumnVector[] vectors = null;
    try (HostMemoryBuffer hostBuffer = HostMemoryBuffer.allocate(dataLen);
         DeviceMemoryBuffer deviceFullBuffer = DeviceMemoryBuffer.allocate(dataLen)) {
      hostBuffer.copyFromStream(0, in, dataLen);
      deviceFullBuffer.copyFromHostBuffer(hostBuffer);
      vectors = sliceOffColumnVectors(types, nc, tu, numRows, deviceFullBuffer, hostBuffer);
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