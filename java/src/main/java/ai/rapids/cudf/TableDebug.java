/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.rapids.cudf;

import java.io.PrintStream;
import java.util.Locale;

public class TableDebug {

  private static final String OUTPUT_STREAM = "ai.rapids.cudf.debug.output";

  enum Output {
    STDOUT,
    STDERR
  }

  public static class Builder {
    private PrintStream print = System.err;

    public Builder() {
      Output outputMode = Output.STDERR;
      try {
        outputMode = Output.valueOf(
            System.getProperty(OUTPUT_STREAM, Output.STDERR.name().toUpperCase(Locale.US))
        );
      } catch (Throwable ignored) {
      }
      switch (outputMode) {
        case STDOUT:
          print = System.out;
          break;
        case STDERR:
        default:
          print = System.err;
      }
    }

    public Builder withPrintStream(PrintStream ps) {
      print = ps;
      return this;
    }

    public final TableDebug build() {
      return new TableDebug(
          print
      );
    }
  }

  public static Builder builder() {
    return new Builder();
  }

  private static final TableDebug DEFAULT_DEBUG = builder().build();

  public static TableDebug get() {
    return DEFAULT_DEBUG;
  }

  private final PrintStream out;

  private TableDebug(PrintStream printStream) {
    out = printStream;
  }

  /**
   * Print the contents of a table. Note that this should never be
   * called from production code, as it is very slow.  Also note that this is not production
   * code.  You might need/want to update how the data shows up or add in support for more
   * types as this really is just for debugging.
   * @param name  the name of the table to print out.
   * @param table the table to print out.
   */
  public synchronized void debug(String name, Table table) {
    out.println("DEBUG " + name + " " + table);
    for (int col = 0; col < table.getNumberOfColumns(); col++) {
      debug(String.valueOf(col), table.getColumn(col));
    }
  }

  /**
   * Print the contents of a column. Note that this should never be
   * called from production code, as it is very slow.  Also note that this is not production
   * code.  You might need/want to update how the data shows up or add in support for more
   * types as this really is just for debugging.
   * @param name the name of the column to print out.
   * @param col  the column to print out.
   */
  public synchronized void debug(String name, ColumnView col) {
    debugGPUAddrs(name, col);
    try (HostColumnVector hostCol = col.copyToHost()) {
      debug(name, hostCol);
    }
  }

  private synchronized void debugGPUAddrs(String name, ColumnView col) {
    try (BaseDeviceMemoryBuffer data = col.getData();
         BaseDeviceMemoryBuffer validity = col.getValid()) {
      out.println("GPU COLUMN " + name + " - NC: " + col.getNullCount()
          + " DATA: " + data + " VAL: " + validity);
    }
    if (col.getType() == DType.STRUCT) {
      for (int i = 0; i < col.getNumChildren(); i++) {
        try (ColumnView child = col.getChildColumnView(i)) {
          debugGPUAddrs(name + ":CHILD_" + i, child);
        }
      }
    } else if (col.getType() == DType.LIST) {
      try (ColumnView child = col.getChildColumnView(0)) {
        debugGPUAddrs(name + ":DATA", child);
      }
    }
  }


  /**
   * Print the contents of a column. Note that this should never be
   * called from production code, as it is very slow.  Also note that this is not production
   * code.  You might need/want to update how the data shows up or add in support for more
   * types as this really is just for debugging.
   * @param name    the name of the column to print out.
   * @param hostCol the column to print out.
   */
  public synchronized void debug(String name, HostColumnVectorCore hostCol) {
    DType type = hostCol.getType();
    out.println("COLUMN " + name + " - " + type);
    if (type.isDecimalType()) {
      for (int i = 0; i < hostCol.getRowCount(); i++) {
        if (hostCol.isNull(i)) {
          out.println(i + " NULL");
        } else {
          out.println(i + " " + hostCol.getBigDecimal(i));
        }
      }
    } else if (DType.STRING.equals(type)) {
      for (int i = 0; i < hostCol.getRowCount(); i++) {
        if (hostCol.isNull(i)) {
          out.println(i + " NULL");
        } else {
          out.println(i + " \"" + hostCol.getJavaString(i) + "\" " +
              hexString(hostCol.getUTF8(i)));
        }
      }
    } else if (DType.INT32.equals(type)
        || DType.INT8.equals(type)
        || DType.INT16.equals(type)
        || DType.INT64.equals(type)
        || DType.TIMESTAMP_DAYS.equals(type)
        || DType.TIMESTAMP_SECONDS.equals(type)
        || DType.TIMESTAMP_MICROSECONDS.equals(type)
        || DType.TIMESTAMP_MILLISECONDS.equals(type)
        || DType.TIMESTAMP_NANOSECONDS.equals(type)
        || DType.UINT8.equals(type)
        || DType.UINT16.equals(type)
        || DType.UINT32.equals(type)
        || DType.UINT64.equals(type)) {
      debugInteger(hostCol, type);
    } else if (DType.BOOL8.equals(type)) {
      for (int i = 0; i < hostCol.getRowCount(); i++) {
        if (hostCol.isNull(i)) {
          out.println(i + " NULL");
        } else {
          out.println(i + " " + hostCol.getBoolean(i));
        }
      }
    } else if (DType.FLOAT64.equals(type)) {
      for (int i = 0; i < hostCol.getRowCount(); i++) {
        if (hostCol.isNull(i)) {
          out.println(i + " NULL");
        } else {
          out.println(i + " " + hostCol.getDouble(i));
        }
      }
    } else if (DType.FLOAT32.equals(type)) {
      for (int i = 0; i < hostCol.getRowCount(); i++) {
        if (hostCol.isNull(i)) {
          out.println(i + " NULL");
        } else {
          out.println(i + " " + hostCol.getFloat(i));
        }
      }
    } else if (DType.STRUCT.equals(type)) {
      for (int i = 0; i < hostCol.getRowCount(); i++) {
        if (hostCol.isNull(i)) {
          out.println(i + " NULL");
        } // The struct child columns are printed out later on.
      }
      for (int i = 0; i < hostCol.getNumChildren(); i++) {
        debug(name + ":CHILD_" + i, hostCol.getChildColumnView(i));
      }
    } else if (DType.LIST.equals(type)) {
      out.println("OFFSETS");
      for (int i = 0; i < hostCol.getRowCount(); i++) {
        if (hostCol.isNull(i)) {
          out.println(i + " NULL");
        } else {
          out.println(i + " [" + hostCol.getStartListOffset(i) + " - " +
              hostCol.getEndListOffset(i) + ")");
        }
      }
      debug(name + ":DATA", hostCol.getChildColumnView(0));
    } else {
      out.println("TYPE " + type + " NOT SUPPORTED FOR DEBUG PRINT");
    }
  }


  private void debugInteger(HostColumnVectorCore hostCol, DType intType) {
    for (int i = 0; i < hostCol.getRowCount(); i++) {
      if (hostCol.isNull(i)) {
        out.println(i + " NULL");
      } else {
        final int sizeInBytes = intType.getSizeInBytes();
        final Object value;
        switch (sizeInBytes) {
          case Byte.BYTES:
            value = hostCol.getByte(i);
            break;
          case Short.BYTES:
            value = hostCol.getShort(i);
            break;
          case Integer.BYTES:
            value = hostCol.getInt(i);
            break;
          case Long.BYTES:
            value = hostCol.getLong(i);
            break;
          default:
            throw new IllegalArgumentException("INFEASIBLE: Unsupported integer-like type " + intType);
        }
        out.println(i + " " + value);
      }
    }
  }


  private static String hexString(byte[] bytes) {
    StringBuilder str = new StringBuilder();
    for (byte b : bytes) {
      str.append(String.format("%02x", b & 0xff));
    }
    return str.toString();
  }
}
