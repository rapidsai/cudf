/*
 *  SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * JNI interface to cudf::chunked_pack.
 *
 * ChunkedPack has an Iterator-like API with the familiar `hasNext` and `next`
 * methods. `next` should be used in a loop until `hasNext` returns false.
 *
 * However, `ChunkedPack.next` is special because it takes a `DeviceMemoryBuffer` as a
 * parameter, which means that the caller can call `next` giving any bounce buffer it
 * may have previously allocated. No requirement exists that the bounce buffer be the
 * same each time, the only requirement is that their sizes are all the same, and match
 * the size that was passed to `Table.makeChunkedPack` (which instantiates this class).
 *
 * The user of `ChunkedPack` must close `.close()` when done using it to clear up both
 * host and device resources.
 */
public class ChunkedPack implements AutoCloseable {
  long nativePtr;

  /**
   * This constructor is invoked by `Table.makeChunkedPack` after creating a native
   * `cudf::chunked_pack`.
   * @param nativePtr pointer to a `cudf::chunked_pack`
   */
  public ChunkedPack(long nativePtr) {
    this.nativePtr = nativePtr;
  }

  /**
   * Get the final contiguous size of the table we are packing. This is
   * the size that the final buffer should be, just like if the user called
   * `cudf::pack` instead.
   * @return the total number of bytes for the table in contiguous layout
   */
  public long getTotalContiguousSize() {
    return chunkedPackGetTotalContiguousSize(nativePtr);
  }

  /**
   * Method to be called to ensure that `ChunkedPack` has work left.
   * This method should be invoked followed by a call to `next`, until
   * `hasNext` returns false.
   * @return true if there is work left to be done (`next` should be called),
   *         false otherwise.
   */
  public boolean hasNext() {
    return chunkedPackHasNext(nativePtr);
  }

  /**
   * Place the next contiguous chunk of our table into `userPtr`.
   *
   * This method throws if `hasNext` is false.
   * @param userPtr the bounce buffer to use for this iteration
   * @return the number of bytes that we were able to place in `userPtr`. This is
   *         at most `userPtr.getLength()`.
   */
  public long next(DeviceMemoryBuffer userPtr) {
    return chunkedPackNext(nativePtr, userPtr.getAddress(), userPtr.getLength());
  }

  /**
   * Generates opaque table metadata that can be unpacked via `cudf::unpack`
   * at a later time.
   * @return a `PackedColumnMetadata` instance referencing cuDF packed table metadata
   */
  public PackedColumnMetadata buildMetadata() {
    return new PackedColumnMetadata(chunkedPackBuildMetadata(nativePtr));
  }

  @Override
  public void close() {
    try {
      chunkedPackDelete(nativePtr);
    } finally {
      nativePtr = 0;
    }
  }

  private static native long chunkedPackGetTotalContiguousSize(long nativePtr);
  private static native boolean chunkedPackHasNext(long nativePtr);
  private static native long chunkedPackNext(long nativePtr, long userPtr, long userPtrSize);
  private static native long chunkedPackBuildMetadata(long nativePtr);
  private static native void chunkedPackDelete(long nativePtr);
}
