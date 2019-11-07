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

import java.util.HashSet;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Provides a callback API that can be used to track device memory usage.  The
 * reported usage is neither exact nor complete, but it should give a picture of memory usage by
 * this library. It does not currently track internal intermediate allocations done by the library.
 * It also does not track any memory fragmentation that might prevent it from being used in the
 * future.
 * <br />
 * The callbacks take the form of <ol>
 *   <li>prediction</li>
 *   <li>allocation</li>
 *   <li>endPrediction</li>
 *   <li>deallocation</li>
 * </ol>
 * An allocation indicates that memory was allocated. One or more allocation calls should typically
 * happen between a prediction and endPrediction call, but it is not guaranteed. It is guaranteed
 * that an endPrediction call will happen for each prediction call. A prediction
 * indicates an educated guess as to how much memory an operation is likely to take. In some cases
 * we know exactly what it takes, but in others it is really just a guess and it may guess too much
 * or too little memory. Typically this is used to try and tell if we are getting close to memory
 * limits and possibly release cached data on the GPU to reduce the amount of memory being used.
 * In those cases a prediction should be
 * treated like a memory reservation that is deducted from when an allocation happens until it is
 * used up.
 * <br />
 * Please note that if your interactions with cuDF are threaded the MemoryListener will be called
 * from each of those threads too. In this case each allocation or endPrediction call is associated
 * with the previous prediction call on the same thread.
 */
public abstract class MemoryListener {
  /**
   * A prediction about how much memory is about to be used for an operation. Predictions will not
   * nest but each thread is independent of other threads.
   * @param amount the number of bytes predicted to be used.
   * @param note a string that can be used for debugging to help keep track of what is making the
   *             prediction.
   */
  public abstract void prediction(long amount, String note);

  /**
   * An actual memory allocation has occurred.
   * @param amount the number of bytes actually allocated.
   * @param id a unique number that can be used for debugging. The same number will be sent to
   *           deallocate.
   */
  public abstract void allocation(long amount, long id);

  /**
   * Indicates that the previous prediction on this thread is no longer in force. Predictions should
   * never nest.
   * @param note a string that can be used for debugging.  This should be the same string that was
   *             passed into prediction.
   */
  public abstract void endPrediction(String note);

  /**
   * Memory was actually deallocated.
   * @param amount the number of bytes released and can be reused.
   * @param id a unique number that can be used for debugging.  The same number would have been sent
   *           to allocate.
   */
  public abstract void deallocation(long amount, long id);

  /**
   * Holds the set of all listeners. An <code>AtomicReference&lt;HashSet&gt;</code> is used here to
   * optimize the common read path over the less common write path (add/remove listeners). Other
   * data structures like a ConcurrentHashMap still involve a lock on the read path where as this
   * eliminates any locking on reads.
   */
  private static final AtomicReference<HashSet<MemoryListener>> listeners =
      new AtomicReference<>(new HashSet<>());

  /**
   * Register a memory listener. If the memory listener is already registered it will not install
   * duplicates. This should be done before any column operations happen or you risk missing those
   * operations.
   * @param listener the listener to start sending events to.
   */
  public static void registerDeviceListener(final MemoryListener listener) {
    listeners.getAndUpdate((orig) -> {
      HashSet<MemoryListener> ret = new HashSet<>(orig);
      ret.add(listener);
      return ret;
    });
  }

  /**
   * Start a prediction.  Don't call this directly please use the DevicePrediction class instead.
   * @param amount number of bytes predicted.
   * @param note what is the prediction for.
   */
  static void devicePrediction(final long amount, String note) {
    listeners.get().forEach((l) -> l.prediction(amount, note));
  }

  /**
   * Memory was actually allocated.  This should typically be done on a per column basis and the id
   * should be the internal id of the column. For most operations this should automatically be done
   * for you unless you are adding in a new way to allocate the device data for a column.
   * @param amount the number of bytes allocated.
   * @param id the id of the column.
   */
  static void deviceAllocation(final long amount, final long id) {
    listeners.get().forEach((l) -> l.allocation(amount, id));
  }

  /**
   * End a prediction.  Don't call this directly please use the DevicePrediction class instead.
   * @param note what is the prediction for.
   */
  static void deviceEndPrediction(String note) {
    listeners.get().forEach((l) -> l.endPrediction(note));
  }

  /**
   * Memory was actually deallocated.  This should typically be done on a per column basis and the
   * id should be the internal id of the column. For most operations this should automatically be
   * done for you unless you are manually freeing the device data for a column.
   * @param amount the number of bytes released.
   * @param id the id of the column.
   */
  static void deviceDeallocation(final long amount, final long id) {
    listeners.get().forEach((l) -> l.deallocation(amount, id));
  }
}