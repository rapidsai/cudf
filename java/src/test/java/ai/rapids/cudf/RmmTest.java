/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

public class RmmTest {
  private static final long TOO_MUCH_MEMORY = 3L * 1024 * 1024 * 1024 * 1024 * 1024 * 1024;

  @BeforeEach
  public void setup() {
    if (Rmm.isInitialized()) {
      Rmm.shutdown();
    }
  }

  @AfterEach
  public void teardown() {
    if (Rmm.isInitialized()) {
      Rmm.shutdown();
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {RmmAllocationMode.CUDA_DEFAULT, RmmAllocationMode.POOL})
  public void testEventHandler(int rmmAllocMode) {
    AtomicInteger invokedCount = new AtomicInteger();
    AtomicLong amountRequested = new AtomicLong();

    RmmEventHandler handler = new BaseRmmEventHandler() {
      @Override
      public boolean onAllocFailure(long sizeRequested) {
        int count = invokedCount.incrementAndGet();
        amountRequested.set(sizeRequested);
        return count != 3;
      }
    };

    Rmm.initialize(rmmAllocMode, false, 512 * 1024 * 1024);
    Rmm.setEventHandler(handler);
    long addr = Rmm.alloc(1024, 0);
    Rmm.free(addr, 0);
    assertTrue(addr != 0);
    assertEquals(0, invokedCount.get());

    // Try to allocate too much
    long requested = TOO_MUCH_MEMORY;
    try {
      addr = Rmm.alloc(requested, 0);
      Rmm.free(addr, 0);
      fail("should have failed to allocate");
    } catch (OutOfMemoryError | RmmException ignored) {
    }

    assertEquals(3, invokedCount.get());
    assertEquals(requested, amountRequested.get());

    // verify after a failure we can still allocate something more reasonable
    requested = 8192;
    addr = Rmm.alloc(requested, 0);
    Rmm.free(addr, 0);
  }

  @Test
  public void testSetEventHandlerTwice() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, false, 0L);
    // installing an event handler the first time should not be an error
    Rmm.setEventHandler(new BaseRmmEventHandler() {
      @Override
      public boolean onAllocFailure(long sizeRequested) {
        return false;
      }
    });

    // installing a second event handler is an error
    RmmEventHandler otherHandler = new BaseRmmEventHandler() {
      @Override
      public boolean onAllocFailure(long sizeRequested) {
        return true;
      }
    };
    assertThrows(RmmException.class, () -> Rmm.setEventHandler(otherHandler));
  }

  @Test
  public void testClearEventHandler() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, false, 0L);
    // clearing the event handler when it isn't set is not an error
    Rmm.clearEventHandler();

    // create an event handler that will always retry
    RmmEventHandler retryHandler = new BaseRmmEventHandler() {
      @Override
      public boolean onAllocFailure(long sizeRequested) {
        return true;
      }
    };

    Rmm.setEventHandler(retryHandler);
    Rmm.clearEventHandler();

    // verify handler is no longer installed, alloc should fail
    try {
      long addr = Rmm.alloc(TOO_MUCH_MEMORY, 0);
      Rmm.free(addr, 0);
      fail("should have failed to allocate");
    } catch (OutOfMemoryError | RmmException ignored) {
    }
  }

  @Test
  public void testAllocOnlyThresholds() {
    final AtomicInteger allocInvocations = new AtomicInteger(0);
    final AtomicInteger deallocInvocations = new AtomicInteger(0);
    final AtomicLong allocated = new AtomicLong(0);

    Rmm.initialize(RmmAllocationMode.POOL, false, 1024 * 1024L);

    RmmEventHandler handler = new RmmEventHandler() {
      @Override
      public boolean onAllocFailure(long sizeRequested) {
        return false;
      }

      @Override
      public long[] getAllocThresholds() {
        return new long[] { 32 * 1024, 8 * 1024 };
      }

      @Override
      public long[] getDeallocThresholds() {
        return null;
      }

      @Override
      public void onAllocThreshold(long totalAllocSize) {
        allocInvocations.getAndIncrement();
        allocated.set(totalAllocSize);
      }

      @Override
      public void onDeallocThreshold(long totalAllocSize) {
        deallocInvocations.getAndIncrement();
      }
    };

    Rmm.setEventHandler(handler);
    long[] addrs = new long[5];
    try {
      addrs[0] = Rmm.alloc(6 * 1024, 0);
      assertEquals(0, allocInvocations.get());
      addrs[1] = Rmm.alloc(2 * 1024, 0);
      assertEquals(1, allocInvocations.get());
      assertEquals(8 * 1024, allocated.get());
      addrs[2] = Rmm.alloc(21 * 1024, 0);
      assertEquals(1, allocInvocations.get());
      addrs[3] = Rmm.alloc(8 * 1024, 0);
      assertEquals(2, allocInvocations.get());
      assertEquals(37 * 1024, allocated.get());
      addrs[4] = Rmm.alloc(8 * 1024, 0);
      assertEquals(2, allocInvocations.get());
    } finally {
      for (long addr : addrs) {
        if (addr != 0) {
          Rmm.free(addr, 0);
        }
      }
    }

    assertEquals(2, allocInvocations.get());
    assertEquals(0, deallocInvocations.get());
  }

  @Test
  public void testThresholds() {
    final AtomicInteger allocInvocations = new AtomicInteger(0);
    final AtomicInteger deallocInvocations = new AtomicInteger(0);
    final AtomicLong allocated = new AtomicLong(0);

    Rmm.initialize(RmmAllocationMode.POOL, false, 1024 * 1024L);

    RmmEventHandler handler = new RmmEventHandler() {
      @Override
      public boolean onAllocFailure(long sizeRequested) {
        return false;
      }

      @Override
      public long[] getAllocThresholds() {
        return new long[] { 8 * 1024 };
      }

      @Override
      public long[] getDeallocThresholds() {
        return new long[] { 6 * 1024 };
      }

      @Override
      public void onAllocThreshold(long totalAllocSize) {
        allocInvocations.getAndIncrement();
        allocated.set(totalAllocSize);
      }

      @Override
      public void onDeallocThreshold(long totalAllocSize) {
        deallocInvocations.getAndIncrement();
        allocated.set(totalAllocSize);
      }
    };

    Rmm.setEventHandler(handler);
    long[] addrs = new long[5];
    try {
      addrs[0] = Rmm.alloc(6 * 1024, 0);
      assertEquals(0, allocInvocations.get());
      assertEquals(0, deallocInvocations.get());
      Rmm.free(addrs[0], 0);
      addrs[0] = 0;
      assertEquals(0, allocInvocations.get());
      assertEquals(1, deallocInvocations.get());
      assertEquals(0, allocated.get());
      addrs[0] = Rmm.alloc(12 * 1024, 0);
      assertEquals(1, allocInvocations.get());
      assertEquals(1, deallocInvocations.get());
      assertEquals(12 * 1024, allocated.get());
      addrs[1] = Rmm.alloc(6 * 1024, 0);
      assertEquals(1, allocInvocations.get());
      assertEquals(1, deallocInvocations.get());
      Rmm.free(addrs[0], 0);
      addrs[0] = 0;
      assertEquals(1, allocInvocations.get());
      assertEquals(1, deallocInvocations.get());
      addrs[0] = Rmm.alloc(4 * 1024, 0);
      assertEquals(2, allocInvocations.get());
      assertEquals(1, deallocInvocations.get());
      assertEquals(10 * 1024, allocated.get());
      Rmm.free(addrs[1], 0);
      addrs[1] = 0;
      assertEquals(2, allocInvocations.get());
      assertEquals(2, deallocInvocations.get());
      assertEquals(4 * 1024, allocated.get());
      Rmm.free(addrs[0], 0);
      addrs[0] = 0;
      assertEquals(2, allocInvocations.get());
      assertEquals(2, deallocInvocations.get());
    } finally {
      for (long addr : addrs) {
        if (addr != 0) {
          Rmm.free(addr, 0);
        }
      }
    }

    assertEquals(2, allocInvocations.get());
    assertEquals(2, deallocInvocations.get());
  }

  @Test
  public void testExceptionHandling() {
    Rmm.initialize(RmmAllocationMode.POOL, false, 1024 * 1024L);

    RmmEventHandler handler = new RmmEventHandler() {
      @Override
      public boolean onAllocFailure(long sizeRequested) {
        throw new AllocFailException();
      }

      @Override
      public long[] getAllocThresholds() {
        return new long[] { 8 * 1024 };
      }

      @Override
      public long[] getDeallocThresholds() {
        return new long[] { 6 * 1024 };
      }

      @Override
      public void onAllocThreshold(long totalAllocSize) {
        throw new AllocThresholdException();
      }

      @Override
      public void onDeallocThreshold(long totalAllocSize) {
        throw new DeallocThresholdException();
      }
    };

    Rmm.setEventHandler(handler);
    long addr = Rmm.alloc(6 * 1024, 0);
    assertThrows(DeallocThresholdException.class, () -> Rmm.free(addr, 0));
    assertThrows(AllocThresholdException.class, () -> Rmm.alloc(12 * 1024, 0));
    assertThrows(AllocFailException.class, () -> Rmm.alloc(TOO_MUCH_MEMORY, 0));
  }

  private static class AllocFailException extends RuntimeException {}
  private static class AllocThresholdException extends RuntimeException {}
  private static class DeallocThresholdException extends RuntimeException {}

  private static abstract class BaseRmmEventHandler implements RmmEventHandler {
    @Override
    public long[] getAllocThresholds() {
      return null;
    }

    @Override
    public long[] getDeallocThresholds() {
      return null;
    }

    @Override
    public void onAllocThreshold(long totalAllocSize) {
    }

    @Override
    public void onDeallocThreshold(long totalAllocSize) {
    }
  }
}
