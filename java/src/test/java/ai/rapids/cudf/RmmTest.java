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
  public void testRmmEventHandler(int rmmAllocMode) {
    AtomicInteger invokedCount = new AtomicInteger();
    AtomicLong amountRequested = new AtomicLong();

    RmmEventHandler handler = new RmmEventHandler() {
      @Override
      public boolean onAllocFailure(long sizeRequested) {
        int count = invokedCount.incrementAndGet();
        amountRequested.set(sizeRequested);
        return count != 3;
      }
    };

    Rmm.initialize(rmmAllocMode, false, 512 * 1024 * 1024, handler);
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
      ignored.printStackTrace(System.err);
    }

    assertEquals(3, invokedCount.get());
    assertEquals(requested, amountRequested.get());

    // verify after a failure we can still allocate something more reasonable
    requested = 8192;
    addr = Rmm.alloc(requested, 0);
    Rmm.free(addr, 0);
  }

  @Test
  public void testRmmSetEventHandler() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, false, 0L);
    // installing an event handler the first time should not be an error
    Rmm.setEventHandler(new RmmEventHandler() {
      @Override
      public boolean onAllocFailure(long sizeRequested) {
        return false;
      }
    });

    // installing a second event handler is an error
    RmmEventHandler otherHandler = new RmmEventHandler() {
      @Override
      public boolean onAllocFailure(long sizeRequested) {
        return true;
      }
    };
    assertThrows(RmmException.class, () -> Rmm.setEventHandler(otherHandler));
  }

  @Test
  public void testRmmClearEventHandler() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, false, 0L);
    // clearing the event handler when it isn't set is not an error
    Rmm.clearEventHandler();

    // create an event handler that will always retry
    RmmEventHandler retryHandler = new RmmEventHandler() {
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
      ignored.printStackTrace(System.err);
    }
  }
}
