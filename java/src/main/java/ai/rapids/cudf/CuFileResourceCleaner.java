/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Keeps track and cleans a cuFile native resource.
 */
final class CuFileResourceCleaner extends MemoryCleaner.Cleaner {
  private static final Logger log = LoggerFactory.getLogger(CuFileResourceCleaner.class);

  private long pointer;
  private final CuFileResourceDestroyer destroyer;
  private boolean closed = false;

  CuFileResourceCleaner(long pointer, CuFileResourceDestroyer destroyer) {
    this.pointer = pointer;
    this.destroyer = destroyer;
    addRef();
  }

  long getPointer() {
    return pointer;
  }

  synchronized void close(Object resource) {
    delRef();
    if (closed) {
      logRefCountDebug("double free " + resource);
      throw new IllegalStateException("Close called too many times " + resource);
    }
    clean(false);
    closed = true;
  }

  @Override
  protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
    boolean neededCleanup = false;
    long origAddress = pointer;
    if (pointer != 0) {
      try {
        destroyer.destroy(pointer);
      } finally {
        // Always mark the resource as freed even if an exception is thrown.
        // We cannot know how far it progressed before the exception, and
        // therefore it is unsafe to retry.
        pointer = 0;
      }
      neededCleanup = true;
    }
    if (neededCleanup && logErrorIfNotClean) {
      log.error("A CUFile RESOURCE WAS LEAKED (ID: " + id + " " + Long.toHexString(origAddress) + ")");
      logRefCountDebug("Leaked cuFile resource");
    }
    return neededCleanup;
  }

  @Override
  public boolean isClean() {
    return pointer == 0;
  }
}
