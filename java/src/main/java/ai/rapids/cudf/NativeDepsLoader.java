/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;

/**
 * This class will load the native dependencies.
 */
public class NativeDepsLoader {
  private static final Logger log = LoggerFactory.getLogger(NativeDepsLoader.class);
  private static final Loader[] loadOrder = new Loader[] {
      new Optional("nvrtc"),
      new Required("boost_filesystem"),
      new Required("rmm"),
      new Required("cudf"),
      new Required("cudfjni")
  };
  private static ClassLoader loader = NativeDepsLoader.class.getClassLoader();
  private static boolean loaded = false;

  private static abstract class Loader {
    protected final String baseName;
    public Loader(String baseName) {
      this.baseName = baseName;
    }

    public abstract void load(String os, String arch) throws IOException;
  }

  private static class Required extends Loader {
    public Required(String baseName) {
      super(baseName);
    }

    @Override
    public void load(String os, String arch) throws IOException {
      loadDep(os, arch, this.baseName, true);
    }
  }

  private static class Optional extends Loader {
    public Optional(String baseName) {
      super(baseName);
    }

    @Override
    public void load(String os, String arch) throws IOException {
      loadDep(os, arch, this.baseName, false);
    }
  }

  static synchronized void loadNativeDeps() {
    if (!loaded) {
      String os = System.getProperty("os.name");
      String arch = System.getProperty("os.arch");
      try {
        for (Loader toLoad : loadOrder) {
          toLoad.load(os, arch);
        }
        loaded = true;
      } catch (Throwable t) {
        log.error("Could not load cudf jni library...", t);
      }
    }
  }

  private static void loadDep(String os, String arch, String baseName, boolean required)
          throws IOException {
    String path = arch + "/" + os + "/" + System.mapLibraryName(baseName);
    File loc;
    URL resource = loader.getResource(path);
    if (resource == null) {
      // It looks like we are not running from the jar, or there are issues with the jar
      File f = new File("./target/native-deps/" + path);
      if (!f.exists()) {
        if (required) {
          throw new FileNotFoundException("Could not locate native dependency " + path);
        }
        // Not required so we will skip it
        return;
      }
      resource = f.toURL();
    }
    try (InputStream in = resource.openStream()) {
      loc = File.createTempFile(baseName, ".so");
      try (OutputStream out = new FileOutputStream(loc)) {
        byte[] buffer = new byte[1024 * 16];
        int read = 0;
        while ((read = in.read(buffer)) >= 0) {
          out.write(buffer, 0, read);
        }
      }
    }
    loc.deleteOnExit();
    System.load(loc.getAbsolutePath());
    loc.delete();
  }

  public static boolean libraryLoaded() {
    if (!loaded) {
      loadNativeDeps();
    }
    return loaded;
  }
}
