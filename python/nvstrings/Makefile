#
# Builds just the python module interface for nvstrings.py, nvcategory.py, nvtext.py
#
BUILD=build
PY_HEADER_PATH=$(PYNI_PATH)/include/python${PYTHON_VERSION}m
PY_LIB_PATH=$(PYNI_PATH)/lib/libpython${PYTHON_VERSION}m.so
CFLAGS=-shared -std=c++14 -fPIC -Isrc -Iinclude -I$(PY_HEADER_PATH) -I/usr/local/cuda/include
LFLAGS=-L. $(PY_LIB_PATH) -lrt -L/usr/local/cuda/lib64 -lcudart
RPATH=-Wl,-rpath,.

all: pyniNVStrings.so pyniNVCategory.so pyniNVText.so

pyniNVStrings.so: cpp/pystrings.cpp ../cpp/include/NVStrings.h
	g++ $(CFLAGS) $< $(LFLAGS) -lNVStrings -o $@ -Wl,-soname,$@ $(RPATH)
	cp $@ $(BUILD)/.

pyniNVCategory.so: cpp/pycategory.cpp ../cpp/include/NVStrings.h ../cpp/include/NVCategory.h
	g++ $(CFLAGS) $< $(LFLAGS) -lNVCategory -lNVStrings -o $@ -Wl,-soname,$@ $(RPATH)
	cp $@ $(BUILD)/.

pyniNVText.so: cpp/pytext.cpp ../cpp/include/NVText.h ../cpp/include/NVStrings.h ../cpp/include/NVCategory.h
	g++ $(CFLAGS) $< $(LFLAGS) -lNVText -lNVStrings -o $@ -Wl,-soname,$@ $(RPATH)
	cp $@ $(BUILD)/.

.PHONY: clean

clean:
	rm -f *.so $(BUILD)/*.so
