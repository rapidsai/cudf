all: libgdf.so

libgdf.so: src/column.cpp src/unaryops.cu src/binaryops.cu
	nvcc -Iinclude -shared -o $@ $+

test:
	python setup.py build_ext --inplace
	py.test -v

clean:
	rm -f libgdf.so