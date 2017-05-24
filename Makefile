all: libgdf.so

libgdf.so: src/column.cpp src/unaryops.cu
	nvcc -Iinclude -shared -o $@ $+

clean:
	rm -f libgdf.so