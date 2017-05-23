all: libgdf.so

libgdf.so: src/gdf.cu
	nvcc -Iinclude -shared -o $@ $<

clean:
	rm -f libgdf.so