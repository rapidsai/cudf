objects = rolling.o rolling_example.o

all: $(objects)
	nvcc $(objects) -o rolling_example

%.o: %.cu
	nvcc -std=c++11 -dc $< -o $@

clean:
	rm $(objects) rolling_example
