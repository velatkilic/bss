GCC=nvcc

main: bmp.o main.o
	$(GCC) -o main main.o bmp.o

main.o: managed.h main.cu
	$(GCC) -c main.cu

bmp.o: bmp.h bmp.cu managed.h
	$(GCC) -c bmp.cu

clean:
	rm -rf *.o main
