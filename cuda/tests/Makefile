GXX=nvcc
FLAGS=-lineinfo
test_mean: mean.o test_mean.o
	$(GXX) -o test_mean mean.o test_mean.o

test_mean.o: mean.h
	$(GXX) -c test_mean.cu $(FLAGS)

mean.o: mean.h mean.cu
	$(GXX) -c mean.cu $(FLAGS)

clean:
	rm -rf *.o test_mean
