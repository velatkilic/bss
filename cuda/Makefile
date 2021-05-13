GXX=nvcc

analyze_main: main
	cuda-memcheck ./main
	# nsys profile --stats=true ./main

main: bmp.o main.o utils.o
	$(GXX) -o main main.o bmp.o utils.o

main.o: utils.h bmp.h main.cu
	$(GXX) -c main.cu

utils.o: bmp.h utils.h utils.cu
	$(GXX) -c utils.cu

bmp.o: bmp.h bmp.cu managed.h
	$(GXX) -c bmp.cu

clean:
	rm -rf *.o main m1.bmp m2.bmp out1.bmp out2.bmp
