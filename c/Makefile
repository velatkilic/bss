GCC=gcc
CFLAGS=-std=c99 -Wall -Wextra -pedantic

main: ImageStuff.o main.o utils.o
	$(GCC) -o main main.o ImageStuff.o utils.o -lm

main.o: ImageStuff.h main.c
	$(GCC) -c main.c $(CFLAGS)

utils.o: utils.h utils.c ImageStuff.h
	$(GCC) -c utils.c $(CFLAGS)

ImageStuff.o: ImageStuff.h ImageStuff.c
	$(GCC) -c ImageStuff.c $(CFLAGS)

clean:
	rm -rf *.o main out1.bmp out2.bmp m1.bmp m2.bmp