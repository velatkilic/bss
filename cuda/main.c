#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "ImageStuff.h"
#include "utils.h"

struct ImgProp ip;

#define Nthreads 512
#define Nblocks 512

int main() {
	// read image data
	unsigned char** h_lenna = ReadBMP("lenna.bmp");
	unsigned char** h_baboo = ReadBMP("baboon.bmp");

	// Create measurements
	float** h_m1 = create_meas(lenna, baboo, 0.6, 0.4);
	float** h_m2 = create_meas(lenna, baboo, 0.4, 0.6);

	// copy measurements to gpu
	cudaMalloc(

	// convert results to unsigned char
	unsigned char** tst1 = float2char(h_m1);
	unsigned char** tst2 = float2char(h_m2);

	// write to BMP
	WriteBMP(tst1 , "m1.bmp");
	WriteBMP(tst2 , "m2.bmp");

	// allocate memory for demixing results
	float** res1 = dalloc();
	float** res2 = dalloc();

	// calculate demixing
	int N = 1000; // number of iterations
	float lr = 1; // learning rate
	/*****************************************
	Time demixing here
	*******************************************/
	time_t t1 = time(NULL);
	demix(m1, m2, res1, res2, N, lr);
	time_t t2 = time(NULL);
	printf("\n Elapsed time: %ld seconds, number of iterations: %d \n", t2-t1, N);
	////////////////////////////////////////////

	// convert results to unsigned char
	unsigned char** out1 = float2char(res1);
	unsigned char** out2 = float2char(res2);

	// write to BMP
	WriteBMP(out1 , "out1.bmp");
	WriteBMP(out2 , "out2.bmp");

	// deallocate memory
	for(int i = 0; i < ip.Vpixels; i++) {
		free(lenna[i]);
		free(baboo[i]);
		free(m1[i]);
		free(m2[i]);
		free(res1[i]);
		free(res2[i]);
		free(tst1[i]);
		free(tst2[i]);
	}
	free(lenna);
	free(baboo);
	free(m1);
	free(m2);
	free(res1);
        free(res2);
        free(tst1);
        free(tst2);
	printf("\n"); // looks nicer 

	return 0;
}
