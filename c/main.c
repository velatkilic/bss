#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "ImageStuff.h"
#include "utils.h"

struct ImgProp ip;

int main() {
	// read image data
	unsigned char** lenna = ReadBMP("lenna.bmp");
	unsigned char** baboo = ReadBMP("baboon.bmp");

	// Create measurements
	double** m1 = create_meas(lenna, baboo, 0.6, 0.4);
	double** m2 = create_meas(lenna, baboo, 0.4, 0.6);

	// convert results to unsigned char
	unsigned char** tst1 = double2char(m1);
	unsigned char** tst2 = double2char(m2);

	// write to BMP
	WriteBMP(tst1 , "m1.bmp");
	WriteBMP(tst2 , "m2.bmp");

	// allocate memory for demixing results
	double** res1 = dalloc();
	double** res2 = dalloc();

	// calculate demixing
	int N = 1000; // number of iterations
	double lr = 1; // learning rate
	demix(m1, m2, res1, res2, N, lr);

	// convert results to unsigned char
	unsigned char** out1 = double2char(res1);
	unsigned char** out2 = double2char(res2);

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

	return 0;
}