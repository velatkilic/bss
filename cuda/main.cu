#include <iostream>
#include "bmp.h"
#include "utils.h"

#define NBLOCKS 512
#define NTHREADS 512

int main() {
	// read bmp images
	BmpImage lenna("lenna.bmp");
	BmpImage baboo("baboon.bmp");

	int length = lenna.length;

	// create measurements
	BmpImage m1 = create_meas(lenna, baboo, 0.6);
	BmpImage m2 = create_meas(lenna, baboo, 0.4);

	m1.writeBmp("m1.bmp");
	m2.writeBmp("m2.bmp");

	// init output images
	BmpImage out1(m1);
	BmpImage out2(m2);

	// subtract mean
	m1.demean();
	m2.demean();

	// initial demixing coefficient guess
	float c12(0.51), c21(0.49), lr(1.0);

	// allocate gpu space
	int size = length*sizeof(float);
	float *d_m1, *d_m2, *d_out1, *d_out2, *d_temp;
	cudaMalloc((void **)&d_m1, size);
	cudaMalloc((void **)&d_m2, size);
	cudaMalloc((void **)&d_out1, size);
	cudaMalloc((void **)&d_out2, size);
	cudaMalloc((void **)&d_temp, size);

	// copy contents of m1 and m2
	cudaMemcpy(d_m1, m1.imgdata, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_m2, m2.imgdata, size, cudaMemcpyHostToDevice);

	// demix loop
	float * h_temp = new float[length];
	float mu;
	for (int i=0; i<100; i++) {
		cu_update_once<<<NBLOCKS,NTHREADS>>>(d_m1, d_m2, d_out1, d_out2, c12, c21, length);
		cudaDeviceSynchronize();

		// update c12
		cu_calc_update<<<NBLOCKS,NTHREADS>>>(d_m1, d_m2, d_temp, length);
		cudaMemcpy(h_temp, d_temp, size, cudaMemcpyDeviceToHost);
		mu = calc_mean(h_temp, length);
		c12 += lr*mu;

		cu_calc_update<<<NBLOCKS,NTHREADS>>>(d_m2, d_m1, d_temp, length);
		cudaMemcpy(h_temp, d_temp, size, cudaMemcpyDeviceToHost);
		mu = calc_mean(h_temp, length);
		c21 += lr*mu;

		if (i%50 ==0) {
			std::cout << "Optimization step : " << i << std::endl;
		}
	}

	// copy result back to cpu
	float *h_out1 = new float[length];
	float *h_out2 = new float[length];
	cudaMemcpy(h_out1, d_out1, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_out2, d_out2, size, cudaMemcpyDeviceToHost);

	// set BmpImage class
	out1.setData(h_out1, length);
	out2.setData(h_out2, length);

	// write to file
	out1.writeBmp("out1.bmp");
	out2.writeBmp("out2.bmp");

	// free memory
	cudaFree(d_m1); cudaFree(d_m2); cudaFree(d_out1);
	cudaFree(d_out2); cudaFree(d_temp);
	delete [] h_out1;
	delete [] h_out2;
	delete [] h_temp;
	return 0;
}