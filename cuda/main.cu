#include <iostream>
#include "bmp.h"
#include "utils.h"

#define NBLOCKS 1
#define NTHREADS 1

int main() {
	// read bmp images (cudaMallocManaged under the hood)
	BmpImage lenna("lenna.bmp");
	BmpImage baboo("baboon.bmp");

	// create measurements
	BmpImage m1 = lenna*.4f + baboo*0.6f;
	BmpImage m2 = lenna*.6f + baboo*0.4f;

	m1.writeBmp("m1.bmp");
	m2.writeBmp("m2.bmp");

	// initial demixing coefficient guess
	float *c12, *c21;
	cudaMallocManaged(&c12,sizeof(float));
	cudaMallocManaged(&c21,sizeof(float));
	*c12 = 0.49f;
	*c21 = 0.51f;

	float lr  = 1.0f; // learning rate

	// subtract mean
	m1.demean();
	m2.demean();

	// demix loop
	BmpImage out1(m1);
	BmpImage out2(m2);
	BmpImage temp(m1);
	float mu;
	for (int i=0; i<1; i++) {
		cu_update_once<<<NBLOCKS,NTHREADS>>>(m1, m2, out1, out2, *c12, *c21);
		cudaDeviceSynchronize();

		// // update c12
		// cu_calc_update<<<NBLOCKS,NTHREADS>>>(m1, m2, temp);
		// cudaDeviceSynchronize();
		// mu = temp.calc_mean();
		// (*c12) += lr*mu;

		// // update c21
		// cu_calc_update<<<NBLOCKS,NTHREADS>>>(m2, m1, temp);
		// cudaDeviceSynchronize();
		// mu = temp.calc_mean();
		// (*c21) += lr*mu;
	}

	out1.writeBmp("out1.bmp");
	out2.writeBmp("out2.bmp");

	cudaFree(c12); cudaFree(c21);

		// // Calculate average for images
	// float *mu1, *mu2;
	// cudaMallocManaged(&mu1,sizeof(float));
	// cu_sum_atomic<<<NBLOCKS,NTHREADS>>>(m1, mu1);
	// cudaDeviceSynchronize();

	// std::cout << "update mu1 = " << *mu1 << std::endl;

	// cudaMallocManaged(&mu2,sizeof(float));
	// cu_sum_atomic<<<NBLOCKS,NTHREADS>>>(m2, mu2);
	// cudaDeviceSynchronize();

	// std::cout << "update mu" << std::endl;

	// *mu1 = *mu1/((float) m1.length);
	// std::cout << "update mu1 = " << *mu1 << std::endl;
	// *mu2 = *mu2/((float) m2.length);
	// std::cout << "update mu2 = " << *mu1 << std::endl;

	// // subtract the mean from images
	// demean<<<NBLOCKS,NTHREADS>>>(m1,mu1);
	// cudaDeviceSynchronize();
	// demean<<<NBLOCKS,NTHREADS>>>(m2,mu2);
	// cudaDeviceSynchronize();
	return 0;
}