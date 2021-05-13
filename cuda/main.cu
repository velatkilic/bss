#include <iostream>
#include "bmp.h"
#include "utils.h"

#define NBLOCKS 512
#define NTHREADS 512

int main() {
	// read bmp images (cudaMallocManaged under the hood)
	BmpImage lenna("lenna.bmp");
	BmpImage baboo("baboon.bmp");

	// create measurements
	BmpImage m1 = lenna*.4f + baboo*0.6f;
	BmpImage m2 = lenna*.6f + baboo*0.4f;

	// initial demixing coefficient guess and learning rate
	// TODO: derive a cuFloat class from managed class
	float *c12, *c21, *lr;
	cudaMallocManaged(&c12,sizeof(float));
	cudaMallocManaged(&c21,sizeof(float));
	cudaMallocManaged(&lr,sizeof(float));

	*c12 = 0.49f;
	*c21 = 0.51f;
	*lr  = 1.0f;

	// Calculate average for images
	float *mu1, *mu2;
	cudaMallocManaged(&mu1,sizeof(float));
	cu_sum_atomic<<<NBLOCKS,NTHREADS>>>(m1, mu1);
	cudaDeviceSynchronize();

	cudaMallocManaged(&mu2,sizeof(float));
	cu_sum_atomic<<<NBLOCKS,NTHREADS>>>(m2, mu2);
	cudaDeviceSynchronize();

	*mu1 /= m1.length;
	*mu2 /= m2.length;

	// subtract the mean from images
	demean<<<NBLOCKS,NTHREADS>>>(m1,mu1);
	cudaDeviceSynchronize();
	demean<<<NBLOCKS,NTHREADS>>>(m2,mu2);
	cudaDeviceSynchronize();

	// demix loop
	BmpImage out1(m1);
	BmpImage out2(m2);
	BmpImage temp(m1);
	for (int i=0; i<1000; i++) {
		cu_update_once<<<NBLOCKS,NTHREADS>>>(m1, m2, out1, out2, c12, c21);
		cudaDeviceSynchronize();

		// update c12
		cu_calc_update<<<NBLOCKS,NTHREADS>>>(m1, m2, temp);
		cudaDeviceSynchronize();
		cu_sum_atomic<<<NBLOCKS,NTHREADS>>>(temp, mu1); // reuse mu because why not
		cudaDeviceSynchronize();
		mu1 /= temp.length;
		c12 += (*lr) * (*mu1);

		// update c21
		cu_calc_update<<<NBLOCKS,NTHREADS>>>(m2, m1, temp);
		cudaDeviceSynchronize();
		cu_sum_atomic<<<NBLOCKS,NTHREADS>>>(temp, mu2); // reuse mu because why not
		cudaDeviceSynchronize();
		mu2 /= temp.length;
		c21 += (*lr) * (*mu2);
	}


	// free memory
	cudaFree(c21);
	cudaFree(c12);
	cudaFree(lr);
	cudaFree(mu1);
	cudaFree(mu2);

	
	return 0;
}