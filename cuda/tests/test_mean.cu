#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>

#include "mean.h"

#define NTHREADS 512
#define NBLOCKS 512

int main() {
  int npix = 512;
  int Ntot = npix*npix;
  int size = Ntot*sizeof(float);
  
  // allocate host array
  float* arr = (float *) malloc(size);
  float* h_sum = (float *) malloc(sizeof(float));
  *h_sum = 0; //init to zero

  // assign random numbers
  for (int i=0; i<Ntot; ++i) {
     arr[i] = (float) (rand() % 255);
  }

  // allocate memory on device
  float* d_arr, *d_sum;
  cudaMalloc((void**) &d_arr, size);
  cudaMalloc((void**) &d_sum, sizeof(float));

  // copy host data to device
  cudaMemcpy(d_arr,arr, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sum,h_sum,sizeof(float), cudaMemcpyHostToDevice);

  // calculate average on cpu for comparison
  float sum = 0;
  for (int i=0; i<Ntot; ++i) {
    sum += arr[i];
  } 
  
  // launch cuda mean thread that uses atomics and a grid stride
  cu_sum_atomic<<<NBLOCKS,NTHREADS>>>(d_arr, d_sum, Ntot);

  // copy result back to host
  cudaMemcpy(h_sum,d_sum,sizeof(float), cudaMemcpyDeviceToHost);
  
  // compare results
  printf("Results are cpu: %f , gpu: %f \n", sum, *h_sum);
  // assert(*h_sum == sum);

  // free memory
  free(arr); free(h_sum);
  cudaFree(d_arr); cudaFree(d_sum);

  return 0;
}
