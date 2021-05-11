#include "mean.h"

__global__ void cu_sum_atomic(float* m, float* res, int Ntot) {
  float dmmy=0;
  int idx = threadIdx.x + blockDim.x * blockIdx.x; 
  int stride = blockDim.x * gridDim.x;
  // grid stride loop
  for (int i = idx; i<Ntot; i += stride) dmmy += m[i]; 
  // atomic add prevents simultan. read/write for res
  if (idx < Ntot) atomicAdd(res,dmmy);
}
