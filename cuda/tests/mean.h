#ifndef MEAN_H
#define MEAN_H

#include <stdio.h>

__global__ void cu_sum_atomic(float* m, float* res, int Ntot);

#endif
