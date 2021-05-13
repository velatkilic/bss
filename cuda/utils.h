#ifndef UTILS_H
#define UTILS_H

#include "bmp.h"

__global__ void cu_calc_update(const float* img1, const float* img2, float* out, int N);
__global__ void cu_update_once(const float* img1, const float* img2, float* out1, float* out2, float c12, float c21, int N);
float calc_mean(float * dat, int N);

#endif