#ifndef UTILS_H
#define UTILS_H

#include "bmp.h"

__global__ void cu_sum_atomic(const BmpImage& img, float* res) ;
__global__ void demean(BmpImage& img, float* mu);
__global__ void cu_calc_update(const BmpImage& img1, const BmpImage& img2, BmpImage& out);
__global__ void cu_update_once(const BmpImage& img1, const BmpImage& img2, BmpImage& out1, BmpImage& out2, float* c12, float* c21);


#endif