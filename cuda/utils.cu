#include "utils.h"

__global__ void cu_sum_atomic(const BmpImage& img, float* res) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x; 
	// atomic add prevents simultan. read/write for res
	if (idx < img.length) atomicAdd(res,img.imgdata[idx]);
}

// subtract the mean
__global__ void demean(BmpImage& img, float* mu) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x; 
	if (idx < img.length) img.imgdata[idx] = img.imgdata[idx] - (*mu);
}


__global__ void cu_calc_update(const BmpImage& img1, const BmpImage& img2, BmpImage& out) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x; 
	if (idx < img1.length)
		out.imgdata[idx] = powf(img1.imgdata[idx],3) * atanf(img2.imgdata[idx]);
}

__global__ void cu_update_once(const BmpImage& img1, const BmpImage& img2, BmpImage& out1, BmpImage& out2, float& c12, float& c21) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	float den = 1.0 - c12*c21;
	if (idx < img1.length) {
		out1.imgdata[idx] = (img1.imgdata[idx] - c12*img2.imgdata[idx])/den;
		out2.imgdata[idx] = (img2.imgdata[idx] - c21*img1.imgdata[idx])/den;
	}
}