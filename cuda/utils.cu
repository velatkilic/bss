#include "utils.h"

__global__
void cu_calc_update(const float* img1, const float* img2, float* out, int N) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x; 
	if (idx < N)
		out[idx] = powf(img1[idx],3) * atanf(img2[idx]);
}

__global__
void cu_update_once(const float* img1, const float* img2, float* out1, float* out2, float c12, float c21, int N) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	float den = 1.0 - c12*c21;
	if (idx < N) {
		out1[idx] = (img1[idx] - c12*img2[idx])/den;
		out2[idx] = (img2[idx] - c21*img1[idx])/den;
	}
}

float calc_mean(float * dat, int N) {
	float out = 0;
	for (int i=0; i<N; i++) {
		out += dat[i];
	}
	return out/N;
}