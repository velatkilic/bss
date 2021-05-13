#include "utils.h"

__global__
void cu_calc_update(const float* img1, const float* img2, float* out, int N) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i=idx; i<N; i+=stride) {
		out[i] = powf(img1[i],3) * atanf(img2[i]);
	}
}

__global__
void cu_update_once(const float* img1, const float* img2, float* out1, float* out2, float c12, float c21, int N) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;
	float den = 1.0 - c12*c21;
	for (int i=idx; i<N; i+=stride) {
		out1[i] = (img1[i] - c12*img2[i])/den;
		out2[i] = (img2[i] - c21*img1[i])/den;
	}
}

float calc_mean(float * dat, int N) {
	float out = 0;
	for (int i=0; i<N; i++) {
		out += dat[i];
	}
	return out/N;
}