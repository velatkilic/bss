#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include "ImageStuff.h"
struct ImgProp ip;

// allocate 2D float memory on host
float** dalloc() {
	float** output = (float**) malloc(ip.Vpixels * sizeof(float*));
	for(int i=0; i<ip.Vpixels; i++) {
		output[i] = (float *) malloc(ip.Hbytes * sizeof(float));
	}
	return output;
}

// allocate 2D float memory on device
float** cu_dalloc() {
	float** output;
	cudaMalloc( (void **) output, ip.Vpixels * sizeof(float*));
	for(int i=0; i<ip.Vpixels; i++) {
		cudaMalloc((void *) output[i], ip.Hbytes * sizeof(float));
	}
	return output;
}

// Free host memory
void dfree(float** inp) {
	for(int i=0; i<ip.Vpixels; i++) {
		free(inp[i]);
	}
	free(inp);
}

// free device memory
void cu_dfree(float** inp) {
	for(int i=0; i<ip.Vpixels; i++) {
		cudaFree(inp[i]);
	}
	cudaFree(inp);
}

// create 2 measurements from given mixing values a_i and input images inp_i
// just host version is needed
float** create_meas(unsigned char** inp1, unsigned char** inp2, float a1, float a2) {
	// allocate memory
	float** output = dalloc();

	// create measurement
	for (int i = 0; i<ip.Vpixels; i++) {
		for (int j=0; j<ip.Hbytes; j++) {
			output[i][j]   = a1*inp1[i][j]/255.0   + a2*inp2[i][j]/255.0;
		}
	}

	return output;
}

// calculate the mean of the float 2D array
float calc_mean(float ** m) {
	float out = 0;
	for (int i = 0; i<ip.Vpixels; i++) {
		for (int j=0; j<ip.Hbytes; j++) {
			out += m[i][j];
		}
	}

	return out/(ip.Hbytes * ip.Vpixels);
}

// cuda mean
__global__ float cu_calc_mean(float ** m) {
	float out = 0;
	for (int i = 0; i<ip.Vpixels; i++) {
		for (int j=0; j<ip.Hbytes; j++) {
			out += m[i][j];
		}
	}

	return out/(ip.Hbytes * ip.Vpixels);
}

// calculate max
float calc_max(float ** m) {
	float out = -10000;
	for (int i = 0; i<ip.Vpixels; i++) {
		for (int j=0; j<ip.Hbytes; j++) {
			if (m[i][j] > out) out = m[i][j];
		}
	}

	return out;
}

// calculate min
float calc_min(float ** m) {
	float out = 10000;
	for (int i = 0; i<ip.Vpixels; i++) {
		for (int j=0; j<ip.Hbytes; j++) {
			if (m[i][j] < out) out = m[i][j];
		}
	}

	return out;
}

// normalize image to range (0 1)
void norm_img(float **m) {
	float max = calc_max(m);
	float min = calc_min(m);
	float del = max - min;

	// printf("min : %f \n max  : %f \n", min, max);

	for (int i = 0; i<ip.Vpixels; i++) {
		for (int j=0; j<ip.Hbytes; j++) {
			m[i][j] = (m[i][j] - min)/del;
		}
	}
}

// subtract mean
void subtract_mean(float **m) {
	float mean = calc_mean(m);
	for (int i = 0; i<ip.Vpixels; i++) {
		for (int j=0; j<ip.Hbytes; j++) {
			m[i][j]   -= mean;
		}
	}
}


// calculate update term
float calc_update(float **res1, float **res2) {
	float** allup = dalloc(); // allocate 2D array
	// calculate update
	for (int i=0; i<ip.Hpixels; i++) {
		for (int j=0; j<ip.Hbytes; j++) {
			allup[i][j]   = pow(res1[i][j],3) * atan(res2[i][j]);
		}
	}

	// calculate average update
	float output = calc_mean(allup);
	dfree(allup); // free memory
	return output;
}

// demix two signals
void demix(float** m1, float** m2, float** res1, float** res2, int N, float lr) {
	// init demixing coefficients
	float c12 = 0.49;
	float c21 = 0.51;

	// de-mean the signals
	subtract_mean(m1);
	subtract_mean(m2);

	// demixing iteration for N times
	float den; //denominator
	for (int n=0; n<N; n++) {
		den = 1.0 - c12*c21;
		// demix once
		for (int i=0; i<ip.Hpixels; i++) {
			for (int j=0; j<ip.Hbytes; j+=3) {
				res1[i][j]   = (m1[i][j]   - c12*m2[i][j])/den;
				res1[i][j+1] = (m1[i][j+1] - c12*m2[i][j+1])/den;
				res1[i][j+2] = (m1[i][j+2] - c12*m2[i][j+2])/den;

				res2[i][j]   = (m2[i][j]   - c21*m1[i][j])/den;
				res2[i][j+1] = (m2[i][j+1] - c21*m1[i][j+1])/den;
				res2[i][j+2] = (m2[i][j+2] - c21*m1[i][j+2])/den;
			}
		}
		// update c
		c12 += lr*calc_update(res1, res2);
		c21 += lr*calc_update(res2, res1);
	}

}

// cast float to unsigned char to write to BMP
unsigned char** float2char(float** res) {
	// allocate memory for char
	unsigned char** output = (unsigned char**) malloc(ip.Hpixels * sizeof(unsigned char*));
	for(int i=0; i<ip.Vpixels; i++) {
		output[i] = (unsigned char *) malloc(ip.Hbytes * sizeof(unsigned char));
	}

	// normalize
	norm_img(res);

	// case float to char
	for (int i=0; i<ip.Vpixels; i++) {
		for (int j=0; j<ip.Hbytes; j++) {
			output[i][j]   = (unsigned char) (255.0*res[i][j]);
		}
	}

	return output;
}
