#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include "ImageStuff.h"
struct ImgProp ip;

// allocate 2D double memory
double** dalloc() {
	double** output = (double**) malloc(ip.Vpixels * sizeof(double*));
	for(int i=0; i<ip.Vpixels; i++) {
		output[i] = (double *) malloc(ip.Hbytes * sizeof(double));
	}
	return output;
}

void dfree(double** inp) {
	for(int i=0; i<ip.Vpixels; i++) {
		free(inp[i]);
	}
	free(inp);
}

// create 2 measurements from given mixing values a_i and input images inp_i
double** create_meas(unsigned char** inp1, unsigned char** inp2, double a1, double a2) {
	// allocate memory
	double** output = dalloc();

	// create measurement
	for (int i = 0; i<ip.Vpixels; i++) {
		for (int j=0; j<ip.Hbytes; j++) {
			output[i][j]   = a1*inp1[i][j]/255.0   + a2*inp2[i][j]/255.0;
		}
	}

	return output;
}

// calculate the mean of the double 2D array
double calc_mean(double ** m) {
	double out = 0;
	for (int i = 0; i<ip.Vpixels; i++) {
		for (int j=0; j<ip.Hbytes; j++) {
			out += m[i][j];
		}
	}

	return out/(ip.Hbytes * ip.Vpixels);
}

// calculate max
double calc_max(double ** m) {
	double out = -10000;
	for (int i = 0; i<ip.Vpixels; i++) {
		for (int j=0; j<ip.Hbytes; j++) {
			if (m[i][j] > out) out = m[i][j];
		}
	}

	return out;
}

// calculate min
double calc_min(double ** m) {
	double out = 10000;
	for (int i = 0; i<ip.Vpixels; i++) {
		for (int j=0; j<ip.Hbytes; j++) {
			if (m[i][j] < out) out = m[i][j];
		}
	}

	return out;
}

// normalize image to range (0 1)
void norm_img(double **m) {
	double max = calc_max(m);
	double min = calc_min(m);
	double del = max - min;

	// printf("min : %f \n max  : %f \n", min, max);

	for (int i = 0; i<ip.Vpixels; i++) {
		for (int j=0; j<ip.Hbytes; j++) {
			m[i][j] = (m[i][j] - min)/del;
		}
	}
}

// subtract mean
void subtract_mean(double **m) {
	double mean = calc_mean(m);
	for (int i = 0; i<ip.Vpixels; i++) {
		for (int j=0; j<ip.Hbytes; j++) {
			m[i][j]   -= mean;
		}
	}
}


// calculate update term
double calc_update(double **res1, double **res2) {
	double** allup = dalloc(); // allocate 2D array
	// calculate update
	for (int i=0; i<ip.Hpixels; i++) {
		for (int j=0; j<ip.Hbytes; j++) {
			allup[i][j]   = pow(res1[i][j],3) * atan(res2[i][j]);
		}
	}

	// calculate average update
	double output = calc_mean(allup);
	dfree(allup); // free memory
	return output;
}

// demix two signals
void demix(double** m1, double** m2, double** res1, double** res2, int N, double lr) {
	// init demixing coefficients
	double c12 = 0.49;
	double c21 = 0.51;

	// de-mean the signals
	subtract_mean(m1);
	subtract_mean(m2);

	// demixing iteration for N times
	double den; //denominator
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

// cast double to unsigned char to write to BMP
unsigned char** double2char(double** res) {
	// allocate memory for char
	unsigned char** output = (unsigned char**) malloc(ip.Hpixels * sizeof(unsigned char*));
	for(int i=0; i<ip.Vpixels; i++) {
		output[i] = (unsigned char *) malloc(ip.Hbytes * sizeof(unsigned char));
	}

	// normalize
	norm_img(res);

	// case double to char
	for (int i=0; i<ip.Vpixels; i++) {
		for (int j=0; j<ip.Hbytes; j++) {
			output[i][j]   = (unsigned char) (255.0*res[i][j]);
		}
	}

	return output;
}