#ifndef UTILS_H
#define UTILS_H

float** dalloc();
void dfree(float** inp);
float** create_meas(unsigned char** inp1, unsigned char** inp2, float a1, float a2);

float calc_mean(float ** m);
float calc_min(float ** m);
float calc_max(float ** m);
void norm_img(float **m);
void subtract_mean(float **m);
float calc_update(float **res1, float **res2);

void demix(float** m1, float** m2, float** res1, float** res2, int N, float lr);
unsigned char** float2char(float** res);

#endif
