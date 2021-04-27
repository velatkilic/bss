#ifndef UTILS_H
#define UTILS_H

double** dalloc();
void dfree(double** inp);
double** create_meas(unsigned char** inp1, unsigned char** inp2, double a1, double a2);

double calc_mean(double ** m);
double calc_min(double ** m);
double calc_max(double ** m);
norm_img(double **m);
void subtract_mean(double **m);
double calc_update(double **res1, double **res2);

void demix(double** m1, double** m2, double** res1, double** res2, int N, double lr);
unsigned char** double2char(double** res);

#endif