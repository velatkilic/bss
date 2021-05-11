#include "utils.h"

void demix_cpu(cv::Mat& m1, cv::Mat& m2, cv::Mat& res1, cv::Mat& res2, int N, double lr) {
  // init demixing coeff
  float c12 = 0.49;
  float c21 = 0.51;

  // demean the signals
  m1 = m1 - cv::mean(m1);
  m2 = m2 - cv::mean(m2);

  // demixing iterations N times
  float den;
  for (int n=0; n<N; ++n) {
    // calculate results
    den  = 1.0f - c12*c21;
    res1 = (m1 - c12*m2)/den;
    res2 = (m2 - c21*m1)/den;
    
    // update demixing coefficients
    c12 += lr*cv::mean(f(res1)*g(res2));
    c21 += lr*cv::mean(f(res2)*g(res1));
  }
}
