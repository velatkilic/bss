#include <iostream>
#include <opencv2/opencv.hpp>
int main(int argc, char *argv[]) {
  // read image data
  cv::Mat lenna = cv::imread("lenna.bmp");
  cv::Mat baboo = cv::imread("baboon.bmp");
  
  // conver to float
  lenna.convertTo(lenna, CV_32F);
  baboo.convertTo(baboo, CV_32F);

  // create measurements
  cv::Mat m1 = lenna*0.6+baboo*0.4;
  cv::Mat m2 = lenna*0.4+baboo*0.6;

  // save measurements
  cv::imwrite("m1.png",m1);
  cv::imwrite("m2.png",m2);

  return 0;
}
