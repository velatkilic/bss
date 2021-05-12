#include <iostream>
#include "bmp.h"

int main() {
	BmpImage lenna("lenna.bmp");
	BmpImage baboo("baboon.bmp");

	BmpImage m1;
	std::cout << m1.length << std::endl;
	m1 = lenna;// + baboo*.6f;
	std::cout << m1.length << std::endl;
	std::cout << lenna.length << std::endl;
	lenna.writeBmp("lenna2.bmp");
	m1.writeBmp("m1.bmp");
	return 0;
}