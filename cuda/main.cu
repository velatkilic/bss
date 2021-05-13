#include <iostream>
#include "bmp.h"

int main() {
	BmpImage lenna("lenna.bmp");
	BmpImage baboo("baboon.bmp");

	BmpImage m1 = lenna*.4f + baboo*0.6f;
	m1.writeBmp("m1.bmp");
	return 0;
}