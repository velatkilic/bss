#include <iostream>
#include "bmp.h"

int main() {
	BmpImage lenna("lenna.bmp");
	lenna.writeBmp("lenna2.bmp");
	return 0;
}