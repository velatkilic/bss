#include "bmp.h"

// clear contents of the class
void BmpImage::clear() {
	delete [] imgdata;
	Hpixels=0; Vpixels=0; length=0;
	HeaderInfo.clear();
}

// deep copy constructor
BmpImage::BmpImage(const BmpImage& img) {

	// clear previous data
	this->clear();

	// copy values of the argument
	Hpixels = img.Hpixels;
	Vpixels = img.Vpixels;
	HeaderInfo = img.HeaderInfo;

	cudaMallocManaged(&imgdata, length);
	memcpy(imgdata, img.imgdata, length);

}

// deep assignment
BmpImage& BmpImage::operator=(const BmpImage& img) {
	if (this != &img) {
		// clear previous data
		this->clear();

		// copy values of the argument
		Hpixels = img.Hpixels;
		Vpixels = img.Vpixels;
		HeaderInfo = img.HeaderInfo;

		cudaMallocManaged(&imgdata, length);
		memcpy(imgdata, img.imgdata, length);
	}

	return *this;
}

// read bitmap file
void BmpImage::readBmp(const std::string fname) {
	// open file for read
	std::ifstream file;
	file.open(fname);
	if (!file.is_open()) {
		std::cout << "Error: failed to open " << fname << std::endl;
	}

	// read header info: 54 chars
	for (int i=0; i<54; i++) {
		HeaderInfo[i] = file.get();
	}

	// from Tolga's book:
	// extract image height and width from header
	Hpixels = *(int*)&HeaderInfo[18];
	Vpixels = *(int*)&HeaderInfo[22];
	int RowBytes = (Hpixels*3 + 3) & (~3);

	length = RowBytes*Vpixels;

	imgdata = new char[length];
	for (int i=0; i<length; i++) {
		imgdata[i] = file.get();
	}

	file.close();

}

// read bitmap file
void BmpImage::writeBmp(const std::string fname) const {
	// open file for read
	std::ofstream file;
	file.open(fname);
	if (!file.is_open()) {
		std::cout << "Error: failed to open " << fname << std::endl;
	}

	// write header info: 54 chars
	for (int i=0; i<54; i++) {
		file.put(HeaderInfo[i]);
	}

	// write image data
	for (int i=0; i<length; i++) {
		file.put(imgdata[i]);
	}

	file.close();

}