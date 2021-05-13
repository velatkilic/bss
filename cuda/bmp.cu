#include "bmp.h"

// clear contents of the class
void BmpImage::clear() {
	cudaFree(imgdata);

	Hpixels=0; Vpixels=0; length=0;
	// HeaderInfo.clear();
}

// deep copy constructor
BmpImage::BmpImage(const BmpImage& img) {

	// clear previous data
	this->clear();

	// copy values of the argument
	Hpixels    = img.Hpixels;
	Vpixels    = img.Vpixels;
	HeaderInfo = img.HeaderInfo;
	length     = img.length;

	imgdata = new float;
	cudaMallocManaged(&imgdata, length*sizeof(float));
	memcpy(imgdata, img.imgdata, length*sizeof(float));
}

// deep assignment
BmpImage& BmpImage::operator=(const BmpImage& img) {
	if (this != &img) {
		// clear previous data
		this->clear();

		// copy values of the argument
		Hpixels    = img.Hpixels;
		Vpixels    = img.Vpixels;
		HeaderInfo = img.HeaderInfo;
		length     = img.length;

		imgdata = new float;
		cudaMallocManaged(&imgdata, length*sizeof(float));
		memcpy(imgdata, img.imgdata, length*sizeof(float));
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
		HeaderInfo.push_back(file.get());
	}

	// from Tolga's book:
	// extract image height and width from header
	Hpixels = *(int*)&HeaderInfo[18];
	Vpixels = *(int*)&HeaderInfo[22];
	int RowBytes = (Hpixels*3 + 3) & (~3);

	std::cout<< "Read: " << Hpixels << " x " << Vpixels << " pixels" << std::endl;

	length = RowBytes*Vpixels;

	imgdata = new float[length];
	for (int i=0; i<length; i++) {
		imgdata[i] = (float) file.get();
	}

	file.close();

}

// normalize float image data to [0 255]
char* BmpImage::normalize(void) const {
	char * out = new char[length];
	for (int i=0; i<length; i++) {
		out[i] = (char) imgdata[i];
	}
	return out;
}

// read bitmap file
void BmpImage::writeBmp(const std::string fname) const {
	std::cout<< "Writing: Hpixels x Vpixels " << Hpixels << " x " << Vpixels << std::endl;
	// open file for read
	std::ofstream file;
	file.open(fname);
	if (!file.is_open()) {
		std::cout << "Error: failed to open " << fname << std::endl;
	}
	// write header info: 54 chars
	for (int i=0; i<54; i++) {
		file.put(HeaderInfo.at(i));
	}
	// write image data
	char* nrm_imgdata = normalize();
	for (int i=0; i<length; i++) {
		file.put(nrm_imgdata[i]);
	}

	file.close();
	delete [] nrm_imgdata;

}

// overload math operators
BmpImage& BmpImage::operator*(const float c) {
	for (int i=0; i<length; i++) {
		imgdata[i] *= c;
	}
	return *this;
}

BmpImage& BmpImage::operator+(const BmpImage& img) {
	if (length == img.length) {
		for (int i=0; i<length; i++) {
			imgdata[i] += img.imgdata[i];
		}
	} else {
		std::cout << "Error: image size must match" << std::endl;
	}

	return *this;
}