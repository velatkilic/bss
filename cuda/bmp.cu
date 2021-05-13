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

	imgdata = new float;
	cudaMallocManaged(&imgdata, length*sizeof(float));
	for (int i=0; i<length; i++) {
		imgdata[i] = (float) file.get();
	}

	file.close();

}

// calculate max
float BmpImage::calc_max() const {
	float out = -10000;
	for (int i = 0; i<length; i++) {
		if (imgdata[i] > out) out = imgdata[i];
	}

	return out;
}

// calculate min
float BmpImage::calc_min() const {
	float out = 10000;
	for (int i = 0; i<length; i++) {
		if (imgdata[i] < out) out = imgdata[i];
	}

	return out;
}

// calculate min
float BmpImage::calc_mean() const {
	float out = 0.0f;
	for (int i = 0; i<length; i++) {
		out += imgdata[i];
	}
	out = out/length;
	return out;
}

// subtract mean
void BmpImage::demean() {
	float mu = this->calc_mean();
	for (int i = 0; i<length; i++) {
		imgdata[i] -= mu;
	}
}

// normalize float image data to [0 255]
char* BmpImage::normalize(void) const {
	float min = calc_min();
	float max = calc_max();
	float del = max - min;
	char * out = new char[length];

	for (int i=0; i<length; i++) {
		out[i] = (char) (255.0f*(imgdata[i] - min)/del);
	}
	return out;
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