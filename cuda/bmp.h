#ifndef BMP_H
#define BMP_H

#include <iostream>
#include <string>
#include <fstream>
#include <vector>

// Bitmap image class
class BmpImage {
	public:
		float* imgdata;
		int Hpixels, Vpixels, length;
		std::vector<char> HeaderInfo;

		// constructors
		BmpImage(): Hpixels(0), Vpixels(0), length(0), HeaderInfo(), imgdata(nullptr) {}
		BmpImage(std::string fname){
			Hpixels=0; Vpixels=0; length=0;
			HeaderInfo.reserve(54);
			readBmp(fname);
		}
		
		// deep copy constructor
		BmpImage(const BmpImage& img);
		// deep assignment operator
		BmpImage& operator=(const BmpImage& img);

		// nondefault destructor
		~BmpImage() { clear();}

		// member functions
		void readBmp(const std::string fname);
		float calc_max() const;
		float calc_min() const;
		float calc_mean() const;
		void demean();
		void writeBmp(const std::string fname) const;
		void clear();

		BmpImage& operator*(const float c);
		BmpImage& operator+(const BmpImage& img);
	private:
		char* normalize(void) const; // because needs memory management

};

#endif