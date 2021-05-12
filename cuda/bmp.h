#ifndef BMP_H
#define BMP_H

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "managed.h"

// umString class extends cuda managed memory class
class umString: public Managed {
	public:
		int length;
		char *data;

		umString(): length(0) {data=nullptr;}
		umString(int length): length(length) {cudaMallocManaged(&data,length);}
		umString(const umString &s) {
			length = s.length;
			cudaMallocManaged(&data,length);
			memcpy(data,s.data,length);
		}
};

class BmpImage: public Managed {
	public:
		umString imgdata;
		int Hpixels, Vpixels, length;
		std::vector<char> HeaderInfo;

		// constructors
		BmpImage(): Hpixels(0), Vpixels(0), length(0), HeaderInfo(), imgdata() {}
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
		void writeBmp(const std::string fname) const;
		void clear();

		BmpImage& operator*(const float c);
		BmpImage& operator+(const BmpImage& img);

};

#endif