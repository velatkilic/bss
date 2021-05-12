#ifndef BMP_H
#define BMP_H

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "managed.h"

// Bitmap image class extends cuda managed memory class
class BmpImage: public Managed {
	public:
		char* imgdata;
		int Hpixels, Vpixels, length;
		std::vector<char> HeaderInfo;

		// constructors
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
};

#endif