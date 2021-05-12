// overload new and delete operators for unified memory
// source: https://www.olcf.ornl.gov/calendar/cuda-managed-memory/
class Managed {
	public:
		// new operator
		void* operator new(size_t len) {
			void *ptr;
			cudaMallocManaged(&ptr, len);
			cudaDeviceSynchronize();
			return ptr;
		}

		// delete operator
		void operator delete(void *ptr) {
			cudaDeviceSynchronize();
			cudaFree(ptr);
		}

};

// // umString class extends cuda managed memory class
// class umString: public Managed {
// 	public:
// 		int length;
// 		char *data;

// 		umString(): length(0) {data=nullptr;}
// 		umString(int length): length(length) {cudaMallocManaged(&data,length);}
// 		umString(const umString &s) {
// 			length = s.length;
// 			cudaMallocManaged(&data,length);
// 			memcpy(data,s.data,length);
// 		}
// 		umString& operator=(const umString &s) {
// 			if (this != &s) {
// 				cudaFree(this->data);
// 				(*this)(s);
// 			}
// 			return *this;
// 		}
// 		~umString() {cudaFree(data);}
// };