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