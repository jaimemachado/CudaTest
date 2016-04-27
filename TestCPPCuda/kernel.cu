#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>

class ManagedAllocationPolicy {
public:
	void* operator new(size_t len){
		void *ptr;
		cudaMallocManaged(&ptr, len);
		return ptr;
	}

		void operator delete(void *ptr) {
		cudaFree(ptr);
	}
};

class DefaultAllocationPolicy {
public:
	void* operator new(size_t len){
		return malloc(len);
	}

		void operator delete(void *ptr) {
		free(ptr);
	}
};

template <typename Allocator>
class test : public Allocator
{
public:
	int prop3;
};

// C++ now handles our deep copies
template <typename Allocator>
//template <typedef Allocator>
struct dataElem : public Allocator {
	int prop1;
	int prop2;
	float val;
	test<Allocator> propTest;
};

template <typename Allocator>
__global__ void foo_by_ref(dataElem<Allocator> &e) {
	printf("Thread %d of %d read prop1=%d, prop2=%d, val=%f, propTest=%d \n",
		threadIdx.x, blockIdx.x, e.prop1, e.prop2, e.val, e.propTest.prop3);
}

template <typename Allocator>
__global__ void foo_by_val(dataElem<Allocator> e) {
	printf("Thread %d of %d read prop1=%d, prop2=%d, val=%f, propTest=%d \n",
		threadIdx.x, blockIdx.x, e.prop1, e.prop2, e.val, e.propTest.prop3);
}

int main(void) {
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
	}

	cudaSetDevice(0);
	void *ptr;
	auto a = cudaMallocManaged(&ptr, 10);

	auto *managedElem = new dataElem<ManagedAllocationPolicy>;
	auto *unmanagedElem = new dataElem<DefaultAllocationPolicy>;
	managedElem->prop1 = 1; managedElem->prop2 = 2; managedElem->val = 3.0f; managedElem->propTest.prop3 = 4;
	unmanagedElem->prop1 = 100; unmanagedElem->prop2 = 200; unmanagedElem->val = 300.0f;

	foo_by_ref << <1, 1 >> >(*managedElem); // works
	cudaDeviceSynchronize();
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));

	foo_by_val << <1, 1 >> >(*managedElem); // works
	//foo << <1, 1 >> >(*unmanagedElem); // illegal memory access -- attempt to access host mem from device
	cudaDeviceSynchronize();
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));

	cudaDeviceReset();
}