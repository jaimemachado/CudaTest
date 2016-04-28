#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>

#define MULTI_DEVICES

#ifdef MULTI_DEVICES
#define ENABLE_GPU
#define ENABLE_CPU
#define ENABLE_MULTI_DEVICES
#define MEMORY_ALLOCATOR ManagedAllocationPolicy
#elif defined GPU_DEVICE
#define ENABLE_GPU
#define MEMORY_ALLOCATOR ManagedAllocationPolicy
#else
#define ENABLE_CPU
#define MEMORY_ALLOCATOR DefaultAllocationPolicy
#endif

enum class DeviceSelector{
	GPU,
	CPU
};

enum class MemType{
	Mannaged,
	NotMannaged
};
template <typename Allocator>
class DeviceSelectionPolicyBase : public Allocator {
	DeviceSelector m_eProcessLocation;
public:
	virtual DeviceSelector GetProcessLocation(){
		return m_eProcessLocation;
	}
	bool SetProcessLocation(DeviceSelector location){
		if (location == DeviceSelector::GPU && GetMemType() == MemType::NotMannaged)
		{
			return false;
		}
		m_eProcessLocation = location;
		return true;
	}
};

class ManagedAllocationPolicy{
public:
	MemType GetMemType(){
		return MemType::Mannaged;
	}
	void* operator new(size_t len){
		void *ptr;
		cudaMallocManaged(&ptr, len);
		return ptr;
	}

	void operator delete(void *ptr) {
		cudaFree(ptr);
	}
};

class DefaultAllocationPolicy{
public:
	MemType GetMemType(){
		return MemType::NotMannaged;
	}
	void* operator new(size_t len){
		return malloc(len);
	}

	void operator delete(void *ptr) {
		free(ptr);
	}
};

template <typename Allocator>
class test : public DeviceSelectionPolicyBase<Allocator>
{
public:
	int prop3;
};

// C++ now handles our deep copies
template <typename Allocator>
//template <typedef Allocator>
struct dataElem : public DeviceSelectionPolicyBase<Allocator> {
	DeviceSelector GetProcessLocation(){
		return m_eProcessLocation;
	}
	bool SetProcessLocation(DeviceSelector location){
		if (!DeviceSelectionPolicyBase::SetProcessLocation(location))
		{
			return false;
		}
		return propTest.SetProcessLocation(location);
	}

	dataElem(){
		if (GetMemType() == MemType::NotMannaged){
			m_eProcessLocation = DeviceSelector::CPU;
		}
		else{
			m_eProcessLocation = DeviceSelector::GPU;
		}
	}
	int prop1;
	int prop2;
	float val;
	test<Allocator> propTest;
private:
	DeviceSelector m_eProcessLocation;
};

template <typename Allocator>
__global__ void g_global_foo2(test<Allocator> &e) {
	g_foo2(e);
}

template <typename Allocator>
__device__ void g_foo2(test<Allocator> &e) {
	printf("GPU: Thread %d of %d read propTest=%d \n",
		threadIdx.x, blockIdx.x, e.prop3);
}

template <typename Allocator>
void d_foo2(test<Allocator> &e) {
	printf("CPU: read propTest=%d \n",
		e.prop3);
}

template <typename Allocator>
void foo2(test<Allocator> &e) {
	if (e.GetProcessLocation() == DeviceSelector::GPU)
	{
		g_global_foo2 << <1, 1 >> >(e); // works
		cudaDeviceSynchronize();
		printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	}
	else{
		d_foo2(e); // works
	}
}

template <typename Allocator>
__global__ void g_foo(dataElem<Allocator> &e) {
	printf("GPU: Thread %d of %d read prop1=%d, prop2=%d, val=%f\n",
		threadIdx.x, blockIdx.x, e.prop1, e.prop2, e.val);
	g_foo2(e.propTest);
}

template <typename Allocator>
void d_foo(dataElem<Allocator> &e) {
	printf("CPU: read prop1=%d, prop2=%d, val=%f\n",
		e.prop1, e.prop2, e.val);
	foo2(e.propTest);
}

template <typename Allocator>
void foo(dataElem<Allocator> &e) {
	if (e.GetProcessLocation() == DeviceSelector::GPU)
	{
		g_foo << <1, 1 >> >(e); // works
		cudaDeviceSynchronize();
		printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	}
	else{
		d_foo(e); // works
	}
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

	auto *managedElem = new dataElem<MEMORY_ALLOCATOR>;
	auto *unmanagedElem = new dataElem<DefaultAllocationPolicy>;
	managedElem->prop1 = 1; managedElem->prop2 = 2; managedElem->val = 3.0f; managedElem->propTest.prop3 = 4;
	unmanagedElem->prop1 = 100; unmanagedElem->prop2 = 200; unmanagedElem->val = 300.0f; unmanagedElem->propTest.prop3 = 20;

	//managedElem->SetProcessLocation(DeviceSelector::CPU);

	foo(*managedElem); // works
	foo(*unmanagedElem); // illegal memory access -- attempt to access host mem from device

	cudaDeviceReset();
}