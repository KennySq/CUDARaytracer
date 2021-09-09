//#ifdef __cplusplus
//extern "C" {
//#endif

#include"Util.cuh"


template<typename _Ty>
class Memory
{
public:

	Memory(unsigned int count);
	Memory(const Memory& o) = delete;
	Memory(const Memory&& o) = delete;
	~Memory();

	inline unsigned int GetCount() { return mCount; }

private:
	_Ty* mRaw;
	unsigned int mCount;

};

template<typename _Ty>
Memory<_Ty>::Memory(unsigned int count)
{
	cudaError error = cudaMalloc((void**)&mRaw, sizeof(_Ty) * count);
	cudaErrorCheck(error);
}

template<typename _Ty>
Memory<_Ty>::~Memory()
{
	cudaError error = cudaFree((void*)mRaw);
	cudaErrorCheck(error);
}
