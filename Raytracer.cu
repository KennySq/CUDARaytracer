#include "Raytracer.cuh"

__device__ DWORD* gPixels;
__device__ HittableList* gWorld;


//void kernelInitConstData(double* constScalar, Vec3* constVector)
//{
//	double aspectRatio = 4.0 / 3.0;
//
//	Point3 origin = Point3(0, 0, 0);
//	Vec3 vertical = Vec3(0, 2.0, 0);
//	Vec3 horizontal = Vec3(vertical.y() * aspectRatio, 0, 0);
//	Vec3 lowerLeft = origin - horizontal / 2 - vertical / 2 - Vec3(0, 0, 1.0);
//
//	double scalars[] = { aspectRatio };
//	Vec3 vectors[] = { origin, vertical, horizontal, lowerLeft };
//
//	cudaError error = cudaMemcpy(constScalar, scalars, sizeof(double) * ARRAYSIZE(scalars), cudaMemcpyHostToDevice);
//	cudaErrorCheck(error);
//
//	error = cudaMemcpy(constVector, vectors, sizeof(Vec3) * ARRAYSIZE(vectors), cudaMemcpyHostToDevice);
//	cudaErrorCheck(error);
//
//}

inline __device__ void setColor(LPDWORD pixels, int width, int height, Color color)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int tid = y * width + x;

	int writeColor = 0;

	int r = color[0] * 255.999;
	int g = color[1] * 255.999;
	int b = color[2] * 255.999;

	int ir = r << 16;
	int ig = g << 8;
	int ib = b;

	writeColor |= ir;
	writeColor |= ig;
	writeColor |= ib;

	pixels[tid] = writeColor;
}


__global__ void kernelClearScreen(LPDWORD pixels, Color color, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int tid = y * width + x;

	int writeColor = 0;

	int r = color[0] * 255.999;
	int g = color[1] * 255.999;
	int b = color[2] * 255.999;

	int ir = r << 16;
	int ig = g << 8;
	int ib = b;

	writeColor |= ir;
	writeColor |= ig;
	writeColor |= ib;

	pixels[tid] = writeColor;
	
}

__global__ void kernelBackground(LPDWORD pixels, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	double u = double(x) / (width);
	double v = double(y) / (height);

	double aspectRatio = 4.0 / 3.0;

	Point3 origin = Point3(0, 0, 0);
	Vec3 vertical = Vec3(0, 2.0, 0);
	Vec3 horizontal = Vec3(vertical.y() * aspectRatio, 0, 0);
	Vec3 lowerLeft = origin - horizontal / 2 - vertical / 2 - Vec3(0, 0, 1.0);

	Ray r(origin, lowerLeft + u * horizontal + v * vertical - origin);

	Color outColor{};
	Vec3 unitDirection = UnitVector(r.mDirection);

	double t = 0.5 * (unitDirection.e[1] + 1.0);
	outColor = (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);

	setColor(pixels, width, height, outColor);

}

template<typename _Ty>
void CopyDeviceToHost(void* device, void* host, unsigned int count)
{
	cudaError error = cudaMemcpy(host, device, count * sizeof(_Ty), cudaMemcpyDeviceToHost);
	cudaErrorCheck(error);
}

__device__ Color RayColor(LPDWORD pixels, const Ray& r, const HittableList& world, Hittable** worldObjects, int width, int height)
{
	HitRecord rec;

//	printf("%p\n", worldObjects);

	if (world.Hit(r, 0, infinity, rec, worldObjects))
	{
		return 0.5 * (rec.normal + Color(1, 1, 1));
	}

	Vec3 UnitDirection = UnitVector(r.mDirection);
	double t = 0.5 * (UnitDirection.y() + 1.0);

	return (1.0 - t) * Color(1, 1, 1) + t * Color(0.5, 0.7, 1.0);

}

__global__ void kernelRender(LPDWORD pixels, int width, int height, const HittableList& world, Hittable** worldObjects)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	double u = double(x) / (width);
	double v = double(y) / (height);

	double aspectRatio = 4.0 / 3.0;

	Point3 origin = Point3(0, 0, 0);
	Vec3 vertical = Vec3(0, 2.0, 0);
	Vec3 horizontal = Vec3(vertical.y() * aspectRatio, 0, 0);
	Vec3 lowerLeft = origin - horizontal / 2 - vertical / 2 - Vec3(0, 0, 1.0);

	Ray r(origin, lowerLeft + u * horizontal + v * vertical - origin);

	printf("%p\n", &world);

	Color out = RayColor(pixels, r, world, worldObjects, width, height);

	setColor(pixels, width, height, out);

	return;
}


Raytracer::Raytracer(HWND handle, HINSTANCE instance, unsigned int width, unsigned int height)
	: mHandle(handle), mInst(instance), mWidth(width), mHeight(height)
{
	BITMAPINFO bitInfo{};

	bitInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bitInfo.bmiHeader.biWidth = width;
	bitInfo.bmiHeader.biHeight = height;
	bitInfo.bmiHeader.biBitCount = 32;
	bitInfo.bmiHeader.biPlanes = 1;
	bitInfo.bmiHeader.biCompression = BI_RGB;

	HDC dc = GetDC(mHandle);

	mBitmap = CreateDIBSection(dc, &bitInfo, DIB_RGB_COLORS, (void**)(&mPixels), nullptr, 0);
	mMemoryDC = CreateCompatibleDC(dc);
	SelectObject(mMemoryDC, mBitmap);
	ReleaseDC(mHandle, dc);

	cudaError error = cudaMalloc((void**)&gPixels, sizeof(DWORD) * width * height);
	cudaErrorCheck(error);

	error = cudaMalloc((void**)&gWorld, sizeof(HittableList));
	cudaErrorCheck(error);

	error = cudaMalloc((void**)&deviceScene, sizeof(Hittable*) * 1);
	cudaErrorCheck(error);

	error = cudaMalloc((void**)&deviceSpheres, sizeof(Sphere) * 1);
	cudaErrorCheck(error);


	//kernelMakeShared << <1, 1 >> > (1, deviceScene, deviceSpheres);


	//AddSphere << <1, 1 >> > (Vec3(0, 0, 0), 5.0, gWorld, deviceScene);
	
	cudaDeviceSynchronize();
}


void Raytracer::Run()
{
	dim3 grids = dim3(50, 50, 1);
	dim3 blocks = dim3(16, 12, 1);
	
	//kernelBackground << <grids, blocks >> > (gPixels, mWidth ,mHeight);
	
	kernelRender << <grids, blocks >> > (gPixels, mWidth, mHeight, *gWorld, deviceScene);
	cudaDeviceSynchronize();


	
	CopyDeviceToHost<DWORD>(gPixels, mPixels, mWidth * mHeight);

}

void Raytracer::Release()
{
	DeleteDC(mMemoryDC);
	DeleteObject(mBitmap);
}
