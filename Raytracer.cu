#include "Raytracer.cuh"

__device__ DWORD* gPixels;
__device__ Hittable** gWorld;
__device__ Hittable** gSpheres;
__device__ 	curandState* gRandStates;

const unsigned int gSphereCount = 2;
const unsigned int gSampleCount = 50;

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

	float u = float(x) / (width);
	float v = float(y) / (height);

	float aspectRatio = 4.0 / 3.0;

	Point3 origin = Point3(0, 0, 0);
	Vec3 vertical = Vec3(0, 2.0, 0);
	Vec3 horizontal = Vec3(vertical.y() * aspectRatio, 0, 0);
	Vec3 lowerLeft = origin - horizontal / 2 - vertical / 2 - Vec3(0, 0, 1.0);

	Ray r(origin, lowerLeft + u * horizontal + v * vertical - origin);

	Color outColor{};
	Vec3 unitDirection = UnitVector(r.mDirection);

	float t = 0.5 * (unitDirection.e[1] + 1.0);
	outColor = (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);

	setColor(pixels, width, height, outColor);

}

__global__ void randInit(int width, int height, curandState* state)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int index = j * width + i;

	curand_init(1984, index, 0, &state[index]);

}

__global__ void makeResources(Hittable** world, Hittable** spheres, unsigned int count)
{
	if (threadIdx.x == 0)
	{
		(*world) = new HittableList();

		for (unsigned int i = 0; i < count; i++)
		{
			(spheres)[i] = new Sphere();
		}

		printf("makeResources => %p\n", spheres[0]);
		printf("makeResources => %p\n", spheres[1]);
	}

	__syncthreads();

	return;
}

__global__ void AddSphere(Vec3 center, float radius, unsigned int index, Hittable** spheres, Hittable** world)
{
	if (threadIdx.x == 0)
	{
		Sphere* sph = (Sphere*)spheres[index];

		sph->mCenter = center;
		sph->mRadius = radius;
	}

	__syncthreads();

}

template<typename _Ty>
void CopyDeviceToHost(void* device, void* host, unsigned int count)
{
	cudaError error = cudaMemcpy(host, device, count * sizeof(_Ty), cudaMemcpyDeviceToHost);
	cudaErrorCheck(error);
}

inline __device__ Vec3 RandVec(curandState* randState)
{
	return Vec3(curand_uniform(randState), curand_uniform(randState), curand_uniform(randState));
}

#define RANDVEC Vec3(curand_uniform(localRand), curand_uniform(localRand), curand_uniform(localRand))

inline __device__ Vec3 RandomUnitSphere(curandState* localRand)
{
	Vec3 p;

	do
	{
		p = 2.0 * RANDVEC - Vec3(1, 1, 1);
	} while (p.LengthSquared() >= 1.0f);

	return p;

}


__device__ Color RayColor(LPDWORD pixels, Ray& r, Hittable** world, Hittable** spheres, int width, int height, int sphereCount, curandState* randStates, int tid)
{
	Ray currentRay = r;

	float atten = 1.0f;

	for (int i = 0; i < 50; i++)
	{
		HitRecord rec;


		if ((*world)->Hit(currentRay, 0, infinity, rec, spheres, sphereCount))
		{
			Vec3 target = rec.p + rec.normal + RandomUnitSphere(&randStates[tid]);
			atten *= 0.5f;
			
			currentRay = Ray(rec.p, target - rec.p);
		}

		//__syncthreads(); // deadlock.
		else
		{
			Vec3 UnitDirection = UnitVector(currentRay.mDirection);
			float t = 0.5 * (UnitDirection.y() + 1.0);

			Vec3 c = (1.0f - t) * Vec3(1, 1, 1) + t * Vec3(0.5, 0.7, 1.0);
			return atten * c;
		}

	

	}

	return Vec3(0, 0, 0);


}

__global__ void kernelRender(LPDWORD pixels, int width, int height, Hittable** world, Hittable** spheres, unsigned int sphereCount, curandState* randStates, unsigned int sampleCount)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	curandState localRand = randStates[y * width + x];
	Color out(0,0,0);

	float aspectRatio = 4.0 / 3.0;

	const Point3 origin = Point3(0, 0, 0);
	const Vec3 vertical = Vec3(0, 2.0, 0);
	const Vec3 horizontal = Vec3(vertical.y() * aspectRatio, 0, 0);
	const Vec3 lowerLeft = origin - horizontal / 2 - vertical / 2 - Vec3(0, 0, 1.0);


	for (int s = 0; s < sampleCount; s++)
	{
		float u = float(x + curand_uniform(&localRand)) / float(width);
		float v = float(y + curand_uniform(&localRand)) / float(height);

		Ray r(origin, lowerLeft + u * horizontal + v * vertical - origin);

		out += RayColor(pixels, r, world, spheres, width, height, sphereCount, randStates, y*width+x);
	}

	randStates[y * width + x] = localRand;

	out /= sampleCount;
	out[0] = sqrt(out[0]);
	out[1] = sqrt(out[1]);
	out[2] = sqrt(out[2]);

	setColor(pixels, width, height, out);

	return;
}


Raytracer::Raytracer(HWND handle, HINSTANCE instance, unsigned int width, unsigned int height)
	: mHandle(handle), mInst(instance), mWidth(width), mHeight(height)
{
	dim3 grids = dim3(50, 50, 1);
	dim3 blocks = dim3(16, 12, 1);

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

	error = cudaMalloc((void**)&gWorld, sizeof(Hittable**));
	cudaErrorCheck(error);

	//error = cudaMalloc((void**)&gRawScene, sizeof(Hittable***) * gSphereCount);
	//cudaErrorCheck(error);

	error = cudaMalloc((void**)&gSpheres, sizeof(Hittable**) * gSphereCount);
	cudaErrorCheck(error);

	makeResources << <1, 1 >> > (gWorld, gSpheres, gSphereCount);

	AddSphere << <1, 1 >> > (Vec3(0, 0, -1), 0.5, 0, gSpheres, gWorld);
	AddSphere << <1, 1 >> > (Vec3(0, -100.5, -1), 100, 1, gSpheres, gWorld);


	cudaMalloc((void**)&gRandStates, sizeof(curandState) * (width * height));
	cudaErrorCheck(error);

	cudaDeviceSynchronize();

	randInit << <grids, blocks >> > (width, height, gRandStates);
	error = cudaGetLastError();
	cudaErrorCheck(error);
}

void Raytracer::Run()
{
	dim3 grids = dim3(50, 50, 1);
	dim3 blocks = dim3(16, 12, 1);
	
	//kernelBackground << <grids, blocks >> > (gPixels, mWidth ,mHeight);
	
	kernelRender << <grids, blocks >> > (gPixels, mWidth, mHeight, gWorld, gSpheres, gSphereCount, gRandStates, gSampleCount);
	cudaDeviceSynchronize();
	

	CopyDeviceToHost<DWORD>(gPixels, mPixels, mWidth * mHeight);
//	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

}

void Raytracer::Release()
{
	DeleteDC(mMemoryDC);
	DeleteObject(mBitmap);
}
