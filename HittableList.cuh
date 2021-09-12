#pragma once
#include"Hittable.cuh"
#include"Sphere.cuh"

__device__ Hittable** deviceScene = nullptr;
__device__ Sphere* deviceSpheres;

inline __global__ void kernelMakeShared(int count, Hittable** scenePtr, Sphere* sceneSpheres)
{

	//	scenePtr = (Hittable**)malloc(sizeof(Hittable*) * count);
	sceneSpheres = (Sphere*)malloc(sizeof(Sphere) * count);

	//for (unsigned int i = 0; i < count; i++)
	//{
	//	scenePtr[i] = &sceneSpheres[i];
	//}

	printf("%p\n", scenePtr);
}


class HittableList : public Hittable
{
public:
	__host__ __device__ HittableList() : mCount(0)
	{
	}

	virtual __device__ bool Hit(const Ray& r, double tMin, double tMax, HitRecord& rec, Hittable** world) const override
	{
		HitRecord tempRec;
		bool hitAnything = false;

		double closest = tMax;

		//printf("%p\n", world[0]);

		//for (unsigned int i = 0; i < mCount; i++)
		//{
		//	if (world[i]->Hit(r, tMin, closest, tempRec, world))
		//	{
		//		hitAnything = true;
		//		closest = tempRec.t;
		//		rec = tempRec;
		//	}

		//	__syncthreads();
		//}

		return hitAnything;
	}

	int mCount;

};


inline __global__ void AddSphere(Vec3 center, double radius, HittableList* deviceScene, Hittable** rawScene)
{

	Sphere* sph = (Sphere*)rawScene[deviceScene->mCount];

	sph->mCenter = center;
	sph->mRadius = radius;

	deviceScene->mCount++;

	return;
}