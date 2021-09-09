#pragma once
#include"Hittable.cuh"
#include"Sphere.cuh"

extern __shared__ Hittable* sharedSceneMemory[];
__shared__ Hittable** deviceScene;

inline __global__ void kernelMakeShared()
{
	if (threadIdx.x == 0)
	{
		deviceScene = (Hittable**)sharedSceneMemory;
	}

	printf("%p\n", deviceScene);

	__syncthreads();
}


class HittableList : public Hittable
{
public:
	__host__ __device__ HittableList()
	{
//		kernelMakeShared << <1, 1, 49152 >> > ();
	}
//	__device__ HittableList();


// Hittable을(를) 통해 상속됨
	virtual __device__ bool Hit(const Ray& r, double tMin, double tMax, HitRecord& rec, Hittable**  world) const override
	{
		HitRecord tempRec;
		bool hitAnything = false;

		double closest = tMax;

		for (unsigned int i = 0; i < mCount; i++)
		{
			if (world[i]->Hit(r, tMin, closest, tempRec, world))
			{
				hitAnything = true;
				closest = tempRec.t;
				rec = tempRec;
			}
		}

		return hitAnything;
	}

	int mCount;

};


inline __global__ void AddSphere(Sphere* sphere, HittableList* deviceScene, Hittable** rawScene)
{

	rawScene[deviceScene->mCount] = sphere;
	deviceScene->mCount++;

	return;
}