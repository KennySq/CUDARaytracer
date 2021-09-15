#pragma once
#include"Hittable.cuh"
#include"Sphere.cuh"


//inline __global__ void kernelMakeShared(int count, Hittable** scenePtr, Sphere* sceneSpheres)
//{
//
//	//	scenePtr = (Hittable**)malloc(sizeof(Hittable*) * count);
//	sceneSpheres = (Sphere*)malloc(sizeof(Sphere) * count);
//
//	//for (unsigned int i = 0; i < count; i++)
//	//{
//	//	scenePtr[i] = &sceneSpheres[i];
//	//}
//
//	printf("%p\n", scenePtr);
//}


class HittableList : public Hittable
{
public:
	__host__ __device__ HittableList()
	{
	}

	__device__ virtual bool Hit(Ray& r, double tMin, double tMax, HitRecord& rec, Hittable** world, unsigned int count) const override
	{
		HitRecord tempRec;
		bool hitAnything = false;

		double closest = tMax;


		//printf("world[0] => %p\n", *world[0]);
		//printf("world[1] => %p\n", *world[1]);

		for (unsigned int i = 0; i < 1; i++)
		{
			Sphere* sph = (Sphere*)(world[i]);
			if (sph->Hit(r, tMin, closest, tempRec, world, count))
			{
				hitAnything = true;
				closest = tempRec.t;
				rec = tempRec;
			}

			__syncthreads();
		}

		return hitAnything;
	}

};


//inline __global__ void AddSphere(Vec3 center, double radius, HittableList* deviceScene, Hittable** rawScene, Sphere* spheres)
//{
//	unsigned int count = deviceScene->mCount;
//
//	Sphere* sph = &spheres[count];
//
//	sph->mCenter = center;
//	sph->mRadius = radius;
//
//	deviceScene->mCount++;
//
//	rawScene[count] = (Hittable*)sph;
//
//	return;
//}