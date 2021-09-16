#pragma once
#include"Hittable.cuh"
#include"Sphere.cuh"


class HittableList : public Hittable
{
public:
	__host__ __device__ HittableList()
	{
	}

	__device__ virtual bool Hit(Ray& r, float tMin, float tMax, HitRecord& rec, Hittable** world, unsigned int count) const override
	{
		HitRecord tempRec;
		bool hitAnything = false;

		float closest = tMax;

		for (unsigned int i = 0; i < count; i++)
		{
			Sphere* sph = (Sphere*)(world[i]);
			if (sph->Hit(r, tMin, closest, tempRec, world, count))
			{
				hitAnything = true;
				closest = tempRec.t;
				rec = tempRec;
			}

		}

		return hitAnything;
	}

};