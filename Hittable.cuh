#pragma once
#include"Util.h"
#include"Ray.cuh"
struct HitRecord
{
	Point3 p;
	Vec3 normal;
	double t;
	bool bFrontFace;

	inline __device__ void SetFaceNormal(Ray r, Vec3 outwardNormal)
	{
		bFrontFace = Dot(r.mDirection, outwardNormal) < 0;
		normal = bFrontFace ? outwardNormal : -outwardNormal;
	}
};

class Hittable
{
public:
	__device__ virtual bool Hit(Ray& r, double tMin, double tMax, HitRecord& rec, Hittable** world, unsigned int count) const = 0;
};