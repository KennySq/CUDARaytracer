#pragma once

#include"Util.h"

class Ray
{
public:
	__device__ Ray() {}
	__device__ Ray(Point3 origin, Vec3 direction)
		: mOrigin(origin), mDirection(direction)
	{}

	inline __device__ Point3 At(float t)
	{
		return mOrigin + t * mDirection;
	}

public:
	Point3 mOrigin;
	Vec3 mDirection;
};