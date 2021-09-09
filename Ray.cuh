#pragma once

#include"Util.h"

class Ray
{
public:
	__device__ Ray() {}
	__device__ Ray(const Point3& origin, const Vec3& direction)
		: mOrigin(origin), mDirection(direction)
	{}

	inline __device__ Point3 At(double t) const
	{
		return mOrigin + t * mDirection;
	}

public:
	Point3 mOrigin;
	Vec3 mDirection;
};