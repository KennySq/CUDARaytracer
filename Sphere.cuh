#pragma once
#include"Hittable.cuh"

class Sphere : public Hittable
{
public:
	__device__ Sphere() {}
	__device__ Sphere(Point3 cen, float r) : mCenter(cen), mRadius(r) {}

	__device__ virtual bool Hit(Ray& r, float tMin, float tMax, HitRecord& rec, Hittable** world, unsigned int count) const override
	{
		Vec3 oc = r.mOrigin - mCenter;
		float a = Dot(r.mDirection, r.mDirection);
		float b = Dot(oc, r.mDirection);
		float c = Dot(oc,oc) - mRadius * mRadius;
	
		float discriminant = b * b - a * c;

		if (discriminant > 0)
		{
			float temp = (-b - sqrt(discriminant)) / a;

			if (temp < tMax && temp > tMin)
			{
				rec.t = temp;
				rec.p = r.At(rec.t);
				rec.normal = (rec.p - mCenter) / mRadius;
				return true;
			}

			temp = (-b + sqrt(discriminant)) / a;

			if (temp < tMax && temp > tMin)
			{
				rec.t = temp;
				rec.p = r.At(rec.t);
				rec.normal = (rec.p - mCenter) / mRadius;
				return true;
			}

		}
		return false;
	
	}

public:
	Point3 mCenter;
	float mRadius;
};