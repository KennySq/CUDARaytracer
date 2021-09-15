#pragma once
#include"Hittable.cuh"

// 32 + 8 (virtual?)
class Sphere : public Hittable
{
public:
	__device__ Sphere() {}
	__device__ Sphere(Point3 cen, double r) : mCenter(cen), mRadius(r) {}

	__device__ virtual bool Hit(Ray& r, double tMin, double tMax, HitRecord& rec, Hittable** world, unsigned int count) const override
	{

		Vec3 oc = r.mOrigin - mCenter;
		double a = r.mDirection.LengthSquared();
		double bHalf = Dot(oc, r.mDirection);
		double c = oc.LengthSquared() - mRadius * mRadius;
	
		double discriminant = bHalf * bHalf - a * c;

		if (discriminant < 0)
		{
			return false;
		}

		double sqrtd = sqrtf(discriminant);

		double root = (-bHalf - sqrtd) / a;
		if (root < tMin || root > tMax)
		{
			root = (-bHalf + sqrtd) / a;
			if (root < tMin || tMax < root)
			{
				return false;
			}
		}

		//rec.t = root;
		rec.p = r.At(rec.t); // ??

		Vec3 outwardNormal = (rec.p - mCenter) / mRadius;
		//rec.SetFaceNormal(r, outwardNormal);
		return true;
	
	}

public:
	Point3 mCenter; // 24
	double mRadius; // 8
};