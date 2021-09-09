#include<cuda_runtime_api.h>
#include<device_functions.h>
#include<cmath>

class Vec3
{
public:
	__host__ __device__ Vec3() : e{ 0,0,0 } {}
	__host__ __device__ Vec3(double x, double y, double z) : e{ x,y,z, } {}

	inline __device__ double x() const { return e[0]; }
	inline __device__ double y() const { return e[1]; }
	inline __device__ double z() const { return e[2]; }


	inline __device__ Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
	inline __device__ double operator[](int i) const { return e[i]; }
	inline __device__ double& operator[](int i) { return e[i]; }

	inline __device__ Vec3& operator+=(const Vec3& v)
	{
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];

		return *this;
	}

	inline __device__ Vec3& operator*=(const double t)
	{
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;

		return *this;
	}

	inline __device__ Vec3& operator/=(const double t)
	{
		return *this *= 1 / t;
	}

	inline __device__ double LengthSquared() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
	inline __device__ double Length() const { return sqrtf(LengthSquared()); }



public:
	double e[3];
};

using Point3 = Vec3;
using Color = Vec3;

inline __device__ Vec3 operator+(const Vec3& u, const Vec3& v)
{
	return Vec3(u[0] + v[0], u[1] + v[1], u[2] + v[2]);
}

inline __device__ Vec3 operator-(const Vec3& u, const Vec3& v)
{
	return Vec3(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
}

inline __device__ Vec3 operator*(const Vec3& u, const Vec3& v)
{
	return Vec3(u[0] * v[0], u[1] * v[1], u[2] * v[2]);
}

inline __device__ Vec3 operator*(double t, const Vec3& v)
{
	return Vec3(t * v[0], t * v[1], t * v[2]);
}

inline __device__ Vec3 operator*(const Vec3& v, double t)
{
	return t * v;
}

inline __device__ Vec3 operator/(Vec3 v,  double t)
{
	return (1 / t) * v;
}

inline __device__ double Dot(const Vec3& u, const Vec3& v)
{
	return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

inline __device__ Vec3 Cross(const Vec3& u, const Vec3& v)
{
	return Vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline __device__ Vec3 UnitVector(Vec3 v)
{
	return v / v.Length();
}