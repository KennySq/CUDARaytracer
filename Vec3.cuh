#include<cuda_runtime_api.h>
#include<device_functions.h>
#include<cmath>

class Vec3
{
public:
	__host__ __device__ Vec3() : e{ 0,0,0 } {}
	__host__ __device__ Vec3(float x, float y, float z) : e{ x,y,z, } {}
//	__host__ __device__ Vec3(Vec3& vec) : e{ vec.e[0], vec.e[1], vec.e[2] } {}


	inline __host__ __device__ float x() const { return e[0]; }
	inline __host__ __device__ float y() const { return e[1]; }
	inline __host__ __device__ float z() const { return e[2]; }


	inline __device__ Vec3 operator-() { return Vec3(-e[0], -e[1], -e[2]); }
	inline __device__ float operator[](int i) const { return e[i]; }
	inline __device__ float& operator[](int i) { return e[i]; }

	inline __host__ __device__ Vec3& operator+=(const Vec3& v)
	{
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];

		return *this;
	}

	inline __host__ __device__ Vec3& operator*=(const float t)
	{
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;

		return *this;
	}

	inline __host__ __device__ Vec3& operator/=(const float t)
	{
		return *this *= 1 / t;
	}

	inline __host__ __device__ float LengthSquared() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
	inline __host__ __device__ float Length() const { return sqrtf(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }



public:
	float e[3];
};

using Point3 = Vec3;
using Color = Vec3;

inline __device__ Vec3 operator+(Vec3 u, Vec3 v)
{
	return Vec3(u[0] + v[0], u[1] + v[1], u[2] + v[2]);
}

inline __device__ Vec3 operator-(Vec3 u, Vec3 v)
{
	return Vec3(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
}

inline __device__ Vec3 operator*(Vec3 u, Vec3 v)
{
	return Vec3(u[0] * v[0], u[1] * v[1], u[2] * v[2]);
}

inline __device__ Vec3 operator*(float t, Vec3 v)
{
	return Vec3(t * v[0], t * v[1], t * v[2]);
}

inline __device__ Vec3 operator*(Vec3 v, float t)
{
	return t * v;
}

inline __device__ Vec3 operator/(Vec3 v,  float t)
{
	return (1 / t) * v;
}

inline __device__ float Dot(Vec3 u, Vec3 v)
{
	return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

inline __device__ Vec3 Cross(Vec3 u, Vec3 v)
{
	return Vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline __device__ Vec3 UnitVector(Vec3& v)
{
	return v / v.Length();
}

