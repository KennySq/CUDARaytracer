#pragma once
#include<cuda_runtime_api.h>
#include<device_functions.h>
#include<device_launch_parameters.h>
#include<curand_kernel.h>
#include<limits>
#include<stdio.h>

__constant__ const float infinity = std::numeric_limits<float>::infinity();
__constant__ const float pi = 3.1415926535897932385;

inline float Deg2Rad(float degrees) {
	return degrees * pi / 180.0;
}


static void cudaErrorCheck(const cudaError& error)
{
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));

	}
}

#include"Vec3.cuh"