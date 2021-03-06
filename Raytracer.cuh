#pragma once
#include"Ray.cuh"
#include"HittableList.cuh"

#include<Windows.h>
#include<iostream>

#ifdef __cplusplus
extern "C" {
#endif
	class Raytracer
	{
	public:
		Raytracer(HWND handle, HINSTANCE instance, unsigned int width, unsigned int height);
		
		Raytracer() = delete;
		Raytracer(const Raytracer& rhs) = delete;
		Raytracer(const Raytracer&& rhs) = delete;

		void Run();

		HDC GetMemoryDC() const { return mMemoryDC; }


		void Release();



	private:


		unsigned int mWidth;
		unsigned int mHeight;

		HWND mHandle;
		HBITMAP mBitmap;

		HDC mMemoryDC;

		LPDWORD mPixels;

		HINSTANCE mInst;
	};

#ifdef __cplusplus
}
#endif