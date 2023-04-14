#ifndef CUDA_CUH
#define CUDA_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "camera.h"

constexpr unsigned int bytesPerPixel = 4;
constexpr unsigned int spheresCount = 100;
constexpr unsigned int lightsCount = 25;
constexpr float sceneSize = 1000.0f;
constexpr float avgRadius = 50.0f;
constexpr float Ka = 0.05f;
constexpr float Kd = 0.1f;
constexpr float Ks = 0.8f;
constexpr int m = 80;

struct Spheres {
	float x[spheresCount];
	float y[spheresCount];
	float z[spheresCount];
	float rad[spheresCount];
	float R[spheresCount];
	float G[spheresCount];
	float B[spheresCount];
};

struct Lights {
	float x[lightsCount];
	float y[lightsCount];
	float z[lightsCount];
	float R[lightsCount];
	float G[lightsCount];
	float B[lightsCount];
};

void allocCuda(Spheres** spheres, Lights** lights, Spheres* initialSpheres, Lights* initialLights);
void freeCuda(Spheres* spheres, Lights* lights);
void runKernel(Spheres* spheres, Lights* lights, const Camera& camera, unsigned char* texture,
	unsigned int width, unsigned int height);
__global__ void textureKernel(Spheres* spheres, Lights* lights, Vec3 cameraPos, Vec3 cameraUp,
	Vec3 cameraRight, Vec3 rayIntersect, unsigned char* texture, unsigned int width,
	unsigned int height);
__device__ Vec3 calcPixelPosition(Vec3 cameraPos, Vec3 right, Vec3 up, int width,
	int height, int tid);
__device__ Vec3 calcPixelDir(Vec3 rayIntersect, Vec3 pixelPos);
__device__ void findClosest(float* outT, unsigned int* outSphereIndex, Vec3 pixelPos, Vec3 pixelDir,
	Spheres* spheres);
__device__ float calcIntersect(Vec3 pixelPos, Vec3 pixelDir, Vec3 spherePos, float sphereRad);
__device__ void calcColor(float* outR, float* outG, float* outB, Vec3 pixelPos, Vec3 pixelDir,
	float t, Spheres* spheres, unsigned int sphereIndex, Lights* lights);
__device__ Vec3 getNormalizedCuda(Vec3 vec);
__device__ float dotProduct(Vec3 vec1, Vec3 vec2);
__device__ unsigned char f2c(float color);

#endif
