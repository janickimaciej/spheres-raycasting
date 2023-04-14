#include "cuda.cuh"
#include <iostream>
#include <chrono>
#include <cmath>

void allocCuda(Spheres** spheres, Lights** lights, Spheres* initialSpheres, Lights* initialLights) {
	cudaMalloc(spheres, sizeof(Spheres));
	cudaMalloc(lights, sizeof(Lights));
	cudaMemcpy(*spheres, initialSpheres, sizeof(Spheres), cudaMemcpyHostToDevice);
	cudaMemcpy(*lights, initialLights, sizeof(Lights), cudaMemcpyHostToDevice);
}

void freeCuda(Spheres* spheres, Lights* lights) {
	cudaFree(spheres);
	cudaFree(lights);
}

void runKernel(Spheres* spheres, Lights* lights, const Camera& camera, unsigned char* texture,
	unsigned int width, unsigned int height) {
	static const int blockSize = 256;
	static const int numBlocks = (width*height + blockSize - 1)/blockSize;
	textureKernel<<<numBlocks, blockSize>>>(spheres, lights, camera.getPos(), camera.getUp(),
		camera.getRight(), camera.getRayIntersect(), texture, width, height);
	cudaDeviceSynchronize();
}

__global__ void textureKernel(Spheres* spheres, Lights* lights, Vec3 cameraPos, Vec3 cameraUp,
	Vec3 cameraRight, Vec3 rayIntersect, unsigned char* texture, unsigned int width,
	unsigned int height) {
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid >= width*height) return;
	texture[4*tid + 3] = 255;

	Vec3 pixelPos = calcPixelPosition(cameraPos, cameraRight, cameraUp, width, height, tid);
	Vec3 pixelDir = calcPixelDir(rayIntersect, pixelPos);

	float closestT;
	unsigned int closestSphereIndex;
	findClosest(&closestT, &closestSphereIndex, pixelPos, pixelDir, spheres);

	if(closestSphereIndex == INT_MAX) {
		texture[4*tid] = 0;
		texture[4*tid + 1] = 0;
		texture[4*tid + 2] = 0;
		return;
	}

	float R;
	float G;
	float B;
	calcColor(&R, &G, &B, pixelPos, pixelDir, closestT, spheres, closestSphereIndex, lights);
	texture[4*tid] = f2c(R);
	texture[4*tid + 1] = f2c(G);
	texture[4*tid + 2] = f2c(B);
}

__device__ Vec3 calcPixelPosition(Vec3 cameraPos, Vec3 cameraRight, Vec3 cameraUp, int width,
	int height, int tid) {
	int row = tid/width;
	int col = tid - width*row;

	float coefRight = (col - (float)width/2)*pixelGap;
	float coefUp = (row - (float)height/2)*pixelGap;

	Vec3 pixelPos = cameraPos;
	pixelPos.x += coefRight*cameraRight.x + coefUp*cameraUp.x;
	pixelPos.y += coefRight*cameraRight.y + coefUp*cameraUp.y;
	pixelPos.z += coefRight*cameraRight.z + coefUp*cameraUp.z;
	return pixelPos;
}

__device__ Vec3 calcPixelDir(Vec3 rayIntersect, Vec3 pixelPos) {
	return getNormalizedCuda(Vec3 {
		pixelPos.x - rayIntersect.x,
		pixelPos.y - rayIntersect.y,
		pixelPos.z - rayIntersect.z
	});
}

__device__ void findClosest(float* outT, unsigned int* outSphereIndex, Vec3 pixelPos, Vec3 pixelDir,
	Spheres* spheres) {
	float minT = FLT_MAX;
	unsigned int minSphereIndex = INT_MAX;
	float t;
	Vec3 spherePos;
	for(int i = 0; i < spheresCount; i++) {
		spherePos = { spheres->x[i], spheres->y[i], spheres->z[i] };
		t = calcIntersect(pixelPos, pixelDir, spherePos, spheres->rad[i]);
		if(t > 0 && t < minT) {
			minT = t;
			minSphereIndex = i;
		}
	}
	*outT = minT;
	*outSphereIndex = minSphereIndex;
}

__device__ float calcIntersect(Vec3 pixelPos, Vec3 pixelDir, Vec3 spherePos, float sphereRad) {
	float dx = pixelPos.x - spherePos.x;
	float dy = pixelPos.y - spherePos.y;
	float dz = pixelPos.z - spherePos.z;

	float a = pixelDir.x*pixelDir.x + pixelDir.y*pixelDir.y + pixelDir.z*pixelDir.z;
	float b = 2*(pixelDir.x*dx + pixelDir.y*dy + pixelDir.z*dz);
	float c = dx*dx + dy*dy + dz*dz - sphereRad*sphereRad;

	float delta = b*b - 4*a*c;
	if(delta <= 0) return FLT_MAX;
	else return (-b - sqrt(delta))/(2*a);
}

__device__ void calcColor(float* outR, float* outG, float* outB, Vec3 pixelPos, Vec3 pixelDir,
	float t, Spheres* spheres, unsigned int sphereIndex, Lights* lights) {
	*outR = 0;
	*outG = 0;
	*outB = 0;

	float sphereR = spheres->R[sphereIndex];
	float sphereG = spheres->G[sphereIndex];
	float sphereB = spheres->B[sphereIndex];
	float lightR;
	float lightG;
	float lightB;
	float brightness;

	Vec3 intersection = {
		pixelPos.x + pixelDir.x*t,
		pixelPos.y + pixelDir.y*t,
		pixelPos.z + pixelDir.z*t
	};

	Vec3 vecN = {
		intersection.x - spheres->x[sphereIndex],
		intersection.y - spheres->y[sphereIndex],
		intersection.z - spheres->z[sphereIndex]
	};
	vecN = getNormalizedCuda(vecN);
	Vec3 vecV = { -pixelDir.x, -pixelDir.y, -pixelDir.z };
	Vec3 vecL;
	Vec3 vecR;

	float cosNL;
	float cosVR;

	*outR += sphereR*Ka;
	*outG += sphereG*Ka;
	*outB += sphereB*Ka;

	for(int i = 0; i < lightsCount; i++) {
		lightR = lights->R[i];
		lightG = lights->G[i];
		lightB = lights->B[i];

		vecL = {
			lights->x[i] - intersection.x,
			lights->y[i] - intersection.y,
			lights->z[i] - intersection.z
		};
		vecL = getNormalizedCuda(vecL);

		cosNL = dotProduct(vecN, vecL);
		
		vecR = {
			2*cosNL*vecN.x - vecL.x,
			2*cosNL*vecN.y - vecL.y,
			2*cosNL*vecN.z - vecL.z
		};
		vecR = getNormalizedCuda(vecR);

		cosVR = dotProduct(vecV, vecR);

		if(cosNL < 0) cosNL = 0;
		if(cosVR < 0) cosVR = 0;

		brightness = Kd*cosNL + Ks*pow(cosVR, m);

		*outR += sphereR*lightR*brightness;
		*outG += sphereG*lightG*brightness;
		*outB += sphereB*lightB*brightness;
	}
}

__device__ Vec3 getNormalizedCuda(Vec3 vec) {
	float norm = sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
	return Vec3 { vec.x/norm, vec.y/norm, vec.z/norm };
}

__device__ float dotProduct(Vec3 vec1, Vec3 vec2) {
	return vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z;
}

__device__ unsigned char f2c(float color) {
	if(color < 0) return 0;
	if(color > 1) return 255;
	return color*255;
}
