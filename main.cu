#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cstdio>
#include <iostream>
#include <ctime>
#include "shader_program.h"
#include "cuda.cuh"

constexpr unsigned int canvasCount = 6*3;
constexpr Vec3 cameraPos = { 0.0f, -1100.0f, 0.0f };
constexpr Vec3 cameraDir = { 0.0f, 1.0f, 0.0f };
constexpr Vec3 cameraUp = { 0.0f, 0.0f, 1.0f };
constexpr float cameraFOVDeg = 90.0f;
Camera gCamera = Camera(cameraPos, cameraDir, cameraUp, cameraFOVDeg);
unsigned int gVAO;
unsigned int gTexture;
unsigned int gAuxBuffer;
cudaGraphicsResource_t gCudaAux;
bool gWasResized = false;

GLFWwindow* initializeWindow(unsigned int width, unsigned int height);
void resizeWindow(GLFWwindow* window, int width, int height);
void postResizeWindow();
void initializeSpheres(Spheres* spheres);
void initializeLights(Lights* lights);
void createVAO(unsigned int* VAO);
void createTexture(unsigned int* texture, unsigned int* auxBuffer, cudaGraphicsResource_t* cudaAux,
	unsigned int width, unsigned int height);
void processInput(GLFWwindow* window);
float getRandPosSphere();
float getRandPosLight();
float getRad();
float getRandCol();

int main() {
	GLFWwindow* window = initializeWindow(gWidth, gHeight);
	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

	Spheres* initialSpheres = new Spheres();
	Lights* initialLights = new Lights();

	initializeSpheres(initialSpheres);
	initializeLights(initialLights);

	Spheres* spheres;
	Lights* lights;
	allocCuda(&spheres, &lights, initialSpheres, initialLights);
	delete initialSpheres;
	delete initialLights;
	
	createVAO(&gVAO);
	createTexture(&gTexture, &gAuxBuffer, &gCudaAux, gWidth, gHeight);
	unsigned char* cudaTexture;

	ShaderProgram shaderProgram = ShaderProgram("vertexShader.glsl", "fragmentShader.glsl");

	//glfwSwapInterval(0); // turn off vsync
	while(!glfwWindowShouldClose(window)) {
		processInput(window);

		cudaGraphicsMapResources(1, &gCudaAux, 0);
		size_t size;
		cudaGraphicsResourceGetMappedPointer((void**)&cudaTexture, &size, gCudaAux);
		runKernel(spheres, lights, gCamera, cudaTexture, gWidth, gHeight);
		cudaGraphicsUnmapResources(1, &gCudaAux, 0);

		glBindTexture(GL_TEXTURE_2D, gTexture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, gWidth, gHeight, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		shaderProgram.use();
		glBindVertexArray(gVAO);
		glDrawArrays(GL_TRIANGLES, 0, canvasCount);

		glfwSwapBuffers(window);
		glfwPollEvents();
		if(gWasResized) postResizeWindow();
	}

	freeCuda(spheres, lights);
	glfwTerminate();
	return 0;
}

GLFWwindow* initializeWindow(unsigned int width, unsigned int height) {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	GLFWwindow* window = glfwCreateWindow(width, height, "Spheres", nullptr, nullptr);
	glfwSetWindowPos(window, 0, 38);
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, resizeWindow);
	return window;
}

void resizeWindow(GLFWwindow* window, int width, int height) {
	gWidth = width;
	gHeight = height;

	glViewport(0, 0, width, height);

	gWasResized = true;
}

void postResizeWindow() {
	gCamera.setProjPlaneWidth(pixelGap*gWidth);
	gCamera.setProjPlaneHeight(pixelGap*gHeight);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gAuxBuffer);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, gWidth*gHeight*bytesPerPixel, nullptr, GL_DYNAMIC_COPY);
	cudaGraphicsGLRegisterBuffer(&gCudaAux, gAuxBuffer, cudaGraphicsMapFlagsNone);

	glBindTexture(GL_TEXTURE_2D, gTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, gWidth, gHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	gWasResized = false;
}

void initializeSpheres(Spheres* spheres) {
	srand((unsigned int)time(nullptr));
	for(int i = 0; i < spheresCount; i++) {
		spheres->x[i] = getRandPosSphere();
		spheres->y[i] = getRandPosSphere();
		spheres->z[i] = getRandPosSphere();
		spheres->rad[i] = getRad();
		spheres->R[i] = getRandCol();
		spheres->G[i] = getRandCol();
		spheres->B[i] = getRandCol();
	}
}

void initializeLights(Lights* lights) {
	srand((unsigned int)time(nullptr));
	for(int i = 0; i < lightsCount; i++) {
		lights->x[i] = getRandPosLight();
		lights->y[i] = getRandPosLight();
		lights->z[i] = getRandPosLight();
		lights->R[i] = getRandCol();
		lights->G[i] = getRandCol();
		lights->B[i] = getRandCol();
	}
}

void createVAO(unsigned int* VAO) {
	float canvas[canvasCount] = {
		-1.0f, -1.0f, 0.0f,
		-1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		1.0f, 1.0f, 0.0f
	};

	unsigned int VBO;
	glGenBuffers(1, &VBO);
	glGenVertexArrays(1, VAO);
	glBindVertexArray(*VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, canvasCount*sizeof(float), canvas, GL_DYNAMIC_COPY);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
}

void createTexture(unsigned int* texture, unsigned int* auxBuffer, cudaGraphicsResource_t* cudaAux,
	unsigned int width, unsigned int height) {
	glGenBuffers(1, auxBuffer);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *auxBuffer);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, width*height*bytesPerPixel, nullptr, GL_DYNAMIC_COPY);
	cudaGraphicsGLRegisterBuffer(cudaAux, *auxBuffer, cudaGraphicsMapFlagsNone);
	
	glGenTextures(1, texture);
	glBindTexture(GL_TEXTURE_2D, *texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void processInput(GLFWwindow* window) {
	if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
	}
	if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		gCamera.moveForwards();
	}
	if(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		gCamera.moveBackwards();
	}
	if(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		gCamera.tiltLeft();
	}
	if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		gCamera.tiltRight();
	}
	if(glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
		gCamera.turnUp();
	}
	if(glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
		gCamera.turnDown();
	}
	if(glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
		gCamera.turnLeft();
	}
	if(glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
		gCamera.turnRight();
	}
}

float getRandPosSphere() {
	return (2*(float)rand()/RAND_MAX - 1)*sceneSize;
}

float getRandPosLight() {
	return (2*(float)rand()/RAND_MAX - 1)*sceneSize;
}

float getRad() {
	return avgRadius;
}

float getRandCol() {
	return (float)rand()/RAND_MAX;
}
