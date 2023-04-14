#ifndef CAMERA
#define CAMERA

#include "matrix.h"

extern unsigned int gWidth;
extern unsigned int gHeight;
constexpr float Pi = 3.14159265358979323846f;
constexpr float pixelGap = 0.05f;

struct Vec3 {
	float x;
	float y;
	float z;
};

float deg2Rad(float deg);
Vec3 getNormalized(const Vec3& vec);
Vec3 crossProduct(const Vec3& v1, const Vec3& v2);

class Camera {
	enum class Axis {
		Dir,
		Up,
		Right
	};

	enum class Direction {
		Negative,
		Positive
	};

	Vec3 pos;
	Vec3 dir;
	Vec3 up;
	Vec3 right;
	float projPlaneWidth = pixelGap*gWidth;
	float projPlaneHeight = pixelGap*gHeight;
	float FOVRad;
	float vel = 10.0f;
	float angVelRad = 0.02f;

	Matrix getRotationMatrix(Vec3 rotationVec, float angVelRadWithDir);
	void Rotate(Axis axis, Direction direction);
public:
	Camera(Vec3 pos, Vec3 dir, Vec3 up, float FOVDeg);
	void moveForwards();
	void moveBackwards();
	void turnUp();
	void turnDown();
	void turnLeft();
	void turnRight();
	void tiltLeft();
	void tiltRight();

	Vec3 getPos() const;
	Vec3 getDir() const;
	Vec3 getUp() const;
	Vec3 getRight() const;
	Vec3 getRayIntersect() const;
	void setProjPlaneWidth(float projPlaneWidth);
	void setProjPlaneHeight(float projPlaneHeight);
};

#endif
