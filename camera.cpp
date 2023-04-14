#include "camera.h"
#include <cmath>

unsigned int gWidth = 1920;
unsigned int gHeight = 1000;

float deg2Rad(float deg) {
	return deg/180.0f*Pi;
}

Vec3 getNormalized(const Vec3& vec) {
	float norm = sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
	return Vec3 { vec.x/norm, vec.y/norm, vec.z/norm };
}

Vec3 crossProduct(const Vec3& vec1, const Vec3& vec2) {
	return Vec3 {
		vec1.y*vec2.z - vec1.z*vec2.y,
		vec1.z*vec2.x - vec1.x*vec2.z,
		vec1.x*vec2.y - vec1.y*vec2.x
	};
}

Matrix Camera::getRotationMatrix(Vec3 rotationVec, float angVelRadWithDir) {
	float auxVec[] = {
		0, -rotationVec.z, rotationVec.y,
		rotationVec.z, 0, -rotationVec.x,
		-rotationVec.y, rotationVec.x, 0
	};
	Matrix auxMatrix(3, 3, auxVec);

	float identityVec[] = {
		1, 0, 0,
		0, 1, 0,
		0, 0, 1
	};
	Matrix rotationMatrix(3, 3, identityVec);

	rotationMatrix += sin(angVelRadWithDir)*auxMatrix +
		(1 - cos(angVelRadWithDir))*auxMatrix*auxMatrix;

	return rotationMatrix;
}

void Camera::Rotate(Axis axis, Direction direction) {
	Vec3 rotationVec;
	Vec3* vec1;
	Vec3* vec2;
	
	switch(axis) {
	case Axis::Dir:
		rotationVec = dir;
		vec1 = &up;
		vec2 = &right;
		break;
	case Axis::Up:
		rotationVec = up;
		vec1 = &right;
		vec2 = &dir;
		break;
	default:
		rotationVec = right;
		vec1 = &dir;
		vec2 = &up;
		break;
	}

	Matrix rotationMatrix = getRotationMatrix(rotationVec,
		direction == Direction::Negative ? -angVelRad : angVelRad);

	float vec1Vec[] = { vec1->x, vec1->y, vec1->z };
	Matrix vec1Matrix(3, 1, vec1Vec);
	float vec2Vec[] = { vec2->x, vec2->y, vec2->z };
	Matrix vec2Matrix(3, 1, vec2Vec);

	vec1Matrix = rotationMatrix*vec1Matrix;
	vec2Matrix = rotationMatrix*vec2Matrix;

	vec1->x = vec1Matrix.val[0][0];
	vec1->y = vec1Matrix.val[1][0];
	vec1->z = vec1Matrix.val[2][0];
	vec2->x = vec2Matrix.val[0][0];
	vec2->y = vec2Matrix.val[1][0];
	vec2->z = vec2Matrix.val[2][0];

	*vec1 = getNormalized(*vec1);
	*vec2 = getNormalized(*vec2);
}

Camera::Camera(Vec3 pos, Vec3 dir, Vec3 up, float FOVDeg) : pos(pos), dir(getNormalized(dir)),
	up(getNormalized(up)), right(crossProduct(this->dir, this->up)), FOVRad(deg2Rad(FOVDeg)) { }

void Camera::moveForwards() {
	pos.x += dir.x*vel;
	pos.y += dir.y*vel;
	pos.z += dir.z*vel;
}

void Camera::moveBackwards() {
	pos.x -= dir.x*vel;
	pos.y -= dir.y*vel;
	pos.z -= dir.z*vel;
}

void Camera::turnUp() {
	Rotate(Axis::Right, Direction::Positive);
}

void Camera::turnDown() {
	Rotate(Axis::Right, Direction::Negative);
}

void Camera::turnLeft() {
	Rotate(Axis::Up, Direction::Positive);
}

void Camera::turnRight() {
	Rotate(Axis::Up, Direction::Negative);
}

void Camera::tiltLeft() {
	Rotate(Axis::Dir, Direction::Negative);
}

void Camera::tiltRight() {
	Rotate(Axis::Dir, Direction::Positive);
}

Vec3 Camera::getPos() const {
	return pos;
}

Vec3 Camera::getDir() const {
	return dir;
}

Vec3 Camera::getUp() const {
	return up;
}

Vec3 Camera::getRight() const {
	return right;
}

Vec3 Camera::getRayIntersect() const {
	float dist = projPlaneWidth/(2*tan(FOVRad/2));
	return Vec3{ pos.x - dir.x*dist, pos.y - dir.y*dist, pos.z - dir.z*dist };
}

void Camera::setProjPlaneWidth(float projPlaneWidth) {
	this->projPlaneWidth = projPlaneWidth;
}

void Camera::setProjPlaneHeight(float projPlaneHeight) {
	this->projPlaneHeight = projPlaneHeight;
}
