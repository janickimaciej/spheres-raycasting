#ifndef MATRIX
#define MATRIX

class Matrix {
	const float eps = 1e-5f;

	void swapRows(float* r1, float* r2, int n);
	bool isZero(float x);
	bool isInteger(float x, int* i = nullptr);
public:
	int m;
	int n;
	float** val;
	Matrix(int m, int n);
	Matrix(int m, int n, float vec[]);
	Matrix(const Matrix& M);
	~Matrix();
	void Print();
	void Gauss();
	void Inverse();
	Matrix& operator=(const Matrix& M);
	Matrix operator-() const;
	Matrix& operator+=(const Matrix& M);
	Matrix& operator-=(const Matrix& M);
	friend Matrix operator+(const Matrix& M1, const Matrix& M2);
	friend Matrix operator-(const Matrix& M1, const Matrix& M2);
	Matrix& operator*=(float s);
	Matrix& operator/=(float s);
	friend Matrix operator*(const float s, const Matrix& M);
	friend Matrix operator*(const Matrix& M, const float s);
	friend Matrix operator/(const Matrix& M, float s);
	friend Matrix operator*(const Matrix& M1, const Matrix& M2);
};

#endif
