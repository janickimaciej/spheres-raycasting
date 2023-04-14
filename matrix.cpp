#include "matrix.h"
#include <cstdio>
#include <stdexcept>

Matrix::Matrix(int m, int n) : m(m), n(n) {
	val = new float*[m];
	for(int r = 0; r < m; r++) {
        val[r] = new float[n];
        for(int c = 0; c < n; c++) {
            val[r][c] = 0;
        }
    }
}

Matrix::Matrix(int m, int n, float vec[]) : Matrix(m, n) {
	for(int r = 0; r < m; r++) {
        for(int c = 0; c < n; c++) {
            val[r][c] = vec[r*n + c];
        }
    }
}

Matrix::Matrix(const Matrix& M) : Matrix(M.m, M.n) {
    for(int r = 0; r < m; r++) {
        for(int c = 0; c < n; c++) {
            val[r][c] = M.val[r][c];
        }
    }
}

Matrix::~Matrix() {
    for(int r = 0; r < m; r++) {
        delete [] val[r];
    }
    delete [] val;
}

void Matrix::Print() {
    printf("\n");
    for(int r = 0; r < m; r++) {
        printf("|");
        for(int c = 0; c < n; c++) {
            int i;
            if(isInteger(val[r][c], &i)) printf("%6d", i);
            else printf("%6.2f", val[r][c]);
        }
        printf("  |\n");
    }
    printf("\n");
}

void Matrix::Gauss() {
    int r1 = 0;
    int c1 = 0;
    while(r1 < m && c1 < n) {
        if(isZero(val[r1][c1])) {
            bool f = false;
            for(int r2 = r1 + 1; r2 < m; r2++) {
                if(!isZero(val[r2][c1])) {
                    swapRows(val[r1], val[r2], n);
                    f = true;
                    break;
                }
            }
            if(!f) {
                c1++;
                continue;
            }
        }

        for(int c2 = n - 1; c2 >= c1; c2--) {
            val[r1][c2] /= val[r1][c1];
        }

        for(int r2 = 0; r2 < m; r2++) {
            if(r2 == r1) continue;
            for(int c2 = n - 1; c2 >= r1; c2--) {
                val[r2][c2] -= val[r1][c2]*val[r2][c1];
            }
        }
        r1++;
        c1++;
    }
}

void Matrix::Inverse() {
    if(m != n) throw std::invalid_argument("Matrix must be square");
    Matrix Aug = Matrix(m, 2*m);
    for(int r = 0; r < m; r++) {
        for(int c = 0; c < m; c++) {
            Aug.val[r][c] = val[r][c];
        }
        for(int c = m; c < 2*m; c++) {
            Aug.val[r][c] = c - m == r ? 1 : 0;
        }
    }
    Aug.Gauss();
    for(int r = 0; r < m; r++) {
        for(int c = 0; c < m; c++) {
            val[r][c] = Aug.val[r][c + m];
        }
    }
}

void Matrix::swapRows(float* r1, float* r2, int n) {
    for(int c = 0; c < n; c++) {
        float temp = r1[c];
        r1[c] = r2[c];
        r2[c] = temp;
    }
}

bool Matrix::isZero(float x) {
    return x < eps && x > -eps;
}

bool Matrix::isInteger(float x, int* i) {
    bool isNegative = false;
    if(x < 0) {
        isNegative = true;
        x *= -1;
    } if(x - (int)x < eps) {
        if(i != nullptr) {
            *i = (int)x;
            if(isNegative) *i *= -1;
        }
        return true;
    } else if((int)x + 1 - x < eps) {
        if(i != nullptr) {
            *i = (int)x + 1;
            if(isNegative) *i *= -1;
        }
        return true;
    }
    else return false;
}

Matrix& Matrix::operator=(const Matrix& M) {
    if(M.m != this->m || M.n != this->n) throw std::invalid_argument("Matrix dimensions must match");
    for(int r = 0; r < m; r++) {
        for(int c = 0; c < n; c++) {
            val[r][c] = M.val[r][c];
        }
    }
    return *this;
}

Matrix Matrix::operator-() const {
    Matrix R(*this);
    for(int r = 0; r < R.m; r++) {
        for(int c = 0; c < R.n; c++) {
            R.val[r][c] *= -1;
        }
    }
    return R;
}

Matrix& Matrix::operator+=(const Matrix& M) {
    *this = *this + M;
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& M) {
    *this += -M;
    return *this;
}

Matrix operator+(const Matrix& M1, const Matrix& M2) {
    if(M1.m != M2.m || M1.n != M2.n) throw std::invalid_argument("Matrix dimensions must match");
    Matrix R(M1);
    for(int r = 0; r < R.m; r++) {
        for(int c = 0; c < R.n; c++) {
            R.val[r][c] += M2.val[r][c];
        }
    }
    return R;
}

Matrix operator-(const Matrix& M1, const Matrix& M2) {
    if(M1.m != M2.m || M1.n != M2.n) throw std::invalid_argument("Matrix dimensions must match");
    return M1 + (-M2);
}

Matrix& Matrix::operator*=(float s) {
    *this = *this*s;
    return *this;
}

Matrix& Matrix::operator/=(float s) {
    *this *= 1/s;
    return *this;
}

Matrix operator*(float s, const Matrix& M) {
    Matrix R(M);
    for(int r = 0; r < R.m; r++) {
        for(int c = 0; c < R.n; c++) {
            R.val[r][c] *= s;
        }
    }
    return R;
}

Matrix operator*(const Matrix& M, float s) {
    return s*M;
}

Matrix operator/(const Matrix& M, float s) {
    return (1/s)*M;
}

Matrix operator*(const Matrix& M1, const Matrix& M2) {
    if(M1.n != M2.m) throw std::invalid_argument(
        "Number of columns of the first matrix must be equal to number of rows of the second matrix"
    );
    Matrix R(M1.m, M2.n);
    for(int r = 0; r < R.m; r++) {
        for(int c = 0; c < R.n; c++) {
            R.val[r][c] = 0;
            for(int cr = 0; cr < M1.n; cr++) {
                R.val[r][c] += M1.val[r][cr]*M2.val[cr][c];
            }
        }
    }
    return R;
}
