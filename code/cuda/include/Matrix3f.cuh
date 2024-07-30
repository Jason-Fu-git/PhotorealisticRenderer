//
// Created by Jason Fu on 24-7-12.
//

#ifndef FINALPROJECTCUDA_MATRIX3F_CUH
#define FINALPROJECTCUDA_MATRIX3F_CUH

#include <cstdio>
#include "Vector3f.cuh"

// 3x3 Matrix, stored in column major order (OpenGL style)
class Matrix3f {
public:

    // Fill a 3x3 matrix with "fill", default to 0.
    __device__ __host__ inline Matrix3f(float fill = 0.f) {
        for (int i = 0; i < 9; ++i) {
            m_elements[i] = fill;
        }
    }

    __device__ __host__ inline Matrix3f(float m00, float m01, float m02,
                                        float m10, float m11, float m12,
                                        float m20, float m21, float m22) {
        m_elements[0] = m00;
        m_elements[1] = m10;
        m_elements[2] = m20;

        m_elements[3] = m01;
        m_elements[4] = m11;
        m_elements[5] = m21;

        m_elements[6] = m02;
        m_elements[7] = m12;
        m_elements[8] = m22;
    }

    // setColumns = true ==> sets the columns of the matrix to be [v0 v1 v2]
    // otherwise, sets the rows
    __device__ __host__ inline Matrix3f(const Vector3f &v0, const Vector3f &v1, const Vector3f &v2,
                                        bool setColumns = true) {
        if (setColumns) {
            setCol(0, v0);
            setCol(1, v1);
            setCol(2, v2);
        } else {
            setRow(0, v0);
            setRow(1, v1);
            setRow(2, v2);
        }
    }

    // copy constructor
    __device__ __host__ inline Matrix3f(const Matrix3f &rm) {
        memcpy(m_elements, rm.m_elements, 9 * sizeof(float));
    }

    // assignment operator
    __device__ __host__ inline Matrix3f &operator=(const Matrix3f &rm) {
        if (this != &rm) {
            memcpy(m_elements, rm.m_elements, 9 * sizeof(float));
        }
        return *this;
    }

    // no destructor necessary

    __device__ __host__ inline const float &operator()(int i, int j) const {
        return m_elements[j * 3 + i];
    }

    __device__ __host__ inline float &operator()(int i, int j) {
        return m_elements[j * 3 + i];
    }

    __device__ __host__ inline Vector3f getRow(int i) const {
        return {
                m_elements[i],
                m_elements[i + 3],
                m_elements[i + 6]
        };
    }

    __device__ __host__ inline void setRow(int i, const Vector3f &v) {
        m_elements[i] = v._x;
        m_elements[i + 3] = v._y;
        m_elements[i + 6] = v._z;
    }

    __device__ __host__ inline Vector3f getCol(int j) {
        int colStart = 3 * j;

        return {
                m_elements[colStart],
                m_elements[colStart + 1],
                m_elements[colStart + 2]
        };
    }

    __device__ __host__ inline void setCol(int j, const Vector3f &v) {
        int colStart = 3 * j;

        m_elements[colStart] = v._x;
        m_elements[colStart + 1] = v._y;
        m_elements[colStart + 2] = v._z;
    }

    // ---- Utility ----

    __device__ __host__ inline static Matrix3f ones() {
        Matrix3f m;
        for (int i = 0; i < 9; ++i) {
            m.m_elements[i] = 1;
        }

        return m;
    }

    __device__ __host__ inline static Matrix3f identity() {
        Matrix3f m;

        m(0, 0) = 1;
        m(1, 1) = 1;
        m(2, 2) = 1;

        return m;
    }


    __device__ __host__ inline static Matrix3f rotateX(float radians) {
        float c = cosf(radians);
        float s = sinf(radians);

        return {
                1, 0, 0,
                0, c, -s,
                0, s, c
        };
    }

    __device__ __host__ inline static Matrix3f rotateY(float radians) {
        float c = cosf(radians);
        float s = sinf(radians);

        return {
                c, 0, s,
                0, 1, 0,
                -s, 0, c
        };
    }

    __device__ __host__ inline static Matrix3f rotateZ(float radians) {
        float c = cosf(radians);
        float s = sinf(radians);

        return {
                c, -s, 0,
                s, c, 0,
                0, 0, 1
        };
    }

    __device__ __host__ inline static Matrix3f scaling(float sx, float sy, float sz) {
        return {
                sx, 0, 0,
                0, sy, 0,
                0, 0, sz
        };
    }

    __device__ __host__ inline static Matrix3f uniformScaling(float s) {
        return {
                s, 0, 0,
                0, s, 0,
                0, 0, s
        };
    }

    __device__ __host__ inline static Matrix3f rotation(const Vector3f &rDirection, float radians) {
        Vector3f normalizedDirection = rDirection.normalized();

        float cosTheta = cosf(radians);
        float sinTheta = sinf(radians);

        float x = normalizedDirection._x;
        float y = normalizedDirection._y;
        float z = normalizedDirection._z;

        return {
                x * x * (1.0f - cosTheta) + cosTheta, y * x * (1.0f - cosTheta) - z * sinTheta,
                z * x * (1.0f - cosTheta) + y * sinTheta,
                x * y * (1.0f - cosTheta) + z * sinTheta, y * y * (1.0f - cosTheta) + cosTheta,
                z * y * (1.0f - cosTheta) - x * sinTheta,
                x * z * (1.0f - cosTheta) - y * sinTheta, y * z * (1.0f - cosTheta) + x * sinTheta,
                z * z * (1.0f - cosTheta) + cosTheta
        };
    }

    __device__ __host__ inline float determinant() const {
        return determinant3x3(m_elements[0], m_elements[3], m_elements[6],
                              m_elements[1], m_elements[4], m_elements[7],
                              m_elements[2], m_elements[5], m_elements[8]);

    }

    __device__ __host__ static float inline determinant3x3(float m00, float m01, float m02,
                                                           float m10, float m11, float m12,
                                                           float m20, float m21, float m22) {
        return
                (
                        m00 * (m11 * m22 - m12 * m21)
                        - m01 * (m10 * m22 - m12 * m20)
                        + m02 * (m10 * m21 - m11 * m20)
                );
    }

private:

    float m_elements[9];

};

// Matrix-Vector multiplication
// 3x3 * 3x1 ==> 3x1
__device__ __host__ inline Vector3f operator*(const Matrix3f &m, const Vector3f &v) {
    Vector3f output(0, 0, 0);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            output[i] += m(i, j) * v[j];
        }
    }

    return output;
}

// Matrix-Matrix multiplication
__device__ __host__ inline Matrix3f operator*(const Matrix3f &x, const Matrix3f &y) {
    Matrix3f product; // zeroes

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                product(i, k) += x(i, j) * y(j, k);
            }
        }
    }

    return product;
}

#endif //FINALPROJECTCUDA_MATRIX3F_CUH
