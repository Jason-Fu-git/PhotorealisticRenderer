//
// Created by Jason Fu on 24-7-14.
//

#include "Matrix4f.cuh"

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */

#include <cmath>
#include <cstring>

#include "Matrix3f.cuh"
#include "Vector3f.cuh"
#include "Vector4f.cuh"

__device__ __host__ Matrix4f::Matrix4f(float fill) {
    for (int i = 0; i < 16; ++i) {
        m_elements[i] = fill;
    }
}

__device__ __host__ Matrix4f::Matrix4f(float m00, float m01, float m02, float m03,
                                       float m10, float m11, float m12, float m13,
                                       float m20, float m21, float m22, float m23,
                                       float m30, float m31, float m32, float m33) {
    m_elements[0] = m00;
    m_elements[1] = m10;
    m_elements[2] = m20;
    m_elements[3] = m30;

    m_elements[4] = m01;
    m_elements[5] = m11;
    m_elements[6] = m21;
    m_elements[7] = m31;

    m_elements[8] = m02;
    m_elements[9] = m12;
    m_elements[10] = m22;
    m_elements[11] = m32;

    m_elements[12] = m03;
    m_elements[13] = m13;
    m_elements[14] = m23;
    m_elements[15] = m33;
}

__device__ __host__ Matrix4f &Matrix4f::operator/=(float d) {
    for (int ii = 0; ii < 16; ii++) {
        m_elements[ii] /= d;
    }
    return *this;
}

__device__ __host__ Matrix4f::Matrix4f(const Matrix4f &rm) {
    memcpy(m_elements, rm.m_elements, 16 * sizeof(float));
}

__device__ __host__ Matrix4f &Matrix4f::operator=(const Matrix4f &rm) {
    if (this != &rm) {
        memcpy(m_elements, rm.m_elements, 16 * sizeof(float));
    }
    return *this;
}

__device__ __host__ const float &Matrix4f::operator()(int i, int j) const {
    return m_elements[j * 4 + i];
}

__device__ __host__ float &Matrix4f::operator()(int i, int j) {
    return m_elements[j * 4 + i];
}

__device__ __host__ Matrix3f Matrix4f::getSubmatrix3x3(int i0, int j0) const {
    Matrix3f out;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out(i, j) = (*this)(i + i0, j + j0);
        }
    }

    return out;
}

__device__ __host__ void Matrix4f::setSubmatrix3x3(int i0, int j0, const Matrix3f &m) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            (*this)(i + i0, j + j0) = m(i, j);
        }
    }
}

__device__ __host__ float Matrix4f::determinant() const {
    float m00 = m_elements[0];
    float m10 = m_elements[1];
    float m20 = m_elements[2];
    float m30 = m_elements[3];

    float m01 = m_elements[4];
    float m11 = m_elements[5];
    float m21 = m_elements[6];
    float m31 = m_elements[7];

    float m02 = m_elements[8];
    float m12 = m_elements[9];
    float m22 = m_elements[10];
    float m32 = m_elements[11];

    float m03 = m_elements[12];
    float m13 = m_elements[13];
    float m23 = m_elements[14];
    float m33 = m_elements[15];

    float cofactor00 = Matrix3f::determinant3x3(m11, m12, m13, m21, m22, m23, m31, m32, m33);
    float cofactor01 = -Matrix3f::determinant3x3(m12, m13, m10, m22, m23, m20, m32, m33, m30);
    float cofactor02 = Matrix3f::determinant3x3(m13, m10, m11, m23, m20, m21, m33, m30, m31);
    float cofactor03 = -Matrix3f::determinant3x3(m10, m11, m12, m20, m21, m22, m30, m31, m32);

    return (m00 * cofactor00 + m01 * cofactor01 + m02 * cofactor02 + m03 * cofactor03);
}

__device__ __host__ Matrix4f Matrix4f::inverse(bool *pbIsSingular, float epsilon) const {
    float m00 = m_elements[0];
    float m10 = m_elements[1];
    float m20 = m_elements[2];
    float m30 = m_elements[3];

    float m01 = m_elements[4];
    float m11 = m_elements[5];
    float m21 = m_elements[6];
    float m31 = m_elements[7];

    float m02 = m_elements[8];
    float m12 = m_elements[9];
    float m22 = m_elements[10];
    float m32 = m_elements[11];

    float m03 = m_elements[12];
    float m13 = m_elements[13];
    float m23 = m_elements[14];
    float m33 = m_elements[15];

    float cofactor00 = Matrix3f::determinant3x3(m11, m12, m13, m21, m22, m23, m31, m32, m33);
    float cofactor01 = -Matrix3f::determinant3x3(m12, m13, m10, m22, m23, m20, m32, m33, m30);
    float cofactor02 = Matrix3f::determinant3x3(m13, m10, m11, m23, m20, m21, m33, m30, m31);
    float cofactor03 = -Matrix3f::determinant3x3(m10, m11, m12, m20, m21, m22, m30, m31, m32);

    float cofactor10 = -Matrix3f::determinant3x3(m21, m22, m23, m31, m32, m33, m01, m02, m03);
    float cofactor11 = Matrix3f::determinant3x3(m22, m23, m20, m32, m33, m30, m02, m03, m00);
    float cofactor12 = -Matrix3f::determinant3x3(m23, m20, m21, m33, m30, m31, m03, m00, m01);
    float cofactor13 = Matrix3f::determinant3x3(m20, m21, m22, m30, m31, m32, m00, m01, m02);

    float cofactor20 = Matrix3f::determinant3x3(m31, m32, m33, m01, m02, m03, m11, m12, m13);
    float cofactor21 = -Matrix3f::determinant3x3(m32, m33, m30, m02, m03, m00, m12, m13, m10);
    float cofactor22 = Matrix3f::determinant3x3(m33, m30, m31, m03, m00, m01, m13, m10, m11);
    float cofactor23 = -Matrix3f::determinant3x3(m30, m31, m32, m00, m01, m02, m10, m11, m12);

    float cofactor30 = -Matrix3f::determinant3x3(m01, m02, m03, m11, m12, m13, m21, m22, m23);
    float cofactor31 = Matrix3f::determinant3x3(m02, m03, m00, m12, m13, m10, m22, m23, m20);
    float cofactor32 = -Matrix3f::determinant3x3(m03, m00, m01, m13, m10, m11, m23, m20, m21);
    float cofactor33 = Matrix3f::determinant3x3(m00, m01, m02, m10, m11, m12, m20, m21, m22);

    float determinant = m00 * cofactor00 + m01 * cofactor01 + m02 * cofactor02 + m03 * cofactor03;

    bool isSingular = (fabsf(determinant) < epsilon);
    if (isSingular) {
        if (pbIsSingular != nullptr) {
            *pbIsSingular = true;
        }
        return {};
    } else {
        if (pbIsSingular != nullptr) {
            *pbIsSingular = false;
        }

        float reciprocalDeterminant = 1.0f / determinant;

        return {
                        cofactor00 * reciprocalDeterminant, cofactor10 * reciprocalDeterminant,
                        cofactor20 * reciprocalDeterminant, cofactor30 * reciprocalDeterminant,
                        cofactor01 * reciprocalDeterminant, cofactor11 * reciprocalDeterminant,
                        cofactor21 * reciprocalDeterminant, cofactor31 * reciprocalDeterminant,
                        cofactor02 * reciprocalDeterminant, cofactor12 * reciprocalDeterminant,
                        cofactor22 * reciprocalDeterminant, cofactor32 * reciprocalDeterminant,
                        cofactor03 * reciprocalDeterminant, cofactor13 * reciprocalDeterminant,
                        cofactor23 * reciprocalDeterminant, cofactor33 * reciprocalDeterminant
                };
    }
}

__device__ __host__ void Matrix4f::transpose() {
    float temp;

    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            temp = (*this)(i, j);
            (*this)(i, j) = (*this)(j, i);
            (*this)(j, i) = temp;
        }
    }
}

__device__ __host__ Matrix4f Matrix4f::transposed() const {
    Matrix4f out;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            out(j, i) = (*this)(i, j);
        }
    }

    return out;
}

// static
__device__ __host__ Matrix4f Matrix4f::ones() {
    Matrix4f m;
    for (int i = 0; i < 16; ++i) {
        m.m_elements[i] = 1;
    }

    return m;
}

// static
__device__ __host__ Matrix4f Matrix4f::identity() {
    Matrix4f m;

    m(0, 0) = 1;
    m(1, 1) = 1;
    m(2, 2) = 1;
    m(3, 3) = 1;

    return m;
}

// static
__device__ __host__ Matrix4f Matrix4f::translation(float x, float y, float z) {
    return {
                    1, 0, 0, x,
                    0, 1, 0, y,
                    0, 0, 1, z,
                    0, 0, 0, 1
            };
}

// static
__device__ __host__ Matrix4f Matrix4f::translation(const Vector3f &rTranslation) {
    return {
                    1, 0, 0, rTranslation._x,
                    0, 1, 0, rTranslation._y,
                    0, 0, 1, rTranslation._z,
                    0, 0, 0, 1
            };
}

// static
__device__ __host__ Matrix4f Matrix4f::rotateX(float radians) {
    float c = cosf(radians);
    float s = sinf(radians);

    return {
                    1, 0, 0, 0,
                    0, c, -s, 0,
                    0, s, c, 0,
                    0, 0, 0, 1
            };
}

// static
__device__ __host__ Matrix4f Matrix4f::rotateY(float radians) {
    float c = cosf(radians);
    float s = sinf(radians);

    return {
                    c, 0, s, 0,
                    0, 1, 0, 0,
                    -s, 0, c, 0,
                    0, 0, 0, 1
            };
}

// static
__device__ __host__ Matrix4f Matrix4f::rotateZ(float radians) {
    float c = cosf(radians);
    float s = sinf(radians);

    return {
                    c, -s, 0, 0,
                    s, c, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1
            };
}

// static
__device__ __host__ Matrix4f Matrix4f::rotation(const Vector3f &rDirection, float radians) {
    Vector3f normalizedDirection = rDirection.normalized();

    float cosTheta = cosf(radians);
    float sinTheta = sinf(radians);

    float x = normalizedDirection._x;
    float y = normalizedDirection._y;
    float z = normalizedDirection._z;

    return {
                    x * x * (1.0f - cosTheta) + cosTheta, y * x * (1.0f - cosTheta) - z * sinTheta,
                    z * x * (1.0f - cosTheta) + y * sinTheta, 0.0f,
                    x * y * (1.0f - cosTheta) + z * sinTheta, y * y * (1.0f - cosTheta) + cosTheta,
                    z * y * (1.0f - cosTheta) - x * sinTheta, 0.0f,
                    x * z * (1.0f - cosTheta) - y * sinTheta, y * z * (1.0f - cosTheta) + x * sinTheta,
                    z * z * (1.0f - cosTheta) + cosTheta, 0.0f,
                    0.0f, 0.0f, 0.0f, 1.0f
            };
}

// static
__device__ __host__ Matrix4f Matrix4f::rotation(const Vector4f &q) {
    Vector4f qq = q.normalized();

    float xx = qq.x() * qq.x();
    float yy = qq.y() * qq.y();
    float zz = qq.z() * qq.z();

    float xy = qq.x() * qq.y();
    float zw = qq.z() * qq.w();

    float xz = qq.x() * qq.z();
    float yw = qq.y() * qq.w();

    float yz = qq.y() * qq.z();
    float xw = qq.x() * qq.w();

    return {
                    1.0f - 2.0f * (yy + zz), 2.0f * (xy - zw), 2.0f * (xz + yw), 0.0f,
                    2.0f * (xy + zw), 1.0f - 2.0f * (xx + zz), 2.0f * (yz - xw), 0.0f,
                    2.0f * (xz - yw), 2.0f * (yz + xw), 1.0f - 2.0f * (xx + yy), 0.0f,
                    0.0f, 0.0f, 0.0f, 1.0f
            };
}

// static
__device__ __host__ Matrix4f Matrix4f::scaling(float sx, float sy, float sz) {
    return {
                    sx, 0, 0, 0,
                    0, sy, 0, 0,
                    0, 0, sz, 0,
                    0, 0, 0, 1
            };
}

// static
__device__ __host__ Matrix4f Matrix4f::uniformScaling(float s) {
    return {
                    s, 0, 0, 0,
                    0, s, 0, 0,
                    0, 0, s, 0,
                    0, 0, 0, 1
            };
}

// static
__device__ __host__ Matrix4f
Matrix4f::orthographicProjection(float width, float height, float zNear, float zFar, bool directX) {
    Matrix4f m;

    m(0, 0) = 2.0f / width;
    m(1, 1) = 2.0f / height;
    m(3, 3) = 1.0f;

    m(0, 3) = -1;
    m(1, 3) = -1;

    if (directX) {
        m(2, 2) = 1.0f / (zNear - zFar);
        m(2, 3) = zNear / (zNear - zFar);
    } else {
        m(2, 2) = 2.0f / (zNear - zFar);
        m(2, 3) = (zNear + zFar) / (zNear - zFar);
    }

    return m;
}

// static
__device__ __host__ Matrix4f
Matrix4f::orthographicProjection(float left, float right, float bottom, float top, float zNear, float zFar,
                                 bool directX) {
    Matrix4f m;

    m(0, 0) = 2.0f / (right - left);
    m(1, 1) = 2.0f / (top - bottom);
    m(3, 3) = 1.0f;

    m(0, 3) = (left + right) / (left - right);
    m(1, 3) = (top + bottom) / (bottom - top);

    if (directX) {
        m(2, 2) = 1.0f / (zNear - zFar);
        m(2, 3) = zNear / (zNear - zFar);
    } else {
        m(2, 2) = 2.0f / (zNear - zFar);
        m(2, 3) = (zNear + zFar) / (zNear - zFar);
    }

    return m;
}

// static
__device__ __host__ Matrix4f Matrix4f::perspectiveProjection(float fLeft, float fRight,
                                                             float fBottom, float fTop,
                                                             float fZNear, float fZFar,
                                                             bool directX) {
    Matrix4f projection; // zero matrix

    projection(0, 0) = (2.0f * fZNear) / (fRight - fLeft);
    projection(1, 1) = (2.0f * fZNear) / (fTop - fBottom);
    projection(0, 2) = (fRight + fLeft) / (fRight - fLeft);
    projection(1, 2) = (fTop + fBottom) / (fTop - fBottom);
    projection(3, 2) = -1;

    if (directX) {
        projection(2, 2) = fZFar / (fZNear - fZFar);
        projection(2, 3) = (fZNear * fZFar) / (fZNear - fZFar);
    } else {
        projection(2, 2) = (fZNear + fZFar) / (fZNear - fZFar);
        projection(2, 3) = (2.0f * fZNear * fZFar) / (fZNear - fZFar);
    }

    return projection;
}

// static
__device__ __host__ Matrix4f
Matrix4f::perspectiveProjection(float fovYRadians, float aspect, float zNear, float zFar, bool directX) {
    Matrix4f m; // zero matrix

    float yScale = 1.f / tanf(0.5f * fovYRadians);
    float xScale = yScale / aspect;

    m(0, 0) = xScale;
    m(1, 1) = yScale;
    m(3, 2) = -1;

    if (directX) {
        m(2, 2) = zFar / (zNear - zFar);
        m(2, 3) = zNear * zFar / (zNear - zFar);
    } else {
        m(2, 2) = (zFar + zNear) / (zNear - zFar);
        m(2, 3) = 2.f * zFar * zNear / (zNear - zFar);
    }

    return m;
}

// static
__device__ __host__ Matrix4f Matrix4f::infinitePerspectiveProjection(float fLeft, float fRight,
                                                                     float fBottom, float fTop,
                                                                     float fZNear, bool directX) {
    Matrix4f projection;

    projection(0, 0) = (2.0f * fZNear) / (fRight - fLeft);
    projection(1, 1) = (2.0f * fZNear) / (fTop - fBottom);
    projection(0, 2) = (fRight + fLeft) / (fRight - fLeft);
    projection(1, 2) = (fTop + fBottom) / (fTop - fBottom);
    projection(3, 2) = -1;

    // infinite view frustum
    // just take the limit as far --> inf of the regular frustum
    if (directX) {
        projection(2, 2) = -1.0f;
        projection(2, 3) = -fZNear;
    } else {
        projection(2, 2) = -1.0f;
        projection(2, 3) = -2.0f * fZNear;
    }

    return projection;
}

//////////////////////////////////////////////////////////////////////////
// Operators
//////////////////////////////////////////////////////////////////////////

__device__ __host__ Vector4f operator*(const Matrix4f &m, const Vector4f &v) {
    Vector4f output(0, 0, 0, 0);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            output[i] += m(i, j) * v[j];
        }
    }

    return output;
}

__device__ __host__ Matrix4f operator*(const Matrix4f &x, const Matrix4f &y) {
    Matrix4f product; // zeroes

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                product(i, k) += x(i, j) * y(j, k);
            }
        }
    }

    return product;
}
