//
// Created by Jason Fu on 24-7-11.
//

#ifndef FINALPROJECTCUDA_VECTOR3F_CUH
#define FINALPROJECTCUDA_VECTOR3F_CUH


class Vector3f {
public:

    __device__ __host__ inline Vector3f() {
        _x = _y = _z = 0.0f;
    }

    __device__ __host__ inline Vector3f(float x, float y, float z) {
        _x = x;
        _y = y;
        _z = z;
    }

    __device__ __host__ inline Vector3f(const float v[3]) {
        _x = v[0];
        _y = v[1];
        _z = v[2];
    }

    // copy constructors
    __device__ __host__ inline Vector3f(const Vector3f &rv) {
        _x = rv._x;
        _y = rv._y;
        _z = rv._z;
    }

    // assignment operators
    __device__ __host__ inline Vector3f &operator=(const Vector3f &rv) {
        if (this != &rv) {
            _x = rv._x;
            _y = rv._y;
            _z = rv._z;
        }
        return *this;
    }

    // no destructor necessary

    // returns the ith element
    __device__ __host__ inline const float &operator[](int i) const {
        if (i == 0) return _x;
        else if (i == 1) return _y;
        else return _z;
    }

    __device__ __host__ inline float &operator[](int i) {
        if (i == 0) return _x;
        else if (i == 1) return _y;
        else return _z;
    }

    __device__ __host__ inline float length() const {
        return sqrtf(_x * _x + _y * _y + _z * _z);
    }

    __device__ __host__ inline float squaredLength() const {
        return
                (
                        _x * _x +
                        _y * _y +
                        _z * _z
                );
    }

    __device__ __host__ inline void normalize() {
        float norm = length();
        _x /= norm;
        _y /= norm;
        _z /= norm;
    }

    __device__ __host__ inline Vector3f normalized() const {
        float norm = length();
        return {
                _x / norm,
                _y / norm,
                _z / norm
        };
    }

    __device__ __host__ inline void negate() {
        _x = -_x;
        _y = -_y;
        _z = -_z;
    }

    __device__ __host__ inline Vector3f &operator+=(const Vector3f &v) {
        _x += v._x;
        _y += v._y;
        _z += v._z;
        return *this;
    }

    __device__ __host__ inline Vector3f &operator-=(const Vector3f &v) {
        _x -= v._x;
        _y -= v._y;
        _z -= v._z;
        return *this;
    }

    __device__ __host__ inline Vector3f &operator*=(float f) {
        _x *= f;
        _y *= f;
        _z *= f;
        return *this;
    }

    __device__ __host__ inline float dot(const Vector3f &v) const {
        return dot(*this, v);
    }

    __device__ __host__ inline Vector3f cross(const Vector3f &v) const {
        return cross(*this, v);
    }

    __device__ __host__ inline static float dot(const Vector3f &v0, const Vector3f &v1) {
        return v0._x * v1._x + v0._y * v1._y + v0._z * v1._z;
    }

    __device__ __host__ inline static Vector3f cross(const Vector3f &v0, const Vector3f &v1) {
        return {
                v0._y * v1._z - v0._z * v1._y,
                v0._z * v1._x - v0._x * v1._z,
                v0._x * v1._y - v0._y * v1._x
        };
    }

    __device__ __host__ inline static Vector3f ZERO() {
        return {0, 0, 0};
    }

    __device__ __host__ inline static Vector3f ONE() {
        return {1, 1, 1};
    }

    __device__ __host__ inline static Vector3f UP() {
        return {0, 1, 0};
    }

    __device__ __host__ inline static Vector3f RIGHT() {
        return {1, 0, 0};
    }

    __device__ __host__ inline static Vector3f FORWARD() {
        return {0, 0, -1};
    }

    float _x;
    float _y;
    float _z;
};

// component-wise operators
__device__ __host__ inline Vector3f operator+(const Vector3f &v0, const Vector3f &v1) {
    return {v0._x + v1._x, v0._y + v1._y, v0._z + v1._z};
}

__device__ __host__ inline Vector3f operator-(const Vector3f &v0, const Vector3f &v1) {
    return {v0._x - v1._x, v0._y - v1._y, v0._z - v1._z};
}

__device__ __host__ inline Vector3f operator*(const Vector3f &v0, const Vector3f &v1) {
    return {v0._x * v1._x, v0._y * v1._y, v0._z * v1._z};
}

__device__ __host__ inline Vector3f operator/(const Vector3f &v0, const Vector3f &v1) {
    return {v0._x / v1._x, v0._y / v1._y, v0._z / v1._z};
}

// unary negation
__device__ __host__ inline Vector3f operator-(const Vector3f &v) {
    return {-v._x, -v._y, -v._z};
}

// multiply and divide by scalar
__device__ __host__ inline Vector3f operator*(float f, const Vector3f &v) {
    return {v._x * f, v._y * f, v._z * f};
}

__device__ __host__ inline Vector3f operator*(const Vector3f &v, float f) {
    return {v._x * f, v._y * f, v._z * f};
}

__device__ __host__ inline Vector3f operator/(const Vector3f &v, float f) {
    return {v._x / f, v._y / f, v._z / f};
}

__device__ __host__ inline bool operator==(const Vector3f &v0, const Vector3f &v1) {
    return (v0._x == v1._x && v0._y == v1._y && v0._z == v1._z);
}

__device__ __host__ inline bool operator!=(const Vector3f &v0, const Vector3f &v1) {
    return !(v0 == v1);
}

#endif //FINALPROJECTCUDA_VECTOR3F_CUH
