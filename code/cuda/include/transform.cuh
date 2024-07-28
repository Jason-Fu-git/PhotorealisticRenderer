/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */

#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "Vector3f.cuh"
#include "Vector4f.cuh"
#include "Matrix4f.cuh"

#include "object3d.cuh"

// transforms a 3D point using a matrix, returning a 3D point
__device__ static inline Vector3f transformPoint(const Matrix4f &mat, const Vector3f &point) {
    return (mat * Vector4f(point, 1)).xyz();
}

// transform a 3D direction using a matrix, returning a direction
__device__ static inline Vector3f transformDirection(const Matrix4f &mat, const Vector3f &dir) {
    return (mat * Vector4f(dir, 0)).xyz();
}

/**
 * @brief Transform class
 *
 * Transform is a class that represents a transformation in 3D space.
 * It can be used to translate, rotate, or scale an object.
 *
 * @note It will transform the ray instead. And because of the way we implement the class, please configurate
 * scale in the last.
 *
 * @var Scale : scale the coordinates of the obj file, a small number's impact may be trivial
 * @var Translate : translate the coordinates of the obj file, calculate (a+x, b+y, c+z)
 */
class Transform : public Object3D {
public:
    __device__  __host__ Transform() {}

    __device__ __host__ Transform(const Matrix4f &m, Object3D *obj) : o(obj) {
        transform = m.inverse();
    }

    __device__ __host__ ~Transform() override {
        delete o;
    }

    __device__  inline bool intersect(const Ray &r, Hit &h, float tmin) override {
        Vector3f trSource = transformPoint(transform, r.getOrigin());
        Vector3f trDirection = transformDirection(transform, r.getDirection());
        Ray tr(trSource, trDirection);
        bool inter = o->intersect(tr, h, tmin);
        if (inter) {
            h.set(h.getT(), h.getObject(), transformDirection(transform.transposed(), h.getNormal()).normalized(),
                  h.isInside());
        }
        return inter;
    }

    __device__ __host__ Object3D *getObject() const {
        return o;
    }

    __device__ __host__ Matrix4f getTransformMatrix() const {
        return transform;
    }

protected:
    Object3D *o; //un-transformed object
    Matrix4f transform;
};

#endif //TRANSFORM_H
