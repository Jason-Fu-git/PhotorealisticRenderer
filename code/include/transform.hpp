/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */

#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <vecmath.h>
#include "object3d.hpp"

// transforms a 3D point using a matrix, returning a 3D point
static Vector3f transformPoint(const Matrix4f &mat, const Vector3f &point) {
    return (mat * Vector4f(point, 1)).xyz();
}

// transform a 3D direction using a matrix, returning a direction
static Vector3f transformDirection(const Matrix4f &mat, const Vector3f &dir) {
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
    Transform() {}

    Transform(const Matrix4f &m, Object3D *obj) : o(obj) {
        transform = m.inverse();
    }

    ~Transform() {
    }

    virtual bool intersect(const Ray &r, Hit &h, float tmin) {
        Vector3f trSource = transformPoint(transform, r.getOrigin());
        Vector3f trDirection = transformDirection(transform, r.getDirection());
        Ray tr(trSource, trDirection);
        bool inter = o->intersect(tr, h, tmin);
        if (inter) {
//            printf("Ray r o: %f %f %f d : %f %f %f "
//                   "Ray tr o: %f %f %f d : %f %f %f \n", r.getOrigin().x(), r.getOrigin().y(), r.getOrigin().z(),
//                   r.getDirection().x(), r.getDirection().y(), r.getDirection().z(),
//                   tr.getOrigin().x(), tr.getOrigin().y(), tr.getOrigin().z(),
//                   tr.getDirection().x(), tr.getDirection().y(), tr.getDirection().z());
            h.set(h.getT(), h.getMaterial(), transformDirection(transform.transposed(), h.getNormal()).normalized(),
                  h.isInside());
        }
        return inter;
    }

protected:
    Object3D *o; //un-transformed object
    Matrix4f transform;
};

#endif //TRANSFORM_H
