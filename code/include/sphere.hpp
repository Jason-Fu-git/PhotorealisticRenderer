#ifndef SPHERE_H
#define SPHERE_H

#include "object3d.hpp"
#include <vecmath.h>
#include <cmath>

// DONE: Implement functions and add more fields as necessary

/**
 * @copybrief 项目所有者独立实现
 * @author Jason Fu
 * @brief Sphere class
 * @var _center
 * @var _radius
 */
class Sphere : public Object3D {
public:
    Sphere() {
        // unit ball at the center
        _center = Vector3f::ZERO;
        _radius = 1.0;
    }

    Sphere(const Vector3f &center, float radius, Material *material) : Object3D(material) {
        // constructor
        _center = center;
        _radius = radius;
    }

    ~Sphere() override = default;

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        // 判断光源与球面的位置关系
        float original_length = r.getDirection().length();
        // 利用几何法判断交点
        Vector3f l = _center - r.getOrigin();
        float side = Vector3f::dot(l, l) - _radius * _radius; // side > 0 : outer ; side < 0 : inner
        float tp = Vector3f::dot(l, r.getDirection().normalized());
        float t2 = _radius * _radius - (Vector3f::dot(l, l) - (tp * tp));
        float t = -1;
        Vector3f normal = Vector3f::ZERO;
        if (side > 0 && tp >= 0 && t2 >= 0) { // 光源在球体外部，有交点
            t = (tp - sqrt(t2)) / original_length;
            normal = (r.pointAtParameter(t) - _center).normalized();
        } else if (side < 0 && t2 >= 0) { // 光源在球体内部，有交点
            t = (tp + sqrt(t2)) / original_length;
            normal = - (r.pointAtParameter(t) - _center).normalized();
        } // 注：光源在球面上的情况未处理
        if (t > tmin && t < h.getT()) { // 是最近的点
            // 更新hit
            h.set(t, material, normal);
            return true;
        }
        return false;
    }

protected:
    Vector3f _center;
    float _radius;
};


#endif
