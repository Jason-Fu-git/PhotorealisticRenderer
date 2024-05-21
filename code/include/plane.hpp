#ifndef PLANE_H
#define PLANE_H

#include "object3d.hpp"
#include <vecmath.h>
#include <cmath>


// function: ax+by+cz=d

/**
 * @copybrief 项目所有者独立实现
 * @author Jason Fu
 * @brief Plane class, function : _normal.x*x + _normal.y*y + _normal.z*z = _d
 * @var _normal
 * @var _d
 */
class Plane : public Object3D {
public:
    Plane() {
        // default plane : z = 0 (xy plane)
        _normal = Vector3f(0, 0, 1);
        _d = 0;
    }

    Plane(const Vector3f &normal, float d, Material *m) : Object3D(m) {
        _normal = normal.normalized();
        _d = d;
    }

    ~Plane() override = default;

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        float original_length = r.getDirection().length();
        float a = Vector3f::dot(_normal, r.getDirection().normalized());
        if (a != 0) {
            float t = -(-_d + Vector3f::dot(_normal, r.getOrigin())) / a / original_length;
            if (t > 0 && t > tmin && t < h.getT()) { // valid intersection
                h.set(t, material, _normal);
                return true;
            }
        }
        return false;
    }

    Vector3f _normal;
protected:
    float _d;


};

#endif //PLANE_H
		

