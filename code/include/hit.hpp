/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */

#ifndef HIT_H
#define HIT_H

#include <vecmath.h>
#include "ray.hpp"

class Material;

/**
 * @var t: distance
 * @var normal: surface normal at the hit point， note : should be normalized!
 * @var material: pointer to the material of the surface that is hit
 */
class Hit {
public:

    // constructors
    Hit() {
        material = nullptr;
        t = 1e38;
    }

    Hit(float _t, Material *m, const Vector3f &n) {
        t = _t;
        material = m;
        normal = n;
    }

    Hit(const Hit &h) {
        t = h.t;
        material = h.material;
        normal = h.normal;
    }

    // destructor
    ~Hit() = default;

    float getT() const {
        return t;
    }

    Material *getMaterial() const {
        return material;
    }

    const Vector3f &getNormal() const {
        return normal;
    }

    void set(float _t, Material *m, const Vector3f &n) {
        t = _t;
        material = m;
        normal = n;
    }

private:
    float t;
    Material *material;
    Vector3f normal;

};

inline std::ostream &operator<<(std::ostream &os, const Hit &h) {
    os << "Hit <" << h.getT() << ", " << h.getNormal() << ">";
    return os;
}

#endif // HIT_H
