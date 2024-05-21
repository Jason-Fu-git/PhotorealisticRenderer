/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */

#ifndef OBJECT3D_H
#define OBJECT3D_H

#include "ray.hpp"
#include "hit.hpp"
#include "material.hpp"

// Base class for all 3d entities.
/**
 * @var material: Pointer to the material of the object.
 */
class Object3D {
public:
    Object3D() : material(nullptr) {}

    virtual ~Object3D() = default;

    explicit Object3D(Material *material) {
        this->material = material;
    }

    // Intersect Ray with this object. If hit, store information in hit structure.
    virtual bool intersect(const Ray &r, Hit &h, float tmin) = 0;
protected:

    Material *material;
};

#endif

