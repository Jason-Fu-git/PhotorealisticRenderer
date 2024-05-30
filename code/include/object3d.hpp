/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */

#ifndef OBJECT3D_H
#define OBJECT3D_H

#include "ray.hpp"
#include "hit.hpp"
#include "material.hpp"
#include <utility>

#define DISTURBANCE 0.01

// Base class for all 3d entities.
/**
 * @var material: Pointer to the material of the object.
 */
class Object3D {
public:
    Object3D() : material(nullptr) {}

    // material 在合适时机析构
    virtual ~Object3D() = default;

    explicit Object3D(Material *material) {
        this->material = material;
    }

    // Intersect Ray with this object. If hit, store information in hit structure.
    virtual bool intersect(const Ray &r, Hit &h, float tmin) = 0;

    /**
     * Calculate the texture coordinate of a point on the object.
     * In brief, the map (x,y) -> (u,v)
     * @param objectX coordinate x on the object
     * @param objectY coordinate y on the object
     * @param objectZ coordinate z on the object
     * @param textureWidth the width of the texture
     * @param textureHeight the height of the texture
     * @return a point on the texture
     */
    virtual std::pair<int, int>
    textureMap(float objectX, float objectY, float objectZ, int textureWidth, int textureHeight) {
        return std::make_pair(-1, -1);
    }

protected:
    Material *material;
};

#endif

