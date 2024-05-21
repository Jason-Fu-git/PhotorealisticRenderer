#ifndef MATERIAL_H
#define MATERIAL_H

#include <cassert>
#include <vecmath.h>
#include <cmath>

#include "ray.hpp"
#include "hit.hpp"
#include <iostream>

/**
 * @copybrief 项目所有者独立实现
 * @author Jason Fu
 * @brief Phong模型着色
 *
 */
class Material {
public:

    explicit Material(const Vector3f &d_color, const Vector3f &s_color = Vector3f::ZERO, float s = 0) :
            diffuseColor(d_color), specularColor(s_color), shininess(s) {

    }

    virtual ~Material() = default;

    virtual Vector3f getDiffuseColor() const {
        return diffuseColor;
    }


    /**
     * Shade(...) computes the color of a ray when it hits a surface.
     * All the parameters need to be normalized! (except for colors)
     */
    Vector3f Shade(const Ray &ray, const Hit &hit,
                   const Vector3f &dirToLight, const Vector3f &lightColor) {
        Vector3f R = (2 * Vector3f::dot(dirToLight, hit.getNormal()) * hit.getNormal() - dirToLight).normalized();
        Vector3f shaded = diffuseColor * std::max(Vector3f::dot(dirToLight, hit.getNormal()), 0.0f)
                          + specularColor * std::pow(std::max(-Vector3f::dot(R, ray.getDirection()), 0.0f), shininess);
        shaded = lightColor * shaded;
        return shaded;
    }

protected:
    Vector3f diffuseColor;
    Vector3f specularColor;
    float shininess;
};


#endif // MATERIAL_H
