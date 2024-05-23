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
 * @brief Phong模型着色，可以设置反射与折射的光学性质，注意若折射系数/反射系数为0，则不会进行折射/反射计算
 *
 */
class Material {
public:

    explicit Material(const Vector3f &d_color, const Vector3f &s_color = Vector3f::ZERO, float s = 0) :
            diffuseColor(d_color), specularColor(s_color), shininess(s), reflective_coefficient(0.0),
            refractive_coefficient(0.0), refractive_index(0.0) {

    }

    virtual ~Material() = default;

    virtual void setReflectiveProperties(float _reflective_coefficient) {
        reflective_coefficient = _reflective_coefficient;
    }

    virtual void setRefractiveProperties(float _refractive_coefficient, float _refractive_index) {
        refractive_coefficient = _refractive_coefficient;
        refractive_index = _refractive_index;
    }

    virtual bool isReflective() const {
        return reflective_coefficient > 0.0;
    }

    virtual bool isRefractive() const {
        return refractive_coefficient > 0.0;
    }

    virtual float getReflectiveCoefficient() const {
        return reflective_coefficient;
    }

    virtual float getRefractiveCoefficient() const {
        return refractive_coefficient;
    }

    virtual float getRefractiveIndex() const {
        return refractive_index;
    }


    /**
     * Shade(...) computes the color of a ray when it hits a surface.
     * All the parameters need to be normalized! (except for colors)
     */
    virtual Vector3f Shade(const Ray &ray, const Hit &hit,
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
    float refractive_index; // 折射率
    float refractive_coefficient; // 折射系数
    float reflective_coefficient; // 反射系数
};

/**
 * 透明材料，漫反射系数和镜面反射系数为0，环境光项非0
 * @author Jason Fu
 *
 */

class TransparentMaterial : public Material {
public:
    explicit TransparentMaterial(float _refractive_index, float _refractive_coefficient,
                                 float _reflective_coefficient) : Material(Vector3f::ZERO) {
        refractive_coefficient = _refractive_coefficient;
        refractive_index = _refractive_index;
        reflective_coefficient = _reflective_coefficient;
    }

    // 透明物体，无Shade
    Vector3f Shade(const Ray &ray, const Hit &hit,
                           const Vector3f &dirToLight, const Vector3f &lightColor) override {
        return Vector3f::ZERO;
    }
};


#endif // MATERIAL_H
