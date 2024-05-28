#ifndef MATERIAL_H
#define MATERIAL_H

#include <cassert>
#include <vecmath.h>
#include <cmath>
#include <cstring>
#include <iostream>

#include "ray.hpp"
#include "hit.hpp"
#include "image.hpp"

class Object3D;

/**
 * @copybrief 项目所有者独立实现
 * @author Jason Fu
 * @brief Phong模型着色，可以设置反射与折射的光学性质，注意若折射系数/反射系数为0，则不会进行折射/反射计算
 *
 */
class Material {
public:
    enum MaterialType {
        DIFFUSE = 0,
        SPECULAR = 1,
        TRANSPARENT = 2,
    };

    explicit Material(const Vector3f &d_color, float rfr_c, float rfl_c, float rfr_i) :
            diffuseColor(d_color), reflective_coefficient(rfl_c),
            refractive_coefficient(rfr_c), refractive_index(rfr_i), type(DIFFUSE), object(nullptr), texture(nullptr) {

    }

    Material(const Material &m) {
        diffuseColor = m.diffuseColor;
        reflective_coefficient = m.reflective_coefficient;
        refractive_coefficient = m.refractive_coefficient;
        refractive_index = m.refractive_index;
        type = m.type;
        object = m.object;
        texture = m.texture;
    }

    virtual ~Material() = default;


    virtual void setReflectiveProperties(float _reflective_coefficient) {
        reflective_coefficient = _reflective_coefficient;
    }

    virtual void setRefractiveProperties(float _refractive_coefficient, float _refractive_index) {
        refractive_coefficient = _refractive_coefficient;
        refractive_index = _refractive_index;
    }

    virtual void setObject(Object3D *object3D) {
        object = object3D;
    }

    void setTexture(const char *filename) ;

    virtual bool isEmitter() const {
        return false;
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

    virtual Vector3f getDiffuseColor() const {
        return diffuseColor;
    }

    virtual Vector3f getEmissionColor() const {
        return Vector3f::ZERO;
    }

    virtual Object3D *getObject() const {
        return object;
    }

    virtual int getType() const {
        return type;
    }


    /**
     * Shade(...) computes the color of a ray when it hits a surface
     * @note
     * 1. All the parameters need to be normalized! (except for colors)
     * 2. For BRDF materials, DO NOT call this function.
     *
     */
    virtual Vector3f Shade(const Ray &ray, const Hit &hit,
                           const Vector3f &dirToLight, const Vector3f &lightColor) = 0;

protected:
    Vector3f diffuseColor;
    Object3D *object{}; // the object that this material belongs to
    Image *texture{};
    float refractive_index{}; // 折射率
    float refractive_coefficient{}; // 折射系数
    float reflective_coefficient{}; // 反射系数
    int type{}; // 材料类型
};

/**
 * Phong模型着色，可以设置反射与折射的光学性质，注意若折射系数/反射系数为0，则不会进行折射/反射计算
 * @author Jason Fu
 */
class PhongMaterial : public Material {
public:

    explicit PhongMaterial(const Vector3f &d_color, const Vector3f &s_color = Vector3f::ZERO, float s = 0) :
            Material(d_color, 0.0, 0.0, 0.0), specularColor(s_color), shininess(s) {

    }

    PhongMaterial(const PhongMaterial& m) : Material(m){
        specularColor = m.specularColor;
        shininess = m.shininess;
    }

    ~PhongMaterial() override = default;


    /**
     * Shade(...) computes the color of a ray when it hits a surface.
     * All the parameters need to be normalized! (except for colors)
     */
    Vector3f Shade(const Ray &ray, const Hit &hit,
                   const Vector3f &dirToLight, const Vector3f &lightColor) override;

protected:
    Vector3f specularColor;
    float shininess;
};

/**
 * 简单的BRDF材质实现
 * @author Jason Fu
 */
class BRDFMaterial : public Material {
public:
    explicit BRDFMaterial(const Vector3f &d_color, float rfr_c, float rfl_c, float rfr_i, int _type,
                          const Vector3f &e_color = Vector3f::ZERO)
            : Material(d_color, rfr_c, rfl_c, rfr_i), emissionColor(e_color) {
        type = _type;
    }

    BRDFMaterial(const BRDFMaterial& m) : Material(m){
        emissionColor = m.emissionColor;
    }

    ~BRDFMaterial() override = default;

    bool isEmitter() const override {
        return emissionColor.x() > 0 || emissionColor.y() > 0 || emissionColor.z() > 0;
    }

    Vector3f getEmissionColor() const override {
        return emissionColor;
    }

    Vector3f Shade(const Ray &ray, const Hit &hit,
                   const Vector3f &dirToLight, const Vector3f &color) override;

protected:
    Vector3f emissionColor;
};


#endif // MATERIAL_H
