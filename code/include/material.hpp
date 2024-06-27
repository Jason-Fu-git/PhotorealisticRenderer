#ifndef MATERIAL_H
#define MATERIAL_H

#include <cassert>
#include <vecmath.h>
#include <cmath>
#include <cstring>
#include <iostream>

#include "ray.hpp"
#include "hit.hpp"
#include "utils.hpp"
#include "image.hpp"

#define ALPHA_THRESHOLD 0.1

class Object3D;

/**
 * Base class for material.
 * @author Jason Fu
 *
 */
class Material {
public:
    enum MaterialType {
        DIFFUSE = 0,
        SPECULAR = 1,
        TRANSPARENT = 2,
        GLOSSY = 3,
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

    void setDiffuseColor(const Vector3f &d_color) {
        diffuseColor = d_color;
    }

    virtual void setObject(Object3D *object3D) {
        object = object3D;
    }

    void setTexture(const char *filename);

    // Note : this constructor will use a public image, in order to save memory
    void setTexture(Image *img) {
        texture = img;
    }

    void setRoughness(float roughness) {
        m = roughness;
    }

    float getRoughness() {
        return m;
    }

    // Judge whether the tay transmits
    // NOTE : u, v should be in [0, 1]!
    bool isTransmit(float u, float v) {
        assert(texture != nullptr);
        auto alpha = texture->GetAlpha(std::floor(u * texture->Width()), std::floor(v * texture->Height()));
        return alpha < ALPHA_THRESHOLD;

    }

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
    float refractive_index{};
    float refractive_coefficient{};
    float reflective_coefficient{};
    int type{};
    float m{};
};

/**
 *
 * Typical Phong material. You can set its reflectivity and transparency properties.
 * Reflection and refraction will be calculated only when the corresponding coefficient is greater than zero.
 * @author Jason Fu
 *
 */
class PhongMaterial : public Material {
public:

    explicit PhongMaterial(const Vector3f &d_color, const Vector3f &s_color = Vector3f::ZERO, float s = 0) :
            Material(d_color, 0.0, 0.0, 0.0), specularColor(s_color), shininess(s) {

    }

    PhongMaterial(const PhongMaterial &m) : Material(m) {
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
 * Simple Lambertian Material
 * @author Jason Fu
 */
class LambertianMaterial : public Material {
public:
    explicit LambertianMaterial(const Vector3f &d_color, float rfr_c, float rfl_c, float rfr_i, int _type,
                                const Vector3f &e_color = Vector3f::ZERO)
            : Material(d_color, rfr_c, rfl_c, rfr_i), emissionColor(e_color) {
        type = _type;
    }

    LambertianMaterial(const LambertianMaterial &m) : Material(m) {
        emissionColor = m.emissionColor;
    }

    ~LambertianMaterial() override = default;

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


class CookTorranceMaterial : public Material {
public:
    explicit CookTorranceMaterial(const Vector3f& d_color, float s, float d, float m, const Vector3f& F0)
            : Material(d_color, 0.0, 0.0, 0.0) {
        this->m = m;
        this->s = s;
        this->d = d;
        this->F0 = F0;
        this->type = MaterialType::GLOSSY;
    }

    CookTorranceMaterial(const CookTorranceMaterial &m) : Material(m) {
        this->m = m.m;
        this->s = m.s;
        this->d = m.d;
        this->F0 = m.F0;
        this->type = MaterialType::GLOSSY;
    }

    Vector3f Shade(const Ray &ray, const Hit &hit, const Vector3f &dirToLight, const Vector3f &lightColor) override;

    Vector3f CookTorranceBRDF(const Vector3f &L, const Vector3f &V, const Vector3f &N) const {
        // half vector
        Vector3f H = (L + V).normalized();
        // Fresnel
        Vector3f F = fresnelSchlick(H, V);
        // distribution
        float D = distributionGGX(N, H);
        // geometry
        float G = geometrySmith(N, V, L);

        auto specular = (D * F * G) / (4 *
                    std::max(Vector3f::dot(N, V), 0.0f) *
                    std::max(Vector3f::dot(N, L), 0.0f) + 0.001f
                );

        return (d * diffuseColor + s * specular);
    }


    /**
     * Calculate the Fresnel Reflection Coefficient
     */
    Vector3f fresnelSchlick(const Vector3f &H, const Vector3f &V) const{
        float hv = std::max(Vector3f::dot(H, V), 0.0f);
        return F0 + (1 - F0) * pow(1 - hv, 5);
    }

    /**
     * GGX Distribution
     */
    float distributionGGX(const Vector3f &N, const Vector3f &H) const {
        float a2 = m * m;
        float nh = std::max(Vector3f::dot(N, H), 0.0f);
        float b = nh * nh * (a2 - 1) + 1;
        return a2 / (M_PI * b * b);
    }

    /**
     * Geometry Smith
     */
    float geometrySmith(const Vector3f &N, const Vector3f &V, const Vector3f &L) const {
        return geometrySchlickGGX(N, V) * geometrySchlickGGX(N, L);
    }

    /**
     * Geometry Schlick-GGX
     */
    float geometrySchlickGGX(const Vector3f &N, const Vector3f &V) const {
        float k = (m + 1) * (m + 1) / 8;
        float nv = std::max(Vector3f::dot(N, V), 0.0f);
        return nv / (nv * (1 - k) + k);
    }

    /**
     * Sampling in the GGX Hemisphere
     */
    Vector3f sampleGGXHemisphere(const Vector3f &N) {
        float r1 = uniform01();
        float r2 = uniform01();

        float a = m * m;
        float phi = 2 * M_PI * r1;
        float cosTheta = std::sqrt((1.0 - r2) / (1.0 + (a * a - 1.0) * r2));
        float sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);

        // Convert to Cartesian coordinates
        Vector3f H;
        H.x() = sinTheta * std::cos(phi);
        H.y() = sinTheta * std::sin(phi);
        H.z() = cosTheta;

        // Transform to world space
        Vector3f up = (std::fabs(N.z()) < 0.999) ? Vector3f(0.0, 0.0, 1.0) : Vector3f(1.0, 0.0, 0.0);
        Vector3f tangentX = Vector3f::cross(up, N).normalized();
        Vector3f tangentY = Vector3f::cross(N, tangentX);

        return (tangentX * H.x() + tangentY * H.y() + N * H.z()).normalized();
    }


protected:
    float s;  // specular coefficient
    float d;  // diffuse coefficient
    float m;  // roughness
    Vector3f F0; // Fresnel coefficient


};


#endif // MATERIAL_H
