//
// Created by Jason Fu on 24-7-12.
//

#ifndef FINALPROJECTCUDA_MATERIAL_CUH
#define FINALPROJECTCUDA_MATERIAL_CUH

#include <cassert>
#include "Vector3f.cuh"
#include <cmath>
#include <cstring>
#include <iostream>

#include "ray.cuh"
#include "hit.cuh"
#include "utils.cuh"
#include "image.cuh"

#define ALPHA_THRESHOLD 0.1

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

    __device__ __host__ inline explicit Material(const Vector3f &d_color, float rfr_c, float rfl_c, float rfr_i) :
            diffuseColor(d_color), reflective_coefficient(rfl_c),
            refractive_coefficient(rfr_c), refractive_index(rfr_i), type(DIFFUSE), texture(nullptr) {
    }


    __device__ __host__ inline Material(const Material &m) {
        diffuseColor = m.diffuseColor;
        reflective_coefficient = m.reflective_coefficient;
        refractive_coefficient = m.refractive_coefficient;
        refractive_index = m.refractive_index;
        type = m.type;
        texture = m.texture;
    }

    __device__ __host__ inline virtual ~Material() {

    }


    __device__ __host__ inline virtual void setReflectiveProperties(float _reflective_coefficient) {
        reflective_coefficient = _reflective_coefficient;
    }

    __device__ __host__ inline virtual void
    setRefractiveProperties(float _refractive_coefficient, float _refractive_index) {
        refractive_coefficient = _refractive_coefficient;
        refractive_index = _refractive_index;
    }

    __device__ __host__ inline virtual void setDiffuseColor(const Vector3f &d_color) {
        diffuseColor = d_color;
    }

    // Note : this constructor will use a public image, in order to save memory
    __device__ __host__ inline virtual void setTexture(Image *img) {
        texture = img;
    }

    __host__ virtual void setTexture(const char *filename);

    // Judge whether the tay transmits
    // NOTE : u, v should be in [0, 1]!
    __device__ __host__ inline virtual bool isTransmit(float u, float v) {
        assert(texture != nullptr);
        auto alpha = texture->GetAlpha(floorf(u * texture->Width()), floorf(v * texture->Height()));
        return alpha < ALPHA_THRESHOLD;

    }

    __device__ __host__ inline virtual bool isEmitter() const {
        return false;
    }

    __device__ __host__ inline virtual bool isReflective() const {
        return reflective_coefficient > 0.0;
    }

    __device__ __host__ inline virtual bool isRefractive() const {
        return refractive_coefficient > 0.0;
    }

    __device__ __host__ inline virtual float getReflectiveCoefficient() const {
        return reflective_coefficient;
    }

    __device__ __host__ inline virtual float getRefractiveCoefficient() const {
        return refractive_coefficient;
    }

    __device__ __host__ inline virtual float getRefractiveIndex() const {
        return refractive_index;
    }

    __device__ __host__ inline virtual Vector3f getDiffuseColor() const {
        return diffuseColor;
    }

    __device__ __host__ inline virtual Vector3f getEmissionColor() const {
        return Vector3f::ZERO();
    }

    __device__ __host__ inline virtual int getType() const {
        return type;
    }

    __device__ virtual Vector3f sampleGGXHemisphere(const Vector3f &N, curandState *state) {
        return Vector3f::ZERO();
    }

    __device__ virtual Vector3f CookTorranceBRDF(const Vector3f &L, const Vector3f &V, const Vector3f &N) const {
        return Vector3f::ZERO();
    }


    /**
     * Shade(...) computes the color of a ray when it hits a surface
     * @note
     * 1. All the parameters need to be normalized! (except for colors)
     * 2. For BRDF materials, DO NOT call this function.
     *
     */
    __device__ virtual Vector3f Shade(const Ray &ray, const Hit &hit,
                                      const Vector3f &dirToLight, const Vector3f &lightColor) = 0;

protected:
    float refractive_index{};
    float refractive_coefficient{};
    float reflective_coefficient{};
    int type{};
    Vector3f diffuseColor;
    Image *texture{};
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

    __device__ __host__ inline explicit PhongMaterial(const Vector3f &d_color,
                                                      const Vector3f &s_color = Vector3f::ZERO(),
                                                      float s = 0) :
            Material(d_color, 0.0, 0.0, 0.0), specularColor(s_color), shininess(s) {
    }

    __device__ __host__ inline PhongMaterial(const PhongMaterial &m) : Material(m) {
        specularColor = m.specularColor;
        shininess = m.shininess;
    }


    __device__ __host__ ~PhongMaterial() override {

    }

    __device__ __host__ Vector3f getSpecularColor() const {
        return specularColor;
    }

    __device__ __host__ float getShininess() const {
        return shininess;
    }


    /**
     * Shade(...) computes the color of a ray when it hits a surface.
     * All the parameters need to be normalized! (except for colors)
     */
    __device__ Vector3f Shade(const Ray &ray, const Hit &hit,
                              const Vector3f &dirToLight, const Vector3f &lightColor) override;

protected:
    float shininess;
    Vector3f specularColor;
};

/**
 * Simple Lambertian Material
 * @author Jason Fu
 */
class LambertianMaterial : public Material {
public:
    __device__ __host__ inline explicit LambertianMaterial(const Vector3f &d_color, float rfr_c, float rfl_c,
                                                           float rfr_i,
                                                           int _type,
                                                           const Vector3f &e_color = Vector3f::ZERO())
            : Material(d_color, rfr_c, rfl_c, rfr_i),
              emissionColor(e_color) {
        type = _type;
    }

    __device__ __host__ inline LambertianMaterial(const LambertianMaterial &m) : Material(m) {
        emissionColor = m.emissionColor;
    }

    __device__ __host__  ~LambertianMaterial() override {

    }

    __device__ __host__ inline bool isEmitter() const override {
        return emissionColor._x > 0 || emissionColor._y > 0 || emissionColor._z > 0;
    }

    __device__ __host__ inline Vector3f getEmissionColor() const override {
        return emissionColor;
    }

    __device__ Vector3f Shade(const Ray &ray, const Hit &hit,
                              const Vector3f &dirToLight, const Vector3f &color) override;

protected:
    Vector3f emissionColor;
};

/**
 * Cook Torrance Model for Glossy Material
 * @author Jason Fu
 * @acknowledgement BRDF章节PPT
 *
 */
class CookTorranceMaterial : public Material {
public:
    __device__ __host__ inline explicit CookTorranceMaterial(const Vector3f &d_color, float s, float d, float m,
                                                             const Vector3f &F0) : Material(d_color, 0.0, 0.0, 0.0) {
        this->m = m;
        this->s = s;
        this->d = d;
        this->F0 = F0;
        this->type = MaterialType::GLOSSY;
    }

    __device__ __host__ ~CookTorranceMaterial() override {

    }

    __device__ __host__ inline CookTorranceMaterial(const CookTorranceMaterial &m) : Material(m) {
        this->m = m.m;
        this->s = m.s;
        this->d = m.d;
        this->F0 = m.F0;
        this->type = MaterialType::GLOSSY;
    }

    __device__ Vector3f
    Shade(const Ray &ray, const Hit &hit, const Vector3f &dirToLight, const Vector3f &lightColor) override;

    __device__ Vector3f CookTorranceBRDF(const Vector3f &L, const Vector3f &V, const Vector3f &N) const override;


    /**
     * Calculate the Fresnel Reflection Coefficient
     */
    __device__  inline Vector3f fresnelSchlick(const Vector3f &H, const Vector3f &V) const {
        float hv = fmaxf(Vector3f::dot(H, V), 0.0f);
        return F0 + (Vector3f(1, 1, 1) - F0) * powf(1 - hv, 5);
    }

    /**
     * GGX Distribution
     */
    __device__ inline float distributionGGX(const Vector3f &N, const Vector3f &H) const {
        float a2 = m * m;
        float nh = fmaxf(Vector3f::dot(N, H), 0.0f);
        float b = nh * nh * (a2 - 1) + 1;
        return a2 / (M_PI * b * b);
    }

    /**
     * Geometry Smith
     */
    __device__ inline float geometrySmith(const Vector3f &N, const Vector3f &V, const Vector3f &L) const {
        return geometrySchlickGGX(N, V) * geometrySchlickGGX(N, L);
    }

    /**
     * Geometry Schlick-GGX
     */
    __device__  inline float geometrySchlickGGX(const Vector3f &N, const Vector3f &V) const {
        float k = (m + 1) * (m + 1) / 8;
        float nv = fmaxf(Vector3f::dot(N, V), 0.0f);
        return nv / (nv * (1 - k) + k);
    }

    /**
     * Sampling in the GGX Hemisphere
     */
    __device__ inline Vector3f sampleGGXHemisphere(const Vector3f &N, curandState *state) override {
        float r1 = uniform01(state);
        float r2 = uniform01(state);

        float a = m * m;
        float phi = 2.0f * M_PI * r1;
        float cosTheta = sqrtf((1.0f - r2) / (1.0f + (a * a - 1.0f) * r2));
        float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

        // Convert to Cartesian coordinates
        Vector3f H;
        H._x = sinTheta * cosf(phi);
        H._y = sinTheta * sinf(phi);
        H._z = cosTheta;

        // Transform to world space
        Vector3f up = (fabsf(N._z) < 0.999) ? Vector3f(0.0, 0.0, 1.0) : Vector3f(1.0, 0.0, 0.0);
        Vector3f tangentX = Vector3f::cross(up, N).normalized();
        Vector3f tangentY = Vector3f::cross(N, tangentX);

        return (tangentX * H._x + tangentY * H._y + N * H._z).normalized();
    }

    __device__ __host__ float getSpecularCoefficient() const {
        return s;
    }

    __device__ __host__ float getDiffuseCoefficient() const {
        return d;
    }

    __device__ __host__ float getRoughness() const {
        return m;
    }

    __device__ __host__ Vector3f getFresnelCoefficient() const {
        return F0;
    }


protected:
    float s;  // specular coefficient
    float d;  // diffuse coefficient
    float m;  // roughness
    Vector3f F0; // Fresnel coefficient


};

__global__ void createPhongMaterialOnDevice(Material **material,
                                            Vector3f d_color,
                                            Vector3f s_color = Vector3f::ZERO(),
                                            float s = 0.0f);

__global__ void createLambertianMaterialOnDevice(Material **material, Vector3f d_color, float rfr_c, float rfl_c,
                                                 float rfr_i, int _type,
                                                 Vector3f e_color = Vector3f::ZERO());

__global__ void
createCookTorranceMaterialOnDevice(Material **material, Vector3f d_color, float s, float d, float m,
                                   Vector3f F0);

__global__ void freeMaterialOnDevice(Material **material);

#endif //FINALPROJECTCUDA_MATERIAL_CUH
