#ifndef LIGHT_H
#define LIGHT_H

#include <cmath>
#include "Vector3f.cuh"

#include "utils.cuh"
#include "object3d.cuh"
#include "group.cuh"
#include "hit.cuh"


#define TOLERANCE 0.01

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
class Light {
public:
    __device__ __host__ inline Light() = default;

    __device__ __host__ inline virtual ~Light() {}

    /**
     * Get the illumination intensity at a given point p
     * @param p the point in the scene
     * @param dir WILL BE MODIFIED : vector from p to origin
     * @param col WILL BE MODIFIED : the color of the light
     */
    __device__ inline virtual void
    getIllumination(const Vector3f &p, Vector3f &dir, Vector3f &col, curandState *state) const = 0;

    /**
     * Determine whether p is in the shadow
     * @param p the point in the scene
     * @param dir MAYBE UNUSED : vector from p to origin
     * @param group the group of objects in the scene
     * @return
     */
    __device__ inline virtual bool isInShadow(const Vector3f &p, Group *group, const Vector3f &dir) const = 0;
};

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
class DirectionalLight : public Light {
public:
    __device__ __host__ inline DirectionalLight() = delete;

    __device__ __host__ inline DirectionalLight(const Vector3f &d, const Vector3f &c) {
        direction = d.normalized();
        color = c;
    }

    __device__ __host__ inline ~DirectionalLight() override {}

    ///@param p not used in this function
    ///@param distanceToLight not well defined because it's not a point light
    __device__ inline void
    getIllumination(const Vector3f &p, Vector3f &dir, Vector3f &col, curandState *state) const override {
        // the direction to the light is the opposite of the
        // direction of the directional light source
        dir = -direction;
        col = color;
    }

    // 方向光，无阴影
    __device__ inline bool isInShadow(const Vector3f &p, Group *group, const Vector3f &dir) const override {
        return false;
    }

    // getters
    __device__ __host__ inline Vector3f getDirection() const { return direction; }

    __device__ __host__ inline Vector3f getColor() const { return color; }

private:

    Vector3f direction;
    Vector3f color;

};

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
class PointLight : public Light {
public:
    __device__ __host__ inline PointLight() = delete;

    __device__ __host__ inline PointLight(const Vector3f &p, const Vector3f &c) {
        position = p;
        color = c;
    }

    __device__ __host__ inline ~PointLight() override {}

    __device__ inline void
    getIllumination(const Vector3f &p, Vector3f &dir, Vector3f &col, curandState *state) const override {
        // the direction to the light is the opposite of the
        // direction of the directional light source
        dir = (position - p).normalized();
        col = color;
    }

    __device__ inline bool isInShadow(const Vector3f &p, Group *group, const Vector3f &dir) const override {
        Ray r(position, (p - position).normalized());
        Hit hit;
        if (group->intersect(r, hit, 0)) {
            Vector3f intersectionPoint = r.pointAtParameter(hit.getT());
            if (Vector3f::dot(intersectionPoint - p, intersectionPoint - p) <= TOLERANCE) {
                return false;
            }
        }
        return true;
    }

    // getters

    __device__ __host__ inline Vector3f getPosition() const { return position; }

    __device__ __host__ inline Vector3f getColor() const { return color; }

private:

    Vector3f position;
    Vector3f color;

};

/**
 * Sphere light
 * NOTE : CURRENTLY, NEE ONLY SUPPORTS SPHERE LIGHT. THIS MEANS IF YOU WANT TO USE NEE, YOU MUST
 * EXPLICITLY DECLARE A SPHERE LIGHT IN THE SCENE CONFIGURATION FILE.
 * @author Jason FU
 *
 */
class SphereLight : public Light {
public:
    __device__ __host__ inline SphereLight(const Vector3f &p, float r, const Vector3f &e_color)
            : position(p), radius(r), emissionColor(e_color) {}

    __device__ __host__ inline ~SphereLight() override {}

    // NOTE : the color it returns should be multiplied by <dir, normal> externally
    __device__ inline void
    getIllumination(const Vector3f &p, Vector3f &dir, Vector3f &col, curandState *state) const override {
        // construct (w, u, v)
        Vector3f w = (position - p).normalized();
        Vector3f u = Vector3f::cross((std::fabs(w._x) > 0.1 ? Vector3f(0, 1, 0) : Vector3f(1, 0, 0)), w).normalized();
        Vector3f v = Vector3f::cross(w, u).normalized();
        float cos_theta_max = std::sqrt(1 - (radius * radius) / Vector3f::dot(position - p, position - p));
        // Randomly generate the ray
        float eps1 = uniform01(state), eps2 = uniform01(state);
        float cos_theta = 1 - eps1 + eps1 * cos_theta_max;
        if (cos_theta > 1) // illegal
        {
            dir = Vector3f::ZERO();
            col = Vector3f::ZERO();
            return;
        }
        float sin_theta = std::sqrt(1 - cos_theta * cos_theta);
        float phi = 2.0f * M_PI * eps2;
        dir = u * std::cos(phi) * sin_theta + v * std::sin(phi) * sin_theta + w * cos_theta;
        dir = dir.normalized();
        // 2 * PI * (1 - cos_theta_max) = 1 / p,    1 / PI is the requirement of BRDF material
        col = emissionColor * 2 * M_PI * (1 - cos_theta_max) * M_1_PI;
    }

    // NOTE : dir is used!
    __device__ inline bool isInShadow(const Vector3f &p, Group *group, const Vector3f &dir) const override {
        if (dir != Vector3f::ZERO()) {
            // aid
            float tmin = (position - p).length();
            Vector3f ldir = -dir;
            Ray r(p, ldir);
            Hit hit;
            if (group->intersect(r, hit, TOLERANCE)) {
                Vector3f intersectionPoint = r.pointAtParameter(hit.getT());
                // exceeds (due to precision error)
                if (hit.getT() > tmin) {
                    return false;
                }
                // intersects
                if (abs(radius * radius - Vector3f::dot(intersectionPoint - position, intersectionPoint - position)) <=
                    TOLERANCE * 10) {
                    return false;
                }
            }
        }
        return true;
    }

    // getters

    __device__ __host__ inline Vector3f getPosition() const { return position; }

    __device__ __host__ inline Vector3f getEmissionColor() const { return emissionColor; }

    __device__ __host__ inline float getRadius() const { return radius; }

private:
    float radius;
    Vector3f position;
    Vector3f emissionColor;
};

__global__ void createDirectionalLightOnDevice(Light **light, Vector3f d, Vector3f c);

__global__ void createPointLightOnDevice(Light **light, Vector3f p, Vector3f c);

__global__ void
createSphereLightOnDevice(Light **light, Vector3f p, float r, Vector3f c);

__global__ void freeLightOnDevice(Light **light);

#endif // LIGHT_H
