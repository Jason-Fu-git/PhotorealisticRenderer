//
// Created by Jason Fu on 24-7-12.
//

#ifndef FINALPROJECTCUDA_RAY_CUH
#define FINALPROJECTCUDA_RAY_CUH

#include <cassert>
#include <cmath>
#include <cmath>
#include "Vector3f.cuh"


/**
 *
 * Ray class mostly copied from Peter Shirley and Keith Morley
 * Note : the direction of ray cannot be normalized, in case of transformation occurs!
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
class Ray {
public:
    enum Axis {
        X_AXIS,
        Y_AXIS,
        Z_AXIS
    };

    __device__ __host__ inline static Ray ZERO() {
        return Ray(Vector3f::ZERO(), Vector3f::ZERO());
    }

    Ray() = delete;

    __device__ __host__ inline Ray(const Vector3f &orig, const Vector3f &dir) {
        origin = orig;
        direction = dir;
    }

    __device__ __host__ inline Ray(const Ray &r) {
        origin = r.origin;
        direction = r.direction;
    }

    __device__ __host__ inline const Vector3f &getOrigin() const {
        return origin;
    }

    __device__ __host__ inline const Vector3f &getDirection() const {
        return direction;
    }

    __device__ __host__ inline Vector3f pointAtParameter(float t) const {
        return origin + direction * t;
    }

    __device__ __host__ inline float parameterAtPoint(float c, int axis) const {
        return (c - origin[axis]) / direction[axis];
    }

    __device__ __host__ inline bool operator==(const Ray &r) const {
        return origin == r.origin && direction == r.direction;
    }

private:

    Vector3f origin;
    Vector3f direction;

};

/**
 * reflect the ray with respect to the normal vector at the point.
 * All the vectors need to be normalized!
 * @param r ray in, from light source to point
 * @param normal normal vector at the point (should be at the same side with r.origin)
 * @param point point on the surface
 * @return reflected ray, from point to light source
 * @author Jason Fu
 *
 */
__device__ __host__ inline static Ray reflect(const Ray &r, const Vector3f &normal, const Vector3f &point) {
    auto reflected_dir = r.getDirection() - 2 * Vector3f::dot(r.getDirection(), normal) * normal;
    return Ray(point, reflected_dir.normalized());
}

/**
 * refract the ray with respect to the normal vector at the point.
 * All the vectors need to be normalized!
 * @param r ray in, from light source to point
 * @param normal normal vector at the point (should be at the same side with r.origin)
 * @param point point on the surface
 * @param n1 refraction index of the medium where ray comes from
 * @param n2 refraction index of the medium where ray goes to
 * @return refracted ray, from point to light source. If refraction fails, return nullptr
 * @author Jason Fu
 *
 */
__device__ __host__ inline static Ray
refract(const Ray &r, const Vector3f &normal, const Vector3f &point, float n1, float n2) {
    float cos_theta_i = -Vector3f::dot(r.getDirection(), normal);
    float sin_theta_t_square = n1 * n1 * (1 - cos_theta_i * cos_theta_i) / (n2 * n2);
    if (sin_theta_t_square > 1) { // total internal reflection
        return Ray::ZERO();
    } else {
        float cos_theta_t = std::sqrt(1 - sin_theta_t_square);
        Vector3f refracted_dir = n1 / n2 * r.getDirection()
                                 + (-n1 / n2 * Vector3f::dot(r.getDirection(), normal) - cos_theta_t) * normal;
        return Ray(point, refracted_dir.normalized());
    }
}

#endif //FINALPROJECTCUDA_RAY_CUH
