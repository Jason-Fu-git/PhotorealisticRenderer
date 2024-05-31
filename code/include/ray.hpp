/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */

#ifndef RAY_H
#define RAY_H

#include <cassert>
#include <cmath>
#include <iostream>
#include <cmath>
#include <Vector3f.h>


// Ray class mostly copied from Peter Shirley and Keith Morley
// Note : the direction of ray cannot be normalized, in case of transformation occurs!
class Ray {
public:

    Ray() = delete;

    Ray(const Vector3f &orig, const Vector3f &dir) {
        origin = orig;
        direction = dir;
    }

    Ray(const Ray &r) {
        origin = r.origin;
        direction = r.direction;
    }

    const Vector3f &getOrigin() const {
        return origin;
    }

    const Vector3f &getDirection() const {
        return direction;
    }

    Vector3f pointAtParameter(float t) const {
        return origin + direction * t;
    }

private:

    Vector3f origin;
    Vector3f direction;

};

inline std::ostream &operator<<(std::ostream &os, const Ray &r) {
    os << "Ray <" << r.getOrigin() << ", " << r.getDirection() << ">";
    return os;
}

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
static Ray *reflect(const Ray &r, const Vector3f &normal, const Vector3f &point) {
    auto reflected_dir = r.getDirection() - 2 * Vector3f::dot(r.getDirection(), normal) * normal;
    return new Ray(point, reflected_dir.normalized());
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
static Ray *refract(const Ray &r, const Vector3f &normal, const Vector3f &point, float n1, float n2) {
    float cos_theta_i = -Vector3f::dot(r.getDirection(), normal);
    float sin_theta_t_square = n1 * n1 * (1 - cos_theta_i * cos_theta_i) / (n2 * n2);
    if (sin_theta_t_square > 1) { // total internal reflection
        return nullptr;
    } else {
        float cos_theta_t = std::sqrt(1 - sin_theta_t_square);
        Vector3f refracted_dir = n1 / n2 * r.getDirection()
                                 + (-n1 / n2 * Vector3f::dot(r.getDirection(), normal) - cos_theta_t) * normal;
        return new Ray(point, refracted_dir.normalized());
    }
}

#endif // RAY_H
