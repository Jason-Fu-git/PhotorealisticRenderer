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
 * @param r 入射光线，从光源指向反射点
 * @param normal 法向量
 * @param point 反射点
 * @return 反射光线，从反射点射出
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
 * @param r 入射光线，从光源指向折射点
 * @param normal 法向量
 * @param point 折射点
 * @param n1 入射光线所在介质的折射率
 * @param n2 折射光线所在介质的折射率
 * @return 折射光线，从折射点射出；若发生全反射，返回空指针。
 * @author Jason Fu
 *
 */
static Ray *refract(const Ray &r, const Vector3f &normal, const Vector3f &point, float n1, float n2) {
    float cos_theta_i = -Vector3f::dot(r.getDirection(), normal);
    float sin_theta_t_square = n1 * n1 * (1 - cos_theta_i * cos_theta_i) / (n2 * n2);
    if (sin_theta_t_square > 1) { // 发生全反射
        return nullptr;
    } else {
        float cos_theta_t = std::sqrt(1 - sin_theta_t_square);
        // 折射光线方向
        Vector3f refracted_dir = n1 / n2 * r.getDirection()
                                 + (-n1 / n2 * Vector3f::dot(r.getDirection(), normal) - cos_theta_t) * normal;
        return new Ray(point, refracted_dir.normalized());
    }
}

#endif // RAY_H
