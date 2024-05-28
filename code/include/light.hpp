#ifndef LIGHT_H
#define LIGHT_H

#include <cmath>
#include <Vector3f.h>

#include "utils.hpp"
#include "object3d.hpp"
#include "group.hpp"
#include "hit.hpp"


#define TOLERANCE 0.01

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
class Light {
public:
    Light() = default;

    virtual ~Light() = default;

    /**
     * 获取某点的光照
     * @param p 该点的坐标
     * @param dir 将被修改：从该点到光源的方向向量
     * @param col 将被修改：该点的光照颜色
     */
    virtual void getIllumination(const Vector3f &p, Vector3f &dir, Vector3f &col) const = 0;

    /**
     * 判断某点是否在阴影中。
     * @param p 该点的坐标
     * @param dir 从该点到光源的方向向量 (MAYBE UNUSED)
     * @param group 场景
     * @return true 表示在阴影中，false 表示不在阴影中
     */
    virtual bool isInShadow(const Vector3f &p, Group *group, const Vector3f &dir) const = 0;
};

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
class DirectionalLight : public Light {
public:
    DirectionalLight() = delete;

    DirectionalLight(const Vector3f &d, const Vector3f &c) {
        direction = d.normalized();
        color = c;
    }

    ~DirectionalLight() override = default;

    ///@param p not used in this function
    ///@param distanceToLight not well defined because it's not a point light
    void getIllumination(const Vector3f &p, Vector3f &dir, Vector3f &col) const override {
        // the direction to the light is the opposite of the
        // direction of the directional light source
        dir = -direction;
        col = color;
    }

    // 方向光，无阴影
    bool isInShadow(const Vector3f &p, Group *group, const Vector3f &dir) const override {
        return false;
    }

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
    PointLight() = delete;

    PointLight(const Vector3f &p, const Vector3f &c) {
        position = p;
        color = c;
    }

    ~PointLight() override = default;

    void getIllumination(const Vector3f &p, Vector3f &dir, Vector3f &col) const override {
        // the direction to the light is the opposite of the
        // direction of the directional light source
        dir = (position - p).normalized();
        col = color;
    }

    bool isInShadow(const Vector3f &p, Group *group, const Vector3f &dir) const override {
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

private:

    Vector3f position;
    Vector3f color;

};

/**
 * 面光源（球面）
 * @author Jason FU
 *
 */
class SphereLight : public Light {
public:
    SphereLight(Vector3f &p, float r, Vector3f &e_color) : position(p), radius(r), emissionColor(e_color){}

    ~SphereLight() override = default;

    // 计算dir，并返回emissionColor，真正的color需自行计算
    void getIllumination(const Vector3f &p, Vector3f &dir, Vector3f &col) const override {
        // 在采样点p建立空间直角坐标系 (w, u, v)
        Vector3f w = (position - p).normalized();
        Vector3f u = Vector3f::cross((std::fabs(w.x()) > 0.1 ? Vector3f(0, 1, 0) : Vector3f(1, 0, 0)), w).normalized();
        Vector3f v = Vector3f::cross(w, u).normalized();
        // 最大张角
        float cos_theta_max = std::sqrt(1 - (radius * radius) / Vector3f::dot(position - p, position - p));
        // 随机生成光线
        float eps1 = uniform01(), eps2 = uniform01();
        float cos_theta = 1 - eps1 + eps1 * cos_theta_max;
        if (cos_theta > 1) // illegal
        {
            dir = Vector3f::ZERO;
            col = Vector3f::ZERO;
            return;
        }
        float sin_theta = std::sqrt(1 - cos_theta * cos_theta);
        float phi = 2 * M_PI * eps2;
        dir = u * std::cos(phi) * sin_theta + v * std::sin(phi) * sin_theta + w * cos_theta;
        dir = dir.normalized();
        // 2 * PI * (1 - cos_theta_max) = 1 / p,    1 / PI 是 BRDF材质的要求
        col = emissionColor * 2 * M_PI * (1 - cos_theta_max) * M_1_PI;
    }

    // NOTE : dir is used!
    bool isInShadow(const Vector3f &p, Group *group, const Vector3f &dir) const override {
        if (dir != Vector3f::ZERO) {
            Vector3f origin = position + dir * radius;
            Ray r(origin, dir);
            Hit hit;
            if (group->intersect(r, hit, 0)) {
                Vector3f intersectionPoint = r.pointAtParameter(hit.getT());
                if (Vector3f::dot(intersectionPoint - p, intersectionPoint - p) <= TOLERANCE) {
                    return false;
                }
            }
        }
        return true;
    }

private:
    Vector3f position;
    float radius;
    Vector3f emissionColor;
};

#endif // LIGHT_H
