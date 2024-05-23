#ifndef LIGHT_H
#define LIGHT_H

#include <Vector3f.h>
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
     * @param group 场景
     * @return true 表示在阴影中，false 表示不在阴影中
     */
    virtual bool isInShadow(const Vector3f &p, Group *group) const = 0;
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

    ///@param p unsed in this function
    ///@param distanceToLight not well defined because it's not a point light
    void getIllumination(const Vector3f &p, Vector3f &dir, Vector3f &col) const override {
        // the direction to the light is the opposite of the
        // direction of the directional light source
        dir = -direction;
        col = color;
    }

    // 方向光，无阴影
    bool isInShadow(const Vector3f &p, Group *group) const override {
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

    bool isInShadow(const Vector3f &p, Group *group) const override {
        Ray r(position, (p - position).normalized());
        Hit hit;
        if (group->intersect(r, hit, 0)) {
            Vector3f intersectionPoint = r.pointAtParameter(hit.getT());
            if (Vector3f::dot(intersectionPoint - p, intersectionPoint - p) > TOLERANCE) {
                return true;
            }
        }
        return false;
    }

private:

    Vector3f position;
    Vector3f color;

};

#endif // LIGHT_H
