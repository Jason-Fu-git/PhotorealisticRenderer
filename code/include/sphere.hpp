#ifndef SPHERE_H
#define SPHERE_H

#include "object3d.hpp"
#include <utility>
#include <vecmath.h>
#include <cmath>


/**
 * @copybrief 项目所有者独立实现
 * @author Jason Fu
 * @brief Sphere class
 * @var _center
 * @var _radius
 */
class Sphere : public Object3D {
public:
    Sphere() {
        // unit ball at the center
        _center = Vector3f::ZERO;
        _radius = 1.0;
        theta_offset = 0.0;
        phi_offset = 0.0;
    }

    Sphere(const Vector3f &center, float radius, Material *material, float theta_offset = 0.0, float phi_offset = 0.0)
            : Object3D(material) {
        // constructor
        _center = center;
        _radius = radius;
        this->phi_offset = phi_offset;
        this->theta_offset = theta_offset;
    }

    ~Sphere() override = default;

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        // 判断光源与球面的位置关系
        float original_length = r.getDirection().length();
        // 利用几何法判断交点
        Vector3f l = _center - r.getOrigin();
        float side = Vector3f::dot(l, l) - _radius * _radius; // side > 0 : outer ; side < 0 : inner
        float tp = Vector3f::dot(l, r.getDirection().normalized());
        float t2 = _radius * _radius - (Vector3f::dot(l, l) - (tp * tp));
        float t = -1;
        Vector3f normal = Vector3f::ZERO;
        bool is_inside = false;
        if (side > 0 && tp >= 0 && t2 >= 0) { // 光源在球体外部，有交点
            t = (tp - sqrt(t2)) / original_length;
            normal = (r.pointAtParameter(t) - _center).normalized();
            is_inside = false;
        } else if (side < 0 && t2 >= 0) { // 光源在球体内部，有交点
            t = (tp + sqrt(t2)) / original_length;
            normal = -(r.pointAtParameter(t) - _center).normalized();
            is_inside = true;
        } // 注：光源在球面上的情况未处理，需要加微小扰动
        if (t > tmin && t < h.getT()) { // 是最近的点
            // 更新hit
            h.set(t, material, normal, is_inside);
            return true;
        }
        return false;
    }

    std::pair<int, int>
    textureMap(float objectX, float objectY, float objectZ, int textureWidth, int textureHeight) override {
        double x = objectX - _center.x();
        double y = objectY - _center.y();
        double z = objectZ - _center.z();
        double theta = std::atan2(z, x) + theta_offset; // xOz平面
        double phi = std::acos(y / _radius) + phi_offset; // y轴方向
        double u = mod((M_PI + theta) / (2 * M_PI), 1);
        double v = mod((M_PI - phi) / M_PI, 1);
        return std::make_pair((int) (u * textureWidth), (int) (v * textureHeight));
    }

protected:
    Vector3f _center;
    float _radius;
    float theta_offset;
    float phi_offset;
};


#endif
