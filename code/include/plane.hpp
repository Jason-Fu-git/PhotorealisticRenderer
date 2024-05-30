#ifndef PLANE_H
#define PLANE_H

#include "object3d.hpp"
#include "utils.hpp"
#include <vecmath.h>
#include <cmath>


// function: ax+by+cz=d

/**
 * @copybrief 项目所有者独立实现
 * @author Jason Fu
 * @brief Plane class, function : _normal.x*x + _normal.y*y + _normal.z*z = _d
 * @var _normal
 * @var _d
 */
class Plane : public Object3D {
public:
    Plane() {
        // default plane : z = 0 (xy plane)
        _normal = Vector3f(0, 0, 1);
        _u = Vector3f(1, 0, 0);
        _v = Vector3f(0, 1, 0);
        origin = Vector3f(0, 0, 0);
        _d = 0;
        scale = 1.0f;
    }

    Plane(const Vector3f &normal, float d, Material *m, float scale = 1.0f) : Object3D(m) {
        _normal = normal.normalized();
        _d = d;
        this->scale = scale;
        // construct an orthonormal basis
        _u = Vector3f::cross((std::fabs(_normal.x()) > 0.1 ? Vector3f(0, 1, 0) : Vector3f(1, 0, 0)),
                             _normal).normalized();
        _v = Vector3f::cross(_normal, _u).normalized();
        origin = std::fabs(_normal.x()) > 0.1 ? Vector3f(_d / _normal.x(), 0, 0) : (
                std::fabs(_normal.y()) > 0.1 ? Vector3f(0, _d / _normal.y(), 0) : Vector3f(0, 0, _d / _normal.z())
        );
    }

    ~Plane() override = default;

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        float original_length = r.getDirection().length();
        float a = Vector3f::dot(_normal, r.getDirection().normalized());
        if (a != 0) {
            float t = -(-_d + Vector3f::dot(_normal, r.getOrigin())) / a / original_length;
            if (t > 0 && t > tmin && t < h.getT()) { // valid intersection
                // 判断交点在物体内还是物体外
                Vector3f normal = _normal;
                bool isInside = false;
                if (Vector3f::dot(_normal, r.getDirection()) > 0) {
                    normal = -_normal;
                    isInside = true;
                }
                h.set(t, material, normal, isInside);
                return true;
            }
        }
        return false;
    }

    std::pair<int, int>
    textureMap(float objectX, float objectY, float objectZ, int textureWidth, int textureHeight) override {
        // compress the image
        double newWidth = textureWidth * scale;
        double newHeight = textureHeight * scale;
        // calculate the coordinate (x, y) on the plane
        Vector3f dir = Vector3f(objectX, objectY, objectZ) - origin;
        double length = dir.length();
        double cos_theta = Vector3f::dot(dir, _u) / length;
        double sin_theta = Vector3f::dot(dir, _v) / length;
        double x = length * cos_theta;
        double y = length * sin_theta;
        // convert
        double u = mod(x, newWidth) / newWidth;
        double v = mod(y, newHeight) / newHeight;
        // convert it to the texture coordinate
        return std::make_pair(std::floor(u * textureWidth), std::floor(v * textureHeight));
    }

    Vector3f _normal;
protected:
    float _d;
    float scale;
    Vector3f _u, _v;
    Vector3f origin;


};

#endif //PLANE_H
		

