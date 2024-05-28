#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "object3d.hpp"
#include <vecmath.h>
#include <cmath>
#include <iostream>

using namespace std;

// DONE: implement this class and add more fields as necessary,

/**
 * @copybrief 项目所有者独立实现
 * @author Jason Fu
 * @brief Triangle class
 * @var _a : Vertex A
 * @var _b : Vertex B
 * @var _c : Vertex C
 * @var normal : Normal vector of the triangle
 */
class Triangle : public Object3D {

public:
    Triangle() = delete;

    ~Triangle() override = default;

    // a b c are three vertex positions of the triangle
    Triangle(const Vector3f &a, const Vector3f &b, const Vector3f &c, Material *m) : Object3D(m) {
        _a = a;
        _b = b;
        _c = c;
    }

    bool intersect(const Ray &ray, Hit &hit, float tmin) override {
        float original_length = ray.getDirection().length();
        // 采用重心坐标
        Vector3f E1 = _a - _b;
        Vector3f E2 = _a - _c;
        Vector3f S = _a - ray.getOrigin();
        float det1 = Matrix3f(ray.getDirection().normalized(), E1, E2).determinant();
        if (det1 != 0) // valid solution
        {
            float t = Matrix3f(S, E1, E2).determinant() / det1 / original_length;
            float beta = Matrix3f(ray.getDirection().normalized(), S, E2).determinant() / det1;
            float gamma = Matrix3f(ray.getDirection().normalized(), E1, S).determinant() / det1;

            if (t > 0 && beta >= 0 && gamma >= 0 && beta + gamma <= 1) { // has intersection
                // 注：内外情况分类未处理
                if (hit.getT() > t && t >= tmin) {
                    // 判断交点在物体内还是物体外
                    Vector3f _normal = normal;
                    bool isInside = false;
                    if (Vector3f::dot(normal, ray.getDirection()) > 0) {
                        _normal = -normal;
                        isInside = true;
                    }
                    // 更新hit
                    hit.set(t, material, _normal.normalized(), isInside);
                    return true;
                }
            }
        }
        return false;
    }


    Vector3f normal;
protected:
    Vector3f _a;
    Vector3f _b;
    Vector3f _c;

};

#endif //TRIANGLE_H
