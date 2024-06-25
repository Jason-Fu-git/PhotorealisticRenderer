#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "object3d.hpp"
#include <vecmath.h>
#include <cmath>
#include <iostream>

using namespace std;
extern long long COUNT;


/**
 *
 * @brief Triangle class
 * @var _a : Vertex A
 * @var _b : Vertex B
 * @var _c : Vertex C
 * @var normal : Normal vector of the triangle
 * @author Jason Fu
 *
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
        au = 100;
        av = 100;
        bu = 100;
        bv = 100;
        cu = 100;
        cv = 100;
//        printf("Triangle created %f %f %f\n", a[0], a[1], a[2]);
    }

    bool intersect(const Ray &ray, Hit &hit, float tmin) override {
        ++COUNT;
        float original_length = ray.getDirection().length();
        // 采用重心坐标
        Vector3f E1 = _a - _b;
        Vector3f E2 = _a - _c;
        Vector3f S = _a - ray.getOrigin();
        float det1 = Matrix3f(ray.getDirection().normalized(), E1, E2).determinant();
        if (det1 != 0) // valid solution
        {
//            printf("det1 = %f\n", det1);
            float t = Matrix3f(S, E1, E2).determinant() / det1 / original_length;
            float beta = Matrix3f(ray.getDirection().normalized(), S, E2).determinant() / det1;
            float gamma = Matrix3f(ray.getDirection().normalized(), E1, S).determinant() / det1;

            if (t > 0 && beta >= 0 && gamma >= 0 && beta + gamma <= 1) { // has intersection
                // 注：内外情况分类未处理
                if (hit.getT() > t && t >= tmin) {

                    // update texture data
                    if (au != 100) {
                        float alpha = 1 - beta - gamma;
                        u = mod(alpha * au + beta * bu + gamma * cu, 1);
                        v = mod(alpha * av + beta * bv + gamma * cv, 1);

                        // judge whether transmitted
                        // if the alpha on the texture is less than the given threshold, this means it transmits
                        if (material->isTransmit(u, v))
                            return false;
                    }

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

    void setTextureUV(float _au, float _av, float _bu, float _bv, float _cu, float _cv) {
        au = _au;
        av = _av;
        bu = _bu;
        bv = _bv;
        cu = _cu;
        cv = _cv;
    }

    pair<int, int>
    textureMap(float objectX, float objectY, float objectZ, int textureWidth, int textureHeight) override {
        return {
                std::floor(u * textureWidth),
                std::floor(v * textureHeight)
        };
    }


    float getLowerBound(int axis) override {
        return std::min(std::min(_a[axis], _b[axis]), _c[axis]);
    }

    float getUpperBound(int axis) override {
        return std::max(std::max(_a[axis], _b[axis]), _c[axis]);
    }


    Vector3f normal;

    // texture map
    float au;
    float av;
    float bu;
    float bv;
    float cu;
    float cv;

    // last intersection's texture coordinate
    float u;
    float v;

protected:
    Vector3f _a;
    Vector3f _b;
    Vector3f _c;



};

#endif //TRIANGLE_H
