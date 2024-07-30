#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "object3d.cuh"
#include "Vector3f.cuh"
#include "Matrix3f.cuh"
#include <cmath>
#include <iostream>

using namespace std;


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
    __device__ __host__ inline Triangle() : Triangle(Vector3f::ZERO(), Vector3f::ZERO(), Vector3f::ZERO(), nullptr,
                                                     -1) {}

    __device__ __host__ inline ~Triangle() override {}

    // a b c are three vertex positions of the triangle
    __device__ __host__ inline Triangle(const Vector3f &a, const Vector3f &b, const Vector3f &c, Material *m,
                                        int materialIndex)
            : Object3D(m, materialIndex) {
        _a = a;
        _b = b;
        _c = c;
        au = 100;
        av = 100;
        bu = 100;
        bv = 100;
        cu = 100;
        cv = 100;
        _an = Vector3f::ZERO();
        _bn = Vector3f::ZERO();
        _cn = Vector3f::ZERO();
    }

    __device__ inline bool intersectTriangle(const Ray &ray, Hit &hit, float tmin) {
        float original_length = ray.getDirection().length();
        // Using areal coordinates
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
                if (hit.getT() > t && t >= tmin) {

                    float alpha = 1 - beta - gamma;

                    // update texture data
                    if (au != 100) {
                        u = mod(alpha * au + beta * bu + gamma * cu, 1);
                        v = mod(alpha * av + beta * bv + gamma * cv, 1);

                        // judge whether transmitted
                        // if the alpha on the texture is less than the given threshold, this means it transmits
                        if (material->isTransmit(u, v))
                            return false;
                    }

                    // Judge whether the intersection is inside the triangle
                    Vector3f _normal = normal;
                    bool isInside = false;

                    // if vertex normal is specified, use normal interpolation
                    if (_an != Vector3f::ZERO()) {
                        _normal = (_an * alpha + _bn * beta + _cn * gamma).normalized();
                    }

                    if (Vector3f::dot(normal, ray.getDirection()) > 0) {
                        _normal = -normal;
                        isInside = true;
                    }

                    // 更新hit
                    hit.set(t, this, _normal.normalized(), isInside);
                    return true;
                }
            }
        }
        return false;
    }


    __device__ inline bool intersect(const Ray &ray, Hit &hit, float tmin) override {
        float original_length = ray.getDirection().length();
        // Using areal coordinates
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
                if (hit.getT() > t && t >= tmin) {

                    float alpha = 1 - beta - gamma;

                    // update texture data
                    if (au != 100) {
                        u = mod(alpha * au + beta * bu + gamma * cu, 1);
                        v = mod(alpha * av + beta * bv + gamma * cv, 1);

                        // judge whether transmitted
                        // if the alpha on the texture is less than the given threshold, this means it transmits
                        if (material->isTransmit(u, v))
                            return false;
                    }

                    // Judge whether the intersection is inside the triangle
                    Vector3f _normal = normal;
                    bool isInside = false;

                    // if vertex normal is specified, use normal interpolation
                    if (_an != Vector3f::ZERO()) {
                        _normal = (_an * alpha + _bn * beta + _cn * gamma).normalized();
                    }

                    if (Vector3f::dot(normal, ray.getDirection()) > 0) {
                        _normal = -normal;
                        isInside = true;
                    }

                    // 更新hit
                    hit.set(t, this, _normal.normalized(), isInside);
                    return true;
                }
            }
        }
        return false;
    }

    __device__ __host__ inline void setTextureUV(float _au, float _av, float _bu, float _bv, float _cu, float _cv) {
        au = _au;
        av = _av;
        bu = _bu;
        bv = _bv;
        cu = _cu;
        cv = _cv;
    }

    __device__ __host__  inline void setVertexNormals(const Vector3f &an, const Vector3f &bn, const Vector3f &cn) {
        _an = an.normalized();
        _bn = bn.normalized();
        _cn = cn.normalized();
    }

    __device__ inline int2
    textureMap(float objectX, float objectY, float objectZ, int textureWidth, int textureHeight) override {
        return {
                (int) floorf(u * textureWidth),
                (int) floorf(v * textureHeight)
        };
    }

    __device__ __host__ inline float getLowerBound(int axis) override {
        return fminf(fminf(_a[axis], _b[axis]), _c[axis]);
    }

    __device__ __host__ inline float getUpperBound(int axis) override {
        return fmaxf(fmaxf(_a[axis], _b[axis]), _c[axis]);
    }

    __device__ __host__ Vector3f getA() {
        return _a;
    }

    __device__ __host__ Vector3f getB() {
        return _b;
    }

    __device__ __host__ Vector3f getC() {
        return _c;
    }


    Vector3f normal;

    // texture map
    float au;
    float av;
    float bu;
    float bv;
    float cu;
    float cv;

    // vertex normals
    Vector3f _an;
    Vector3f _bn;
    Vector3f _cn;

    // last intersection's texture coordinate
    float u;
    float v;

protected:
    Vector3f _a;
    Vector3f _b;
    Vector3f _c;


};

#endif //TRIANGLE_H
