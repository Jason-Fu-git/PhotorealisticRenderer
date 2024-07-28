#ifndef PLANE_H
#define PLANE_H

#include "object3d.cuh"
#include "utils.cuh"
#include "Vector3f.cuh"
#include <cmath>


// function: ax+by+cz=d

/**
 * @brief Plane class, function : _normal.x*x + _normal.y*y + _normal.z*z = _d
 * @var _normal
 * @var _d
 * @author Jason Fu
 */
class Plane : public Object3D {
public:
    __device__ __host__ inline Plane() {
        // default plane : z = 0 (xy plane)
        _normal = Vector3f(0, 0, 1);
        _u = Vector3f(1, 0, 0);
        _v = Vector3f(0, 1, 0);
        origin = Vector3f(0, 0, 0);
        _d = 0;
        scale = 1.0f;
    }

    __device__ __host__ inline Plane(const Vector3f &normal, float d, Material *m, int materialIndex,
                                     float scale = 1.0f) : Object3D(m, materialIndex) {
        _normal = normal.normalized();
        _d = d;
        this->scale = scale;
        // construct an orthonormal basis
        _u = Vector3f::cross((std::fabs(_normal._x) > 0.1 ? Vector3f(0, 1, 0) : Vector3f(1, 0, 0)),
                             _normal).normalized();
        _v = Vector3f::cross(_normal, _u).normalized();
        origin = std::fabs(_normal._x) > 0.1 ? Vector3f(_d / _normal._x, 0, 0) : (
                std::fabs(_normal._y) > 0.1 ? Vector3f(0, _d / _normal._y, 0) : Vector3f(0, 0, _d / _normal._z)
        );
    }

    __device__ __host__ inline ~Plane() override = default;

    __device__ inline bool intersect(const Ray &r, Hit &h, float tmin) override {
        float original_length = r.getDirection().length();
        float a = Vector3f::dot(_normal, r.getDirection().normalized());
        if (a != 0) {
            float t = -(-_d + Vector3f::dot(_normal, r.getOrigin())) / a / original_length;
            if (t > 0 && t > tmin && t < h.getT()) { // valid intersection
                // Determine whether the intersection point is inside the plane
                Vector3f normal = _normal;
                bool isInside = false;
                if (Vector3f::dot(_normal, r.getDirection()) > 0) {
                    normal = -_normal;
                    isInside = true;
                }
                h.set(t, this, normal, isInside);
                return true;
            }
        }
        return false;
    }

    __device__ inline int2
    textureMap(float objectX, float objectY, float objectZ, int textureWidth, int textureHeight) override {
        // compress the image
        float newWidth = textureWidth * scale;
        float newHeight = textureHeight * scale;
        // calculate the coordinate (x, y) on the plane
        Vector3f dir = Vector3f(objectX, objectY, objectZ) - origin;
        float length = dir.length();
        float cos_theta = Vector3f::dot(dir, _u) / length;
        float sin_theta = Vector3f::dot(dir, _v) / length;
        float x = length * cos_theta;
        float y = length * sin_theta;
        // convert
        float u = mod(x, newWidth) / newWidth;
        float v = mod(y, newHeight) / newHeight;
        // convert it to the texture coordinate
        return {(int) floorf(u * textureWidth), (int) floorf(v * textureHeight)};
    }

    __device__ __host__ Vector3f getNormal() const {
        return _normal;
    }

    __device__ __host__ float getD() const {
        return _d;
    }

    __device__ __host__ float getScale() const {
        return scale;
    }

    Vector3f _normal;
protected:
    float _d;
    float scale;
    Vector3f _u, _v;
    Vector3f origin;


};

#endif //PLANE_H
		

