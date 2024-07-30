/**
 * Bounding Box for an object.
 * @author Jason Fu
 */

#ifndef FINALPROJECT_BOUNDINGBOX_HPP
#define FINALPROJECT_BOUNDINGBOX_HPP

#include <cfloat>
#include <cstdio>
#include "ray.cuh"

#define MINUS_T (-0.01)

/**
 * Bounding Box for an object.
 * Used to fasten the ray-object intersection.
 * @author Jason Fu
 */
class BoundingBox {
public:
    // [x0, x1] * [y0, y1] * [z0, z1]
    __device__ __host__ inline BoundingBox(float _x0, float _x1, float _y0, float _y1, float _z0, float _z1)
            : x0(_x0), x1(_x1), y0(_y0), y1(_y1), z0(_z0), z1(_z1) {
    };

    __device__ __host__ inline ~BoundingBox() = default;

    // iterate through six planes
    // If intersect, save the min and max t value.
    // The original value of tmin and tmax should be set externally
    __device__ inline bool isIntersect(const Ray &ray, float &tmin, float &tmax) const {
        auto origin = ray.getOrigin();
        bool intersect = false;
        // x0 plane
        float t = ray.parameterAtPoint(x0, Ray::X_AXIS);
        auto p = ray.pointAtParameter(t);
        if (p._y >= y0 && p._y <= y1 && p._z >= z0 && p._z <= z1) {
            intersect = true;
            tmin = fminf(tmin, t);
            tmax = fmaxf(tmax, t);
        }
        // x1 plane
        t = ray.parameterAtPoint(x1, Ray::X_AXIS);
        p = ray.pointAtParameter(t);
        if (p._y >= y0 && p._y <= y1 && p._z >= z0 && p._z <= z1) {
            intersect = true;
            tmin = fminf(tmin, t);
            tmax = fmaxf(tmax, t);
        }
        // y0 plane
        t = ray.parameterAtPoint(y0, Ray::Y_AXIS);
        p = ray.pointAtParameter(t);
        if (p._x >= x0 && p._x <= x1 && p._z >= z0 && p._z <= z1) {
            intersect = true;
            tmin = fminf(tmin, t);
            tmax = fmaxf(tmax, t);
        }

        // y1 plane
        t = ray.parameterAtPoint(y1, Ray::Y_AXIS);
        p = ray.pointAtParameter(t);
        if (p._x >= x0 && p._x <= x1 && p._z >= z0 && p._z <= z1) {
            intersect = true;
            tmin = fminf(tmin, t);
            tmax = fmaxf(tmax, t);
        }
        // z0 plane
        t = ray.parameterAtPoint(z0, Ray::Z_AXIS);
        p = ray.pointAtParameter(t);
        if (p._x >= x0 && p._x <= x1 && p._y >= y0 && p._y <= y1) {
            intersect = true;
            tmin = fminf(tmin, t);
            tmax = fmaxf(tmax, t);
        }

        // z1 plane
        t = ray.parameterAtPoint(z1, Ray::Z_AXIS);
        p = ray.pointAtParameter(t);
        if (p._x >= x0 && p._x <= x1 && p._y >= y0 && p._y <= y1) {
            intersect = true;
            tmin = fminf(tmin, t);
            tmax = fmaxf(tmax, t);
        }

        // internal origin?
        if (origin._x >= x0 && origin._x <= x1 && origin._y >= y0 && origin._y <= y1 && origin._z >= z0 &&
            origin._z <= z1) {
            intersect = true;
            tmin = MINUS_T;
        }
        return intersect;
    }

    float x0, x1, y0, y1, z0, z1;
};

#endif //FINALPROJECT_BOUNDINGBOX_HPP
