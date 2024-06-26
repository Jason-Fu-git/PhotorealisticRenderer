/**
 * Bounding Box for an object.
 * @author Jason Fu
 */

#ifndef FINALPROJECT_BOUNDINGBOX_HPP
#define FINALPROJECT_BOUNDINGBOX_HPP

#include <cfloat>
#include "ray.hpp"
#include "plane.hpp"

#define MINUS_T (-0.01)

/**
 * Bounding Box for an object.
 * Used to fasten the ray-object intersection.
 * @author Jason Fu
 */
class BoundingBox {
public:
    // [x0, x1] * [y0, y1] * [z0, z1]
    BoundingBox(float _x0, float _x1, float _y0, float _y1, float _z0, float _z1)
            : x0(_x0), x1(_x1), y0(_y0), y1(_y1), z0(_z0), z1(_z1) {
    };

    ~BoundingBox() = default;

    // iterate through six planes
    // If intersect, save the min and max t value.
    // The original value of tmin and tmax should be set externally
    bool isIntersect(const Ray &ray, float &tmin, float &tmax) const {
        auto origin = ray.getOrigin();
        auto direction = ray.getDirection();
        bool intersect = false;
        // x0 plane
        float t = ray.parameterAtPoint(x0, Ray::X_AXIS);
        auto p = ray.pointAtParameter(t);
        if (p.y() >= y0 && p.y() <= y1 && p.z() >= z0 && p.z() <= z1) {
            intersect = true;
            tmin = std::min(tmin, t);
            tmax = std::max(tmax, t);
        }
        // x1 plane
        t = ray.parameterAtPoint(x1, Ray::X_AXIS);
        p = ray.pointAtParameter(t);
        if (p.y() >= y0 && p.y() <= y1 && p.z() >= z0 && p.z() <= z1) {
            intersect = true;
            tmin = std::min(tmin, t);
            tmax = std::max(tmax, t);
        }
        // y0 plane
        t = ray.parameterAtPoint(y0, Ray::Y_AXIS);
        p = ray.pointAtParameter(t);
        if (p.x() >= x0 && p.x() <= x1 && p.z() >= z0 && p.z() <= z1) {
            intersect = true;
            tmin = std::min(tmin, t);
            tmax = std::max(tmax, t);
        }

        // y1 plane
        t = ray.parameterAtPoint(y1, Ray::Y_AXIS);
        p = ray.pointAtParameter(t);
        if (p.x() >= x0 && p.x() <= x1 && p.z() >= z0 && p.z() <= z1) {
            intersect = true;
            tmin = std::min(tmin, t);
            tmax = std::max(tmax, t);
        }
        // z0 plane
        t = ray.parameterAtPoint(z0, Ray::Z_AXIS);
        p = ray.pointAtParameter(t);
        if (p.x() >= x0 && p.x() <= x1 && p.y() >= y0 && p.y() <= y1) {
            intersect = true;
            tmin = std::min(tmin, t);
            tmax = std::max(tmax, t);
        }

        // z1 plane
        t = ray.parameterAtPoint(z1, Ray::Z_AXIS);
        p = ray.pointAtParameter(t);
        if (p.x() >= x0 && p.x() <= x1 && p.y() >= y0 && p.y() <= y1) {
            intersect = true;
            tmin = std::min(tmin, t);
            tmax = std::max(tmax, t);
        }

        // internal origin?
        if (origin.x() >= x0 && origin.x() <= x1 && origin.y() >= y0 && origin.y() <= y1 && origin.z() >= z0 &&
            origin.z() <= z1) {
            intersect = true;
            tmin = MINUS_T;
        }

        return intersect;
    }

    float x0, x1, y0, y1, z0, z1;
};

#endif //FINALPROJECT_BOUNDINGBOX_HPP
