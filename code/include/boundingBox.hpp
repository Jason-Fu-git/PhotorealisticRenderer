//
// Created by Jason Fu on 24-5-30.
//

#ifndef FINALPROJECT_BOUNDINGBOX_HPP
#define FINALPROJECT_BOUNDINGBOX_HPP

#include <cfloat>
#include "ray.hpp"
#include "plane.hpp"

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
        printf("BoundingBox(%f, %f, %f, %f, %f, %f)\n", x0, x1, y0, y1, z0, z1);
        x0Plane = new Plane({1, 0, 0}, x0, nullptr);
        x1Plane = new Plane({1, 0, 0}, x1, nullptr);
        y0Plane = new Plane({0, 1, 0}, y0, nullptr);
        y1Plane = new Plane({0, 1, 0}, y1, nullptr);
        z0Plane = new Plane({0, 0, 1}, z0, nullptr);
        z1Plane = new Plane({0, 0, 1}, z1, nullptr);
    };

    ~BoundingBox() = default;

    // A variation of Woo's algorithm
    bool isIntersect(const Ray &ray) const {
        auto origin = ray.getOrigin();
        auto direction = ray.getDirection();
        Plane *x, *y, *z;
        // x
        if (origin[0] <= x0)
            x = x0Plane;
        else if (origin[0] >= x1)
            x = x1Plane;
        else {
            if (direction[0] < 0)
                x = x0Plane;
            else
                x = x1Plane;
        }
        // y
        if (origin[1] <= y0)
            y = y0Plane;
        else if (origin[1] >= y1)
            y = y1Plane;
        else {
            if (direction[1] < 0)
                y = y0Plane;
            else
                y = y1Plane;
        }
        // z
        if (origin[2] <= z0)
            z = z0Plane;
        else if (origin[2] >= z1)
            z = z1Plane;
        else {
            if (direction[2] < 0)
                z = z0Plane;
            else
                z = z1Plane;
        }
        // hit
        Hit hit;
        if (x->intersect(ray, hit, 0)) {
            auto p = ray.pointAtParameter(hit.getT());
            if(p.y() >= y0 && p.y() <= y1 && p.z() >= z0 && p.z() <= z1)
                return true;
            hit.reset();
        }

        if (y->intersect(ray, hit, 0)) {
            auto p = ray.pointAtParameter(hit.getT());
            if(p.x() >= x0 && p.x() <= x1 && p.z() >= z0 && p.z() <= z1)
                return true;
            hit.reset();
        }

        if (z->intersect(ray, hit, 0)) {
            auto p = ray.pointAtParameter(hit.getT());
            if(p.x() >= x0 && p.x() <= x1 && p.y() >= y0 && p.y() <= y1)
                return true;
        }
        return false;
    }

    float x0, x1, y0, y1, z0, z1;
    Plane *x0Plane, *x1Plane, *y0Plane, *y1Plane, *z0Plane, *z1Plane;
};

#endif //FINALPROJECT_BOUNDINGBOX_HPP
