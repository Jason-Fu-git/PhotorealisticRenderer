#ifndef CAMERA_H
#define CAMERA_H

#include "ray.hpp"
#include <vecmath.h>
#include <float.h>
#include <cmath>

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
class Camera {
public:
    Camera(const Vector3f &center, const Vector3f &direction, const Vector3f &up, int imgW, int imgH) {
        this->center = center;
        this->direction = direction.normalized();
        this->horizontal = Vector3f::cross(this->direction, up).normalized();
        this->up = Vector3f::cross(this->horizontal, this->direction);
        this->width = imgW;
        this->height = imgH;
    }

    // Generate rays for each screen-space coordinate
    // Direction has been normalized!
    virtual Ray generateRay(const Vector2f &point) = 0;

    virtual ~Camera() = default;

    int getWidth() const { return width; }

    int getHeight() const { return height; }

protected:
    // Extrinsic parameters
    Vector3f center;
    Vector3f direction;
    Vector3f up;
    Vector3f horizontal;
    // Intrinsic parameters
    int width;
    int height;
};

/**
 * A camera based on the pin-hole model.
 * @author Jason Fu
 */
class PerspectiveCamera : public Camera {

public:
    PerspectiveCamera(const Vector3f &center, const Vector3f &direction,
                      const Vector3f &up, int imgW, int imgH, float angle) : Camera(center, direction, up, imgW, imgH) {
        // angle is in radian.
        fx = imgW / (2.0f * std::tan(angle / 2.0f));
        fy = imgH / (2.0f * std::tan(angle / 2.0f));
    }

    Ray generateRay(const Vector2f &point) override {
        Vector3f dRc = Vector3f((point.x() - width / 2.0f) / fx, (point.y() - height / 2.0f) / fy, 1).normalized();
        Vector3f ORw = center;
        Matrix3f R = Matrix3f(horizontal, -up, direction);
        Vector3f dRw = (R * dRc).normalized();
        return Ray(ORw, dRw);
    }

private:
    float fx;
    float fy;
};

#endif //CAMERA_H
