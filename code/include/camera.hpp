#ifndef CAMERA_H
#define CAMERA_H

#include "ray.hpp"
#include "utils.hpp"
#include <vecmath.h>
#include <cfloat>
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
                      const Vector3f &up, int imgW, int imgH, float angle,
                      float _aperture = 0.0f, float _focus_plane=1.0f) : Camera(center, direction, up, imgW, imgH) {
        // angle is in radian.
        aperture = _aperture;
        focus_plane = _focus_plane;
        fx = imgW / (2.0f * std::tan(angle / 2.0f) * focus_plane);
        fy = imgH / (2.0f * std::tan(angle / 2.0f) * focus_plane);
        printf("fx: %f, fy: %f, aperture: %f, focus_plane: %f\n", fx, fy, aperture, focus_plane);
    }

    Ray generateRay(const Vector2f &point) override {
        auto p = randomPointInCircle(aperture);
        Vector3f focusPoint = Vector3f((point.x() - width / 2.0f) / fx, (point.y() - height / 2.0f) / fy,  focus_plane);
        Vector3f ORc = Vector3f(p.first, p.second, 0);
        Vector3f dRc = (focusPoint - ORc).normalized();
        // to the world space
        Vector3f ORw = center + horizontal * p.first - up * p.second;
        Matrix3f R = Matrix3f(horizontal, -up, direction);
        Vector3f dRw = (R * dRc).normalized();
//        printf("dRw: %f, %f, %f, dRc: %f %f %f\n", dRw.x(), dRw.y(), dRw.z(), dRc.x(), dRc.y(), dRc.z());
        return Ray(ORw, dRw);
    }

private:
    float fx;
    float fy;
    float aperture;
    float focus_plane;
};

#endif //CAMERA_H
