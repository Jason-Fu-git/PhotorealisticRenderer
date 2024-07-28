//
// Created by Jason Fu on 24-7-12.
//

#ifndef FINALPROJECTCUDA_CAMERA_CUH
#define FINALPROJECTCUDA_CAMERA_CUH

#include "ray.cuh"
#include "utils.cuh"
#include "Vector3f.cuh"
#include "Matrix3f.cuh"
#include <cfloat>
#include <cmath>

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
class Camera {
public:

    __device__ __host__ inline Camera(const Vector3f &center, const Vector3f &direction, const Vector3f &up, int imgW,
                                      int imgH) {
        this->center = center;
        this->direction = direction.normalized();
        this->horizontal = Vector3f::cross(this->direction, up).normalized();
        this->up = Vector3f::cross(this->horizontal, this->direction);
        this->width = imgW;
        this->height = imgH;
    }

    // Generate rays for each screen-space coordinate
    // Direction has been normalized!
    __device__ inline virtual Ray generateRay(const float2 &point, curandState *state) = 0;

    __device__ __host__ inline virtual ~Camera() {}

    __device__ __host__ inline int getWidth() const { return width; }

    __device__ __host__ inline int getHeight() const { return height; }

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

    __device__ __host__ inline ~PerspectiveCamera() override {}

    __device__ __host__ inline PerspectiveCamera(const Vector3f &center, const Vector3f &direction,
                                                 const Vector3f &up, int imgW, int imgH, float angle,
                                                 float _aperture = 0.0f, float _focus_plane = 1.0f) : Camera(center,
                                                                                                             direction,
                                                                                                             up, imgW,
                                                                                                             imgH) {
        // angle is in radian.
        aperture = _aperture;
        focus_plane = _focus_plane;
        // unit num pixels in camera space
        fx = imgW / (2.0f * tanf(angle / 2.0f) * focus_plane);
        fy = imgH / (2.0f * tanf(angle / 2.0f) * focus_plane);
        printf("fx: %f, fy: %f, aperture: %f, focus_plane: %f\n", fx, fy, aperture, focus_plane);
    }

    __device__ inline Ray generateRay(const float2 &point, curandState *state) override {
        // randomly generate a ray in the aperture, while leave the focus point unchanged
        auto p = randomPointInCircle(aperture, state);
        Vector3f focusPoint = Vector3f((point.x - width / 2.0f) / fx, (point.y - height / 2.0f) / fy, focus_plane);
        Vector3f ORc = Vector3f(p.x, p.y, 0);
        Vector3f dRc = (focusPoint - ORc).normalized();
        // to the world space
        Vector3f ORw = center + horizontal * p.x - up * p.y;
        Matrix3f R = Matrix3f(horizontal, -up, direction);
        Vector3f dRw = (R * dRc).normalized();
        return Ray(ORw, dRw);
    }

private:
    float fx;
    float fy;
    float aperture;
    float focus_plane;
};

__global__ void createPerspectiveCameraOnDevice(Camera **camera, Vector3f center, Vector3f direction,
                                                Vector3f up, int imgW, int imgH, float angle,
                                                float _aperture = 0.0f, float _focus_plane = 1.0f);

__global__ void freeCameraOnDevice(Camera **camera);

#endif //FINALPROJECTCUDA_CAMERA_CUH
