//
// Created by Jason Fu on 24-7-12.
//

#include "camera.cuh"

__global__ void createPerspectiveCameraOnDevice(Camera **camera, Vector3f center, Vector3f direction,
                                                Vector3f up, int imgW, int imgH, float angle,
                                                float _aperture, float _focus_plane) {

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*camera) = new PerspectiveCamera(center, direction, up, imgW, imgH, angle, _aperture, _focus_plane);
    }
}

__global__ void freeCameraOnDevice(Camera **camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *camera;
    }
}
