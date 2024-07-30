//
// Created by Jason Fu on 24-7-14.
//

#include "light.cuh"

__global__ void createDirectionalLightOnDevice(Light **light, Vector3f d, Vector3f c) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // single pointer case
        *light = new DirectionalLight(d, c);
    }
}

__global__ void createPointLightOnDevice(Light **light, Vector3f p, Vector3f c) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *light = new PointLight(p, c);
    }
}

__global__ void createSphereLightOnDevice(Light **light, Vector3f p, float r, Vector3f c) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *light = new SphereLight(p, r, c);
    }
}

__global__ void freeLightOnDevice(Light **light) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *light;
    }
}
