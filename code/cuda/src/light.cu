//
// Created by Jason Fu on 24-7-14.
//

#include "light.cuh"

__global__ void createDirectionalLightOnDevice(Light **lights, const Vector3f &d, const Vector3f &c) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // single pointer case
        *lights = new DirectionalLight(d, c);
    }
}

__global__ void createPointLightOnDevice(Light **lights, const Vector3f &p, const Vector3f &c) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *lights = new PointLight(p, c);
    }
}

__global__ void createSphereLightOnDevice(Light **lights, const Vector3f &p, float r, const Vector3f &c) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *lights = new SphereLight(p, r, c);
    }
}

__global__ void freeLightOnDevice(Light **light) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *light;
    }
}
