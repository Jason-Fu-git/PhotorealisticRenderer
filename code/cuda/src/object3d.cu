//
// Created by Jason Fu on 24-7-12.
//

#include "object3d.cuh"
#include "group.cuh"
#include "plane.cuh"
#include "sphere.cuh"
#include "transform.cuh"
#include "triangle.cuh"
#include "mesh.cuh"

__device__ __host__ Object3D::Object3D() : material(nullptr), materialIndex(-1) {}

__device__ __host__ Object3D::Object3D(Material *material, int materialIndex) {
    this->material = material;
    this->materialIndex = materialIndex;
}

__device__ int2
Object3D::textureMap(float objectX, float objectY, float objectZ, int textureWidth, int textureHeight) {
    return {-1, -1};
}

__device__ __host__ float Object3D::getLowerBound(int axis) {
    return 0;
}

__device__ __host__ float Object3D::getUpperBound(int axis) {
    return 0;
}

// Kernel functions for creating objects on device

__global__ void
createPlaneOnDevice(Object3D **object, Material **materials, int materialIndex, Vector3f normal, float d,
                    float scale) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *object = new Plane(normal, d, materials[materialIndex], -1, scale);
    }
}

__global__ void
createSphereOnDevice(Object3D **object, Material **materials, int materialIndex, Vector3f center, float radius,
                     float theta_offset, float phi_offset) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *object = new Sphere(center, radius, materials[materialIndex], -1, theta_offset, phi_offset);
    }
}

__global__ void
createTriangleOnDevice(Object3D **object, Material **materials, int materialIndex, Vector3f a, Vector3f b,
                       Vector3f c, Vector3f normal, float au, float av, float bu, float bv, float cu, float cv,
                       Vector3f an, Vector3f bn, Vector3f cn) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        auto triangle = new Triangle(a, b, c, materials[materialIndex], -1);
        triangle->setVertexNormals(an, bn, cn);
        triangle->setTextureUV(au, av, bu, bv, cu, cv);
        triangle->normal = normal;
        *object = triangle;
    }
}

__global__ void
createMeshOnDevice(Object3D **object, Material **materials, Triangle *trigs, int trigSize) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        //set materials for each triangle
        for (int i = 0; i < trigSize; i++) {
            trigs[i].setMaterial(materials[trigs[i].getMaterialIndex()]);
        }
        *object = new Mesh(trigs, trigSize);
    }
}

__global__ void createGroupOnDevice(Group **object, int groupSize) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *object = new Group(groupSize);
    }
}

__global__ void createTransformOnDevice(Object3D **object, Matrix4f m, Object3D **obj) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *object = new Transform(m.inverse(), *obj);
    }
}

__global__ void freeGroupOnDevice(Group **group) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        delete *group;
    }
}

__global__ void addObjectToGroup(Object3D **object, Group **group) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        (*group)->addObject(*object);
    }
}




