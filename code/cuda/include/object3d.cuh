//
// Created by Jason Fu on 24-7-12.
//

#ifndef FINALPROJECTCUDA_OBJECT3D_CUH
#define FINALPROJECTCUDA_OBJECT3D_CUH

#include "ray.cuh"
#include "hit.cuh"
#include "material.cuh"
#include "Matrix4f.cuh"
#include <utility>


#define DISTURBANCE 0.01f

class Group;

class Triangle;

// Base class for all 3d entities.
/**
 * @var material: Pointer to the material of the object.
 */
class Object3D {
public:
    __device__ __host__ Object3D();

    // NOTE : material should be deleted externally!
    __device__ __host__ virtual ~Object3D() {}

    __device__ __host__ explicit Object3D(Material *material, int materialIndex);

    // Intersect Ray with this object. If hit, store information in hit structure.
    __device__ virtual bool intersect(const Ray &r, Hit &h, float tmin) = 0;

    /**
     * Calculate the texture coordinate of a point on the object.
     * In brief, the map (x,y) -> (u,v)
     * @param objectX coordinate x on the object
     * @param objectY coordinate y on the object
     * @param objectZ coordinate z on the object
     * @param textureWidth the width of the texture
     * @param textureHeight the height of the texture
     * @return a point on the texture
     */
    __device__ virtual int2
    textureMap(float objectX, float objectY, float objectZ, int textureWidth, int textureHeight);

    __device__ __host__ virtual float getLowerBound(int axis);

    __device__ __host__ virtual float getUpperBound(int axis);

    __device__ __host__ int getMaterialIndex() const {
        return materialIndex;
    }

    __device__ __host__ Material *getMaterial() const {
        return material;
    }

    __device__ __host__ void setMaterial(Material *pMaterial) {
        Object3D::material = pMaterial;
    }

protected:
    Material *material;
    int materialIndex;
};

__global__ void
createPlaneOnDevice(Object3D **object, Material **materials, int materialIndex, Vector3f normal, float d,
                    float scale = 1.0f);

__global__ void
createSphereOnDevice(Object3D **object, Material **materials, int materialIndex, Vector3f center, float radius,
                     float theta_offset, float phi_offset);

__global__ void
createTriangleOnDevice(Object3D **object, Material **materials, int materialIndex,
                       Vector3f a, Vector3f b, Vector3f c,
                       float au, float av, float bu, float bv, float cu, float cv,
                       Vector3f an, Vector3f bn, Vector3f cn);

__global__ void
createTransformOnDevice(Object3D **object, Matrix4f m, Object3D **obj);

__global__ void
createMeshOnDevice(Object3D **object, Triangle **trigs, int trigSize);

__global__ void
createGroupOnDevice(Group **object, int groupSize);

__global__ void
addObjectToGroup(Object3D** object, Group **group);

__global__ void
freeGroupOnDevice(Group **group);

#endif //FINALPROJECTCUDA_OBJECT3D_CUH
