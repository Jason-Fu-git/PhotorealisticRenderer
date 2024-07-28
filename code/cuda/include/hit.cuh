//
// Created by Jason Fu on 24-7-12.
//

#ifndef FINALPROJECTCUDA_HIT_CUH
#define FINALPROJECTCUDA_HIT_CUH

#include "Vector3f.cuh"
#include "ray.cuh"

class Object3D;

/**
 * @var t: distance
 * @var normal: surface normal at the hit point， note : should be normalized!
 * @var material: pointer to the material of the surface that is hit
 */
class Hit {
public:

    // constructors
    __device__ __host__ inline Hit() {
        object = nullptr;
        t = 1e38;
        is_inside = false;
    }


    __device__ __host__ inline Hit(float _t, Object3D *o, const Vector3f &n) {
        t = _t;
        object = o;
        normal = n;
        is_inside = false;
    }

    __device__ __host__ inline Hit(const Hit &h) {
        t = h.t;
        object = h.object;
        normal = h.normal;
        is_inside = h.is_inside;
    }

    // destructor
    __device__ __host__ ~Hit() = default;

    __device__ __host__ inline float getT() const {
        return t;
    }

    __device__ __host__ inline Object3D *getObject() const {
        return object;
    }

    __device__ __host__ inline const Vector3f &getNormal() const {
        return normal;
    }

    __device__ __host__ inline bool isInside() const {
        return is_inside;
    }


    __device__ __host__ inline void set(float _t, Object3D *o, const Vector3f &n, bool _is_inside){
        t = _t;
        object = o;
        normal = n;
        is_inside = _is_inside;
    }

    __device__ __host__ inline void reset(){
         t = 1e38;
    }

private:
    float t;
    Vector3f normal;
    Object3D *object;
    bool is_inside;

};

#endif //FINALPROJECTCUDA_HIT_CUH
