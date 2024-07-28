#ifndef GROUP_H
#define GROUP_H


#include "object3d.cuh"
#include "mesh.cuh"
#include "ray.cuh"
#include "hit.cuh"


/**
 * Container of objects based on vector
 * @author Jason Fu
 *
 */
class Group : public Object3D {

public:

    __device__ __host__ inline explicit Group(int num_objects) {
        size = num_objects;
        top = -1;
        objects = new Object3D *[size];
    }

    __device__ __host__ inline ~Group() override {
        for (int i = 0; i < size; i++)
            delete objects[i];
        delete[] objects;
    }

    __device__  inline bool intersect(const Ray &r, Hit &h, float tmin) override {
        bool inter = false;
        for (int i = 0; i < getGroupSize(); ++i) {
            inter |= objects[i]->intersect(r, h, tmin);
        }
        return inter;
    }

    __device__ __host__ inline void addObject(Object3D *obj) {
        assert(top < size);
        objects[++top] = obj;
    }

    __device__ __host__ inline Object3D *getObject(int index) {
        return objects[index];
    }

    __device__ __host__ inline int getGroupSize() const {
        return size;
    }

private:
    Object3D **objects;
    int size;
    int top;
};

#endif
	
