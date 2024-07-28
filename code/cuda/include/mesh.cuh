/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */

#ifndef MESH_H
#define MESH_H

#include "object3d.cuh"
#include "triangle.cuh"
#include "boundingBox.cuh"
#include "Vector3f.cuh"


class Mesh : public Object3D {
public:

    /**
    * Get the lower bound of coordinates along the given axis
    * @param axis 0(x), 1(y), 2(z)
    * @author Jason Fu
    *
    */
    __device__ __host__ inline float getLowerBound(int axis) override {
        if (axis == Ray::X_AXIS) {
            return bbox->x0;
        } else if (axis == Ray::Y_AXIS) {
            return bbox->y0;
        } else {
            return bbox->z0;
        }
    }


    /**
    * Get the upper bound of coordinates along the given axis
    * @param axis 0(x), 1(y), 2(z)
    * @author Jason Fu
    *
    */
    __device__ __host__ inline float getUpperBound(int axis) override {
        if (axis == Ray::X_AXIS) {
            return bbox->x1;
        } else if (axis == Ray::Y_AXIS) {
            return bbox->y1;
        } else {
            return bbox->z1;
        }
    }

public:

    __device__ __host__ explicit Mesh(Triangle *&trigs, int _size);

    __device__ __host__ ~Mesh() override {
        delete[] triangles;
        delete bbox;
    }

    struct TriangleIndex {
        TriangleIndex() {
            x[0] = 0;
            x[1] = 0;
            x[2] = 0;
        }

        int &operator[](const int i) { return x[i]; }

        // By Computer Graphics convention, counterclockwise winding is front face
        int x[3]{};
    };

    __device__ bool intersect(const Ray &r, Hit &h, float tmin) override;

    __device__ __host__ int getSize() const{
        return size;
    }

    __device__ __host__ BoundingBox* getBoundingBox() const{
        return bbox;
    }

    __device__ __host__ Triangle* getTriangles() const{
        return triangles;
    }

private:
    int size;
    BoundingBox *bbox;
    Triangle *triangles;
};

#endif
