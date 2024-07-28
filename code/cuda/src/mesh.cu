#include "mesh.cuh"

/**
 * Use boundingBox to fasten intersection.
 * @author Jason Fu
 *
 */
__device__ bool Mesh::intersect(const Ray &r, Hit &h, float tmin) {
    float t_min = tmin;
    float t_max = 1e38;
    if (bbox->isIntersect(r, t_min, t_max)) {
        bool res = false;
        for (int i = 0; i < size; i++) {
            res |= triangles[i].intersect(r, h, t_min);
        }
    }
    return false;
}

/**
 * Copy constructor
 * @author Jason Fu
 *
 */
__device__ __host__ Mesh::Mesh(Triangle *&trigs, int _size) {
    size = _size;
    triangles = new Triangle[_size];
    for (int ii = 0; ii < _size; ii++) {
        triangles[ii] = trigs[ii];
    }
    // create bounding box
    float x0 = FLT_MAX, x1 = -FLT_MAX, y0 = FLT_MAX, y1 = -FLT_MAX, z0 = FLT_MAX, z1 = -FLT_MAX;
    for (int ii = 0; ii < _size; ii++) {
        auto triangle = triangles[ii];
        x0 = min(x0, triangle.getLowerBound(Ray::X_AXIS));
        x1 = max(x1, triangle.getUpperBound(Ray::X_AXIS));

        y0 = min(y0, triangle.getLowerBound(Ray::Y_AXIS));
        y1 = max(y1, triangle.getUpperBound(Ray::Y_AXIS));

        z0 = min(z0, triangle.getLowerBound(Ray::Z_AXIS));
        z1 = max(z1, triangle.getUpperBound(Ray::Z_AXIS));
    }
    bbox = new BoundingBox(x0 - 0.01, x1 + 0.01,
                           y0 - 0.01, y1 + 0.01,
                           z0 - 0.01, z1 + 0.01);
    // construct other fields
    printf("Mesh loaded %f %f %f %f %f %f\n", x0, x1, y0, y1, z0, z1);
}

