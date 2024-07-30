#include "mesh.cuh"
#include <vector>

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
            res |= triangles[i].intersectTriangle(r, h, t_min);
        }
    }
    return false;
}

/**
 * Move constructor
 * @author Jason Fu
 *
 */
__device__ __host__ Mesh::Mesh(Triangle *trigs, int _size) {
    size = _size;
    triangles = trigs;
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

__host__ Mesh::Mesh(const char *filename, int materialIndex) : Object3D(nullptr, materialIndex) {
    std::vector<Vector3f> v;
    std::vector<TriangleIndex> t;
    std::vector<Vector3f> n;

    // Optional: Use tiny obj loader to replace this simple one.
    float x0 = FLT_MAX, x1 = -FLT_MAX, y0 = FLT_MAX, y1 = -FLT_MAX, z0 = FLT_MAX, z1 = -FLT_MAX;
    FILE *file = fopen(filename, "r");
    assert(file != nullptr);
    char token[1024];
    while (true) {
        if (getToken(file, token) == 0) {
            break;
        }
        if (token[0] == '#') {
            continue;
        }
        if (strcmp(token, "v") == 0) {
            Vector3f vec = readVector3f(file);
            x0 = std::min(x0, vec[0]);
            x1 = std::max(x1, vec[0]);
            y0 = std::min(y0, vec[1]);
            y1 = std::max(y1, vec[1]);
            z0 = std::min(z0, vec[2]);
            z1 = std::max(z1, vec[2]);
            v.push_back(vec);
        } else if (strcmp(token, "f") == 0) {
            TriangleIndex trig;
            for (int ii = 0; ii < 3; ii++) {
                trig[ii] = readInt(file) - 1;
            }
            t.push_back(trig);
        } else {
            continue;
        }
    }
    // compute normal
    n.resize(t.size());
    for (int triId = 0; triId < (int) t.size(); ++triId) {
        TriangleIndex &triIndex = t[triId];
        Vector3f a = v[triIndex[1]] - v[triIndex[0]];
        Vector3f b = v[triIndex[2]] - v[triIndex[0]];
        b = Vector3f::cross(a, b);
        n[triId] = b / b.length();
    }
    // construct the triangles
    size = t.size();
    triangles = new Triangle[size];
    for (int triId = 0; triId < (int) t.size(); ++triId) {
        TriangleIndex &triIndex = t[triId];
        Triangle triangle(v[triIndex[0]],
                          v[triIndex[1]], v[triIndex[2]], material, materialIndex);
        triangle.normal = n[triId];
        triangles[triId] = triangle;
    }
    // construct the bounding box
    bbox = new BoundingBox(x0 - 0.01, x1 + 0.01,
                           y0 - 0.01, y1 + 0.01,
                           z0 - 0.01, z1 + 0.01);
    printf("Bounding box %f %f %f %f %f %f\n", x0, x1, y0, y1, z0, z1);
    printf("Mesh %s loaded\n", filename);
    fflush(stdout);
}

__global__ void meshIntersectKernel(Triangle *triangles, const Ray &ray, int n, float tmin, float *t) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    printf("index %d\n", index);
    if (index < n) {
        Hit h;
        triangles[index].intersectTriangle(ray, h, tmin);
        t[index] = h.getT();
    }
}
