/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */

#ifndef MESH_H
#define MESH_H

#include <vector>
#include "object3d.hpp"
#include "triangle.hpp"
#include "BSPTree.hpp"
#include "Vector2f.h"
#include "Vector3f.h"


class Mesh : public Object3D {

public:
    Mesh(const char *filename, Material *m);

    ~Mesh() override {
        for (auto obj: triangles) {
            delete obj;
        }
        triangles.clear();
        delete bspTree;
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

    std::vector<Vector3f> v;
    std::vector<TriangleIndex> t;
    std::vector<Vector3f> n;
    std::vector<Object3D *> triangles;

    bool intersect(const Ray &r, Hit &h, float tmin) override;

private:

    // Normal can be used for light estimation
    void computeNormal();

    // construct triangles
    void constructTriangles();

    // construct BSP tree
    void constructBSPTree();

    BSPTree *bspTree;
};

#endif
