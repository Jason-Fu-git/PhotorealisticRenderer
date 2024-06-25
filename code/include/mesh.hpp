/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */

#ifndef MESH_H
#define MESH_H

#include <vector>
#include "object3d.hpp"
#include "triangle.hpp"
#include "boundingBox.hpp"
#include "BSPTree.hpp"
#include "Vector2f.h"
#include "Vector3f.h"


class Mesh : public Object3D {
public:
    float getLowerBound(int axis) override;

    float getUpperBound(int axis) override;

public:
    Mesh(const char *filename, Material *m);

    explicit Mesh(std::vector<Triangle*> &trigs);

    ~Mesh() override {
        for (auto obj: triangles) {
            delete obj;
        }
        triangles.clear();
        delete bspTree;
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

    std::vector<Object3D *> triangles;

    bool intersect(const Ray &r, Hit &h, float tmin) override;

private:


    BSPTree *bspTree;
    BoundingBox *bbox;
};

#endif
