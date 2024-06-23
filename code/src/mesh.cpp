/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */

#include "mesh.hpp"
#include <cfloat>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>


bool Mesh::intersect(const Ray &r, Hit &h, float tmin) {
    // First intersect with the Bounding Box
    float t_min = tmin;
    float t_max = 1e38;
    if (!bbox->isIntersect(r, t_min, t_max)){
        return false;
    }
    // Use BSP to fasten the intersection process
    bool result = bspTree->intersect(r, h, t_min, t_max);
    return result;
}

Mesh::Mesh(const char *filename, Material *material) : Object3D(material) {
    std::vector<Vector3f> v;
    std::vector<TriangleIndex> t;
    std::vector<Vector3f> n;

    // Optional: Use tiny obj loader to replace this simple one.
    float x0 = FLT_MAX, x1 = -FLT_MAX, y0 = FLT_MAX, y1 = -FLT_MAX, z0 = FLT_MAX, z1 = -FLT_MAX;
    std::ifstream f;
    f.open(filename);
    if (!f.is_open()) {
        std::cout << "Cannot open " << filename << "\n";
        return;
    }
    std::string line;
    std::string vTok("v");
    std::string fTok("f");
    std::string texTok("vt");
    char bslash = '/', space = ' ';
    std::string tok;
    int texID;
    while (true) {
        std::getline(f, line);
        if (f.eof()) {
            break;
        }
        if (line.size() < 3) {
            continue;
        }
        if (line.at(0) == '#') {
            continue;
        }
        std::stringstream ss(line);
        ss >> tok;
        if (tok == vTok) {
            Vector3f vec;
            ss >> vec[0] >> vec[1] >> vec[2];
            x0 = std::min(x0, vec[0]);
            x1 = std::max(x1, vec[0]);
            y0 = std::min(y0, vec[1]);
            y1 = std::max(y1, vec[1]);
            z0 = std::min(z0, vec[2]);
            z1 = std::max(z1, vec[2]);
            v.push_back(vec);
        } else if (tok == fTok) {
            if (line.find(bslash) != std::string::npos) {
                std::replace(line.begin(), line.end(), bslash, space);
                std::stringstream facess(line);
                TriangleIndex trig;
                facess >> tok;
                for (int ii = 0; ii < 3; ii++) {
                    facess >> trig[ii] >> texID;
                    trig[ii]--;
                }
                t.push_back(trig);
            } else {
                TriangleIndex trig;
                for (int ii = 0; ii < 3; ii++) {
                    ss >> trig[ii];
                    trig[ii]--;
                }
                t.push_back(trig);
            }
        } else if (tok == texTok) {
            Vector2f texcoord;
            ss >> texcoord[0];
            ss >> texcoord[1];
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
    for (int triId = 0; triId < (int) t.size(); ++triId) {
        TriangleIndex &triIndex = t[triId];
        auto triangle = new Triangle(v[triIndex[0]],
                                     v[triIndex[1]], v[triIndex[2]], material);
        triangle->normal = n[triId];
        triangles.push_back(triangle);
    }
    // construct the bounding box
    bbox = new BoundingBox(x0 - 0.01, x1 + 0.01,
                           y0 - 0.01, y1 + 0.01,
                           z0 - 0.01, z1 + 0.01);
    // construct other fields
    bspTree = new BSPTree(triangles);
    f.close();
    printf("Mesh %s loaded\n", filename);
}

Mesh::Mesh(vector<Triangle*> &trigs, Material *m) {
    triangles.assign(trigs.begin(), trigs.end());
    material = m;
    // create bounding box
    float x0 = FLT_MAX, x1 = -FLT_MAX, y0 = FLT_MAX, y1 = -FLT_MAX, z0 = FLT_MAX, z1 = -FLT_MAX;
    for(auto triangle : triangles) {
        x0 = min(x0, triangle->getLowerBound(Ray::X_AXIS));
        x1 = max(x1, triangle->getUpperBound(Ray::X_AXIS));

        y0 = min(y0, triangle->getLowerBound(Ray::Y_AXIS));
        y1 = max(y1, triangle->getUpperBound(Ray::Y_AXIS));

        z0 = min(z0, triangle->getLowerBound(Ray::Z_AXIS));
        z1 = max(z1, triangle->getUpperBound(Ray::Z_AXIS));
    }
    bbox = new BoundingBox(x0 - 0.01, x1 + 0.01,
                           y0 - 0.01, y1 + 0.01,
                           z0 - 0.01, z1 + 0.01);
    // construct other fields
    bspTree = new BSPTree(triangles);
    printf("Mesh loaded");
}

