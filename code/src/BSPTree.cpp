//
// Created by Jason Fu on 24-5-30.
//
#include "BSPTree.hpp"
#include "ray.hpp"
#include "object3d.hpp"
#include <algorithm>

BSPTree::BSPTree(std::vector<Object3D *> &objects) {
    root = construct(objects, AXIS_X);
}

BSPTree::~BSPTree() {
    delete root;
}

bool BSPTree::intersect(const Ray &r, Hit &h, float tmin) {
    return intersect(root, r, h, tmin);
}

// left : lb <= pivot
// right : ub > pivot
BSPTree::Node *BSPTree::construct(std::vector<Object3D *> &objects, int axis) {
    int n = objects.size();
    if (n <= LEAF_SIZE) {
        // stop recursion
        if (n <= 0)
            return nullptr;
        // otherwise, create a leaf node
        Node *leaf = new Node(axis, 0, n);
        leaf->objects = new Object3D *[n];
        for (int i = 0; i < n; i++) {
            leaf->objects[i] = objects[i];
        }
        return leaf;
    }
    // otherwise, create a non-leaf node
    int median = n / 2;
    // select the pivot
    std::nth_element(objects.begin(), objects.begin() + median, objects.end(),
                     [axis](Object3D *a, Object3D *b) {
                         return a->getLowerBound(axis) < b->getLowerBound(axis);
                     });
    auto pivot = objects[median]->getLowerBound(axis);
    // split
    std::vector<Object3D *> left, right;
    for (auto &obj: objects) {
        if (obj->getLowerBound(axis) <= pivot) {
            left.push_back(obj);
        }
        if (obj->getUpperBound(axis) > pivot) {
            right.push_back(obj);
        }
    }
    if (left.size() == n || right.size() == n) {
        // if all objects are in the same side, create a leaf node
        Node *leaf = new Node(axis, 0, n);
        leaf->objects = new Object3D *[n];
        for (int i = 0; i < n; i++) {
            leaf->objects[i] = objects[i];
        }
        return leaf;
    }
    // construct
    Node *lc = construct(left, (axis + 1) % 3);
    Node *rc = construct(right, (axis + 1) % 3);
    Node *node = new Node(axis, pivot, 0);
    node->lc = lc;
    node->rc = rc;
    return node;
}

bool BSPTree::intersect(BSPTree::Node *node, const Ray &r, Hit &h, float tmin) {
    bool isIntersect = false;
    if (node) {
        // leaf node
        if (node->size > 0) {
            for (int i = 0; i < node->size; i++)
                isIntersect |= node->objects[i]->intersect(r, h, tmin);
            return isIntersect;
        }
        // non-leaf node, make recursive call
        if (intersectLeft(node, r)) {
            isIntersect |= intersect(node->lc, r, h, tmin);
        }
        if (intersectRight(node, r)) {
            isIntersect |= intersect(node->rc, r, h, tmin);
        }
    }
    return isIntersect;
}

bool BSPTree::intersectLeft(BSPTree::Node *node, const Ray &r) {
    return r.getOrigin()[node->axis] <= node->split || r.getDirection()[node->axis] < 0;
}

bool BSPTree::intersectRight(BSPTree::Node *node, const Ray &r) {
    return r.getOrigin()[node->axis] >= node->split || r.getDirection()[node->axis] > 0;
}


