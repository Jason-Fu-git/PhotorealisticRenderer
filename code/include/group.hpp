#ifndef GROUP_H
#define GROUP_H


#include "object3d.hpp"
#include "ray.hpp"
#include "hit.hpp"
#include <iostream>
#include <vector>


/**
 * @copybrief 项目所有者独立实现
 * @author Jason Fu
 * @brief 存储Object3D的数据结构，简单用向量实现
 *
 */
class Group : public Object3D {

public:

    Group() {
        // reserve 10 pointers' spaces on default
        objects.reserve(10);
    }

    explicit Group(int num_objects) {
        objects.reserve(num_objects);
    }

    ~Group() override {
        objects.clear();
    }

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        bool inter = false;
        for (int i = 0; i < getGroupSize(); ++i) {
            inter |= objects[i]->intersect(r, h, tmin);
        }
        return inter;
    }

    void addObject(int index, Object3D *obj) {
        objects.insert(objects.begin() + index, obj);
    }

    Object3D* getObject(int index) {
        return objects[index];
    }

    int getGroupSize() {
        return objects.size();
    }

private:
    std::vector<Object3D *> objects;

};

#endif
	
