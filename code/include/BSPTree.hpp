/**
 * BSP Tree Class Definition
 * @author Jason Fu
 *
 */

#ifndef FINALPROJECT_BSPTREE_HPP
#define FINALPROJECT_BSPTREE_HPP

#include <vector>

/**
 * BSP Tree (Binary Space Partition Tree)
 * Also called a 3-dimension KD Tree.
 * Used in a mesh object.
 * @author Jason Fu
 *
 */

#define LEAF_SIZE 4

class Object3D;

class Ray;

class Hit;

class BSPTree {
public:

    /**
     * Node in the BSP Tree
     */
    struct Node {
        Node *lc; // left children
        Node *rc; // right children
        Object3D **objects; // the objects in this node

        float split; // the value to split
        int axis;   // along which axis to split
        int size; // only leaf node stores data

        Node(int _axis, float _split, int _size = 0) :
                axis(_axis), split(_split), size(_size), lc(nullptr), rc(nullptr), objects(nullptr) {}

        ~Node() {
            delete lc;
            delete rc;
            delete[] objects;
        }
    };

    /**
     * Construct the BSP Tree
     * @param objects the objects that will be stored in the tree
     */
    explicit BSPTree(std::vector<Object3D *> &objects);

    ~BSPTree();

    /**
     * Intersect the ray with the BSP Tree
     * @param r the target ray
     * @param h if hit, the information of the hit point
     * @param tmin the minimum tolerance of the t value
     * @param tmax the maximum tolerance of the t value
     * @return whether the ray intersects with the BSP Tree
     */
    bool intersect(const Ray &r, Hit &h, float tmin, float tmax);

    int getSize() {
        return size;
    }

private:
    /**
     * Construct the BSP Tree recursively
     * @param objects objects that will be stored in the tree
     * @param axis the target axis
     * @return
     */
    Node *construct(std::vector<Object3D *> &objects, int axis);

    bool intersect(Node *node, const Ray &r, Hit &h, float tmin, float tmax);

    Node *root;

    int size;
};

#endif //FINALPROJECT_BSPTREE_HPP
