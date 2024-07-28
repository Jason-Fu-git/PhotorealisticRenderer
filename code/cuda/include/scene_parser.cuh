/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */

#ifndef SCENE_PARSER_H
#define SCENE_PARSER_H

#include <cassert>
#include "Vector3f.cuh"
#include <vector>

class Camera;

class Light;

class Material;

class Object3D;

class Group;

class Sphere;

class Plane;

class Triangle;

class Transform;

class Mesh;

#define MAX_PARSER_TOKEN_LENGTH 1024

class SceneParser {

    friend class Renderer;

public:

    SceneParser() = delete;

    SceneParser(const char *filename);

    ~SceneParser();


private:

    void parseFile();

    void parsePerspectiveCamera();

    void parseBackground();

    void parseLights();

    Light *parsePointLight();

    Light *parseDirectionalLight();

    void parseMaterials();

    Material *parseMaterial();

    Object3D *parseObject(char token[MAX_PARSER_TOKEN_LENGTH]);

    Group *parseGroup();

    Sphere *parseSphere();

    Plane *parsePlane();

    Triangle *parseTriangle();

    Transform *parseTransform();

    Group *parseObjFile(const char *filename, int materialType);

    Group **groupHostToDevice(Group *hGroup);

    Object3D **createObjectOnDevice(Object3D *obj);

    int getToken(char token[MAX_PARSER_TOKEN_LENGTH]);

    Vector3f readVector3f();

    float readFloat();

    int readInt();

    FILE *file;
    Vector3f background_color;
    // camera
    Camera *camera;
    Camera **deviceCamera;
    // lights
    int num_lights;
    Light **lights;
    Light **deviceLights;
    // materials
    int num_materials;
    std::vector<Material *> materials;
    Material **deviceMaterials;
    int currentMaterialIndex;
    // group
    Group *group;
    Group **deviceGroup;

    Light *parseSphereLight();
};


#endif // SCENE_PARSER_H
