#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

#include "scene_parser.hpp"
#include "camera.hpp"
#include "light.hpp"
#include "material.hpp"
#include "object3d.hpp"
#include "group.hpp"
#include "mesh.hpp"
#include "sphere.hpp"
#include "plane.hpp"
#include "triangle.hpp"
#include "transform.hpp"
#include "curve.hpp"
#include "revsurface.hpp"
#include "fast_obj.h"

#define DegreesToRadians(x) ((M_PI * x) / 180.0f)
#define PHONG_MATERIAL 0
#define BRDF_MATERIAL 1
#define COOKTORRANCE_MATERIAL 2

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
SceneParser::SceneParser(const char *filename) {

    // initialize some reasonable default values
    group = nullptr;
    camera = nullptr;
    background_color = Vector3f(0.5, 0.5, 0.5);
    num_lights = 0;
    lights = nullptr;
    num_materials = 0;
    materials = nullptr;
    current_material = nullptr;

    // parse the file
    printf("parsing scene file %s\n", filename);
    assert(filename != nullptr);
    const char *ext = &filename[strlen(filename) - 4];

    if (strcmp(ext, ".txt") != 0) {
        printf("wrong file name extension\n");
        exit(0);
    }
    file = fopen(filename, "r");

    if (file == nullptr) {
        printf("cannot open scene file\n");
        exit(0);
    }
    parseFile();
    fclose(file);
    file = nullptr;

    if (num_lights == 0) {
        printf("WARNING:    No lights specified\n");
    }
}

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
SceneParser::~SceneParser() {

    delete group;
    delete camera;

    int i;
    for (i = 0; i < num_materials; i++) {
        delete materials[i];
    }
    delete[] materials;
    for (i = 0; i < num_lights; i++) {
        delete lights[i];
    }
    delete[] lights;
    for (i = 0; i < materials_vec.size(); i++) {
        delete materials_vec[i];
    }
    materials_vec.clear();
}

// ====================================================================
// ====================================================================

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
void SceneParser::parseFile() {
    //
    // at the top level, the scene can have a camera, 
    // background color and a group of objects
    // (we add lights and other things in future assignments)
    //
    char token[MAX_PARSER_TOKEN_LENGTH];
    while (getToken(token)) {
        if (!strcmp(token, "PerspectiveCamera")) {
            parsePerspectiveCamera();
        } else if (!strcmp(token, "Background")) {
            parseBackground();
        } else if (!strcmp(token, "Lights")) {
            parseLights();
        } else if (!strcmp(token, "Materials")) {
            parseMaterials();
        } else if (!strcmp(token, "Group")) {
            group = parseGroup();
        } else {
            printf("Unknown token in parseFile: '%s'\n", token);
            exit(0);
        }
    }
}

// ====================================================================
// ====================================================================

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
void SceneParser::parsePerspectiveCamera() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    // read in the camera parameters
    getToken(token);
    assert (!strcmp(token, "{"));
    getToken(token);
    assert (!strcmp(token, "center"));
    Vector3f center = readVector3f();
    getToken(token);
    assert (!strcmp(token, "direction"));
    Vector3f direction = readVector3f();
    getToken(token);
    assert (!strcmp(token, "up"));
    Vector3f up = readVector3f();
    getToken(token);
    assert (!strcmp(token, "angle"));
    float angle_degrees = readFloat();
    float angle_radians = DegreesToRadians(angle_degrees);
    getToken(token);
    assert (!strcmp(token, "width"));
    int width = readInt();
    getToken(token);
    assert (!strcmp(token, "height"));
    int height = readInt();
    getToken(token);
    float aperture = 0.0f;
    if (!strcmp(token, "aperture")) {
        aperture = readFloat();
        getToken(token);
    }
    float focus = 1.0f;
    if (!strcmp(token, "focus")) {
        focus = readFloat();
        getToken(token);
    }
    assert (!strcmp(token, "}"));
    camera = new PerspectiveCamera(center, direction, up, width, height, angle_radians, aperture, focus);
}

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
void SceneParser::parseBackground() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    // read in the background color
    getToken(token);
    assert (!strcmp(token, "{"));
    while (true) {
        getToken(token);
        if (!strcmp(token, "}")) {
            break;
        } else if (!strcmp(token, "color")) {
            background_color = readVector3f();
        } else {
            printf("Unknown token in parseBackground: '%s'\n", token);
            assert(0);
        }
    }
}

// ====================================================================
// ====================================================================

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
void SceneParser::parseLights() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert (!strcmp(token, "{"));
    // read in the number of objects
    getToken(token);
    assert (!strcmp(token, "numLights"));
    num_lights = readInt();
    lights = new Light *[num_lights];
    // read in the objects
    int count = 0;
    while (num_lights > count) {
        getToken(token);
        if (strcmp(token, "DirectionalLight") == 0) {
            lights[count] = parseDirectionalLight();
        } else if (strcmp(token, "PointLight") == 0) {
            lights[count] = parsePointLight();
        } else if (strcmp(token, "SphereLight") == 0) {
            lights[count] = parseSphereLight();
        } else {
            printf("Unknown token in parseLight: '%s'\n", token);
            exit(0);
        }
        count++;
    }
    getToken(token);
    assert (!strcmp(token, "}"));
}

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
Light *SceneParser::parseDirectionalLight() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert (!strcmp(token, "{"));
    getToken(token);
    assert (!strcmp(token, "direction"));
    Vector3f direction = readVector3f();
    getToken(token);
    assert (!strcmp(token, "color"));
    Vector3f color = readVector3f();
    getToken(token);
    assert (!strcmp(token, "}"));
    return new DirectionalLight(direction, color);
}

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
Light *SceneParser::parsePointLight() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert (!strcmp(token, "{"));
    getToken(token);
    assert (!strcmp(token, "position"));
    Vector3f position = readVector3f();
    getToken(token);
    assert (!strcmp(token, "color"));
    Vector3f color = readVector3f();
    getToken(token);
    assert (!strcmp(token, "}"));
    return new PointLight(position, color);
}

/**
 * @author Jason Fu
 *
 */
Light *SceneParser::parseSphereLight() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert (!strcmp(token, "{"));
    getToken(token);
    assert (!strcmp(token, "position"));
    Vector3f position = readVector3f();
    getToken(token);
    assert (!strcmp(token, "color"));
    Vector3f color = readVector3f();
    getToken(token);
    assert (!strcmp(token, "radius"));
    float radius = readFloat();
    getToken(token);
    assert (!strcmp(token, "}"));
    return new SphereLight(position, radius, color);
}
// ====================================================================
// ====================================================================

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
void SceneParser::parseMaterials() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert (!strcmp(token, "{"));
    // read in the number of objects
    getToken(token);
    assert (!strcmp(token, "numMaterials"));
    num_materials = readInt();
    materials = new Material *[num_materials];
    // read in the objects
    int count = 0;
    while (num_materials > count) {
        getToken(token);
        if (!strcmp(token, "Material") ||
            !strcmp(token, "PhongMaterial") ||
            !strcmp(token, "LambertianMaterial") ||
            !strcmp(token, "CookTorranceMaterial")) {
            materials[count] = parseMaterial();
        } else {
            printf("Unknown token in parseMaterial: '%s'\n", token);
            exit(0);
        }
        count++;
    }
    getToken(token);
    assert (!strcmp(token, "}"));
}


/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
Material *SceneParser::parseMaterial() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    char filename[MAX_PARSER_TOKEN_LENGTH];
    filename[0] = 0;
    Vector3f diffuseColor(1, 1, 1), specularColor(0, 0, 0), emissionColor(0, 0, 0), F0(0, 0, 0);
    float shininess = 0;
    float rfl_c = 0;
    float rfr_c = 0;
    float rfr_i = 0;
    float m = 0;
    float ks = 0;
    float kd = 0;
    int materialType = PHONG_MATERIAL;
    int surfaceType = Material::DIFFUSE;
    getToken(token);
    assert (!strcmp(token, "{"));
    while (true) {
        getToken(token);
        if (strcmp(token, "diffuseColor") == 0) {
            diffuseColor = readVector3f();
        } else if (strcmp(token, "specularColor") == 0) {
            specularColor = readVector3f();
        } else if (strcmp(token, "emissionColor") == 0) {
            emissionColor = readVector3f();
        } else if (strcmp(token, "shininess") == 0) {
            shininess = readFloat();
        } else if (strcmp(token, "texture") == 0) {
            getToken(filename);
        } else if (strcmp(token, "reflectiveCoefficient") == 0) {
            rfl_c = readFloat();
        } else if (strcmp(token, "refractiveIndex") == 0) {
            rfr_i = readFloat();
        } else if (strcmp(token, "refractiveCoefficient") == 0) {
            rfr_c = readFloat();
        } else if (strcmp(token, "materialType") == 0) {
            materialType = readInt();
        } else if (strcmp(token, "surfaceType") == 0) {
            surfaceType = readInt();
        } else if (strcmp(token, "roughness") == 0) {
            m = readFloat();
        } else if (strcmp(token, "F0") == 0) {
            F0 = readVector3f();
        } else if (strcmp(token, "ks") == 0) {
            ks = readFloat();
        } else if (strcmp(token, "kd") == 0) {
            kd = readFloat();
        } else {
            assert (!strcmp(token, "}"));
            break;
        }
    }
    assert(surfaceType == Material::DIFFUSE || surfaceType == Material::SPECULAR ||
           surfaceType == Material::TRANSPARENT || surfaceType == Material::GLOSSY);
    if (materialType == PHONG_MATERIAL) {
        auto *answer = new PhongMaterial(diffuseColor, specularColor, shininess);
        answer->setReflectiveProperties(rfl_c);
        answer->setRefractiveProperties(rfr_c, rfr_i);
        if (filename[0])
            answer->setTexture(filename);
        return answer;
    } else if (materialType == BRDF_MATERIAL) {
        auto *answer = new LambertianMaterial(diffuseColor, rfr_c, rfl_c, rfr_i, surfaceType, emissionColor);
        if (filename[0])
            answer->setTexture(filename);
        return answer;
    } else if (materialType == COOKTORRANCE_MATERIAL) {
        auto *answer = new CookTorranceMaterial(diffuseColor, ks, kd, m, F0);
        if (filename[0])
            answer->setTexture(filename);
        return answer;
    } else {
        printf("Unknown material type in parseObject: '%d'\n", materialType);
        exit(0);
    }
}



// ====================================================================
// ====================================================================

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
Object3D *SceneParser::parseObject(char token[MAX_PARSER_TOKEN_LENGTH]) {
    Object3D *answer = nullptr;
    if (!strcmp(token, "Group")) {
        answer = (Object3D *) parseGroup();
    } else if (!strcmp(token, "Sphere")) {
        answer = (Object3D *) parseSphere();
    } else if (!strcmp(token, "Plane")) {
        answer = (Object3D *) parsePlane();
    } else if (!strcmp(token, "Triangle")) {
        answer = (Object3D *) parseTriangle();
    } else if (!strcmp(token, "TriangleMesh")) {
        answer = (Object3D *) parseTriangleMesh();
    } else if (!strcmp(token, "Transform")) {
        answer = (Object3D *) parseTransform();
    } else if (!strcmp(token, "RevSurface")) {
        answer = (Object3D *) parseRevSurface();
    } else {
        printf("Unknown token in parseObject: '%s'\n", token);
        exit(0);
    }
    return answer;
}

// ====================================================================
// ====================================================================

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
Group *SceneParser::parseGroup() {
    //
    // each group starts with an integer that specifies
    // the number of objects in the group
    //
    // the material index sets the material of all objects which follow,
    // until the next material index (scoping for the materials is very
    // simple, and essentially ignores any tree hierarchy)
    //
    char token[MAX_PARSER_TOKEN_LENGTH];
    char filename[MAX_PARSER_TOKEN_LENGTH];

    getToken(token);
    assert (!strcmp(token, "{"));

    // use obj parser
    getToken(token);
    if (!strcmp(token, "ObjMaterialType")) {
        // get the material type
        int materialType = readInt();
        getToken(token);
        assert(!strcmp(token, "ObjFile"));
        // get the filename
        getToken(filename);
        getToken(token);
        assert (!strcmp(token, "}"));
        const char *ext = &filename[strlen(filename) - 4];
        assert(!strcmp(ext, ".obj"));
        return parseObjFile(filename, materialType);
    }

    // read in the number of objects
    assert (!strcmp(token, "numObjects"));
    int num_objects = readInt();

    auto *answer = new Group(num_objects);

    // read in the objects
    int count = 0;
    while (num_objects > count) {
        getToken(token);
        if (!strcmp(token, "MaterialIndex")) {
            // change the current material
            int index = readInt();
            if (index == -1)
                current_material = nullptr;
            else {
                current_material = getMaterial(index);
                if (auto phong_m = dynamic_cast<PhongMaterial *>(current_material)) {
                    current_material = new PhongMaterial(*phong_m);
                } else if (auto l_m = dynamic_cast<LambertianMaterial *>(current_material)) {
                    current_material = new LambertianMaterial(*l_m);
                } else if (auto cook_m = dynamic_cast<CookTorranceMaterial *>(current_material)) {
                    current_material = new CookTorranceMaterial(*cook_m);
                } else {
                    printf("Unknown material type in parseObject: '%d'\n", index);
                    exit(-1);
                }
            }
        } else {
            Object3D *object = parseObject(token);
            assert (object != nullptr);
            answer->addObject(count, object);
            if (current_material != nullptr) {
                current_material->setObject(object);
                materials_vec.push_back(current_material);
            }
            count++;
        }
    }
    getToken(token);
    assert (!strcmp(token, "}"));

    // return the group
    return answer;
}

/**
 * Parse .obj file using fast_obj.h
 * @author Jason Fu
 *
 */
Group *SceneParser::parseObjFile(const char *filename, int materialType) {
    fastObjMesh *m = fast_obj_read(filename);
    if (m == nullptr) {
        fprintf(stderr, "Error reading %s\n", filename);
        exit(-1);
    }

    // create the group
    auto *answer = new Group(m->group_count);
    printf("Reading %s: %d faces, %d groups\n", filename, m->face_count, m->group_count);


    // parse textures
    std::vector<Image *> imgs;
    imgs.push_back(nullptr);
    for (int ti = 1; ti < m->texture_count; ti++) {
        assert(m->textures[ti].path != nullptr);
        Image *texture;
        std::string path = std::string(m->textures[ti].path);
        if (hasEnding(path, ".tga")) {
            texture = Image::LoadTGA(m->textures[ti].path);
        } else if (hasEnding(path, ".ppm")) {
            texture = Image::LoadPPM(m->textures[ti].path);
        } else if (hasEnding(path, ".png")) {
            texture = Image::LoadPNG(m->textures[ti].path);
        } else {
            texture = nullptr;
            std::cerr << "Unsupported texture format : must be one of .tga or .ppm" << std::endl;
            exit(1);
        }
        imgs.push_back(texture);
        printf("Texture %d: %s\n", ti, m->textures[ti].path);
    }

    // read the subgroups
    for (int gi = 0; gi < m->group_count; gi++) {
        auto grp = m->groups[gi];

        // create the subgroup
        printf("Group %d: %d faces\n", gi, grp.face_count);

        // parse the faces
        std::vector<Triangle *> triangles;
        triangles.reserve(grp.face_count);

        int idx = 0;
        for (int fi = 0; fi < grp.face_count; fi++) {
            assert(m->face_vertices[grp.face_offset + fi] == 3);

            // parse the material
            auto objMaterial = m->materials[m->face_materials[grp.face_offset + fi]];
            Material *material;
            if (materialType == PHONG_MATERIAL)
                material = new PhongMaterial(Vector3f(objMaterial.Kd), Vector3f(objMaterial.Ks), 10);
            else if (materialType == BRDF_MATERIAL)
                material = new LambertianMaterial(Vector3f(objMaterial.Kd), 0, 0, 0, LambertianMaterial::DIFFUSE);
            else {
                fprintf(stderr, "Unknown material type\n");
                exit(-1);
            }

            materials_vec.push_back(material);

            // parse the vertices
            auto *vertices = new Vector3f[3];
            auto *us = new float[3];
            auto *vs = new float[3];
            us[0] = 100;
            for (int vi = 0; vi < 3; vi++) {
                auto mi = m->indices[grp.index_offset + idx];
                assert(mi.p);
                vertices[vi][0] = m->positions[3 * mi.p + 0];
                vertices[vi][1] = m->positions[3 * mi.p + 1];
                vertices[vi][2] = m->positions[3 * mi.p + 2];
                if (mi.t) {
                    us[vi] = m->texcoords[2 * mi.t + 0];
                    vs[vi] = m->texcoords[2 * mi.t + 1];
                }
                ++idx;
            }

            // create the triangle
            auto *triangle = new Triangle(vertices[0], vertices[1], vertices[2], material);

            // calculate the normal
            Vector3f normal = Vector3f::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]).normalized();
            triangle->normal = normal;

            // calculate the texture coordinates
            if (objMaterial.map_Kd) {
                triangle->setTextureUV(us[0], vs[0], us[1], vs[1], us[2], vs[2]);
                auto texture = imgs[objMaterial.map_Kd];
                material->setTexture(texture);
            }

            material->setObject(triangle);

            // add the triangle to the vector
            triangles.push_back(triangle);

            // release the memory
            delete[] us;
            delete[] vs;
            delete[] vertices;
        }

        // create a mesh
        Mesh *mesh = new Mesh(triangles);
        answer->addObject(mesh);

    }

    // release the memory
    fast_obj_destroy(m);

    printf("Successfully parsed the obj file!\n");

    return answer;

}

// ====================================================================
// ====================================================================

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
Sphere *SceneParser::parseSphere() {
    float thetaOffset = 0.0, phiOffset = 0.0;
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert (!strcmp(token, "{"));
    getToken(token);
    assert (!strcmp(token, "center"));
    Vector3f center = readVector3f();
    getToken(token);
    assert (!strcmp(token, "radius"));
    float radius = readFloat();
    getToken(token);
    if (!strcmp(token, "thetaOffset")) {
        thetaOffset = readFloat();
        getToken(token);
    }
    if (!strcmp(token, "phiOffset")) {
        phiOffset = readFloat();
        getToken(token);
    }
    assert (!strcmp(token, "}"));
    assert (current_material != nullptr);
    return new Sphere(center, radius, current_material, thetaOffset, phiOffset);
}

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
Plane *SceneParser::parsePlane() {
    float scale = 1.0f;
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert (!strcmp(token, "{"));
    getToken(token);
    assert (!strcmp(token, "normal"));
    Vector3f normal = readVector3f();
    getToken(token);
    assert (!strcmp(token, "offset"));
    float offset = readFloat();
    getToken(token);
    if (!strcmp(token, "scale")) {
        scale = readFloat();
        getToken(token);
    }
    assert (!strcmp(token, "}"));
    assert (current_material != nullptr);
    return new Plane(normal, offset, current_material, scale);
}

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
Curve *SceneParser::parseBezierCurve() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert (!strcmp(token, "{"));
    getToken(token);
    assert (!strcmp(token, "controls"));
    vector<Vector3f> controls;
    while (true) {
        getToken(token);
        if (!strcmp(token, "[")) {
            controls.push_back(readVector3f());
            getToken(token);
            assert (!strcmp(token, "]"));
        } else if (!strcmp(token, "}")) {
            break;
        } else {
            printf("Incorrect format for BezierCurve!\n");
            exit(0);
        }
    }
    Curve *answer = new BezierCurve(controls);
    return answer;
}

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
Curve *SceneParser::parseBsplineCurve() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert (!strcmp(token, "{"));
    getToken(token);
    assert (!strcmp(token, "controls"));
    vector<Vector3f> controls;
    while (true) {
        getToken(token);
        if (!strcmp(token, "[")) {
            controls.push_back(readVector3f());
            getToken(token);
            assert (!strcmp(token, "]"));
        } else if (!strcmp(token, "}")) {
            break;
        } else {
            printf("Incorrect format for BsplineCurve!\n");
            exit(0);
        }
    }
    Curve *answer = new BsplineCurve(controls);
    return answer;
}

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
RevSurface *SceneParser::parseRevSurface() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert (!strcmp(token, "{"));
    getToken(token);
    assert (!strcmp(token, "profile"));
    Curve *profile;
    getToken(token);
    if (!strcmp(token, "BezierCurve")) {
        profile = parseBezierCurve();
    } else if (!strcmp(token, "BsplineCurve")) {
        profile = parseBsplineCurve();
    } else {
        printf("Unknown profile type in parseRevSurface: '%s'\n", token);
        exit(0);
    }
    getToken(token);
    assert (!strcmp(token, "}"));
    auto *answer = new RevSurface(profile, current_material);
    return answer;
}

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
Triangle *SceneParser::parseTriangle() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert (!strcmp(token, "{"));
    getToken(token);
    assert (!strcmp(token, "vertex0"));
    Vector3f v0 = readVector3f();
    getToken(token);
    assert (!strcmp(token, "vertex1"));
    Vector3f v1 = readVector3f();
    getToken(token);
    assert (!strcmp(token, "vertex2"));
    Vector3f v2 = readVector3f();
    getToken(token);
    assert (!strcmp(token, "}"));
    assert (current_material != nullptr);
    return new Triangle(v0, v1, v2, current_material);
}

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
Mesh *SceneParser::parseTriangleMesh() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    char filename[MAX_PARSER_TOKEN_LENGTH];
    // get the filename
    getToken(token);
    assert (!strcmp(token, "{"));
    getToken(token);
    assert (!strcmp(token, "obj_file"));
    getToken(filename);
    getToken(token);
    assert (!strcmp(token, "}"));
    const char *ext = &filename[strlen(filename) - 4];
    assert(!strcmp(ext, ".obj"));
    Mesh *answer = new Mesh(filename, current_material);

    return answer;
}

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
Transform *SceneParser::parseTransform() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    Matrix4f matrix = Matrix4f::identity();
    Object3D *object = nullptr;
    getToken(token);
    assert (!strcmp(token, "{"));
    // read in transformations: 
    // apply to the LEFT side of the current matrix (so the first
    // transform in the list is the last applied to the object)
    getToken(token);

    while (true) {
        if (!strcmp(token, "Scale")) {
            Vector3f s = readVector3f();
            matrix = matrix * Matrix4f::scaling(s[0], s[1], s[2]);
        } else if (!strcmp(token, "UniformScale")) {
            float s = readFloat();
            matrix = matrix * Matrix4f::uniformScaling(s);
        } else if (!strcmp(token, "Translate")) {
            matrix = matrix * Matrix4f::translation(readVector3f());
        } else if (!strcmp(token, "XRotate")) {
            matrix = matrix * Matrix4f::rotateX(DegreesToRadians(readFloat()));
        } else if (!strcmp(token, "YRotate")) {
            matrix = matrix * Matrix4f::rotateY(DegreesToRadians(readFloat()));
        } else if (!strcmp(token, "ZRotate")) {
            matrix = matrix * Matrix4f::rotateZ(DegreesToRadians(readFloat()));
        } else if (!strcmp(token, "Rotate")) {
            getToken(token);
            assert (!strcmp(token, "{"));
            Vector3f axis = readVector3f();
            float degrees = readFloat();
            float radians = DegreesToRadians(degrees);
            matrix = matrix * Matrix4f::rotation(axis, radians);
            getToken(token);
            assert (!strcmp(token, "}"));
        } else if (!strcmp(token, "Matrix4f")) {
            Matrix4f matrix2 = Matrix4f::identity();
            getToken(token);
            assert (!strcmp(token, "{"));
            for (int j = 0; j < 4; j++) {
                for (int i = 0; i < 4; i++) {
                    float v = readFloat();
                    matrix2(i, j) = v;
                }
            }
            getToken(token);
            assert (!strcmp(token, "}"));
            matrix = matrix2 * matrix;
        } else {
            // otherwise this must be an object,
            // and there are no more transformations
            object = parseObject(token);
            break;
        }
        getToken(token);
    }

    assert(object != nullptr);
    getToken(token);
    assert (!strcmp(token, "}"));
    return new Transform(matrix, object);
}

// ====================================================================
// ====================================================================

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
int SceneParser::getToken(char token[MAX_PARSER_TOKEN_LENGTH]) {
    // for simplicity, tokens must be separated by whitespace
    assert (file != nullptr);
    int success = fscanf(file, "%s ", token);
    if (success == EOF) {
        token[0] = '\0';
        return 0;
    }
    return 1;
}

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
Vector3f SceneParser::readVector3f() {
    float x, y, z;
    int count = fscanf(file, "%f %f %f", &x, &y, &z);
    if (count != 3) {
        printf("Error trying to read 3 floats to make a Vector3f\n");
        assert (0);
    }
    return Vector3f(x, y, z);
}

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
float SceneParser::readFloat() {
    float answer;
    int count = fscanf(file, "%f", &answer);
    if (count != 1) {
        printf("Error trying to read 1 float\n");
        assert (0);
    }
    return answer;
}

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
int SceneParser::readInt() {
    int answer;
    int count = fscanf(file, "%d", &answer);
    if (count != 1) {
        printf("Error trying to read 1 int\n");
        assert (0);
    }
    return answer;
}
