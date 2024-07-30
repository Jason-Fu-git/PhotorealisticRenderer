#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

#include "scene_parser.cuh"
#include "camera.cuh"
#include "light.cuh"
#include "material.cuh"
#include "object3d.cuh"
#include "group.cuh"
#include "mesh.cuh"
#include "sphere.cuh"
#include "plane.cuh"
#include "triangle.cuh"
#include "transform.cuh"
#include "fast_obj.cuh"

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
    currentMaterialIndex = -1;

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
    materials.clear();

    for (i = 0; i < num_lights; i++) {
        delete lights[i];
    }
    delete[] lights;

    // free up device space

    // free camera
    freeCameraOnDevice<<<1, 1>>>(deviceCamera);
    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error freeing camera: %s\n", cudaGetErrorString(err));
    }
    cudaFree(deviceCamera);

    // free lights
    for (int j = 0; j < num_lights; j++) {
        freeLightOnDevice<<<1, 1>>>(deviceLights + j);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Error freeing lights: %s\n", cudaGetErrorString(err));
        }
    }
    cudaFree(deviceLights);

    // free materials
    for (int k = 0; k < num_materials; k++) {
        freeMaterialOnDevice<<<1, 1>>>(deviceMaterials + k);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Error freeing materials: %s\n", cudaGetErrorString(err));
        }
    }
    cudaFree(deviceMaterials);

    // free group
    freeGroupOnDevice<<<1, 1>>>(deviceGroup);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error freeing materials: %s\n", cudaGetErrorString(err));
    }
    cudaFree(deviceGroup);
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
    // create materials on device
    assert(num_materials == materials.size());
    cudaMalloc(&deviceMaterials, sizeof(Material *) * num_materials);
    for (int i = 0; i < num_materials; ++i) {
        auto material = materials[i];
        if (auto p_m = dynamic_cast<PhongMaterial *>(material)) {
            createPhongMaterialOnDevice<<<1, 1>>>(deviceMaterials + i,
                                                  p_m->getDiffuseColor(), p_m->getSpecularColor(), p_m->getShininess());
        } else if (auto l_m = dynamic_cast<LambertianMaterial *>(material)) {
            createLambertianMaterialOnDevice<<<1, 1>>>(deviceMaterials + i,
                                                       l_m->getDiffuseColor(), l_m->getRefractiveCoefficient(),
                                                       l_m->getReflectiveCoefficient(),
                                                       l_m->getRefractiveIndex(), l_m->getType(),
                                                       l_m->getEmissionColor());
        } else if (auto c_m = dynamic_cast<CookTorranceMaterial *>(material)) {
            createCookTorranceMaterialOnDevice<<<1, 1>>>(deviceMaterials + i,
                                                         c_m->getDiffuseColor(),
                                                         c_m->getSpecularCoefficient(),
                                                         c_m->getDiffuseCoefficient(),
                                                         c_m->getRoughness(),
                                                         c_m->getFresnelCoefficient());
        } else {
            fprintf(stderr, "Unknown material at index %d\n", i);
            exit(1);
        }

        auto err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Error occurred in create material on device : %s\n",
                   cudaGetErrorString(err));
        }
    }

    // create 3D objects on device
    deviceGroup = groupHostToDevice(group);
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
    assert(!strcmp(token, "{"));
    getToken(token);
    assert(!strcmp(token, "center"));
    Vector3f center = readVector3f();
    getToken(token);
    assert(!strcmp(token, "direction"));
    Vector3f direction = readVector3f();
    getToken(token);
    assert(!strcmp(token, "up"));
    Vector3f up = readVector3f();
    getToken(token);
    assert(!strcmp(token, "angle"));
    float angle_degrees = readFloat();
    float angle_radians = DegreesToRadians(angle_degrees);
    getToken(token);
    assert(!strcmp(token, "width"));
    int width = readInt();
    getToken(token);
    assert(!strcmp(token, "height"));
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
    assert(!strcmp(token, "}"));
    camera = new PerspectiveCamera(center, direction, up, width, height, angle_radians, aperture, focus);

    // create camera on device
    cudaMalloc(&deviceCamera, sizeof(Camera **));
    createPerspectiveCameraOnDevice<<<1, 1>>>(deviceCamera,
                                              center, direction, up, width, height, angle_radians, aperture, focus);
    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error parsing camera: %s\n", cudaGetErrorString(err));
    }
}

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
void SceneParser::parseBackground() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    // read in the background color
    getToken(token);
    assert(!strcmp(token, "{"));
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
    assert(!strcmp(token, "{"));
    // read in the number of objects
    getToken(token);
    assert(!strcmp(token, "numLights"));
    num_lights = readInt();
    lights = new Light *[num_lights];
    // allocate lights on device
    cudaMalloc(&deviceLights, num_lights * sizeof(Light **));
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
    assert(!strcmp(token, "}"));
    // copy lights to device
    for (int i = 0; i < num_lights; i++) {
        auto light = lights[i];
        if (auto pointLight = dynamic_cast<PointLight *>(light)) {
            createPointLightOnDevice<<<1, 1>>>(deviceLights + i, pointLight->getPosition(), pointLight->getColor());
        } else if (auto directionalLight = dynamic_cast<DirectionalLight *>(light)) {
            createDirectionalLightOnDevice<<<1, 1>>>(deviceLights + i, directionalLight->getDirection(),
                                                     directionalLight->getColor());
        } else if (auto sphereLight = dynamic_cast<SphereLight *>(light)) {
            createSphereLightOnDevice<<<1, 1>>>(deviceLights + i, sphereLight->getPosition(), sphereLight->getRadius(),
                                                sphereLight->getEmissionColor());
        } else {
            printf("Unknown light type\n");
        }
        auto err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Error parsing light: %s\n", cudaGetErrorString(err));
        }
    }
}

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
Light *SceneParser::parseDirectionalLight() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));
    getToken(token);
    assert(!strcmp(token, "direction"));
    Vector3f direction = readVector3f();
    getToken(token);
    assert(!strcmp(token, "color"));
    Vector3f color = readVector3f();
    getToken(token);
    assert(!strcmp(token, "}"));
    return new DirectionalLight(direction, color);
}

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
Light *SceneParser::parsePointLight() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));
    getToken(token);
    assert(!strcmp(token, "position"));
    Vector3f position = readVector3f();
    getToken(token);
    assert(!strcmp(token, "color"));
    Vector3f color = readVector3f();
    getToken(token);
    assert(!strcmp(token, "}"));
    return new PointLight(position, color);
}

/**
 * @author Jason Fu
 *
 */
Light *SceneParser::parseSphereLight() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));
    getToken(token);
    assert(!strcmp(token, "position"));
    Vector3f position = readVector3f();
    getToken(token);
    assert(!strcmp(token, "color"));
    Vector3f color = readVector3f();
    getToken(token);
    assert(!strcmp(token, "radius"));
    float radius = readFloat();
    getToken(token);
    assert(!strcmp(token, "}"));
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
    assert(!strcmp(token, "{"));
    // read in the number of objects
    getToken(token);
    assert(!strcmp(token, "numMaterials"));
    num_materials = readInt();
    // read in the objects
    int count = 0;
    while (num_materials > count) {
        getToken(token);
        if (!strcmp(token, "Material") ||
            !strcmp(token, "PhongMaterial") ||
            !strcmp(token, "LambertianMaterial") ||
            !strcmp(token, "CookTorranceMaterial")) {
            materials.push_back(parseMaterial());
        } else {
            printf("Unknown token in parseMaterial: '%s'\n", token);
            exit(0);
        }
        count++;
    }
    getToken(token);
    assert(!strcmp(token, "}"));
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
    assert(!strcmp(token, "{"));
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
            assert(!strcmp(token, "}"));
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
    } else if (!strcmp(token, "Transform")) {
        answer = (Object3D *) parseTransform();
    } else if (!strcmp(token, "TriangleMesh")) {
        answer = (Object3D *) parseTriangleMesh();
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
    assert(!strcmp(token, "{"));

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
        assert(!strcmp(token, "}"));
        const char *ext = &filename[strlen(filename) - 4];
        assert(!strcmp(ext, ".obj"));
        return parseObjFile(filename, materialType);
    }

    // read in the number of objects
    assert(!strcmp(token, "numObjects"));
    int num_objects = readInt();

    auto *answer = new Group(num_objects);

    // read in the objects
    int count = 0;
    while (num_objects > count) {
        getToken(token);
        if (!strcmp(token, "MaterialIndex")) {
            // change the current material
            currentMaterialIndex = readInt();
        } else {
            Object3D *object = parseObject(token);
            assert(object != nullptr);
            answer->addObject(object);
            count++;
        }
    }
    getToken(token);
    assert(!strcmp(token, "}"));

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
                material = new PhongMaterial(Vector3f(objMaterial.Kd), Vector3f(objMaterial.Ks), objMaterial.Ns);
            else if (materialType == BRDF_MATERIAL)
                material = new LambertianMaterial(Vector3f(objMaterial.Kd), 0, 0, 0, LambertianMaterial::DIFFUSE);
            else {
                fprintf(stderr, "Unknown material type\n");
                exit(-1);
            }

            materials.push_back(material);
            currentMaterialIndex = num_materials;
            ++num_materials;


            // parse the vertices
            auto *vertices = new Vector3f[3];
            auto *us = new float[3];
            auto *vs = new float[3];
            us[0] = 100;
            auto *ns = new Vector3f[3];
            bool has_n = false;
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
                if (mi.n) {
                    ns[vi][0] = m->normals[3 * mi.n + 0];
                    ns[vi][1] = m->normals[3 * mi.n + 1];
                    ns[vi][2] = m->normals[3 * mi.n + 2];
                    has_n = true;
                }
                ++idx;
            }

            // create the triangle
            auto *triangle = new Triangle(vertices[0], vertices[1], vertices[2], material, currentMaterialIndex);

            // calculate the normal
            Vector3f normal = Vector3f::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]).normalized();
            triangle->normal = normal;

            // load the vertex normal
            if (has_n) {
                triangle->setVertexNormals(ns[0], ns[1], ns[2]);
            }

            // calculate the texture coordinates
            if (objMaterial.map_Kd) {
                triangle->setTextureUV(us[0], vs[0], us[1], vs[1], us[2], vs[2]);
                auto texture = imgs[objMaterial.map_Kd];
                material->setTexture(texture);
            }

            // add the triangle to the vector
            triangles.push_back(triangle);

            // release the memory
            delete[] us;
            delete[] vs;
            delete[] vertices;
        }
        auto triangles_array = triangles.data();

        // create a mesh
        Mesh *mesh = new Mesh(*triangles_array, triangles.size());
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
    assert(!strcmp(token, "{"));
    getToken(token);
    assert(!strcmp(token, "center"));
    Vector3f center = readVector3f();
    getToken(token);
    assert(!strcmp(token, "radius"));
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
    assert(!strcmp(token, "}"));
    assert(currentMaterialIndex != -1);
    return new Sphere(center, radius, materials[currentMaterialIndex], currentMaterialIndex,
                      thetaOffset, phiOffset);
}

/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
Plane *SceneParser::parsePlane() {
    float scale = 1.0f;
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));
    getToken(token);
    assert(!strcmp(token, "normal"));
    Vector3f normal = readVector3f();
    getToken(token);
    assert(!strcmp(token, "offset"));
    float offset = readFloat();
    getToken(token);
    if (!strcmp(token, "scale")) {
        scale = readFloat();
        getToken(token);
    }
    assert(!strcmp(token, "}"));
    assert(currentMaterialIndex != -1);
    return new Plane(normal, offset, materials[currentMaterialIndex], currentMaterialIndex, scale);
}


/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */
Triangle *SceneParser::parseTriangle() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));
    getToken(token);
    assert(!strcmp(token, "vertex0"));
    Vector3f v0 = readVector3f();
    getToken(token);
    assert(!strcmp(token, "vertex1"));
    Vector3f v1 = readVector3f();
    getToken(token);
    assert(!strcmp(token, "vertex2"));
    Vector3f v2 = readVector3f();
    getToken(token);
    assert(!strcmp(token, "}"));
    assert(currentMaterialIndex != -1);
    return new Triangle(v0, v1, v2, materials[currentMaterialIndex], currentMaterialIndex);
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
    assert(!strcmp(token, "{"));
    getToken(token);
    assert(!strcmp(token, "obj_file"));
    getToken(filename);
    getToken(token);
    assert(!strcmp(token, "}"));
    const char *ext = &filename[strlen(filename) - 4];
    assert(!strcmp(ext, ".obj"));
    Mesh *answer = new Mesh(filename, currentMaterialIndex);

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
    assert(!strcmp(token, "{"));
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
            assert(!strcmp(token, "{"));
            Vector3f axis = readVector3f();
            float degrees = readFloat();
            float radians = DegreesToRadians(degrees);
            matrix = matrix * Matrix4f::rotation(axis, radians);
            getToken(token);
            assert(!strcmp(token, "}"));
        } else if (!strcmp(token, "Matrix4f")) {
            Matrix4f matrix2 = Matrix4f::identity();
            getToken(token);
            assert(!strcmp(token, "{"));
            for (int j = 0; j < 4; j++) {
                for (int i = 0; i < 4; i++) {
                    float v = readFloat();
                    matrix2(i, j) = v;
                }
            }
            getToken(token);
            assert(!strcmp(token, "}"));
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
    assert(!strcmp(token, "}"));
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
    assert(file != nullptr);
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
        assert(0);
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
        assert(0);
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
        assert(0);
    }
    return answer;
}

Group **SceneParser::groupHostToDevice(Group *hGroup) {
    // alloc space on device for hGroup
    Group **dGroup;
    cudaMalloc(&dGroup, sizeof(Group **));
    createGroupOnDevice<<<1, 1>>>(dGroup, hGroup->getGroupSize());
    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error creating hGroup on device : %s\n", cudaGetErrorString(err));
    }
    // iterate through the elements of hGroup
    for (int i = 0; i < hGroup->getGroupSize(); i++) {
        auto object = hGroup->getObject(i);
        Object3D **dObject;
        cudaMalloc(&dObject, sizeof(Object3D **));
        if (auto plane = dynamic_cast<Plane *>(object)) {
            createPlaneOnDevice<<<1, 1>>>(dObject, deviceMaterials, plane->getMaterialIndex(),
                                          plane->getNormal(), plane->getD(), plane->getScale());
        } else if (auto sphere = dynamic_cast<Sphere *>(object)) {
            createSphereOnDevice<<<1, 1>>>(dObject, deviceMaterials, sphere->getMaterialIndex(),
                                           sphere->getCenter(), sphere->getRadius(),
                                           sphere->getThetaOffset(), sphere->getPhiOffset());
        } else if (auto triangle = dynamic_cast<Triangle *>(object)) {
            createTriangleOnDevice<<<1, 1>>>(dObject, deviceMaterials, triangle->getMaterialIndex(),
                                             triangle->getA(), triangle->getB(), triangle->getC(), triangle->normal,
                                             triangle->au, triangle->av, triangle->bu, triangle->bv,
                                             triangle->cu, triangle->cv, triangle->_an, triangle->_bn, triangle->_cn);
        } else if (auto mesh = dynamic_cast<Mesh *>(object)) {
            Triangle *dTriangles;
            Triangle *hTriangles = mesh->getTriangles();
            cudaMalloc(&dTriangles, sizeof(Triangle) * mesh->getSize());
            cudaMemcpy(dTriangles, hTriangles, sizeof(Triangle) * mesh->getSize(), cudaMemcpyHostToDevice);
            createMeshOnDevice<<<1, 1>>>(dObject, deviceMaterials, dTriangles, mesh->getSize());
        } else if (auto transform = dynamic_cast<Transform *>(object)) {
            auto obj = transform->getObject();
            auto dObj = createObjectOnDevice(obj);
            createTransformOnDevice<<<1, 1>>>(dObject, transform->getTransformMatrix(), dObj);
        } else if (auto cGroup = dynamic_cast<Group *>(object)) {
            *dObject = *groupHostToDevice(cGroup);
        } else {
            fprintf(stderr, "Unsupported object type at index %d\n", i);
        }


        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Error creating hGroup on device : %s\n", cudaGetErrorString(err));
        }

        addObjectToGroup<<<1, 1>>>(dObject, dGroup);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Error creating hGroup on device : %s\n", cudaGetErrorString(err));
        }
    }
    return dGroup;
}

Object3D **SceneParser::createObjectOnDevice(Object3D *object) {
    Object3D **dObject;
    auto err = cudaMalloc(&dObject, sizeof(Object3D **));
    if (auto plane = dynamic_cast<Plane *>(object)) {
        createPlaneOnDevice<<<1, 1>>>(dObject, deviceMaterials, plane->getMaterialIndex(),
                                      plane->getNormal(), plane->getD(), plane->getScale());
    } else if (auto sphere = dynamic_cast<Sphere *>(object)) {
        createSphereOnDevice<<<1, 1>>>(dObject, deviceMaterials, sphere->getMaterialIndex(),
                                       sphere->getCenter(), sphere->getRadius(),
                                       sphere->getThetaOffset(), sphere->getPhiOffset());
    } else if (auto triangle = dynamic_cast<Triangle *>(object)) {
        createTriangleOnDevice<<<1, 1>>>(dObject, deviceMaterials, triangle->getMaterialIndex(),
                                         triangle->getA(), triangle->getB(), triangle->getC(), triangle->normal,
                                         triangle->au, triangle->av, triangle->bu, triangle->bv,
                                         triangle->cu, triangle->cv, triangle->_an, triangle->_bn, triangle->_cn);
    } else if (auto mesh = dynamic_cast<Mesh *>(object)) {
        fflush(stdout);
        Triangle *dTriangles;
        Triangle *hTriangles = mesh->getTriangles();
        err = cudaMalloc(&dTriangles, sizeof(Triangle) * mesh->getSize());
        err = cudaMemcpy(dTriangles, hTriangles, sizeof(Triangle) * mesh->getSize(), cudaMemcpyHostToDevice);
        createMeshOnDevice<<<1, 1>>>(dObject, deviceMaterials, dTriangles, mesh->getSize());
    } else {
        fprintf(stderr, "Unsupported object type\n");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error creating group on device : %s\n", cudaGetErrorString(err));
    }

    return dObject;
}
