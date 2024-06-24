/*
 * fast_obj
 *
 * Version 1.3
 *
 * MIT License
 *
 * Copyright (c) 2018-2021 Richard Knight
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#ifndef FAST_OBJ_HDR
#define FAST_OBJ_HDR

#define FAST_OBJ_VERSION_MAJOR  1
#define FAST_OBJ_VERSION_MINOR  3
#define FAST_OBJ_VERSION        ((FAST_OBJ_VERSION_MAJOR << 8) | FAST_OBJ_VERSION_MINOR)

#include <stdlib.h>


typedef struct
{
    /* Texture name from .mtl file */
    char*                       name;

    /* Resolved path to texture */
    char*                       path;

} fastObjTexture;


typedef struct
{
    /* Material name */
    char*                       name;

    /* Parameters */
    float                       Ka[3];  /* Ambient */
    float                       Kd[3];  /* Diffuse */
    float                       Ks[3];  /* Specular */
    float                       Ke[3];  /* Emission */
    float                       Kt[3];  /* Transmittance */
    float                       Ns;     /* Shininess */
    float                       Ni;     /* Index of refraction */
    float                       Tf[3];  /* Transmission filter */
    float                       d;      /* Disolve (alpha) */
    int                         illum;  /* Illumination model */

    /* Set for materials that don't come from the associated mtllib */
    int                         fallback;

    /* Texture map indices in fastObjMesh textures array */
    unsigned int                map_Ka;
    unsigned int                map_Kd;
    unsigned int                map_Ks;
    unsigned int                map_Ke;
    unsigned int                map_Kt;
    unsigned int                map_Ns;
    unsigned int                map_Ni;
    unsigned int                map_d;
    unsigned int                map_bump;

} fastObjMaterial;

/* Allows user override to bigger indexable array */
#ifndef FAST_OBJ_UINT_TYPE
#define FAST_OBJ_UINT_TYPE unsigned int
#endif

typedef FAST_OBJ_UINT_TYPE fastObjUInt;

typedef struct
{
    fastObjUInt                 p;
    fastObjUInt                 t;
    fastObjUInt                 n;

} fastObjIndex;


typedef struct
{
    /* Group name */
    char*                       name;

    /* Number of faces */
    unsigned int                face_count;

    /* First face in fastObjMesh face_* arrays */
    unsigned int                face_offset;

    /* First index in fastObjMesh indices array */
    unsigned int                index_offset;

} fastObjGroup;


/* Note: a dummy zero-initialized value is added to the first index
   of the positions, texcoords, normals and textures arrays. Hence,
   valid indices into these arrays start from 1, with an index of 0
   indicating that the attribute is not present. */
typedef struct
{
    /* Vertex data */
    unsigned int                position_count;
    float*                      positions;

    unsigned int                texcoord_count;
    float*                      texcoords;

    unsigned int                normal_count;
    float*                      normals;

    unsigned int                color_count;
    float*                      colors;

    /* Face data: one element for each face */
    unsigned int                face_count;
    unsigned int*               face_vertices;
    unsigned int*               face_materials;

    /* Index data: one element for each face vertex */
    unsigned int                index_count;
    fastObjIndex*               indices;

    /* Materials */
    unsigned int                material_count;
    fastObjMaterial*            materials;

    /* Texture maps */
    unsigned int                texture_count;
    fastObjTexture*             textures;

    /* Mesh objects ('o' tag in .obj file) */
    unsigned int                object_count;
    fastObjGroup*               objects;

    /* Mesh groups ('g' tag in .obj file) */
    unsigned int                group_count;
    fastObjGroup*               groups;

} fastObjMesh;

typedef struct
{
    void*                       (*file_open)(const char* path, void* user_data);
    void                        (*file_close)(void* file, void* user_data);
    size_t                      (*file_read)(void* file, void* dst, size_t bytes, void* user_data);
    unsigned long               (*file_size)(void* file, void* user_data);
} fastObjCallbacks;

#ifdef __cplusplus
extern "C" {
#endif

fastObjMesh*                    fast_obj_read(const char* path);
fastObjMesh*                    fast_obj_read_with_callbacks(const char* path, const fastObjCallbacks* callbacks, void* user_data);
void                            fast_obj_destroy(fastObjMesh* mesh);

#ifdef __cplusplus
}
#endif

#endif