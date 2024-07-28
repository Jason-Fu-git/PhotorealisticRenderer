//
// Created by Jason Fu on 24-7-28.
//

#ifndef FINALPROJECTCUDA_RENDERER_CUH
#define FINALPROJECTCUDA_RENDERER_CUH

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

#include "ray.cuh"
#include "Vector3f.cuh"
#include "material.cuh"
#include "group.cuh"
#include "light.cuh"
#include "vector"
#include "scene_parser.cuh"
#include "utils.cuh"
#include "camera.cuh"

#define MIN_WEIGHT 0.001

/**
 * Renderer class, which handles the rendering process.
 * @author Jason Fu
 */
class Renderer {
public:
    Renderer(SceneParser *parser, std::string output_path, int num_samples)
            : parser(parser), outputFile(std::move(output_path)), samples(num_samples) {}

    __device__ static Vector3f
    intersectColorPt(Group *group, Ray ray, Light **lights, int num_lights, Material **materials,
                     Vector3f backgroundColor,
                     curandState *state);

    virtual void render();

protected:
    SceneParser *parser;
    std::string outputFile;
    int samples;
    bool printProgress;
};

__global__ void
renderPixel(Vector3f *pixels, Camera **dCam, Material **dMat, Group **dGrp, Light **dLgt, Vector3f backgroundColor,
            int width, int height, int num_lights, int num_samples);

#endif //FINALPROJECTCUDA_RENDERER_CUH

