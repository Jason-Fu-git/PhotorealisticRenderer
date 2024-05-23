#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>

#include "scene_parser.hpp"
#include "image.hpp"
#include "camera.hpp"
#include "group.hpp"
#include "light.hpp"
#include "ray_tracing.hpp"
#include "ctime"
#include "chrono"
#include "utils_random.hpp"

#include <string>

using namespace std;

/**
 * @author Jason Fu
 *
 */
int main(int argc, char *argv[]) {
    seed(time(nullptr));
    for (int argNum = 1; argNum < argc; ++argNum) {
        std::cout << "Argument " << argNum << " is: " << argv[argNum] << std::endl;
    }

    if (argc != 3) {
        cout << "Usage: ./bin/FinalProject <input scene file> <output bmp file>" << endl;
        return 1;
    }
    string inputFile = argv[1];
    string outputFile = argv[2];  // only bmp is allowed.

    // First, parse the scene using SceneParser.
    SceneParser parser(inputFile.c_str());
    Camera *camera = parser.getCamera();
    Group *group = parser.getGroup();
    Image image(camera->getWidth(), camera->getHeight());
    // Then, for each light in the scene, add it to a vector.
    std::vector<Light *> lights;
    lights.reserve(parser.getNumLights());
    for (int i = 0; i < parser.getNumLights(); i++) {
        lights.push_back(parser.getLight(i));
    }
    // Then loop over each pixel in the image, shooting a ray through that pixel .
    // Write the color at the intersection to that pixel in your output image.
    auto stime = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
    for (int x = 0; x < camera->getWidth(); x++) {
        for (int y = 0; y < camera->getHeight(); y++) {
            Vector3f color = Vector3f::ZERO;
            // anti-aliasing using 2*2 subpixel
            for (int sx = 0; sx < 2; sx++) {
                for (int sy = 0; sy < 2; sy++) {
                    // use tent filter (inspired by smallpt)
                    double r1 = 2 * uniform01();
                    double r2 = 2 * uniform01();
                    double dx = (r1 < 1) ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                    double dy = (r2 < 1) ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
                    Ray camRay = camera->generateRay(Vector2f(x + (sx + dx) / 2.0, y + (sy + dy) / 2.0));
                    // whitted-style ray tracing
                    color += intersectColor_whitted_style(group, &camRay, lights, parser.getBackgroundColor(), 1, 3) *
                             0.25;
                }
            }
            image.SetPixel(x, camera->getHeight() - 1 - y, color);
        }
    }

    // 存储图片
    image.SaveImage(outputFile.c_str());
    cout << "Dumped" << endl;

    // 计算时间
    auto etime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(etime - stime).count();
    cout << "time: " << (double)duration / 1000 << " s" << endl;
    return 0;
}

