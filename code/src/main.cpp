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

#include <string>

using namespace std;

int main(int argc, char *argv[]) {
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
    for (int x = 0; x < camera->getWidth(); x++) {
        for (int y = 0; y < camera->getHeight(); y++) {
            Ray camRay = camera->generateRay(Vector2f(x, y));
            // whitted-style ray tracing
            Vector3f color = intersectColor_whitted_style(group, &camRay, lights, parser.getBackgroundColor(),
                                                          false, 0);
            image.SetPixel(x, camera->getHeight() - 1 - y, color);

        }
    }

    // 存储图片
    image.SaveImage(outputFile.c_str());
    cout << "Dumped" << endl;
    return 0;
}

