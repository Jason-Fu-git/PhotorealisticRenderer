#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>

#include "scene_parser.hpp"
#include "image.hpp"
#include "camera.hpp"
#include "group.hpp"
#include "light.hpp"
#include "render.hpp"
#include "ctime"
#include "chrono"
#include "utils.hpp"

#include <string>

using namespace std;

long long COUNT = 0;

/**
 * @author Jason Fu
 *
 */
int main(int argc, char *argv[]) {
    seed(time(nullptr));
    cout << "==============" << endl;
    for (int argNum = 1; argNum < argc; ++argNum) {
        std::cout << "Argument " << argNum << " is: " << argv[argNum] << std::endl;
    }

    if (argc < 4) {
        cout << "Usage: ./bin/FinalProject <render> <input scene file> <output bmp file> <samples>" << endl;
        return 1;
    }
    string renderType = argv[1];
    string inputFile = argv[2];
    string outputFile = argv[3];  // only bmp is allowed.
    int samples = (argc >= 5) ? atoi(argv[4]) / 4 : 1;

    // First, parse the scene using SceneParser.
    SceneParser parser(inputFile.c_str());
    auto stime = std::chrono::high_resolution_clock::now();

    // Create the renderer.
    Renderer *renderer;
    if (renderType == "whitted")
        renderer = new WhittedRenderer(&parser, outputFile, samples);
    else if (renderType == "monteCarlo")
        renderer = new MonteCarloRenderer(&parser, outputFile, samples);
    else {
        printf("Unknown render type: %s\n", renderType.c_str());
        exit(-1);
    }
    renderer->render();

    // 计算时间
    auto etime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(etime - stime).count();
    cout << "time: " << (double) duration / 1000 << " s" << endl;
    cout << COUNT << endl;
    return 0;
}

