#include <cstdlib>
#include <iostream>

#include "chrono"
#include "scene_parser.cuh"
#include "renderer.cuh"



int main(int argc, char *argv[]) {

    /**
 * Program entry point.
 * @author Jason Fu
 */
    std::cout << "==============" << std::endl;
    for (int argNum = 1; argNum < argc; ++argNum) {
        std::cout << "Argument " << argNum << " is: " << argv[argNum] << std::endl;
    }

    if (argc < 4) {
        std::cout << "Usage: ./bin/FinalProject <render> <input scene file> <output bmp file> <samples>" << std::endl;
        return 1;
    }
    std::string renderType = argv[1];
    std::string inputFile = argv[2];
    std::string outputFile = argv[3];  // only bmp is allowed.
    int samples = (argc >= 5) ? atoi(argv[4]) / 4 : 1;

    // First, parse the scene using SceneParser.

    SceneParser parser(inputFile.c_str());
    auto stime = std::chrono::high_resolution_clock::now();

    // Create the renderer.
    Renderer renderer(&parser, outputFile, samples);
    renderer.render();

    // timer
    auto etime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(etime - stime).count();
    cout << "time: " << (double) duration / 1000 << " s" << endl;
    return 0;
}