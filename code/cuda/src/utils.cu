//
// Created by Jason Fu on 24-7-12.
//

#include "utils.cuh"


__device__ __host__ int toRGB(float x, float gamma) {
    return int(powf(clamp(x, 0, 1), 1 / gamma) * 255 + 0.5);
}

__host__ bool hasEnding(const std::string &fullString, const std::string &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}


