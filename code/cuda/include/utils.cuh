//
// Created by Jason Fu on 24-7-12.
//

#ifndef FINALPROJECTCUDA_UTILS_CUH
#define FINALPROJECTCUDA_UTILS_CUH

#include "Vector3f.cuh"
#include "string"
#include "curand_kernel.h"

/**
 * random number generator.
 * @param s seed
 */
__device__ inline float randf(unsigned long long &seed) {
    seed = (unsigned long long) 0x5deece66d * seed + 0xb & (((unsigned long long) 1 << 48) - 1);
    return (seed >> 18) / ((float) ((unsigned long long) 1 << 30));
}

/**
 * Generates a random float between 0 and 1 following the uniform distribution.
 * @return a random float [0,1]
 */
__device__ inline float uniform01(curandState *state) {
    auto rng = curand_uniform(state);
    return rng;
}

/**
 * Randomly choose a point in a circle
 * @param radius radius of the circle
 * @return a random point in the circle
 */
__device__ inline float2 randomPointInCircle(float radius, curandState *state) {
    //randomly choose a length
    float r = uniform01(state) * radius;
    //randomly choose an angle
    float theta = uniform01(state) * 2 * M_PI;
    //calculate the x and y
    float x = r * std::cos(theta);
    float y = r * std::sin(theta);
    return {x, y};
}

/**
 * Randomly choose a point in a sphere
 * @param radius radius of the sphere
 * @return a random point in the sphere
 */
__device__ inline Vector3f randomPointInSphere(float radius, curandState *state) {
    // randomly choose two angles
    float theta = uniform01(state) * 2 * M_PI;
    float phi = uniform01(state) * M_PI;
    // randomly choose a length
    float r = uniform01(state) * radius;
    // calculate the x, y, z
    float x = r * std::sin(phi) * std::cos(theta);
    float y = r * std::sin(phi) * std::sin(theta);
    float z = r * std::cos(phi);
    return {x, y, z};
}

/**
 * clamp the data between min and max.
 * 1. if data > max, then return max
 * 2. if data < min, then return min
 * 3  else, return data
 * @param x
 * @param min
 * @param max
 * @return a float between [min, max]
 * @author Jason Fu
 * @copybrief inspired by smallpt
 */
__device__ __host__ inline float clamp(float x, float min = 0, float max = 1) {
    return fminf(max, fmaxf(min, x));
}

__device__ __host__ inline Vector3f clamp(const Vector3f &x, float min = 0, float max = 1) {
    return {clamp(x._x, min, max), clamp(x._y, min, max), clamp(x._z, min, max)};
}

/**
 * convert a float intensity to RGB color using gamma correction
 * @param x intensity
 * @param gamma gamma correction, default = 2.2
 * @return a RGB color [0, 255]
 * @author Jason Fu
 * @copybrief inspired by smallpt
 */
__device__ __host__ int toRGB(float x, float gamma = 2.2);

/**
 * check if the string ends with the ending string
 * @param fullString
 * @param ending
 * @return true if fullString ends with ending, false otherwise
 */
__host__ bool hasEnding(std::string const &fullString, std::string const &ending);

/**
 * calculate x % y
 * @param x
 * @param y
 * @return [0, y]
 */
__device__ __host__ inline float mod(float x, float y) {
    return fmodf(fmodf(x, y) + y, y);
}


/**
 * calculate k!
 * @param k
 * @return
 */
__device__ __host__ inline int fac(int k) {
    int res = 1;
    for (int i = 1; i <= k; i++) {
        res *= i;
    }
    return res;
}


/**
 * calculate the combination number C_n^k
 * @param k
 * @param n
 * @return
 */
__device__ __host__ inline int comb(int k, int n) {
    return fac(n) / (fac(k) * fac(n - k));
}

__device__ __host__ inline float square(float x) {
    return x * x;
}

#endif //FINALPROJECTCUDA_UTILS_CUH
