/**
 * @author Jason Fu
 */

#ifndef UTILS_RANDOM_HPP
#define UTILS_RANDOM_HPP

#include "random"
#include "vecmath.h"

/**
 * Seeds the random number generator.
 * @param s seed
 */
inline void seed(unsigned int s) {
    srand(s);
}

/**
 * Generates a random double between 0 and 1 following the uniform distribution.
 * @return a random double [0,1]
 */
inline double uniform01() {
    return (double) rand() / (double) RAND_MAX;
}

/**
 * Randomly choose a point in a circle
 * @param radius radius of the circle
 * @return a random point in the circle
 */
 inline std::pair<float, float> randomPointInCircle(float radius) {
     //randomly choose a length
     float r = uniform01() * radius;
     //randomly choose an angle
     float theta = uniform01() * 2 * M_PI;
     //calculate the x and y
     float x = r * std::cos(theta);
     float y = r * std::sin(theta);
     return std::make_pair(x, y);
 }

/**
 * clamp the data between min and max.
 * 1. if data > max, then return max
 * 2. if data < min, then return min
 * 3  else, return data
 * @param x
 * @param min
 * @param max
 * @return a double between [min, max]
 * @author Jason Fu
 * @copybrief inspired by smallpt
 */
inline float clamp(float x, float min, float max) {
    return std::min(max, std::max(min, x));
}

inline Vector3f clamp(const Vector3f &x, float min, float max) {
    return Vector3f(clamp(x.x(), min, max), clamp(x.y(), min, max), clamp(x.z(), min, max));
}

/**
 * convert a double intensity to RGB color using gamma correction
 * @param x intensity
 * @param gamma gamma correction, default = 2.2
 * @return a RGB color [0, 255]
 * @author Jason Fu
 * @copybrief inspired by smallpt
 */
inline int toRGB(double x, double gamma = 2.2) {
    return int(std::pow(clamp(x, 0, 1), 1 / gamma) * 255 + 0.5);
}

/**
 * check if the string ends with the ending string
 * @param fullString
 * @param ending
 * @return true if fullString ends with ending, false otherwise
 */
inline bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

/**
 * calculate x % y
 * @param x
 * @param y
 * @return [0, y]
 */
inline double mod(double x, double y) {
    return fmod(fmod(x, y) + y, y);
}


/**
 * calculate k!
 * @param k
 * @return
 */
inline int fac(int k) {
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
inline int comb(int k, int n) {
    return fac(n) / (fac(k) * fac(n - k));
}


inline float square(float x) {
    return x * x;
}


#endif // UTILS_RANDOM_HPP