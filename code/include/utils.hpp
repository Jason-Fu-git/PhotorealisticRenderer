#ifndef UTILS_RANDOM_HPP
#define UTILS_RANDOM_HPP

#include "random"

/**
 * Seeds the random number generator.
 * @param s seed
 * @author Jason Fu
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
inline double clamp(double x, double min, double max) {
    return std::min(max, std::max(min, x));
}

inline Vector3f clamp(const Vector3f &x, double min, double max) {
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


#endif // UTILS_RANDOM_HPP