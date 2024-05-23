#ifndef UTILS_RANDOM_HPP
#define UTILS_RANDOM_HPP

#include "random"

/**
 * Seeds the random number generator.
 * @param s seed
 * @author Jason Fu
 */
void seed(unsigned int s){
    srand(s);
}

/**
 * Generates a random double between 0 and 1 following the uniform distribution.
 * @return a random double [0,1]
 */
double uniform01(){
    return (double)rand() / (double)RAND_MAX;
}

#endif // UTILS_RANDOM_HPP