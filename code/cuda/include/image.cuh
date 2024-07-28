//
// Created by Jason Fu on 24-7-12.
//

#ifndef FINALPROJECTCUDA_IMAGE_CUH
#define FINALPROJECTCUDA_IMAGE_CUH


#include <cassert>
#include "Vector3f.cuh"

// Simple image class
class Image {
public:

    __device__ __host__ inline Image(int w, int h) {
        width = w;
        height = h;
        data = new Vector3f[width * height];
        a = new float[width * height];
        for (int i = 0; i < width * height; i++) {
            a[i] = 1;
        }
    }

    __device__ __host__ inline Image(const Image &img) {
        width = img.width;
        height = img.height;
        data = new Vector3f[width * height];
        a = new float[width * height];
        for (int i = 0; i < width * height; i++) {
            data[i] = img.data[i];
            a[i] = img.a[i];
        }
    }

    __device__ __host__ inline ~Image() {
        delete[] data;
        delete[] a;
    }

    __device__ __host__ inline int Width() const {
        return width;
    }

    __device__ __host__ inline int Height() const {
        return height;
    }

    __device__ __host__ inline const Vector3f &GetPixel(int x, int y) const {
        if (x == width) --x;
        if (y == height) --y;
        if (x < 0 || x >= width || y < 0 || y >= height) {
            printf("Warning: pixel (%d, %d) out of bounds (%d, %d)\n", x, y, width, height);
        }
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        return data[y * width + x];
    }

    __device__ __host__ inline float GetAlpha(int x, int y) const {
        if (x == width) --x;
        if (y == height) --y;
        if (x < 0 || x >= width || y < 0 || y >= height) {
            printf("Warning: pixel (%d, %d) out of bounds (%d, %d)\n", x, y, width, height);
        }
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        return a[y * width + x];
    }

    __device__ __host__ inline void SetAllPixels(const Vector3f &color) {
        for (int i = 0; i < width * height; ++i) {
            data[i] = color;
        }
    }

    __device__ __host__ inline void SetPixel(int x, int y, const Vector3f &color, float alpha = 1.0f) {
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        data[y * width + x] = color;
        a[y * width + x] = alpha;
    }

    __host__ static Image *LoadPPM(const char *filename);

    __host__ void SavePPM(const char *filename) const;

    __host__ static Image *LoadTGA(const char *filename);

    // Powered by https://github.com/lvandeve/lodepng
    __host__ static Image *LoadPNG(const char *filename);

    __host__ void SaveTGA(const char *filename) const;

    __host__ int SaveBMP(const char *filename);

    __host__ void SaveImage(const char *filename);

private:

    int width;
    int height;
    Vector3f *data; // RGB [0,1]
    float *a;       // Alpha [0,1]

};

#endif //FINALPROJECTCUDA_IMAGE_CUH
