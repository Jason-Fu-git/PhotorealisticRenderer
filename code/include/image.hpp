/**
 * @copybrief 清华大学计算机图形学课程提供框架
 *
 */

#ifndef IMAGE_H
#define IMAGE_H

#include <cassert>
#include <vecmath.h>

// Simple image class
class Image {

public:

    Image(int w, int h) {
        width = w;
        height = h;
        data = new Vector3f[width * height];
        a = new float[width * height];
        for (int i = 0; i < width * height; i++) {
            a[i] = 1;
        }
    }

    Image(const Image &img) {
        width = img.width;
        height = img.height;
        data = new Vector3f[width * height];
        a = new float[width * height];
        for (int i = 0; i < width * height; i++) {
            data[i] = img.data[i];
            a[i] = img.a[i];
        }
    }

    ~Image() {
        delete[] data;
        delete[] a;
    }

    int Width() const {
        return width;
    }

    int Height() const {
        return height;
    }

    const Vector3f &GetPixel(int x, int y) const {
        if (x == width) --x;
        if (y == height) --y;
        if (x < 0 || x >= width || y < 0 || y >= height) {
            printf("Warning: pixel (%d, %d) out of bounds (%d, %d)\n", x, y, width, height);
        }
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        return data[y * width + x];
    }

    float GetAlpha(int x, int y) const {
        if (x == width) --x;
        if (y == height) --y;
        if (x < 0 || x >= width || y < 0 || y >= height) {
            printf("Warning: pixel (%d, %d) out of bounds (%d, %d)\n", x, y, width, height);
        }
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        return a[y * width + x];
    }

    void SetAllPixels(const Vector3f &color) {
        for (int i = 0; i < width * height; ++i) {
            data[i] = color;
        }
    }

    void SetPixel(int x, int y, const Vector3f &color, float alpha = 1.0f) {
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        data[y * width + x] = color;
        a[y * width + x] = alpha;
    }

    static Image *LoadPPM(const char *filename);

    void SavePPM(const char *filename) const;

    static Image *LoadTGA(const char *filename);

    // Powered by https://github.com/lvandeve/lodepng
    static Image *LoadPNG(const char *filename);

    void SaveTGA(const char *filename) const;

    int SaveBMP(const char *filename);

    void SaveImage(const char *filename);

private:

    int width;
    int height;
    Vector3f *data; // RGB [0,1]
    float *a;       // Alpha [0,1]

};

#endif // IMAGE_H
