/**
 * Material Class Implementation
 * @author Jason Fu
 *
 */
#include "material.hpp"
#include "object3d.hpp"
#include "utils.hpp"

Vector3f BRDFMaterial::Shade(const Ray &ray, const Hit &hit, const Vector3f &dirToLight, const Vector3f &color) {
    // 如果具有材质，则计算映射
    if (texture != nullptr) {
        // 子物体不能为空
        assert(object != nullptr);
        // 计算纹理映射
        auto oPoint = ray.pointAtParameter(hit.getT());
        auto tPoint = object->textureMap(oPoint.x(), oPoint.y(), oPoint.z(), texture->Width(), texture->Height());
        int x = std::max(0, std::min(tPoint.first, texture->Width() - 1));
        int y = std::max(0, std::min(tPoint.second, texture->Height() - 1));
        // 计算纹理颜色
        return texture->GetPixel(x, y);
    }
    // 否则，直接返回漫反射颜色
    return diffuseColor;
}

Vector3f PhongMaterial::Shade(const Ray &ray, const Hit &hit, const Vector3f &dirToLight, const Vector3f &lightColor) {
    Vector3f diffuse_color = diffuseColor;
    // 如果具有材质，则计算映射
    if (texture != nullptr) {
        // 子物体不能为空
        assert(object != nullptr);
        // 计算纹理映射
        auto oPoint = ray.pointAtParameter(hit.getT());
        auto tPoint = object->textureMap(oPoint.x(), oPoint.y(), oPoint.z(), texture->Width(), texture->Height());
        int x = std::max(0, std::min(tPoint.first, texture->Width() - 1));
        int y = std::max(0, std::min(tPoint.second, texture->Height() - 1));
        // 计算纹理颜色
        return texture->GetPixel(texture->Width() - 1 - x, y);
    }
    // 否则，按Phong模型着色
    Vector3f R = (2 * Vector3f::dot(dirToLight, hit.getNormal()) * hit.getNormal() - dirToLight).normalized();
    Vector3f shaded = diffuse_color * std::max(Vector3f::dot(dirToLight, hit.getNormal()), 0.0f)
                      + specularColor * std::pow(std::max(-Vector3f::dot(R, ray.getDirection()), 0.0f), shininess);
    shaded = lightColor * shaded;
    return shaded;
}

void Material::setTexture(const char *filename) {
    std::string path = std::string(filename);
    if (hasEnding(path, ".tga")) {
        texture = Image::LoadTGA(filename);
    } else if (hasEnding(path, ".ppm")) {
        texture = Image::LoadPPM(filename);
    } else if (hasEnding(path, ".png")) {
        texture = Image::LoadPNG(filename);
    } else {
        texture = nullptr;
        std::cerr << "Unsupported texture format : must be one of .tga or .ppm" << std::endl;
    }
}
