/**
 * Material Class Implementation
 * @author Jason Fu
 *
 */
#include "material.cuh"
#include "object3d.cuh"
#include "utils.cuh"
#include "cstdio"

__device__ Vector3f
LambertianMaterial::Shade(const Ray &ray, const Hit &hit, const Vector3f &dirToLight, const Vector3f &color) {
    // 如果具有材质，则计算映射
    if (texture != nullptr) {
        // 计算纹理映射
        auto oPoint = ray.pointAtParameter(hit.getT());
        auto tPoint = hit.getObject()->textureMap(oPoint._x, oPoint._y, oPoint._z, texture->Width(), texture->Height());
        int x = fmaxf(0, fminf(tPoint.x, texture->Width() - 1));
        int y = fmaxf(0, fminf(tPoint.y, texture->Height() - 1));
        // 计算纹理颜色
        return texture->GetPixel(x, y);
    }
    // 否则，直接返回漫反射颜色
    return diffuseColor;
}

__device__ Vector3f
PhongMaterial::Shade(const Ray &ray, const Hit &hit, const Vector3f &dirToLight, const Vector3f &lightColor) {
    Vector3f diffuse_color = diffuseColor;
    // 如果具有材质，则计算映射
    if (texture != nullptr) {
        // 计算纹理映射
        auto oPoint = ray.pointAtParameter(hit.getT());
        auto tPoint = hit.getObject()->textureMap(oPoint._x, oPoint._y, oPoint._z, texture->Width(), texture->Height());
        int x = fmaxf(0, fminf(tPoint.x, texture->Width() - 1));
        int y = fmaxf(0, fminf(tPoint.y, texture->Height() - 1));
        // 计算纹理颜色
        diffuse_color = texture->GetPixel(texture->Width() - 1 - x, y);
    }
    // 否则，按Phong模型着色
    Vector3f R = (2 * Vector3f::dot(dirToLight, hit.getNormal()) * hit.getNormal() - dirToLight).normalized();
    Vector3f shaded = diffuse_color * fmaxf(Vector3f::dot(dirToLight, hit.getNormal()), 0.0f)
                      + specularColor * powf(fmaxf(-Vector3f::dot(R, ray.getDirection()), 0.0f), shininess);
    shaded = lightColor * shaded;
    return shaded;
}

__host__ void Material::setTexture(const char *filename) {
    std::string path = std::string(filename);
    if (hasEnding(path, ".tga")) {
        texture = Image::LoadTGA(filename);
    } else if (hasEnding(path, ".ppm")) {
        texture = Image::LoadPPM(filename);
    } else if (hasEnding(path, ".png")) {
        texture = Image::LoadPNG(filename);
    } else {
        texture = nullptr;
        fprintf(stderr, "Unsupported texture format : must be one of .tga or .ppm");
    }
}


Vector3f
__device__
CookTorranceMaterial::Shade(const Ray &ray, const Hit &hit, const Vector3f &dirToLight, const Vector3f &lightColor) {
    // 如果具有材质，则计算映射
    if (texture != nullptr) {
        // 计算纹理映射
        auto oPoint = ray.pointAtParameter(hit.getT());
        auto tPoint = hit.getObject()->textureMap(oPoint._x, oPoint._y, oPoint._z, texture->Width(), texture->Height());
        int x = fmaxf(0.0f, fminf(tPoint.x, texture->Width() - 1));
        int y = fmaxf(0.0f, fminf(tPoint.y, texture->Height() - 1));
        // 计算纹理颜色
        return texture->GetPixel(x, y);
    }
    // 否则，直接返回1
    return {1, 1, 1};
}

__device__ Vector3f
CookTorranceMaterial::CookTorranceBRDF(const Vector3f &L, const Vector3f &V, const Vector3f &N) const {
    // half vector
    Vector3f H = (L + V).normalized();
    // Fresnel
    Vector3f F = fresnelSchlick(H, V);
    // distribution
    float D = distributionGGX(N, H);
    // geometry
    float G = geometrySmith(N, V, L);

    auto specular = (D * F * G) / (4 *
                                   fmaxf(Vector3f::dot(N, V), 0.0f) *
                                   fmaxf(Vector3f::dot(N, L), 0.0f) + 0.001f
    );

    return (d * diffuseColor + s * specular);
}

__global__ void
createPhongMaterialOnDevice(Material **material, Vector3f d_color, Vector3f s_color, float s) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *material = new PhongMaterial(d_color, s_color, s);
    }
}

__global__ void
createLambertianMaterialOnDevice(Material **material, Vector3f d_color, float rfr_c, float rfl_c, float rfr_i,
                                 int _type, Vector3f e_color) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*material) = new LambertianMaterial(d_color, rfr_c, rfl_c, rfr_i, _type, e_color);
    }
}

__global__ void
createCookTorranceMaterialOnDevice(Material **material, Vector3f d_color, float s, float d, float m,
                                   Vector3f F0) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *material = new CookTorranceMaterial(d_color, s, d, m, F0);
    }
}

__global__ void freeMaterialOnDevice(Material **material) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *material;
    }
}
