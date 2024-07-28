//
// Created by Jason Fu on 24-7-28.
//

#include "renderer.cuh"

void Renderer::render() {

    Camera **deviceCamera = parser->deviceCamera;
    Group **deviceGroup = parser->deviceGroup;
    Material **deviceMaterials = parser->deviceMaterials;
    Light **deviceLights = parser->deviceLights;
    int num_lights = parser->num_lights;
    Vector3f backgroundColor = parser->background_color;

    int width = parser->camera->getWidth();
    int height = parser->camera->getHeight();
    Image image(width, height);

    // pixels
    auto *hostPixels = new Vector3f[width * height];
    Vector3f *devicePixels;
    auto err = cudaMalloc(&devicePixels,
                          sizeof(Vector3f) * width * height);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Loop over each pixel in the image, shooting a ray through that pixel .
    // Write the color at the intersection to that pixel in your output image.
    assert(width % 16 == 0 && height % 16 == 0);
    assert(samples % 4 == 0);

    dim3 gridSize(width / 16, height / 16, samples / 4);
    dim3 blockSize(16, 16, 1);

    // render
    renderPixel<<<gridSize, blockSize>>>(devicePixels, deviceCamera, deviceMaterials, deviceGroup, deviceLights,
                                         backgroundColor, width, height, num_lights, samples);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
        exit(3);
    }

    // copy the result back
    err = cudaMemcpy(hostPixels, devicePixels, sizeof(Vector3f) * width * height, cudaMemcpyDeviceToHost);

    // set the image file
    for (int i = 0; i < width * height; i++) {
        image.SetPixel(i % width, i / width, hostPixels[i]);
    }

    // dump the image file
    image.SaveImage(outputFile.c_str());
    printf("\nDumped.\n");

    // free up
    cudaFree(devicePixels);
    delete[] hostPixels;

}

__device__ Vector3f
Renderer::intersectColorPt(Group *group, Ray ray, Light **lights, int num_lights, Material **materials,
                           Vector3f backgroundColor, curandState *state) {
    int depth = 0;
    Vector3f finalColor(0, 0, 0);
    Vector3f prevColor(1, 1, 1);
    Vector3f neeColor(0, 0, 0);
    while (true) {
        Hit hit;
        // 求交
        bool intersect = group->intersect(ray, hit, DISTURBANCE);
        if (!intersect) {// 未相交则返回背景色
            finalColor += prevColor * backgroundColor;
            break;
        } else { // 相交则计算交点
            auto obj = hit.getObject();

            auto material = obj->getMaterial();
            int type = material->getType();
            Vector3f color = material->Shade(ray, hit, Vector3f::ZERO(), Vector3f::ZERO());
            Vector3f e_color = material->getEmissionColor() + neeColor;
            neeColor = Vector3f::ZERO();

            // 累积光源颜色
            finalColor += prevColor * e_color;

            float p = fmaxf(color._x, fmaxf(color._y, color._z)) / 1.25f;
            // 根据RR决定是否终止(5层递归之后才开始判断)
            if (++depth > 5) {
                if (uniform01(state) < p) { // 越亮的物体计算次数越多
                    color = color / p;
                } else {
                    break;
                }
            }

            // 更新系数
            prevColor = prevColor * color;

            // 生成下次光线
            if (type == Material::DIFFUSE) { // 漫反射
                // 随机生成一个漫反射曲线
                float r1 = 2.0f * M_PI * uniform01(state);
                float r2 = uniform01(state), r2s = std::sqrt(r2);
                // 生成正交坐标系 (w, u, v)
                Vector3f w = hit.getNormal();
                Vector3f u = (Vector3f::cross((fabs(w._x) > 0.1 ? Vector3f(0, 1, 0) : Vector3f(1, 0, 0)),
                                              w)).normalized();
                Vector3f v = Vector3f::cross(w, u).normalized();
                // 生成漫反射光线
                Vector3f dir = (u * std::cos(r1) * r2s + v * std::sin(r1) * r2s +
                                w * std::sqrt(1 - r2)).normalized();
                ray = Ray(ray.pointAtParameter(hit.getT() - DISTURBANCE), dir);
                // 对光源采样（NEE）
                for (int j = 0; j < num_lights; j++) {
                    auto light = lights[j];
                    Vector3f ldir, lc;
                    // 计算光源方向和颜色
                    light->getIllumination(ray.pointAtParameter(hit.getT()), ldir, lc, state);
                    // 计算光源是否被遮挡
                    if (!light->isInShadow(ray.pointAtParameter(hit.getT()), group, -ldir)) {
                        neeColor += lc * fmaxf(Vector3f::dot(ldir, hit.getNormal()), 0.0f);
                    }
                }
            } else if (type == Material::GLOSSY) { // glossy 材质

                const Vector3f &N = hit.getNormal();
                Vector3f V = -ray.getDirection().normalized();

                // 对H采样
                Vector3f H = material->sampleGGXHemisphere(N, state);

                // 计算出射光线L
                Vector3f L = (2.0F * Vector3f::dot(V, H) * H - V).normalized();

                // 计算出射光线L是否被遮挡
                if (Vector3f::dot(L, N) > 0.0f) {
                    ray = Ray(ray.pointAtParameter(hit.getT() - DISTURBANCE), L);
                    prevColor = prevColor * material->CookTorranceBRDF(L, V, N);
                } else {
                    break;
                }
            } else if (type == Material::SPECULAR) { // 镜面反射
                // 生成反射光线
                ray = reflect(ray, hit.getNormal(), ray.pointAtParameter(hit.getT() - DISTURBANCE));
            } else if (type == Material::TRANSPARENT) { // 折射
                // 注意判断光线是否在物体内部
                bool is_inside = hit.isInside();
                float n1 = (is_inside) ? material->getRefractiveIndex() : 1;
                float n2 = (is_inside) ? 1 : material->getRefractiveIndex();
                // 折射光
                Ray rfr_ray = refract(ray, hit.getNormal(), ray.pointAtParameter(hit.getT() + DISTURBANCE),
                                      n1, n2);
                // 反射光
                Ray rfl_ray = reflect(ray, hit.getNormal(), ray.pointAtParameter(hit.getT() - DISTURBANCE));
                if (rfr_ray == Ray::ZERO()) { // 发生全反射
                    ray = rfl_ray;
                } else { // 根据菲涅尔反射函数计算
                    float a = (n1 > n2) ? (n1 / n2 - 1) : (n2 / n1 - 1);
                    float b = (n1 > n2) ? (n1 / n2 + 1) : (n2 / n1 + 1);
                    float R0 = (a * a) / (b * b);
                    float c = 1 - (is_inside ? fabsf(Vector3f::dot(rfr_ray.getDirection(), hit.getNormal()))
                                             : fabsf(Vector3f::dot(ray.getDirection(), hit.getNormal())));
                    float Re = R0 + (1 - R0) * pow(c, 5);
                    // 使用RR
                    float P = 0.25f + 0.5f * Re;
                    if (uniform01(state) < P) {
                        ray = rfl_ray;
                        prevColor *= Re / P;
                    } else {
                        ray = rfr_ray;
                        prevColor *= (1.0f - Re) / (1.0f - P);
                    }
                }
            }
        }
    }

    return finalColor;
}

__global__ void
renderPixel(Vector3f *pixels, Camera **dCam, Material **dMat, Group **dGrp, Light **dLgt, Vector3f backgroundColor,
            int width, int height, int num_lights, int num_samples) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = (height - y - 1) * width + x;

    unsigned long long seed = (((unsigned long long) y << 32) | ((unsigned long long) x << 16) |
                               blockIdx.z * blockDim.z + threadIdx.z);

    // initialize curand state
    curandState state;
    int rand_idx = threadIdx.x + blockIdx.x * blockDim.x
                   + threadIdx.y + blockIdx.y * blockDim.y
                   + threadIdx.z + blockIdx.z * blockDim.z;
    curand_init(seed, rand_idx, 0, &state);


    Vector3f color = Vector3f::ZERO();
    // anti-aliasing using 2*2 subpixel
    for (int sx = 0; sx < 2; sx++) {
        for (int sy = 0; sy < 2; sy++) {
            // use tent filter (inspired by smallpt)
            float r1 = 2 * uniform01(&state);
            float r2 = 2 * uniform01(&state);
            float dx = (r1 < 1) ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
            float dy = (r2 < 1) ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
            Ray camRay = (*dCam)->generateRay({x + (sx + dx) / 2.0f, y + (sy + dy) / 2.0f}, &state);
            // whitted-style ray tracing
            Vector3f sample_color = Renderer::intersectColorPt(*dGrp, camRay, dLgt, num_lights, dMat,
                                                               backgroundColor, &state)
                                    * (1.0f / num_samples);
            color += clamp(sample_color, 0.0, 1.0) * 0.25f;
        }
    }
    pixels[i] += color;
}

