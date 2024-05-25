/**
 * 光线跟踪函数实现
 * @author Jason Fu
 *
 */

#ifndef RAY_TRACING_HPP
#define RAY_TRACING_HPP

#include <cmath>

#include "ray.hpp"
#include "vecmath.h"
#include "material.hpp"
#include "group.hpp"
#include "light.hpp"
#include "vector"
#include "utils.hpp"

#define MIN_WEIGHT 0.001

/**
 * 光线追踪的Whitted-style实现。
 * 注：两个物体不能挨在一起。
 * @param group 场景中的物体
 * @param ray 射线
 * @param lights 光源
 * @param backgroundColor 背景颜色
 * @param weight 当前着色的权重
 * @param depth 剩余递归深度
 * @return 多次反射/折射的累加，直至达到递归深度
 * @author Jason Fu
 */
Vector3f intersectColor_whitted_style(Group *group, Ray *ray, std::vector<Light *> &lights, Vector3f backgroundColor,
                                      float weight, int depth) {
    if (weight < MIN_WEIGHT || depth == 0)
        return Vector3f::ZERO;

    Hit hit;
    // 求交
    bool intersect = group->intersect(*ray, hit, DISTURBANCE);
    // 如果有交点
    if (intersect) {
        // 累加所有光源的影响
        Vector3f finalColor = Vector3f::ZERO;
        Material *material = hit.getMaterial();
        bool is_inside = hit.isInside();
        for (auto light: lights) {
            Vector3f L, lightColor;
            // 获得光照强度
            light->getIllumination(ray->pointAtParameter(hit.getT()), L, lightColor);
            // 计算局部光强（如果不是在物体内部，且不是在阴影中）
            if (!light->isInShadow(ray->pointAtParameter(hit.getT()), group))
                finalColor += material->Shade(*ray, hit, L, lightColor);
        }
        // 递归计算反射光
        if (material->isReflective()) {
            Ray *reflectionRay = reflect(*ray, hit.getNormal(), ray->pointAtParameter(hit.getT() - DISTURBANCE));
            finalColor += material->getReflectiveCoefficient() *
                          intersectColor_whitted_style(group, reflectionRay, lights, backgroundColor,
                                                       material->getReflectiveCoefficient() * weight, depth - 1);
            delete reflectionRay;
        }
        // 递归计算折射光
        if (material->isRefractive()) {
            // 注意判断光线是否在物体内部
            float n1 = (is_inside) ? material->getRefractiveIndex() : 1;
            float n2 = (is_inside) ? 1 : material->getRefractiveIndex();
            // 折射光
            Ray *refractionRay = refract(*ray, hit.getNormal(), ray->pointAtParameter(hit.getT() + DISTURBANCE),
                                         n1, n2);


            if (refractionRay != nullptr) { // 若不发生全反射
                finalColor += material->getRefractiveCoefficient() *
                              intersectColor_whitted_style(group, refractionRay, lights, backgroundColor,
                                                           material->getRefractiveCoefficient() * weight, depth - 1);
            }

            delete refractionRay;
        }
        return finalColor;
    } else {
        return backgroundColor;
    }

}

/**
 * 光线追踪的Monte-Carlo实现，cos-weighted采样， RR终止
 * @param group 场景中的物体
 * @param ray 射线
 * @param lights 光源
 * @param backgroundColor 背景颜色
 * @param depth 递归深度
 * @return 本次采样得到的颜色
 * @author Kevin Beason (smallpt)
 */
Vector3f intersectColor_monte_carlo(Group *group, const Ray &ray,
                                    const std::vector<Light *> &lights, Vector3f backgroundColor,
                                    int depth) {
    Hit hit;
    // 求交
    bool intersect = group->intersect(ray, hit, DISTURBANCE);
    if (!intersect) return backgroundColor; // 未相交则返回背景色
    Material *material = hit.getMaterial();
    int type = material->getType();
    Vector3f color = material->getDiffuseColor();
    Vector3f e_color = material->getEmissionColor();
    Vector3f final_color = Vector3f::ZERO;
    float p = std::max(color.x(), std::max(color.y(), color.z())) / 1.25;
    // 根据RR决定是否终止(5层递归之后才开始判断)
    if (++depth > 5) {
        if (uniform01() < p) { // 越亮的物体计算次数越多
            color = color / p;
        } else {
            return e_color;
        }
    }

    // 判断材料类型
    if (type == Material::DIFFUSE) { // 漫反射
        // 随机生成一个漫反射曲线
        float r1 = 2 * M_PI * uniform01();
        float r2 = uniform01(), r2s = std::sqrt(r2);
        // 生成正交坐标系 (w, u, v)
        Vector3f w = hit.getNormal();
        Vector3f u = ((std::fabs(w.x()) > 0.1 ? Vector3f(0, 1, 0) : Vector3f::cross(Vector3f(1, 0, 0),
                                                                                    w))).normalized();
        Vector3f v = Vector3f::cross(w, u).normalized();
        // 生成漫反射曲线
        Vector3f dir = (u * std::cos(r1) * r2s + v * std::sin(r1) * r2s + w * std::sqrt(1 - r2)).normalized();
        Ray rfl_ray = Ray(ray.pointAtParameter(hit.getT() - DISTURBANCE), dir);
        // 递归
        final_color = e_color + color * intersectColor_monte_carlo(group, rfl_ray, lights, backgroundColor, depth);
    } else if (type == Material::SPECULAR) { // 镜面反射
        // 生成反射光线
        Ray *rfl_ray = reflect(ray, hit.getNormal(), ray.pointAtParameter(hit.getT() - DISTURBANCE));
        final_color = e_color + color * intersectColor_monte_carlo(group, *rfl_ray, lights, backgroundColor, depth);
        delete rfl_ray;
    } else if (type == Material::TRANSPARENT) { // 折射
        // 注意判断光线是否在物体内部
        bool is_inside = hit.isInside();
        float n1 = (is_inside) ? material->getRefractiveIndex() : 1;
        float n2 = (is_inside) ? 1 : material->getRefractiveIndex();
        // 折射光
        Ray *rfr_ray = refract(ray, hit.getNormal(), ray.pointAtParameter(hit.getT() + DISTURBANCE),
                               n1, n2);
        // 反射光
        Ray *rfl_ray = reflect(ray, hit.getNormal(), ray.pointAtParameter(hit.getT() - DISTURBANCE));
        if (rfr_ray == nullptr) { // 发生全反射
            final_color = e_color + color * intersectColor_monte_carlo(group, *rfl_ray, lights, backgroundColor, depth);
        } else { // 根据菲涅尔反射函数计算
            double a = (n1 > n2) ? (n1 / n2 - 1) : (n2 / n1 - 1);
            double b = (n1 > n2) ? (n1 / n2 + 1) : (n2 / n1 + 1);
            double R0 = (a * a) / (b * b);
            double c = 1 - (is_inside ? std::fabs(Vector3f::dot(rfr_ray->getDirection(), hit.getNormal()))
                                      : std::fabs(Vector3f::dot(ray.getDirection(), hit.getNormal())));
            double Re = R0 + (1 - R0) * std::pow(c, 5);
            if (depth > 2) { // 两层递归后，使用RR
                double P = 0.25 + 0.5 * Re;
                if (uniform01() < P)
                    final_color = e_color +
                                  color * intersectColor_monte_carlo(group, *rfl_ray, lights, backgroundColor, depth) *
                                  Re / P;
                else
                    final_color = e_color +
                                  color * intersectColor_monte_carlo(group, *rfr_ray, lights, backgroundColor, depth) *
                                  (1 - Re) / (1 - P);

            } else { // 递归深度较浅时，使用两次递归
                final_color = e_color
                              + color * intersectColor_monte_carlo(group, *rfl_ray, lights, backgroundColor, depth)
                                * Re
                              + color * intersectColor_monte_carlo(group, *rfr_ray, lights, backgroundColor, depth)
                                * (1 - Re);
            }
        }
        delete rfl_ray;
        delete rfr_ray;
    }
    return final_color;
}


#endif