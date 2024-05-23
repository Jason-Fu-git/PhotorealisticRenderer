/**
 * 光线跟踪函数实现
 * @author Jason Fu
 *
 */

#ifndef RAY_TRACING_HPP
#define RAY_TRACING_HPP

#include "ray.hpp"
#include "vecmath.h"
#include "material.hpp"
#include "group.hpp"
#include "light.hpp"
#include "vector"

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
 */
Vector3f intersectColor_whitted_style(Group *group, Ray *ray, std::vector<Light *> &lights, Vector3f backgroundColor,
                                      float weight, int depth) {
    if (weight < MIN_WEIGHT && depth > 0)
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
        }
        return finalColor;
    } else {
        return backgroundColor;
    }

}


#endif