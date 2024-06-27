#ifndef FINALPROJECT_REVSURFACE_HPP
#define FINALPROJECT_REVSURFACE_HPP

#include "object3d.hpp"
#include "curve.hpp"
#include "boundingBox.hpp"
#include <tuple>

#define GN_MAX_ITER 100
#define GN_ERROR 1e-3
#define GN_STEP 0.1

// Currently, revsurface does not support transparent material
// NOTE : THIS CLASS ONLY SUPPORT SPECIFIC SCENES
class RevSurface : public Object3D {

    Curve *pCurve;
    BoundingBox *bBox;
    int direction; // 1 - up, -1 - down
    float curve_x_max, curve_y_min, curve_y_max;

public:
    RevSurface(Curve *pCurve, Material *material) : pCurve(pCurve), Object3D(material) {
        curve_x_max = 0;
        curve_y_max = pCurve->getControls()[0].y();
        curve_y_min = pCurve->getControls()[0].y();
        // Check flat.
        for (const auto &cp: pCurve->getControls()) {
            if (cp.z() != 0.0) {
                printf("Profile of revSurface must be flat on xy plane.\n");
                exit(0);
            }
            // Update bounds
            if (std::abs(cp.x()) > curve_x_max) curve_x_max = std::abs(cp.x());
            if (cp.y() > curve_y_max) curve_y_max = cp.y();
            if (cp.y() < curve_y_min) curve_y_min = cp.y();
        }
        // create bounding box
        bBox = new BoundingBox(-curve_x_max, curve_x_max, curve_y_min, curve_y_max, -curve_x_max, curve_x_max);
        // judge direction
        if (curve_y_max > pCurve->getControls()[0].y()) direction = 1;
        else direction = -1;
        printf("bbox: %f %f %f %f %f %f\n", bBox->x0, bBox->y0, bBox->z0, bBox->x1, bBox->y1, bBox->z1);

        for (int i = 0; i < 20; i++) {
            Curve::CurvePoint cp;
            pCurve->getDataAt(i * pCurve->max_t / 20, cp);
            printf("%f %f %f\n", cp.V.x(), cp.V.y(), cp.V.z());
        }

    }

    ~RevSurface() override {
        delete pCurve;
    }

    /**
     * Intersect with the revsurface using Newton's method
     * @author Jason Fu
     *
     */
    bool intersect(const Ray &r, Hit &h, float tmin) override {
        // intersect with the bounding box
        float tmax = 1e38;
        if (bBox->isIntersect(r, tmin, tmax)) {
            //find t0
            float t0 = pCurve->min_t, t1 = pCurve->max_t, tr = -1;
            Curve::CurvePoint finalP;
            if (Newton(r, t0)) {
                Curve::CurvePoint cp;
                pCurve->getDataAt(t0, cp);
                float t = (cp.V.y() - r.getOrigin().y()) / r.getDirection().y();
                if (t > tr) {
                    tr = t;
                    finalP = cp;
                }
            }
            if (Newton(r, t1)) {
                Curve::CurvePoint cp;
                pCurve->getDataAt(t1, cp);
                float t = (cp.V.y() - r.getOrigin().y()) / r.getDirection().y();
                if (tr < 0 || t < tr) {
                    tr = t;
                    finalP = cp;
                }
            }
            if (tr > tmin) {
                // calculate the point at t0
                float t = tr;
                if (t > tmin) {
                    // calculate tangent
                    // tangent on zOx
                    auto p = r.pointAtParameter(t);
                    float sin_theta = t * r.getDirection().x() + r.getOrigin().x();
                    float cos_theta = t * r.getDirection().z() + r.getOrigin().z();
                    Vector3f u(cos_theta, 0, -sin_theta);
                    // tangent on xOy
                    Vector3f v(finalP.T.x(), finalP.T.y(), 0);
                    // normal
                    Vector3f n = Vector3f::cross(u, v).normalized() * direction;
                    // save to hit
                    h.set(t, material, n, false);
                    return true;
                }
            }
        }
        return false;
    }

    std::pair<int, int>
    textureMap(float objectX, float objectY, float objectZ, int textureWidth, int textureHeight) override {
        // calculate the texture coordinate
        float theta = std::atan2(objectX, objectZ) + M_PI;
        float mu = (objectY - curve_y_min) / (curve_y_max - curve_y_min);
        return {int(theta / (2 * M_PI) * textureWidth), int(mu * textureHeight)};
    }

private:
    /**
     * Find the intersection point of the ray with the surface using Newton iteration.
     * @param r the ray
     * @param t the parameter on the xy-curve, the original number is t0.
     * @return whether the iteration converges.
     * @author Jason Fu
     */
    bool Newton(const Ray &r, float &t) {
        int iter = 0;
        while (iter < GN_MAX_ITER) {
            // calculate f df
            float f, df;
            fdf(r, t, f, df);
            // judge if the iteration converges
            if (f < GN_ERROR) {
                return true;
            } else {
                // update t
                t = clamp(t - GN_STEP * f / df, pCurve->min_t, pCurve->max_t);
            }
            ++iter;
        }
        return false;
    }

    /**
     * Target function : f(t) = ((y(t)-oy)dz/dy + oz)^2 + ((y(t)-oy)dx/dy + ox)^2 - x(t)^2 )^2
     * @param r
     * @param t
     * @return f(t), df(t)
     * @author Jason Fu
     * @acknowledgement PA2 习题课
     */
    inline void fdf(const Ray &r, float t, float &f, float &df) {
        Curve::CurvePoint cp;
        pCurve->getDataAt(t, cp);
        float xt = cp.V.x();
        float yt = cp.V.y();
        float dxt = cp.T.x();
        float dyt = cp.T.y();
        float ox = r.getOrigin().x();
        float oy = r.getOrigin().y();
        float oz = r.getOrigin().z();
        float dx = r.getDirection().x();
        float dy = r.getDirection().y();
        float dz = r.getDirection().z();
        // calculate f
        float a = (yt - oy) * dz / dy + oz;
        float b = (yt - oy) * dx / dy + ox;
        f = a * a + b * b - xt * xt;
        // calculate df
        df = 2 * f * (2 * a * dyt * dz / dy + 2 * b * dyt * dx / dy - 2 * xt * dxt);
        f = f * f;
    }


};

#endif //FINALPROJECT_REVSURFACE_HPP
