//
// Created by Jason Fu on 24-6-14.
//

#ifndef FINALPROJECT_REVSURFACE_HPP
#define FINALPROJECT_REVSURFACE_HPP

#include "object3d.hpp"
#include "curve.hpp"
#include "boundingBox.hpp"
#include "mesh.hpp"
#include <tuple>

#define GN_MAX_ITER 20
#define GN_ERROR 1e-3
#define GN_STEP 0.1
#define NUM_SEARCHES 100
#define DISCRETE_SIZE 30


typedef Curve::CurvePoint CurvePoint;

// currently, revsurface does not support transparent material
class RevSurface : public Object3D {

    Curve *pCurve;
    BoundingBox *bBox;
    Mesh *mesh;
    std::vector<CurvePoint> curvePoints;


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
        // create mesh
        initMesh();
    }

    ~RevSurface() override {
        delete pCurve;
    }


    bool intersect(const Ray &r, Hit &h, float tmin) override {
//        if (abs(r.getDirection().y()) < TOLERANCE)
//            return false;
        // intersect with the bounding box
        float min_t = 1e38, max_t = -1e38;
        if (bBox->isIntersect(r, min_t, max_t)) {
            if (mesh->intersect(r, h, tmin)) {
                // find tc
                auto mesh_p = r.pointAtParameter(h.getT());
                float t0 = 0;
                for (int i = 1; i < curvePoints.size(); i++) {
                    if ((curvePoints[i].V.y() >= mesh_p.y() && curvePoints[i - 1].V.y() <= mesh_p.y()
                         || curvePoints[i].V.y() <= mesh_p.y() && curvePoints[i - 1].V.y() >= mesh_p.y())) {
                        t0 = pCurve->min_t + i / (float) curvePoints.size() * (pCurve->max_t - pCurve->min_t);
                        break;
                    }
                }
//                printf(" t0: %f ", t0);
                // GN iteration
                if (Newton(r, t0, 0, 1)) {
                    // calculate the point at t0
                    CurvePoint cp;
                    pCurve->getDataAt(t0, cp);
                    float t = (cp.V.y() - r.getOrigin().y()) / r.getDirection().y();
                    float begin = max(pCurve->min_t, t0 - 1 / (float) curvePoints.size() * (pCurve->max_t - pCurve->min_t));
                    float end = min(pCurve->max_t, t0 + 1 / (float) curvePoints.size() * (pCurve->max_t - pCurve->min_t));
//                    printf(" t: %f\n", t0);
//                    printf("ht: %f, t : %f\n", h.getT(), t);

                    // search around
                    for(int i = 0; i < NUM_SEARCHES; i++){
                        float ti = begin + (end - begin) * (float) i / (float) NUM_SEARCHES;
                        if(Newton(r, ti, 0, 1)){
                            CurvePoint cpi;
                            pCurve->getDataAt(ti, cpi);
                            float tri = (cpi.V.y() - r.getOrigin().y()) / r.getDirection().y();
                            if(tri < t){
                                t = tri;
                                cp = cpi;
                                break;
                            }
                        }
                    }
                    if (t > tmin) {
                        // calculate tangent
                        auto p = r.pointAtParameter(t);
                        float sin_theta = t * r.getDirection().x() + r.getOrigin().x();
                        float cos_theta = t * r.getDirection().z() + r.getOrigin().z();
                        Vector3f u(cos_theta, 0, -sin_theta);
                        // tangent on xOy
                        Vector3f v(cp.T.x(), cp.T.y(), 0);
                        // normal
                        Vector3f n = Vector3f::cross(u, v).normalized() * direction;
                        // save to hit
                        h.set(t, material, n, false);
                        return true;
                    }
                }
            }
        }
        return false;
    }

private:

    // intersect with the plane when dy = 0
    bool intersect_y0(const Ray &r, Hit &h, float tmin){
        float yt = r.getOrigin().y();
        // find the corresponding t using newton method


        return false;
    }


    bool GN_iteration_3D(const Ray &r, float &tr, float &tc, float theta){
        int iter = 0;
        while (iter < GN_MAX_ITER) {
            ++iter;

            // calculate f
            auto pr = r.pointAtParameter(tr);
            CurvePoint cp;
            pCurve->getDataAt(tr, cp);
            float x = cp.V.x() * sin(theta);
            float y = cp.V.y();
            float z = cp.V.x() * cos(theta);
            float f = (x - pr.x()) * (x - pr.x()) + (y - pr.y()) * (y - pr.y()) + (z - pr.z()) * (z - pr.z());
            if (f< GN_ERROR)
                return true;

            // calculate df/dtr df/dtc  df/dtheta
            float df_dtr = 2 * (pr.x() - x) * r.getDirection().x()
                            + 2 * (pr.y() - y) * r.getDirection().y()
                                + 2 * (pr.z() - z) * r.getDirection().z();
            float df_dtc = 2 * (x - pr.x()) * sin(theta) * cp.T.x()
                                + 2 * (y - pr.y()) * cp.T.y()
                                    + 2 * (z - pr.z()) * cos(theta) * cp.T.x();
            float df_dtheta = 2 * (x - pr.x()) * cp.V.x() * cos(theta)
                                    - 2 * (z - pr.z()) * cp.V.x() * sin(theta);
            // update the parameters
            tr = tr - GN_STEP * df_dtr;
            tc = tc - GN_STEP * df_dtc;
            theta = theta - GN_STEP * df_dtheta;
        }
    }
    /**
     * Find the intersection point of the ray with the surface using Newton iteration.
     * @param r the ray
     * @param t the parameter on the xy-curve, the original number is t0.
     * @param tmin the minimum t value of the curve segment
     * @param tmax the maximum t value of the curve segment
     * @return whether the iteration converges.
     */
    bool Newton(const Ray &r, float &t, float tmin, float tmax) {
        int iter = 0;
        while (iter < GN_MAX_ITER) {
            // calculate f df
            float f, df;
            fdf(r, t, f, df, tmin, tmax);
//            printf("%d, %f, %f, %f\n", iter, t, f, df);
            // judge if the iteration converges
            if (abs(f) < GN_ERROR) {
                return true;
            } else {
                // update t
                t = t - GN_STEP * f / df;
            }
            ++iter;
        }
        return false;
    }

    /**
     * Target function : f(t) = ((y(t)-oy)dz/dy + oz)^2 + ((y(t)-oy)dx/dy + ox)^2 - x(t)^2
     * @param r
     * @param t
     * @param tmin the minimum t value of the curve segment
     * @param tmax the maximum t value of the curve segment
     * @return f(t), df(t)
     */
    inline void fdf(const Ray &r, float t, float &f, float &df, float tmin, float tmax) {
        // corner case
        if (t <= tmin) {
            CurvePoint cp;
            pCurve->getDataAt(tmin, cp);
            float xt = cp.V.x();
            float yt = cp.V.y();
            float ox = r.getOrigin().x();
            float oy = r.getOrigin().y();
            float oz = r.getOrigin().z();
            float dx = r.getDirection().x();
            float dy = r.getDirection().y();
            float dz = r.getDirection().z();
            // calculate f0
            float a = (yt - oy) * dz / dy + oz;
            float b = (yt - oy) * dx / dy + ox;
            f = a * a + b * b - xt * xt;

            // y = -f(t-t0) + y0
            df = -f;
            f = -f * (t - tmin) + f;

        } else if (t >= tmax) {
            CurvePoint cp;
            pCurve->getDataAt(tmax, cp);
            float xt = cp.V.x();
            float yt = cp.V.y();
            float ox = r.getOrigin().x();
            float oy = r.getOrigin().y();
            float oz = r.getOrigin().z();
            float dx = r.getDirection().x();
            float dy = r.getDirection().y();
            float dz = r.getDirection().z();

            // calculate f0
            float a = (yt - oy) * dz / dy + oz;
            float b = (yt - oy) * dx / dy + ox;
            f = a * a + b * b - xt * xt;

            // y = f(t-t0) + y0
            df = f;
            f = f * (t - tmax) + f;


        } else {
            CurvePoint cp;
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
            df = 2 * a * dyt * dz / dy + 2 * b * dyt * dx / dy - 2 * xt * dxt;
        }

    }

    void initMesh() {
        // Definition for drawable surface.
        typedef std::tuple<unsigned, unsigned, unsigned> Tup3u;
        // Surface is just a struct that contains vertices, normals, and
        // faces.  VV[i] is the position of vertex i, and VN[i] is the normal
        // of vertex i.  A face is a triple i,j,k corresponding to a triangle
        // with (vertex i, normal i), (vertex j, normal j), ...
        // Currently this struct is computed every time when canvas refreshes.
        // You can store this as member function to accelerate rendering.

        struct Surface {
            std::vector<Vector3f> VV;
            std::vector<Vector3f> VN;
            std::vector<Tup3u> VF;
        } surface;

        pCurve->discretize(DISCRETE_SIZE, curvePoints);
        const int steps = 40;
        // iterate through every curve point
        for (unsigned int ci = 0; ci < curvePoints.size(); ++ci) {
            const CurvePoint &cp = curvePoints[ci];
            // rotate
            for (unsigned int i = 0; i < steps; ++i) {
                float t = (float) i / steps;
                Quat4f rot;
                rot.setAxisAngle(t * 2 * 3.14159, Vector3f::UP); // UP = 0, 1, 0
                // find the right position for the point and its normal
                Vector3f pnew = Matrix3f::rotation(rot) * cp.V;
                Vector3f pNormal = Vector3f::cross(cp.T, -Vector3f::FORWARD);
                Vector3f nnew = Matrix3f::rotation(rot) * pNormal;
                surface.VV.push_back(pnew);
                surface.VN.push_back(nnew);
                //  ensure that the surface is closed by repeating the first and last points
                int i1 = (i + 1 == steps) ? 0 : i + 1;
                if (ci != curvePoints.size() - 1) {
                    /**
                     * b -------
                     *   |    /|
                     *   |  /  |
                     *   |/    |
                     * a -------
                     */
                    // triangle 1 : this loop (a_i, a_{i+1}), next loop b_i
                    surface.VF.emplace_back((ci + 1) * steps + i, ci * steps + i1, ci * steps + i);
                    // triangle 2 : this loop a_{i+1}, next loop (b_i, b_{i+1})
                    surface.VF.emplace_back((ci + 1) * steps + i, (ci + 1) * steps + i1, ci * steps + i1);
                }
            }
        }

        // parse the surface to triangle mesh
        std::vector<Triangle *> triangles;
        for (unsigned i = 0; i < surface.VF.size(); i++) {
            auto *tri = new Triangle(
                    surface.VV[std::get<0>(surface.VF[i])],
                    surface.VV[std::get<1>(surface.VF[i])],
                    surface.VV[std::get<2>(surface.VF[i])],
                    material
            );
            tri->normal = ((surface.VN[std::get<0>(surface.VF[i])] +
                            surface.VN[std::get<1>(surface.VF[i])] +
                            surface.VN[std::get<2>(surface.VF[i])]) / 3).normalized();
            triangles.push_back(tri);
        }
        mesh = new Mesh(triangles, material);
    }

};

#endif //FINALPROJECT_REVSURFACE_HPP
