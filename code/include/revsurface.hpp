#ifndef FINALPROJECT_REVSURFACE_HPP
#define FINALPROJECT_REVSURFACE_HPP

#include "object3d.hpp"
#include "curve.hpp"
#include "boundingBox.hpp"
#include "mesh.hpp"
#include <tuple>

#define GN_MAX_ITER 100
#define GN_ERROR 1e-3
#define GN_STEP 0.1

typedef Curve::CurvePoint CurvePoint;

// Currently, revsurface does not support transparent material
// NOTE : THIS CLASS ONLY SUPPORT SPECIFIC SCENES
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
        printf("bbox: %f %f %f %f %f %f\n", bBox->x0, bBox->y0, bBox->z0, bBox->x1, bBox->y1, bBox->z1);
        // judge direction
        if (curve_y_max > pCurve->getControls()[0].y()) direction = 1;
        else direction = -1;
        // create mesh
        initMesh();

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
        // alternative : use mesh
//        if(mesh->intersect(r, h, tmin)){
//            h.set(h.getT(), material, h.getNormal(), h.isInside());
//            return true;
//        }
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

    /**
     * @copydetails PA2
     */
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

        pCurve->discretize(20, curvePoints);
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
        mesh = new Mesh(triangles);
    }
};

#endif //FINALPROJECT_REVSURFACE_HPP
