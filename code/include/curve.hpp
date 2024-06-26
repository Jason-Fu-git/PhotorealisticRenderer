/**
 * Curve class, supports Bezier curve and B spline curve
 */

#ifndef FINALPROJECT_CURVE_HPP
#define FINALPROJECT_CURVE_HPP

#include "object3d.hpp"
#include "utils.hpp"
#include <vecmath.h>
#include <vector>
#include <utility>

#include <algorithm>


/**
 * An abstract curve class
 * @author Jason Fu
 */
class Curve : public Object3D {
protected:
    // control points
    std::vector<Vector3f> controls;
public:

    float min_t, max_t;

    // The CurvePoint object stores information about a point on a curve
    // after it has been tessellated: the vertex (V) and the tangent (T)
    // It is the responsibility of functions that create these objects to fill in all the data.
    struct CurvePoint {
        Vector3f V; // Vertex
        Vector3f T; // Tangent  (unit)

        explicit CurvePoint(const Vector3f &_v = Vector3f::ZERO, const Vector3f &_t = Vector3f::ZERO) : V(_v), T(_t) {}
    };

    explicit Curve(std::vector<Vector3f> points) : controls(std::move(points)) {}


    std::vector<Vector3f> &getControls() {
        return controls;
    }

    /**
    * This function sees the curve as P(t), where t $\in$ [0, 1].
    * Given the parameter t, this function will save the curve point P(t) and the tangent at P(t) into data.
    * @param t a parameter in [0, 1]
    * @param data
    */
    virtual void getDataAt(float t, CurvePoint &data) = 0;

    /**
     * Discretize the curve into a set of CurvePoint objects.
     * @param resolution
     * @param data The results will be saved into "data" vector.
     */
    virtual void discretize(int resolution, std::vector<CurvePoint> &data) = 0;

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        return false;
    }
};

/**
 * 3-order BezierCurve
 * @author Jason Fu
 */
class BezierCurve : public Curve {
public:

    // A structure to store the base function $B_{i, k}(t)$ and its derivative $B'_{i, k}(t)$, given the resolution.
    struct BezierData {
        int resolution;
        std::vector<std::vector<std::pair<double, double> > > t_and_tangent;

        /**
        * Calculate base functions $B_{i, k}(t)$ for a Bezier curve, by De Casteljau's algorithm.
        * @param i
        * @param k
        * @param t
        * @return The value of $B_{i, k}(t)$.
        */
        static double BezierSpline(int i, int k, double t) {
            double res = comb(i, k);
            for (int j = 0; j < k - i; j++) {
                res *= (1 - t);
            }
            for (int j = 0; j < i; j++) {
                res *= t;
            }
            return res;
        }

        /**
         * Calculate $B'_{i,k}(t)$.
         * @param i
         * @param k
         * @param t
         * @return The value of $B'_{i, k}(t)$.
         */
        static double BezierSplineDerivative(int i, int k, double t) {
            return k * (BezierSpline(i - 1, k - 1, t) - BezierSpline(i, k - 1, t));
        }

        // calculate the basis functions and tangent for 3-order Bezier Curve.
        void calculate() {
            t_and_tangent.clear();
            double step = 1.0 / resolution;
            for (int i = 0; i <= resolution; ++i) {
                double t = i * step;
                if (t > 1) t = 1;
                // calculate the basis functions and tangent for 3-order Bezier Curve.
                std::vector<std::pair<double, double> > tmp_vector;
                tmp_vector.reserve(4);
                for (int j = 0; j < 4; j++) {
                    tmp_vector.emplace_back(
                            BezierSpline(j, 3, t),
                            BezierSplineDerivative(j, 3, t)
                    );
                }
                // save the result
                t_and_tangent.push_back(tmp_vector);
            }
        }

        explicit BezierData(int res = 1) : resolution(res) { calculate(); }

        // Every time the resolution is changed, we need to recalculate the base functions.
        void setResolution(int res) {
            if (res != resolution) {
                resolution = res;
                calculate();
            }
        }
    };

    explicit BezierCurve(const std::vector<Vector3f> &points) : Curve(points) {
        if (points.size() < 4 || points.size() % 3 != 1) {
            printf("Number of control points of BezierCurve must be 3n+1!\n");
            exit(0);
        }
        min_t = 0;
        max_t = 1;
    }

    void getDataAt(float t, CurvePoint &data) override {
        // calculate the number of segments. note that the size of controls must be 3n+1
        int segment_count = controls.size() / 3;
        // calculate the index of the segment
        int segment_index = std::floor(t * segment_count);
        if (segment_index == segment_count) segment_index--;
        // calculate the parameter in [0, 1] of the segment
        float segment_t = t * segment_count - segment_index;
        // calculate the basis functions and tangent for 3 order Bezier Curve.
        data.V = Vector3f::ZERO;
        data.T = Vector3f::ZERO;
        for (int j = 0; j < 4; j++) {
            auto P = controls[3 * segment_index + j];
            data.V += P * BezierData::BezierSpline(j, 3, segment_t);
            data.T += P * BezierData::BezierSplineDerivative(j, 3, segment_t);
        }
        data.T = data.T.normalized();
    }

    void discretize(int resolution, std::vector<CurvePoint> &data) override {
        data.clear();
        // According to the resolution, calculate the base functions and tangent.
        // Note : if the resolution is the same as before, we will use the previous values.
        bezierData.setResolution(resolution);
        // According to the control points, calculate the curve points and tangents in each interval.
        int batch_count = controls.size() / 3;
        for (int batch = 0; batch < batch_count; ++batch) {
            // Calculate the curve points and tangents in each interval.
            for (auto &j: bezierData.t_and_tangent) {
                Vector3f V = Vector3f::ZERO;
                Vector3f T = Vector3f::ZERO;
                // 3-order Bezier Curve
                for (int i = 0; i < 4; i++) {
                    auto P = controls[3 * batch + i];
                    V += P * j[i].first;
                    T += P * j[i].second;
                }
                // save the results.
                data.emplace_back(V, T);
            }
        }
    }

protected:
    BezierData bezierData;

};

class BsplineCurve : public Curve {
public:

    // A structure to store the base functions and their derivatives for a B spline, given the resolution.
    // Note : The Calculation will only cover the interval [t_k, t_{n+1}], however, the `t_and_tangent` vector
    // will store the data for the whole interval [t_0, t_{n+1}].
    struct BData {

        int resolution;
        int k;
        int n;
        std::vector<double> ts;
        std::vector<std::pair<double *, double *> > t_and_tangent;

        /**
        * Calculate the base function value $B_{i,_k}(_t)$ for a B Spline and its derivative.
        * @param _k
        * @param _n
        * @param _t
        * @param _ts vector of _t, must be sorted
        * @return ($B_{i,_k}(_t)$, $B'_{i,_k}(_t)$)
        */
        static std::pair<double *, double *> BSplineAndDerivative(int _k, int _n, double _t, std::vector<double> _ts) {
            auto *prev_b = new double[_n + _k + 2];
            auto *b_derivative = new double[_n + _k + 2];
            // initialize : B_{i,0}(_t)
            for (int i = 0; i < _n + _k + 1; ++i) {
                (_t >= _ts[i] && _t < _ts[i + 1]) ? prev_b[i] = 1.0 : prev_b[i] = 0.0;
            }
            // calculate the base function
            for (int j = 1; j <= _k; ++j) {
                auto *b = new double[_n + _k + 2];
                for (int i = 0; i < _n + _k + 1 - j; ++i) {
                    b[i] = (_t - _ts[i]) / (_ts[i + j] - _ts[i]) * prev_b[i]
                           + (_ts[i + j + 1] - _t) / (_ts[i + j + 1] - _ts[i + 1]) * prev_b[i + 1];
                }
                for (int i = _n + _k + 1 - j; i < _n + _k + 1; ++i) {
                    b[i] = 0.0;
                }
                if (j == _k) {
                    // calculate the derivative
                    for (int i = 0; i < _n + 1; ++i) {
                        b_derivative[i] = _k * (
                                prev_b[i] / (_ts[i + _k] - _ts[i])
                                - prev_b[i + 1] / (_ts[i + _k + 1] - _ts[i + 1])
                        );
                    }
                    for (int i = _n + 1; i < _n + _k + 1; ++i) {
                        b_derivative[i] = 0.0;
                    }
                }
                delete[] prev_b;
                prev_b = b;
            }


            return std::make_pair(prev_b, b_derivative);
        }

        // calculate the basis functions and tangent for the given B Spline
        void calculate() {
            // clean previous space
            for (auto &i: t_and_tangent) {
                delete[] i.first;
                delete[] i.second;
            }
            t_and_tangent.clear();

            // calculate the data
            auto res = BSplineAndDerivative(k, n, ts[k], ts);
            t_and_tangent.emplace_back(
                    res.first,
                    res.second
            );

            for (int i = k; i <= n; ++i) {
                double step = (double) (ts[i + 1] - ts[i]) / (double) (resolution);
                for (int j = 1; j <= resolution; ++j) {
                    double t = ts[i] + j * step;
                    // calculate the basis functions and tangent for B Curve.
                    auto _res = BSplineAndDerivative(k, n, t, ts);
                    t_and_tangent.emplace_back(
                            _res.first,
                            _res.second
                    );
                }
            }
        }


        explicit BData(int _k = 0, int _n = 0, int _res = -1) : k(_k), n(_n), resolution(_res) {
        }

        void setResolution(int res) {
            if (res != resolution) {
                resolution = res;
                calculate();
            }
        }

    };

    explicit BsplineCurve(const std::vector<Vector3f> &points) : Curve(points) {
        if (points.size() < 4) {
            printf("Number of control points of BsplineCurve must be more than 4!\n");
            exit(0);
        }
        bData.n = points.size() - 1;
        bData.k = 3;
        for (int i = 0; i <= bData.n + bData.k + 1; i++) {
            bData.ts.push_back((double) i / (double) (bData.n + bData.k + 1));
        }
        min_t = bData.ts[bData.k];
        max_t = bData.ts[bData.n + bData.k + 1];
    }

    void getDataAt(float t, CurvePoint &data) override {
        // Note that only [t_k, t_{n+1}] is valid.
        t = bData.ts[bData.k] + t * (bData.ts[bData.n + 1] - bData.ts[bData.k]);
        // calculate the basis functions and tangent for B Curve.
        data.V = Vector3f::ZERO;
        data.T = Vector3f::ZERO;
        auto t_and_tangent = BData::BSplineAndDerivative(bData.k, bData.n, t, bData.ts);
        for (int i = 0; i < controls.size(); ++i) {
            auto P = controls[i];
            data.V += P * t_and_tangent.first[i];
            data.T += P * t_and_tangent.second[i];
        }
        data.T = data.T.normalized();
    }

    void discretize(int resolution, std::vector<CurvePoint> &data) override {
        data.clear();
        bData.setResolution(resolution);
        // calculate the Bspline value and derivative
        for (auto &j: bData.t_and_tangent) {
            Vector3f V = Vector3f::ZERO;
            Vector3f T = Vector3f::ZERO;
            for (int i = 0; i < controls.size(); i++) {

                auto P = controls[i];
                V += P * (j.first)[i];
                T += P * (j.second)[i];
            }
            // 保存结果
            data.emplace_back(V, T);
        }
    }

protected:
    BData bData;
};

#endif //FINALPROJECT_CURVE_HPP
