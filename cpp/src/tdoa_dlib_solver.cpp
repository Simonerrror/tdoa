// tdoa_dlib_solver.cpp
#include "tdoa_dlib_solver.hpp"
#include <dlib/optimization.h>
#include <cmath>
#include <random>

namespace {
    double distance(const Point& p1, const Point& p2) {
        double dx = p1[0] - p2[0];
        double dy = p1[1] - p2[1];
        return std::sqrt(dx * dx + dy * dy);
    }
}

dlib::matrix<double, 6, 1> solve_tdoa_dlib(const TDOASet& tdoa, const SourceSet& sources, bool apply_direction_penalty) {
    using namespace dlib;

    Point center = {0.0, 0.0};
    for (const auto& src : sources) {
        center[0] += src[0];
        center[1] += src[1];
    }
    center[0] /= sources.size();
    center[1] /= sources.size();

    std::default_random_engine rng;
    std::normal_distribution<double> dist(0.0, 0.5);

    matrix<double, 6, 1> x;
    x = center[0] + dist(rng), center[1] + dist(rng),
        center[0] + dist(rng), center[1] + dist(rng),
        center[0] + dist(rng), center[1] + dist(rng);

    auto loss = [&](const matrix<double, 6, 1>& x) {
        Point A = {x(0), x(1)};
        Point B = {x(2), x(3)};
        Point C = {x(4), x(5)};

        double total = 0.0;
        for (int s = 0; s < 3; ++s) {
            Point S = sources[s];
            double ab = distance(A, S) - distance(B, S);
            double ac = distance(A, S) - distance(C, S);
            double bc = distance(B, S) - distance(C, S);
        
            double dab = ab - tdoa[s][0];
            double dac = ac - tdoa[s][1];
            double dbc = bc - tdoa[s][2];
        
            total += dab * dab;
            total += dac * dac;
            total += dbc * dbc;
        }
        

        if (apply_direction_penalty) {
            const Point& D = sources[0];
            double dx = D[0] - A[0];
            double penalty = (dx < 0.0) ? 100.0 * dx * dx : 0.0;
            total += penalty;
        }

        return total;
    };

    find_min_using_approximate_derivatives(
        bfgs_search_strategy(),
        objective_delta_stop_strategy(1e-9),
        loss,
        x,
        -1 // minimize
    );

    return x;
}