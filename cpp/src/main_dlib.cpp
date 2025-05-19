#include "tdoa_dlib_solver.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    // Истинные координаты (для сравнения)
    std::array<Point, 3> true_receivers = {{{0.0, 0.0}, {5.0, 0.0}, {2.5, 5.0}}};
    SourceSet sources = {{{3.0, 4.0}, {6.0, 1.0}, {1.0, 3.0}}};

    TDOASet tdoa;
    auto dist = [](const Point& p1, const Point& p2) {
        double dx = p1[0] - p2[0];
        double dy = p1[1] - p2[1];
        return std::sqrt(dx * dx + dy * dy);
    };

    for (int s = 0; s < 3; ++s) {
        Point S = sources[s];
        tdoa[s][0] = dist(true_receivers[0], S) - dist(true_receivers[1], S);
        tdoa[s][1] = dist(true_receivers[0], S) - dist(true_receivers[2], S);
        tdoa[s][2] = dist(true_receivers[1], S) - dist(true_receivers[2], S);
    }

    auto result = solve_tdoa_dlib(tdoa, sources, false);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Estimated coordinates:\n";
    std::cout << "A = [" << result(0) << ", " << result(1) << "]\n";
    std::cout << "B = [" << result(2) << ", " << result(3) << "]\n";
    std::cout << "C = [" << result(4) << ", " << result(5) << "]\n";

    std::cout << "\nTrue coordinates:\n";
    std::cout << "A = [0.00, 0.00]\n";
    std::cout << "B = [5.00, 0.00]\n";
    std::cout << "C = [2.50, 5.00]\n";

    return 0;
}
