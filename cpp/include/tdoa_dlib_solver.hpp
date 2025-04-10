
#pragma once

#include <array>
#include <dlib/matrix.h>

using Point = std::array<double, 2>;
using SourceSet = std::array<Point, 3>;
using TDOASet = std::array<std::array<double, 3>, 3>; // [source][pair]

// Возвращает x = [Ax, Ay, Bx, By, Cx, Cy]
dlib::matrix<double, 6, 1> solve_tdoa_dlib(
    const TDOASet& tdoa,
    const SourceSet& sources,
    bool apply_direction_penalty = false
);

