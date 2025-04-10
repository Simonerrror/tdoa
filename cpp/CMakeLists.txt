cmake_minimum_required(VERSION 3.10)
project(tdoa_cpp)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(tdoa_main
    src/main.cpp
    src/tdoa_solver.cpp
)
