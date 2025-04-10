cmake_minimum_required(VERSION 3.10)
project(tdoa_dlib_solver)

set(CMAKE_CXX_STANDARD 17)

find_package(dlib REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(main_dlib
    src/main_dlib.cpp
    src/tdoa_dlib_solver.cpp
)

target_link_libraries(main_dlib dlib::dlib)
