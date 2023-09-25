cmake_minimum_required(VERSION 3.26)
set(CMAKE_CUDA_STANDARD 17)

project(steger)
find_package(OpenCV 3.4 REQUIRED)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_executable(steger steger.cpp)

target_link_libraries(steger ${OpenCV_LIBS} ${CUDA_LIBRARIES})
set_target_properties(steger PROPERTIES CUDA_SEPARABLE_COMPILATION ON)



