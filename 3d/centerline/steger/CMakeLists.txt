cmake_minimum_required(VERSION 3.26)

find_package(OpenCV 3.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
add_executable( steger steger.cpp )

target_link_libraries( steger ${OpenCV_LIBS} )

