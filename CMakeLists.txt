cmake_minimum_required(VERSION 3.15)

project(BlenderFlow
  VERSION 0.0.1
  DESCRIPTION "Library to use 3D models in machine learning"
  LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

include(cmake/utils.cmake)

set(ASSETS "${CMAKE_SOURCE_DIR}/assets")

add_subdirectory(src)
add_subdirectory(app)
