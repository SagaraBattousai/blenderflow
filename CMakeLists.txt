cmake_minimum_required(VERSION 3.15)

project(BlenderFlow
  VERSION 0.0.1
  DESCRIPTION "Library to use 3D models in machine learning"
  LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

include(cmake/utils.cmake)
include(cmake/python_extension_build.cmake)

set(ASSETS "${CMAKE_SOURCE_DIR}/assets")

# Add Interpreter if needed and remove NumPy if not used but as it is so
# common it is left by default.
find_package(Python3 COMPONENTS Development NumPy REQUIRED) 

message(DEBUG ${Python3_NumPy_VERSION})

add_subdirectory(src)
add_subdirectory(app)


