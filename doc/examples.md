@page examples Examples

[TOC]

- @ref ex1 "Example 1: Hello world!"
- @ref ex2 "Example 2: Use monitor to communicate errors"
- @ref ex3 "Example 3: Custom type and operator"

@section compiling Compiling the examples

All examples have been compiled on a MacBook Pro with an Apple M2 Max chip and [open-mpi](https://www.open-mpi.org/) 4.1.5.
We further used clang 16.0.6 together with cmake 3.27.2.

Assuming that the actual example code is in a file `main.cpp`, the following generic `CMakeLists.txt` should work for all examples:

```cmake
cmake_minimum_required(VERSION 3.20)
project(example CXX)

# set required standard
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# fetch mpi from github
set(Build_Tests OFF CACHE BOOL "" FORCE)
include (FetchContent)
FetchContent_Declare(
  mpi
  GIT_REPOSITORY https://github.com/TRIQS/mpi.git
  GIT_TAG        1.2.x
)
FetchContent_MakeAvailable(mpi)

# build the example
add_executable(ex main.cpp)
target_link_libraries(ex mpi::mpi_c)
```
