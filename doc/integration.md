@page integration Integration in C++ projects

[TOC]

**mpi** is a header only library.
To use it in your own `C++` code, you simply have to include the relevant header files and
tell your compiler/build system where it can find the necessary files.
For example:

```cpp
#include <mpi/mpi.hpp>

// use mpi
```

In the following, we describe some common ways to achieve this (with special focus on CMake).

@section cmake CMake

@subsection fetch FetchContent

If you use [CMake](https://cmake.org/) to build your source code, it is recommended to fetch the source code directly from the
[Github repository](https://github.com/TRIQS/mpi) using CMake's [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html)
module:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_project CXX)

# fetch from github
include(FetchContent)
FetchContent_Declare(
  mpi
  GIT_REPOSITORY https://github.com/TRIQS/mpi.git
  GIT_TAG        1.2.x
)
FetchContent_MakeAvailable(mpi)

# declare a target and link to mpi
add_executable(my_executable main.cpp)
target_link_libraries(my_executable mpi::mpi_c)
```

Note that the above will also build [goolgetest](https://github.com/google/googletest) and the unit tests for **mpi**.
To disable this, you can put `set(Build_Tests OFF CACHE BOOL "" FORCE)` before fetching the content or by specifying `-DBuild_Tests=OFF` on the command line.

@subsection find_package find_package

If you have already installed **mpi** on your system by following the instructions from the @ref installation page, you can also make
use of CMake's [find_package](https://cmake.org/cmake/help/latest/command/find_package.html) command.
This has the advantage that you don't need to download anything, i.e. no internet connection is required.

Let's assume that **mpi** has been installed to `path_to_install_dir`.
Then linking your project to **mpi** with CMake is as easy as

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_project CXX)

# find mpi
find_package(mpi REQUIRED CONFIG)

# declare a target and link to mpi
add_executable(my_executable main.cpp)
target_link_libraries(my_executable mpi::mpi_c)
```

In case, CMake cannot find the package, you might have to tell it where to look for the `mpi-config.cmake` file by setting the variable
`mpi_DIR` to `path_to_install_dir/lib/cmake/mpi` or by sourcing the provided `mpivars.sh` before running CMake:

```console
$ source path_to_install_dir/share/mpi/mpivars.sh
```

@subsection add_sub add_subdirectory

You can also integrate **mpi** into our CMake project by placing the entire source tree in a subdirectory and call `add_subdirectory()`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_project CXX)

# add mpi subdirectory
add_subdirectory(deps/mpi)

# declare a target and link to mpi
add_executable(my_executable main.cpp)
target_link_libraries(my_executable mpi::mpi_c)
```

Here, it is assumed that the **mpi** source tree is in a subdirectory `deps/mpi` relative to your `CMakeLists.txt` file.

@section other Other

Since **mpi** is header-only, you can also simply copy the relevant files directly into our project.
For example, you could place the `c++/mpi` directory from the **mpi** source tree into the include path of your project.
You can then build or compile it with any available method.
