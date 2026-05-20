# MPI C++ Interface is header-only
# Create an Interface Target
file(GLOB_RECURSE headers *.hpp)
add_library(${PROJECT_NAME}_c INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME}_c ALIAS ${PROJECT_NAME}_c)

# Configure target and compilation
target_compile_features(${PROJECT_NAME}_c INTERFACE cxx_std_20)
target_include_directories(${PROJECT_NAME}_c INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/c++>)
target_include_directories(${PROJECT_NAME}_c SYSTEM INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
target_compile_definitions(${PROJECT_NAME}_c INTERFACE
				MPI_GIT_HASH=${PROJECT_GIT_HASH}
				$<$<CONFIG:Debug>:MPI_DEBUG>
			  )

# Install library and headers
install(TARGETS ${PROJECT_NAME}_c EXPORT ${PROJECT_NAME}-targets DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")

# ========= Additional Depdencies ==========

# Link against Itertools
target_link_libraries(${PROJECT_NAME}_c INTERFACE itertools::itertools_c)


message(STATUS "-------- MPI detection -------------")

set(MPI_CXX_SKIP_MPICXX TRUE CACHE BOOL "If true, the MPI-2 C++ bindings are disabled using definitions.")
set(MPI_DETERMINE_LIBRARY_VERSION TRUE)
find_package(MPI REQUIRED COMPONENTS CXX)

# Create an interface target
add_library(mpi INTERFACE)
target_link_libraries(mpi INTERFACE ${MPI_CXX_LIBRARIES})
target_include_directories(mpi SYSTEM INTERFACE ${MPI_CXX_INCLUDE_DIRS})
#target_link_options(mpi INTERFACE ${MPI_CXX_LINK_FLAGS}) cmake 3.13+
target_compile_options(mpi INTERFACE ${MPI_CXX_COMPILE_OPTIONS})
target_compile_definitions(mpi INTERFACE ${MPI_CXX_COMPILE_DEFINITIONS})

# Backward compatibility for older FindMPI.cmake (without MPI_CXX_SKIP_MPICXX)
if(CMAKE_VERSION VERSION_LESS 3.10.0)
    # Stop OpenMPI from including cxx binding headers, if installed
    target_compile_definitions(mpi INTERFACE MPICH_SKIP_MPICXX OMPI_SKIP_MPICXX _MPICC_H)
endif()

# Export Interface target
target_link_libraries(${PROJECT_NAME}_c INTERFACE mpi)
install(TARGETS mpi EXPORT ${PROJECT_NAME}-targets)

# Backward compatibility for older FindMPI.cmake
if(NOT MPIEXEC_EXECUTABLE)
  set(MPIEXEC_EXECUTABLE ${MPIEXEC} CACHE FILENAME "MPI Executable")
endif()

# Open-MPI specific MPIEXEC_PREFLAGS tweaks. Downstream add_test() calls
# already thread ${MPIEXEC_PREFLAGS} to mpiexec, so this reaches all of them.
# We rebuild MPIEXEC_PREFLAGS from scratch each configure to keep reconfigures
# idempotent (list(APPEND) on a cached value would otherwise duplicate flags).
if(MPI_CXX_LIBRARY_VERSION_STRING MATCHES "Open MPI v([0-9]+)")
  message(STATUS "MPI Version: ${MPI_CXX_LIBRARY_VERSION_STRING}")
  set(MPI_OMPI_MAJOR_VERSION ${CMAKE_MATCH_1} CACHE INTERNAL "Detected Open MPI major version (only set when Open MPI is detected)")

  set(_mpi_preflags "")

  # Open-MPI 3+ requires --oversubscribe.
  if(MPI_OMPI_MAJOR_VERSION GREATER_EQUAL 3)
    list(APPEND _mpi_preflags --oversubscribe)
  endif()

  # Open-MPI 5+ routes through UCX by default; UCX's libucm malloc hooks
  # collide with LLVM sanitizer interceptors and SEGV inside MPI_Init. See
  # doc/sanitizers.md. Probing via $CACHE{...} works even before downstream
  # projects declare option(ASAN ...).
  if(MPI_OMPI_MAJOR_VERSION GREATER_EQUAL 5 AND ($CACHE{ASAN} OR $CACHE{UBSAN} OR $CACHE{MSAN} OR $CACHE{TSAN}))
    list(APPEND _mpi_preflags --mca pml ob1 --mca btl self,vader --mca osc ^ucx)
    message(STATUS "Sanitizer + Open MPI ${MPI_OMPI_MAJOR_VERSION}.x detected: appending '--mca pml ob1 --mca btl self,vader --mca osc ^ucx' to MPIEXEC_PREFLAGS to avoid UCX/sanitizer SEGV")
  endif()

  set(MPIEXEC_PREFLAGS "${_mpi_preflags}" CACHE STRING "Flags to pass to mpiexec directly before the executable to run." FORCE)
endif()

# ========= Static Analyzer Checks ==========

option(ANALYZE_SOURCES OFF "Run static analyzer checks if found (clang-tidy, cppcheck)")
if(ANALYZE_SOURCES)

  # Locate static analyzer tools
  find_program(CPPCHECK_EXECUTABLE NAMES "cppcheck" PATHS ENV PATH)
  find_program(CLANG_TIDY_EXECUTABLE NAMES "clang-tidy" PATHS ENV PATH)

  # Run clang-tidy if found
  if(CLANG_TIDY_EXECUTABLE)
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXECUTABLE}")
    set_target_properties(${PROJECT_NAME}_c PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE}")
  else()
    message(STATUS "clang-tidy not found in $PATH. Please consider installing clang-tidy for additional checks!")
  endif()

  # Run cppcheck if found
  if(CPPCHECK_EXECUTABLE)
    message(STATUS "cppcheck found: ${CPPCHECK_EXECUTABLE}")
    add_custom_command(
	TARGET ${PROJECT_NAME}_c
	COMMAND ${CPPCHECK_EXECUTABLE}
	  --enable=warning,style,performance,portability
	  --std=c++23
	  --template=gcc
	  --verbose
	  --force
	  --quiet
          ${headers}
	WORKING_DIRECTORY
	  ${CMAKE_CURRENT_SOURCE_DIR}
    )
  else()
    message(STATUS "cppcheck not found in $PATH. Please consider installing cppcheck for additional checks!")
  endif()

endif()

# ========= Dynamic Analyzer Checks ==========

option(ASAN OFF "Compile library and executables with LLVM Address Sanitizer")
if(ASAN)
  if(NOT TARGET asan)
    find_package(sanitizer REQUIRED COMPONENTS asan)
  endif()
  target_link_libraries(${PROJECT_NAME}_c INTERFACE $<BUILD_INTERFACE:asan>)
endif()

option(UBSAN OFF "Compile library and executables with LLVM Undefined Behavior Sanitizer")
if(UBSAN)
  if(NOT TARGET ubsan)
    find_package(sanitizer REQUIRED COMPONENTS ubsan)
  endif()
  target_link_libraries(${PROJECT_NAME}_c INTERFACE $<BUILD_INTERFACE:ubsan>)
endif()

option(MSAN OFF "Compile library and executables with LLVM Memory Sanitizer")
if(MSAN)
  if(NOT TARGET msan)
    find_package(sanitizer REQUIRED COMPONENTS msan)
  endif()
  target_link_libraries(${PROJECT_NAME}_c INTERFACE $<BUILD_INTERFACE:msan>)
endif()

option(TSAN OFF "Compile library and executables with LLVM Thread Sanitizer")
if(TSAN)
  if(NOT TARGET tsan)
    find_package(sanitizer REQUIRED COMPONENTS tsan)
  endif()
  target_link_libraries(${PROJECT_NAME}_c INTERFACE $<BUILD_INTERFACE:tsan>)
endif()
