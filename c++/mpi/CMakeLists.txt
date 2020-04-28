# MPI C++ Interface is header-only
# Create an Interface Target
file(GLOB_RECURSE headers *.hpp)
add_library(mpi_c INTERFACE)

# Configure target and compilation
target_compile_options(mpi_c INTERFACE -fPIC -std=c++17)
target_include_directories(mpi_c INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/c++>)
target_include_directories(mpi_c SYSTEM INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
target_compile_definitions(mpi_c INTERFACE
				MPI_GIT_HASH=${PROJECT_GIT_HASH}
				$<$<CONFIG:Debug>:MPI_DEBUG>
			  )

# Install library and headers
install(TARGETS mpi_c EXPORT mpi-targets DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")


# ========= Additional Depdencies ==========

# Link against Itertools
target_link_libraries(mpi_c INTERFACE itertools_c)


message(STATUS "-------- MPI detection -------------")

find_package(MPI REQUIRED)

# Create an interface target
add_library(mpi INTERFACE)
target_include_directories(mpi SYSTEM INTERFACE ${MPI_C_INCLUDE_PATH})
target_link_libraries(mpi INTERFACE ${MPI_C_LIBRARIES})
separate_arguments(MPI_C_COMPILE_FLAGS) # Convert to list
target_compile_options(mpi INTERFACE ${MPI_C_COMPILE_FLAGS})

# Export Interface target
target_link_libraries(mpi_c INTERFACE mpi)
install(TARGETS mpi EXPORT mpi-targets)

# Backward compatibility for older FindMPI.cmake
if(NOT MPIEXEC_EXECUTABLE)
  set(MPIEXEC_EXECUTABLE ${MPIEXEC} CACHE FILENAME "MPI Executable")
endif()

# Compatibility to Open-MPI 3.0.0: check whether MPI executable has option --oversubscribe and add it 
execute_process(COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ls ${MPIEXEC_POSTFLAGS} RESULT_VARIABLE HAS_NO_OVERSUBSCRIBE OUTPUT_QUIET ERROR_QUIET)
if(NOT HAS_NO_OVERSUBSCRIBE)
  list(APPEND MPIEXEC_PREFLAGS --oversubscribe)
  set(MPIEXEC_PREFLAGS ${MPIEXEC_PREFLAGS} CACHE STRING "These flags will be directly before the executable that is being run by mpiexec." FORCE)
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
    set_target_properties(mpi_c PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE}")
  else()
    message(STATUS "clang-tidy not found in $PATH. Please consider installing clang-tidy for additional checks!")
  endif()

  # Run cppcheck if found
  if(CPPCHECK_EXECUTABLE)
    message(STATUS "cppcheck found: ${CPPCHECK_EXECUTABLE}")
    add_custom_command(
	TARGET mpi_c
	COMMAND ${CPPCHECK_EXECUTABLE}
	  --enable=warning,style,performance,portability
	  --std=c++17
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
option(UBSAN OFF "Compile library and executables with LLVM Undefined Behavior Sanitizer")

if(ASAN)
  if(NOT TARGET asan)
    find_package(sanitizer REQUIRED "asan")
  endif()
  target_link_libraries(mpi_c INTERFACE asan)
  install(TARGETS asan EXPORT mpi-targets)
endif()
if(UBSAN)
  if(NOT TARGET ubsan)
    find_package(sanitizer REQUIRED "ubsan")
  endif()
  target_link_libraries(mpi_c INTERFACE ubsan)
  install(TARGETS ubsan EXPORT mpi-targets)
endif()
