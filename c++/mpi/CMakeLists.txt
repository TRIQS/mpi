# MPI C++ Interface is header-only
# Create an Interface Target
add_library(mpi_interface INTERFACE)
target_include_directories(mpi_interface
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/c++>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
)
target_compile_options(mpi_interface INTERFACE -std=c++17 $<$<CXX_COMPILER_ID:GNU>:-fconcepts>)

# Install library and headers
install(TARGETS mpi_interface EXPORT mpi-targets)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")


# ========= Additional Depdencies ==========

message(STATUS "-------- MPI detection -------------")

find_package(MPI REQUIRED)

# Create an interface target
add_library(mpi INTERFACE)
target_include_directories(mpi SYSTEM INTERFACE ${MPI_C_INCLUDE_PATH})
target_link_libraries(mpi INTERFACE ${MPI_C_LIBRARIES})
separate_arguments(MPI_C_COMPILE_FLAGS) # Convert to list
target_compile_options(mpi INTERFACE ${MPI_C_COMPILE_FLAGS})

# Export Interface target
target_link_libraries(mpi_interface INTERFACE mpi)
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
