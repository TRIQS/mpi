configure_file(mpi-config.cmake.in mpi-config.cmake @ONLY)
configure_file(mpi-config-version.cmake.in mpi-config-version.cmake @ONLY)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/mpi-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/mpi-config-version.cmake
  DESTINATION lib/cmake/mpi
)

install(EXPORT mpi-targets NAMESPACE mpi:: DESTINATION lib/cmake/mpi)
