add_subdirectory(cmake)

configure_file(mpi.modulefile.in mpi.modulefile @ONLY)
configure_file(mpivars.sh.in mpivars.sh @ONLY)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/mpi.modulefile
    ${CMAKE_CURRENT_BINARY_DIR}/mpivars.sh
  DESTINATION share
)

message(STATUS "***************************************************************")
message(STATUS "* Use:                                                         ")
message(STATUS "*                                                              ")
message(STATUS "*   source ${CMAKE_INSTALL_PREFIX}/share/mpivars.sh            ")
message(STATUS "*                                                              ")
message(STATUS "* to set up the environment variables                          ")
message(STATUS "***************************************************************")
