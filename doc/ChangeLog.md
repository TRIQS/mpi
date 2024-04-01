@page changelog Changelog

## Version 1.2.0

This is version 1.2.0 of mpi, a high-level C++ interface to the [Message Passing Interface].

We thank all contributors: Thomas Hahn, Alexander Hampel, Dylan Simon, Hugo U.R. Strand, Nils Wentzell

Find below an itemized list of changes in this release.

### General
* Force mpi initialization when FORCE_MPI_INIT is set in environment
* Add support for cray MPICH environments
* Fix compiler warnings
* clang-format all source files

### cmake
* Add compiler warnings for IntelLLVM
* Do not build documentation as subproject
* Synchronize deps/CMakeLists.txt with nda
* Remove redundant PythonSupport check
* Update Findsanitizer.cmake to include TSAN and MSAN

### fixes
* Make sure to specialize mpi_type<..> also for constant builtin types
* Use value initialization for also for MPI_Op and MPI_Datatype
* Fix #8 restore compatibility against MPICH
* Protect mpi::environment construction outside of mpirun
* Demote MPI CXX types to C


## Version 1.1.0

This is version 1.1.0 of mpi, a high-level C++ interface to the [Message Passing Interface].

We thank all contributors: Philipp Dumitrescu, Alexander Hampel, Olivier Parcollet, Dylan Simon, Nils Wentzell

Find below an itemized list of changes in this release.

### General
* Add mpi::monitor class to allow stopping all ranks when exception is thrown on any one of them
* Added layer to use triqs without MPI + tests
* Make sure to specialize mpi_type<..> also for constant builtin types
* Regenerate Apache copyright headers
* Restore compatibility against itertools/unstable: Allow sentinel types for std::end/cend(range)
* Add simple test for vector<non-pod> scatter/gather
* Add mpi_broadcast and mpi_reduce for std::pair + test
* Change mpi_gather/scatter for std::vector of custom type to just gather/scatter the vector and not its values
* New poll barrier as alternative to mpi.barrier
* mpi::details::convert should be recursive, trigger explicit conversion in else branch
* Make sure to std::abort in mpi abort when not in an mpi env
* Simplify check for mpi env vars, provide single static mpi::has_env
* Fix generic mpi operations for vector/lazy types, Protect in_place operations against rvalues and const
* Change type of mpi::lazy to match lazy implementation for triqs::gfs, add is_mpi_lazy checker
* Check that all vector sizes match in all_reduce(std::vector)
* Call to all_reduce should not take a root argument
* Type bool should be associated with MPI_CXX_BOOL
* Use MPI_CXX_DOUBLE_COMPLEX over Fortran-specific type MPI_DOUBLE_COMPLEX

### doc
* Add link to reference doc to README.md
* Minor doc cleanups for doxygen generation, add Doxyfile and update .gitignore

### cmake
* Do not run mpi_monitor as nompi test
* Set CXX standard using target_compile_features
* Use unstable branch of itertools
* Bump Version number to 1.1.0


## Version 1.0.0

mpi is a high-level C++ interface to the [Message Passing Interface].

This is the initial release for this project.
