@page documentation API Documentation

[TOC]

**mpi** implements various high-level C++ wrappers around their low-level C counterparts.
It is not intended as a full replacement for the C implementation.
Instead it tries to help the user with the most common tasks like initializing and finalizing
an @ref mpi::environment "MPI environment" or sending data via @ref coll_comm "collective communications".

The following provides a detailed reference documentation grouped into logical units.

If you are looking for a specific function, class, etc., try using the search bar in the top left corner.

## MPI essentials

@ref mpi_essentials provide the user with two classes necessary for any MPI program:

* The mpi::environment class is used to initialize and finialize the MPI execution environment.
    It calls `MPI_Init` in its constructor and `MPI_Finalize` in its destructor.
    There should be at most one instance in every program and it is usually created at the very beginning of the `main` function.

* The mpi::communicator class is a simple wrapper around an `MPI_Comm` object.
    Besides storing the `MPI_Comm` object, it also provides some convient functions for getting the size of the communicator,
    the rank of the current process or for splitting an existing communicator.

## MPI datatypes and operations

@ref mpi_types_ops map various C++ datatypes to MPI datatypes and help the user with registering their own datatypes to be
used in MPI communications.
Furthermore, it offers tools to simplify the creation of custom MPI operations usually required in `MPI_Reduce` or `MPI_Accumulate` functions.

## Collective MPI communication

The following generic collective communications are defined in @ref coll_comm "Collective MPI communication":

* @ref mpi::all_gather "all_gather"
* @ref mpi::all_reduce "all_reduce"
* @ref mpi::all_reduce_in_place "all_reduce_in_place"
* @ref mpi::broadcast "broadcast"
* @ref mpi::gather "gather"
* @ref mpi::reduce "reduce"
* @ref mpi::reduce_in_place "reduce_in_place"
* @ref mpi::scatter "scatter"

They offer a much simpler interface than their MPI C library analogs. For example, the following broadcasts a `std::vector<double>`
from the process with rank 0 to all others:

```cpp
mpi::broadcast(vec);
```

Compare this with the call to the C library:

```cpp
MPI_Bcast(vec.data(), static_cast<int>(vec.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
```

Under the hood, the generic mpi::broadcast implementation calls the specialized @ref "mpi::mpi_broadcast(std::vector< T >&, mpi::communicator, int)".
The other generic functions are implement in the same way.
See the "Functions" section in @ref coll_comm to check which datatypes are supported out of the box.

In case your datatype is not supported, you are free to provide your own specialization.

## Lazy MPI communication

@ref mpi_lazy can be used to provied collective MPI communication for lazy expression types.
Most users probably won't need to use this functionality directly.

We refer the interested reader to [TRIQS/nda](https://github.com/TRIQS/nda/blob/unstable/c%2B%2B/nda/mpi/reduce.hpp) for more details.

## Event handling

@ref event_handling provides the mpi::monitor class which can be used to communicate and handle events across multiple
processes.

@ref ex2 shows a simple use case.

## Utilities

@ref utilities is a collection of various other tools which do not fit into any other category above.

The following utilities are defined in **mpi**:

* @ref mpi::regular_t "regular_t"
* @ref mpi::chunk "chunk"
* @ref mpi::chunk_length "chunk_length"
