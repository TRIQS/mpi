@page documentation API Documentation

[TOC]

**mpi** implements various high-level C++ wrappers around their low-level C counterparts.
It is not intended as a full replacement for the C implementation.
Instead it tries to help the user with the most common tasks like initializing and finalizing an @ref mpi::environment
"MPI environment" or sending data via @ref coll_comm "collective communications".

The following provides a detailed reference documentation grouped into logical units.

If you are looking for a specific function, class, etc., try using the search bar in the top left corner.

## MPI essentials

@ref mpi_essentials provide the user with two classes necessary for any MPI program:

* The mpi::environment class is used to initialize and finialize the MPI execution environment.
  It calls `MPI_Init` in its constructor and `MPI_Finalize` in its destructor.
  There should be at most one instance in every program and it is usually created at the very beginning of the `main`
  function.

* The mpi::communicator class is a simple wrapper around an `MPI_Comm` object.
  Besides storing the `MPI_Comm` object, it also provides some convient functions for getting the size of the
  communicator, the rank of the current process or for splitting an existing communicator.

* The mpi::group class is a simple wrapper around an `MPI_Group` object.
  Besides storing the `MPI_Group` object, it also provides some convient functions for getting the size of the
  group, the rank of the current process or for splitting the group based on include rules.

It further contains the convenient function mpi::is_initialized and the static boolean mpi::has_env.

## MPI datatypes and operations

@ref mpi_types_ops map various C++ datatypes to MPI datatypes and help the user with registering their own datatypes to
be used in MPI communications.
Furthermore, it offers tools to simplify the creation of custom MPI operations usually required in `MPI_Reduce` or
`MPI_Accumulate` functions.

## Collective MPI communication

**mpi** provides several generic @ref coll_comm "Collective MPI communication".
They offer a much simpler interface than their MPI C library analogs.
For example, the following broadcasts a `std::vector<double>` from the process with rank 0 to all others:

```cpp
mpi::broadcast(vec);
```

Compare this with the call to the C library:

```cpp
MPI_Bcast(vec.data(), static_cast<int>(vec.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
```

Under the hood, the generic mpi::broadcast implementation calls the specialized
@ref "mpi::mpi_broadcast(std::vector< T >&, mpi::communicator, int)".
Other generic functions in **mpi** work similarly.
See the "Functions" section in @ref coll_comm to check which datatypes and MPI operations are supported out of the box.

In case your datatype is not supported, you are free to provide your own specialization.

## MPI one-sided communication and shared memory

@ref mpi_osc_shm can be used to get data from or put data directly to the memory
of another process.  This can be done without the involvement of processes that
are unaffected by the data transfer, i.e. no collective call is required, only
the origin and target process of the data transfer must cooperate.

Another use-case of @ref mpi_osc_shm is the shared memory aspect by which
MPI applications can reduce their memory requirements through the deduplication
of replicated data between MPI ranks that are executed on the same SMP node.

## Lazy MPI communication

@ref mpi_lazy can be used to provied collective MPI communication for lazy expression types.
Most users probably won't need to use this functionality directly.

We refer the interested reader to [TRIQS/nda](https://triqs.github.io/nda/latest/group__av__mpi.html) for more details.

## Event handling

@ref event_handling provides the mpi::monitor class which can be used to communicate and handle events across multiple
processes.

@ref ex2 shows a simple use case.

## Utilities

@ref utilities is a collection of various other tools in **mpi** which do not fit into any other category above.

For users, the most useful of them is probably mpi::check_mpi_call.
A wrapper function that checks the error code returned by MPI C library routines and throws an exception in case the
code is `!= MPI_SUCCESS`.
