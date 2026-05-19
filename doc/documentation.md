@page documentation API Documentation

[TOC]

**mpi** implements various high-level C++ wrappers around their low-level C counterparts.
It is not intended as a full replacement for the C implementation.
Instead it tries to help the user with the most common tasks like initializing and finalizing an @ref mpi::environment
"MPI environment" or sending data via @ref coll_comm "collective communications".

The following provides a detailed reference documentation grouped into logical units.

If you are looking for a specific function, class, etc., try using the search bar in the top left corner.

## MPI essentials

@ref mpi_essentials provides the user with the classes that are necessary for any MPI program:

* The mpi::environment class is used to initialize and finalize the MPI execution environment.
  It calls `MPI_Init` in its constructor and `MPI_Finalize` in its destructor.
  There should be at most one instance in every program and it is usually created at the very beginning of the `main`
  function.

* The mpi::communicator class is a simple wrapper around an `MPI_Comm` object.
  Besides storing the `MPI_Comm` object, it also provides convenient functions for getting the size of the communicator
  and the rank of the current process (mpi::communicator::size, mpi::communicator::rank), for creating new
  communicators (mpi::communicator::split, mpi::communicator::split_shared, mpi::communicator::duplicate), for freeing
  them (mpi::communicator::free) and for synchronization (mpi::communicator::barrier, which supports a polling interval
  to reduce CPU load) or aborting (mpi::communicator::abort).
  The mpi::shared_communicator subtype is returned by mpi::communicator::split_shared and exists at the type level so
  that shared-memory APIs cannot be called on regular communicators by accident.

* The mpi::group class is a simple wrapper around an `MPI_Group` object.
  Besides storing the `MPI_Group` object, it also provides convenient functions for getting the size of the group, the
  rank of the current process and for creating a sub-group from a list of ranks via mpi::group::include.

It further contains the convenient functions mpi::is_initialized and mpi::is_finalized and the static boolean 
mpi::has_env.

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

A reduction reads similarly. The following sums an integer across all ranks and returns the result on rank 0:

```cpp
mpi::communicator world;
int sum = mpi::reduce(world.rank(), world);
// on rank 0: sum == 0 + 1 + ... + (size - 1)
// on other ranks: sum is default constructed
```

Use mpi::all_reduce (or pass `all = true`) if every rank needs the result.

## MPI one-sided communication and shared memory

@ref mpi_osc_shm can be used to get data from or put data directly to the memory
of another process.  This can be done without the involvement of processes that
are unaffected by the data transfer, i.e. no collective call is required, only
the origin and target process of the data transfer must cooperate.

This is provided through the move-only class template mpi::window, which wraps `MPI_Win` and exposes the usual
synchronization (fence / flush / sync / lock-unlock / post-start-complete-wait) and data-movement (get / put) primitives.

Another use-case of @ref mpi_osc_shm is the shared memory aspect by which
MPI applications can reduce their memory requirements through the deduplication
of replicated data between MPI ranks that are executed on the same SMP node.

For this use case the library provides mpi::shared_window, an `MPI_Win_allocate_shared`-backed specialization built on
top of a mpi::shared_communicator. The per-rank base pointer and byte size can be retrieved with
mpi::shared_window::query.

## Event handling

@ref event_handling provides the mpi::monitor class which can be used to communicate and handle events across multiple
processes.

@ref ex2 shows a simple use case.

## Utilities

@ref utilities is a collection of various other tools in **mpi** which do not fit into any other category above.

For users, the most useful entries are:

* mpi::check_mpi_call wraps a return code from an MPI C-library routine and throws a `std::runtime_error` if it is
  `!= MPI_SUCCESS`. It is used internally by every direct MPI call in the library.

* mpi::chunk and mpi::chunk_length distribute a range across the processes of a communicator. mpi::chunk takes a range
  and returns the slice assigned to the calling rank; mpi::chunk_length is the integer-range variant and accepts an
  optional `min_size` granularity.

* mpi::MPICompatibleRange is the concept that gates the contiguous-buffer fast paths in the generic range
  communication functions: it holds for contiguous, sized ranges whose value type has a corresponding mpi::mpi_type.
