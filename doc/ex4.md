@page ex4 Example 4: Provide custom spezializations

[TOC]

In this example, we show how to write a specialized `mpi_reduce_into` for a custom type.

@include ex4.cpp

Output (running with `-n 4`):

```
Reduced vector: 4 8 12 16 20
```

Note that by providing a simple `mpi_reduce_into` for our custom `foo` type, we are able to reduce a `std::vector` of
`foo` objects without any additional work.

Under the hood, each `foo` object is reduced spearately using the above specialization.
For large amounts of data or in performance critical code sections, this might not be desired.
In such a case, it is usally better to make the type MPI compatible such that the reduction can be done with a single
call to MPI C library.
See @ref ex3 for more details.
