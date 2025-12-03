@page ex3 Example 3: Custom type and operator

[TOC]

In this example, we show 
- how to register a new MPI datatype by providing a `tie_data` function for our C++ type and 
- how to use mpi::map_C_function and mpi::map_add to define MPI operations for it.

@include ex3.cpp

Output (running with `-n 5`):

```
sum = (15, 10)
product = (-185, 180)
```