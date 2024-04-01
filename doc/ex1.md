@page ex1 Example 1: Hello world!

[TOC]

In this example, we show how to implement the standard `Hello world` program using **mpi**.

```cpp
#include <mpi/mpi.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
  // initialize MPI environment and communicator
  mpi::environment env(argc, argv);
  mpi::communicator world;

  // get rank and greet world
  int rank = world.rank();
  std::cout << "Hello from processor " << rank << "\n";
}
```

Output (depends on the number of processes and the order is arbitrary):

```
Hello from processor 2
Hello from processor 3
Hello from processor 0
Hello from processor 1
```
