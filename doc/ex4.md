@page ex4 Example 4: Provide custom spezializations

[TOC]

In this example, we show how to write a specialized `mpi_reduce_into` for a custom type.

```cpp
#include <mpi/mpi.hpp>
#include <iostream>
#include <vector>

// Custom type.
class foo {
  public:
  // Constructor.
  foo(int x = 5) : x_(x) {}

  // Get the value stored in the class.
  int x() const { return x_; }

  // Specialization of mpi_reduce_into for the custom type.
  friend void mpi_reduce_into(foo const &f_in, foo &f_out, mpi::communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    mpi::reduce_into(f_in.x_, f_out.x_, c, root, all, op);
  }

  private:
  int x_;
};

int main(int argc, char *argv[]) {
  // initialize MPI environment
  mpi::environment env(argc, argv);
  mpi::communicator world;

  // create a vector of foo objects
  std::vector<foo> vec {foo{1}, foo{2}, foo{3}, foo{4}, foo{5}};

  // reduce the vector of foo objects
  auto result = mpi::reduce(vec, world);

  // print the result on rank 0
  if (world.rank() == 0) {
    std::cout << "Reduced vector: ";
    for (auto const &f : result) std::cout << f.x() << " ";
    std::cout << "\n";
  }
}
```

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
