@page ex3 Example 3: Custom type and operator

[TOC]

In this example, we show how to use mpi::mpi_type_from_tie, mpi::map_C_function and mpi::map_add to register a new MPI datatype and to define MPI operations for it.

```cpp
#include <mpi/mpi.hpp>
#include <iostream>

// define a custom complex type
struct my_complex {
  double real;
  double imag;
};

// define an addition for my_complex
inline my_complex operator+(const my_complex& z1, const my_complex& z2) {
  return { z1.real + z2.real, z1.imag + z2.imag };
}

// define a tie_data function for mpi_type_from_tie
inline auto tie_data(const my_complex& z) {
  return std::tie(z.real, z.imag);
}

// register my_complex as an MPI type
template <> struct mpi::mpi_type<my_complex> : mpi::mpi_type_from_tie<my_complex> {};

int main(int argc, char *argv[]) {
  // initialize MPI environment
  mpi::environment env(argc, argv);
  mpi::communicator world;

  // create complex number z
  my_complex z = { world.rank() + 1.0, static_cast<double>(world.rank()) };

  // sum z over all processes
  auto sum = mpi::reduce(z, world, 0, false, mpi::map_add<my_complex>());

  // define a product for my_complex
  auto my_product = [](const my_complex& z1, const my_complex& z2) {
    return my_complex { z1.real * z2.real - z1.imag * z2.imag, z1.real * z2.imag + z1.imag * z2.real };
  };

  // multiply z over all processes
  auto product = mpi::reduce(z, world, 0, false, mpi::map_C_function<my_complex, my_product>());

  // print result
  if (world.rank() == 0) {
    std::cout << "sum = (" << sum.real << ", " << sum.imag << ")\n";
    std::cout << "product = (" << product.real << ", " << product.imag << ")\n";
  }
}
```

Output (running with `-n 5`):

```
sum = (15, 10)
product = (-185, 180)
```