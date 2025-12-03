#include <mpi/mpi.hpp>
#include <iostream>

// define a custom complex type
struct my_complex {
  double real;
  double imag;
};

// define an addition for my_complex
inline auto operator+(const my_complex &z1, const my_complex &z2) { return my_complex{.real = z1.real + z2.real, .imag = z1.imag + z2.imag}; }

// define a tie_data function for my_complex to make it MPI compatible
inline auto tie_data(const my_complex &z) { return std::tie(z.real, z.imag); }

int main(int argc, char *argv[]) {
  // initialize MPI environment
  mpi::environment env(argc, argv);
  mpi::communicator world;

  // create complex number z
  my_complex z = {.real = world.rank() + 1.0, .imag = static_cast<double>(world.rank())};

  // sum z over all processes
  auto sum = mpi::reduce(z, world, 0, false, mpi::map_add<my_complex>());

  // define a product for my_complex
  auto my_product = [](const my_complex &z1, const my_complex &z2) {
    return my_complex{.real = z1.real * z2.real - z1.imag * z2.imag, .imag = z1.real * z2.imag + z1.imag * z2.real};
  };

  // multiply z over all processes
  auto product = mpi::reduce(z, world, 0, false, mpi::map_C_function<my_complex, my_product>());

  // print result
  if (world.rank() == 0) {
    std::cout << "sum = (" << sum.real << ", " << sum.imag << ")\n";
    std::cout << "product = (" << product.real << ", " << product.imag << ")\n";
  }
}