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